"""Scran 池化标准化（单细胞）

基于池化反卷积逻辑的标准化策略:
    1. 将表达特征相近的细胞群进行池化（pooling）
    2. 克服零值导致的局部方差问题
    3. 通过线性方程组逆向解算出独立细胞的大小因子

参考: Lun, Bach & Marioni (2016), Genome Biology
      "Pooling across cells to normalize single-cell RNA sequencing data"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData


class ScranNormalizer:
    """Scran 基于池化的单细胞归一化

    核心机制:
        - 细胞池化 (cell pooling): 将相似表达的细胞随机分组，对每组求和
        - 线性解卷积 (linear deconvolution): 用线性方程组求解每个细胞的大小因子
        - 与简单 library-size 归一化不同，Scran 能更好地处理细胞间的异质性

    参数:
        min_mean: 低表达基因过滤阈值
        pool_sizes: 池化大小列表（自动计算时使用）

    用法:
        norm = ScranNormalizer()
        norm.run(adata)
    """

    def __init__(
        self,
        min_mean: float = 1e-3,
        pool_sizes: list[int] | None = None,
    ) -> None:
        self.min_mean = min_mean
        self.pool_sizes = pool_sizes

    def run(self, adata: AnnData, **kwargs: Any) -> AnnData:
        """执行 Scran 归一化

        Args:
            adata: 输入 AnnData
            **kwargs: 覆盖默认参数

        Returns:
            归一化后的 AnnData (.layers['normalized'])
        """
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

        # 尝试 R 接口（完整实现）
        try:
            return self._run_r(adata, X, **kwargs)
        except Exception as exc:
            logg.warning(f"Scran R 接口失败 ({exc})，使用 Python 原生实现")

        # Python 原生实现
        return self._run_python(adata, X, **kwargs)

    # ------------------------------------------------------------------
    # R 接口（通过 rpy2 调用 scran::computeSumFactors）
    # ------------------------------------------------------------------

    def _run_r(self, adata: AnnData, X: np.ndarray, **kwargs: Any) -> AnnData:
        """通过 rpy2 调用 R scran 包"""
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, pandas2ri
        from rpy2.robjects.packages import importr

        numpy2ri.activate()
        pandas2ri.activate()

        # 导入 R 包
        scran = importr("scran")
        sce_pkg = importr("SingleCellExperiment")
        base = importr("base")
        stats = importr("stats")

        # 1. 过滤低表达基因
        gene_means = np.mean(X, axis=0)
        keep_genes = gene_means >= self.min_mean
        X_filtered = X[:, keep_genes]

        if X_filtered.shape[1] == 0:
            logg.warning("Scran: 所有基因都被过滤，使用全基因集")
            X_filtered = X

        # 2. 构建 SingleCellExperiment 对象
        # counts 矩阵: gene × cell (R convention: rows=genes, cols=cells)
        counts_r = ro.r.matrix(
            ro.FloatVector(X_filtered.T.astype(float).ravel()),
            nrow=X_filtered.shape[1],
            ncol=X_filtered.shape[0],
            byrow=False,
        )

        # 创建 colData (细胞元数据)
        col_data = ro.DataFrame({
            "cell_id": ro.StrVector([f"cell_{i}" for i in range(X_filtered.shape[0])]),
        })

        sce = sce_pkg.SingleCellExperiment(
            assays=base.list(counts=counts_r),
            colData=col_data,
        )

        # 3. 快速预聚类（用于池化分组）
        clusters = scran.quickCluster(sce, min_size=base.as_integer(100))
        sce = sce_pkg.SingleCellExperiment(
            assays=base.list(counts=counts_r),
            colData=ro.DataFrame({
                "cluster": clusters,
            }),
        )

        # 4. 计算 size factors
        sce = scran.computeSumFactors(sce, clusters=clusters, min_mean=ro.FloatVector([self.min_mean]))

        # 5. 提取 size factors
        size_factors = np.array(sce.rx2("sizeFactors"))

        # 6. 应用归一化
        X_norm = X / (size_factors.reshape(-1, 1) + 1e-8)
        X_norm = np.log1p(X_norm * 10000)  # log-normalize to 10k

        self._save(adata, X_norm, "scran")
        return adata

    # ------------------------------------------------------------------
    # Python 原生实现（不依赖 R）
    # ------------------------------------------------------------------

    def _run_python(self, adata: AnnData, X: np.ndarray, **kwargs: Any) -> AnnData:
        """Python 原生 Scran-like 实现

        模仿 Scran 核心逻辑:
            1. 预聚类（k-means）
            2. 在每个簇内随机池化细胞
            3. 计算每个池的 library size
            4. 线性解卷积求解细胞级别因子
        """
        n_obs, n_vars = X.shape

        # 1. 过滤低表达基因
        gene_means = np.mean(X, axis=0)
        keep = gene_means >= self.min_mean
        X_filt = X[:, keep]
        logg.info(f"Scran: 保留 {keep.sum()}/{n_vars} 个基因 (min_mean={self.min_mean})")

        if X_filt.shape[1] < 10:
            logg.warning("Scran: 过滤后基因数不足，回退到 normalize_total")
            return self._fallback_normalize_total(adata)

        # 2. 预聚类（将细胞分为相似表达组）
        n_clusters = min(50, max(2, int(np.sqrt(n_obs))))
        from sklearn.cluster import MiniBatchKMeans

        # log-normalize 用于聚类
        X_log = np.log1p(X_filt)
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=3,
            batch_size=min(1024, n_obs),
        )
        clusters = kmeans.fit_predict(X_log)

        # 3. 在每个簇内进行池化
        pool_sizes = self.pool_sizes or [20, 40, 60, 80]
        all_pools: list[tuple[list[int], ...]] = []  # 每个元素是 (pool_cell_indices, pool_sum, cluster_id)

        rng = np.random.RandomState(42)
        for cluster_id in range(n_clusters):
            cluster_cells = np.where(clusters == cluster_id)[0]
            if len(cluster_cells) < 2:
                continue

            for size in pool_sizes:
                if len(cluster_cells) < size:
                    continue
                rng.shuffle(cluster_cells)
                n_pools = max(1, len(cluster_cells) // size)
                for p in range(n_pools):
                    pool_cells = cluster_cells[p * size:(p + 1) * size]
                    if len(pool_cells) > 1:
                        pool_sum = X_filt[pool_cells, :].sum(axis=0)
                        all_pools.append((pool_cells.tolist(), pool_sum, cluster_id))

        if len(all_pools) < n_obs // 2:
            logg.warning("Scran: 池化组数不足，回退到 normalize_total")
            return self._fallback_normalize_total(adata)

        # 4. 线性反卷积：解方程 A * sf = b
        # A[i,j] = 1 if cell_j in pool_i else 0
        # b[i] = pool_i total count / pool_i mean library size
        # 求解 size_factors sf_j
        n_pools = len(all_pools)

        A = np.zeros((n_pools, n_obs), dtype=np.float32)
        b = np.zeros(n_pools, dtype=np.float32)
        pool_ref = np.median([np.sum(pool[1]) for pool in all_pools])

        for i, (pool_cells, pool_sum, _) in enumerate(all_pools):
            for cell_idx in pool_cells:
                A[i, cell_idx] = 1.0
            b[i] = np.sum(pool_sum) / (pool_ref + 1e-8)

        # 最小二乘求解
        from scipy.linalg import lstsq as scipy_lstsq

        sf, residuals, rank, sv = scipy_lstsq(A, b, lapack_driver="gelsd")
        sf = np.maximum(sf, 1e-6)  # 非负

        # 5. 应用 size factors
        X_norm = X / (sf.reshape(-1, 1) + 1e-8)
        X_norm = np.log1p(X_norm * 10000)

        self._save(adata, X_norm, "scran_python")
        logg.info(f"Scran (Python) 归一化完成 (clusters={n_clusters}, pools={n_pools})")
        return adata

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_normalize_total(self, adata: AnnData) -> AnnData:
        """回退到 scanpy normalize_total"""
        import scanpy as sc

        adata_copy = adata.copy()
        sc.pp.normalize_total(adata_copy, target_sum=1e4)
        sc.pp.log1p(adata_copy)

        X_norm = adata_copy.X.toarray() if hasattr(adata_copy.X, "toarray") else adata_copy.X
        self._save(adata, X_norm, "scran_fallback")
        return adata

    def _save(self, adata: AnnData, X_norm: np.ndarray, method: str) -> None:
        """保存归一化结果"""
        adata.layers["normalized"] = X_norm.astype(np.float32)
        adata.uns["standardization"] = adata.uns.get("standardization", {})
        adata.uns["standardization"]["normalization"] = {
            "method": method,
            "min_mean": self.min_mean,
        }
        logg.info(f"Scran ({method}) 归一化完成")
