"""TMM / DESeq2 归一化实现（bulk RNA-seq）"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import _logging as logg
from ._utils import check_r_available, ensure_dense

if TYPE_CHECKING:
    from anndata import AnnData


class TMMNormalizer:
    """TMM（Trimmed Mean of M-values）归一化

    通过 rpy2 调用 edgeR 的 calcNormFactors()。

    参考: Robinson & Oshlack (2010), Genome Biology
    """

    def __init__(self, ref_column: str | None = None) -> None:
        self.ref_column = ref_column

    def run(self, adata: AnnData, **kwargs: Any) -> AnnData:
        """执行 TMM 归一化"""
        if check_r_available("edgeR"):
            return self._run_r(adata)
        else:
            logg.warning("rpy2/edgeR 未安装，使用 Python 原生 TMM 实现")
            return self._run_tmm_python(adata)

    def _run_r(self, adata: AnnData) -> AnnData:
        """通过 rpy2 调用 edgeR TMM"""
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr

        numpy2ri.activate()
        edgeR = importr("edgeR")

        X = ensure_dense(adata.X)

        # 调用 edgeR::calcNormFactors -> TMM
        dge = edgeR.DGEList(counts=X.T)
        dge = edgeR.calcNormFactors(dge, method="TMM")
        norm_factors = np.array(dge.rx2("samples").rx2("norm.factors"))

        X_norm = X / (norm_factors.reshape(-1, 1) + 1e-8)

        _save_normalization(adata, X_norm, "tmm")
        return adata

    def _run_tmm_python(self, adata: AnnData) -> AnnData:
        """Python 原生 TMM 实现

        标准 TMM 算法:
        1. 计算伪参考样本（各基因的几何均值）
        2. 对每个样本计算 M-value 和 A-value
        3. Trim M 值 30% 和 A 值 5% 双尾
        4. 计算加权裁剪均值作为归一化因子
        5. 应用归一化因子
        """
        X = ensure_dense(adata.X)

        try:
            n_obs, n_vars = X.shape

            # 1. 伪参考样本: 每个基因的几何均值
            X_safe = X + 1e-8
            ref = np.exp(np.mean(np.log(X_safe), axis=0))  # shape: (n_vars,)
            ref_lib = ref.sum()

            # 2-4. 对每个样本计算归一化因子
            norm_factors = np.ones(n_obs)

            for i in range(n_obs):
                lib_i = X[i, :].sum()
                # M-values: log2 ratios vs reference
                M = np.log2((X[i, :] / lib_i) + 1e-8) - np.log2((ref / ref_lib) + 1e-8)
                # A-values: mean log2 intensities
                A = 0.5 * (np.log2((X[i, :] / lib_i) + 1e-8) + np.log2((ref / ref_lib) + 1e-8))

                # Trim: 30% M-values, 5% A-values (双尾对称)
                M_lower, M_upper = np.percentile(M, [30, 70])
                A_lower, A_upper = np.percentile(A, [5, 95])

                keep = (M >= M_lower) & (M <= M_upper) & (A >= A_lower) & (A <= A_upper)

                if keep.sum() < 10:
                    keep = np.ones(n_vars, dtype=bool)

                # 加权均值: 权重为逆渐近方差
                w = 1.0 / (1.0 / (X[i, :] + 1e-8) + 1.0 / (ref + 1e-8))
                w[~keep] = 0.0
                w_sum = w.sum()
                if w_sum > 0:
                    norm_factors[i] = 2.0 ** (np.average(M, weights=w))
                else:
                    norm_factors[i] = 1.0

            # 5. 应用归一化因子
            X_norm = X / (norm_factors.reshape(-1, 1) + 1e-8)

            _save_normalization(adata, X_norm, "tmm (python)")
            return adata

        except Exception:
            logg.warning("Python TMM 计算失败，回退到 CPM")
            return self._run_cpm_fallback(adata)

    def _run_cpm_fallback(self, adata: AnnData) -> AnnData:
        """最后手段: CPM（Counts Per Million）"""
        X = ensure_dense(adata.X)
        lib_sizes = X.sum(axis=1, keepdims=True)
        X_norm = X / (lib_sizes + 1e-8) * 1e6

        _save_normalization(adata, X_norm, "tmm (cpm fallback)")
        logg.warning("使用了 CPM 作为 TMM 最后回退（非标准实现），建议安装 rpy2 和 edgeR")
        return adata


class DESeq2Normalizer:
    """DESeq2 归一化

    通过 rpy2 调用 DESeq2::estimateSizeFactors()。

    参考: Love, Huber & Anders (2014), Genome Biology
    """

    def __init__(self, design: list[str] | None = None) -> None:
        """初始化 DESeq2 归一化器

        Args:
            design: 实验设计公式的列名列表，来自 adata.obs。
                    None 时使用截距模型 ~1（仅估计 size factors，无条件间比较）。
                    例如 design=["condition", "batch"]。
        """
        self.design = design

    def run(self, adata: AnnData, **kwargs: Any) -> AnnData:
        """执行 DESeq2 归一化"""
        if check_r_available("DESeq2"):
            return self._run_r(adata)
        else:
            logg.warning("rpy2/DESeq2 未安装，使用简化实现")
            return self._run_simple(adata)

    def _run_r(self, adata: AnnData) -> AnnData:
        """通过 rpy2 调用 DESeq2"""
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr

        numpy2ri.activate()
        base = importr("base")

        X = ensure_dense(adata.X).astype(int)

        # 构建设计公式
        if self.design is None:
            # 截距模型: ~1
            design_formula = "~1"
            col_data = ro.DataFrame(
                {
                    "intercept": ro.FloatVector([1.0] * X.shape[0]),
                }
            )
            method_label = "deseq2 (~1)"
        else:
            # 从 adata.obs 提取设计列
            design_str = " + ".join(self.design)
            design_formula = f"~{design_str}"
            col_data_dict = {}
            for col_name in self.design:
                if col_name not in adata.obs.columns:
                    raise ValueError(
                        f"DESeq2: 设计列 '{col_name}' 不在 adata.obs 中。"
                        f" 可用列: {list(adata.obs.columns)}"
                    )
                col_vals = adata.obs[col_name].values
                # R factor 用于分类变量
                if col_vals.dtype.kind in ("U", "S", "O"):
                    col_data_dict[col_name] = ro.StrVector([str(v) for v in col_vals])
                else:
                    col_data_dict[col_name] = ro.FloatVector(col_vals.astype(float))
            col_data = ro.DataFrame(col_data_dict)
            method_label = f"deseq2 ({design_formula})"

        deseq2 = importr("DESeq2")

        dds = deseq2.DESeqDataSetFromMatrix(
            countData=X.T.astype(int),
            colData=col_data,
            design=base.as_formula(design_formula),
        )
        dds = deseq2.estimateSizeFactors(dds)
        size_factors = np.array(dds.rx2("sizeFactors"))

        X_norm = X / (size_factors.reshape(-1, 1) + 1e-8)

        _save_normalization(adata, X_norm, method_label)
        return adata

    def _run_simple(self, adata: AnnData) -> AnnData:
        """简化版 DESeq2: median-of-ratios"""
        X = ensure_dense(adata.X)
        geo_means = np.exp(np.mean(np.log(X + 1), axis=0))
        ratios = X / (geo_means + 1e-8)
        size_factors = np.median(ratios, axis=1)
        X_norm = X / (size_factors.reshape(-1, 1) + 1e-8)

        _save_normalization(adata, X_norm, "deseq2_simple")
        return adata


def _save_normalization(adata: AnnData, X_norm: np.ndarray, method: str) -> None:
    """将归一化结果保存到 .layers 和 .uns"""
    adata.layers["normalized"] = X_norm.astype(np.float32)
    adata.uns["standardization"] = adata.uns.get("standardization", {})
    adata.uns["standardization"]["normalization"] = {"method": method}
    logg.info(f"{method} 归一化完成")
