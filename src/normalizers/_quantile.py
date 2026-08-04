"""Quantile / VSN 归一化（代谢/蛋白质组）"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import _logging as logg
from ._utils import check_r_available, ensure_dense

if TYPE_CHECKING:
    from anndata import AnnData


class QuantileNormalizer:
    """分位数归一化

    使所有样本的特征分布一致，常用在代谢组/蛋白质组数据中。

    参考: Bolstad et al. (2003), Bioinformatics
    """

    def run(self, adata: AnnData, **kwargs: Any) -> AnnData:
        """执行分位数归一化"""
        X = ensure_dense(adata.X)

        # 对每列排序，取平均值作为参考分布
        X_sorted = np.sort(X, axis=0)
        reference = np.mean(X_sorted, axis=1)

        # 将每个样本的值映射到参考分布
        X_norm = np.zeros_like(X)
        for i in range(X.shape[0]):
            ranks = np.argsort(np.argsort(X[i, :]))
            X_norm[i, :] = reference[ranks.astype(int)]

        adata.layers["normalized"] = X_norm.astype(np.float32)
        adata.uns["standardization"] = adata.uns.get("standardization", {})
        adata.uns["standardization"]["normalization"] = {"method": "quantile"}

        logg.info("分位数归一化完成")
        return adata


class VSNNormalizer:
    """VSN（Variance Stabilizing Normalization）归一化

    通过 rpy2 调用 limma::normalizeVSN()。

    参考: Huber et al. (2002), Bioinformatics
    """

    def run(self, adata: AnnData, **kwargs: Any) -> AnnData:
        """执行 VSN 归一化"""
        X = ensure_dense(adata.X)

        if check_r_available("limma"):
            return self._run_r(adata, X)

        # 回退: arcsinh-based variance stabilization
        logg.warning("rpy2/limma 未安装，使用 arcsinh 方差稳定化")
        X_norm, method_detail = self._run_arcsinh(X)
        if X_norm is None:
            logg.warning("arcsinh 参数拟合失败，回退到 log2 变换")
            X_norm = np.log2(X + 1)
            method_detail = "vsn (log2 fallback)"

        adata.layers["normalized"] = X_norm.astype(np.float32)
        adata.uns["standardization"] = adata.uns.get("standardization", {})
        adata.uns["standardization"]["normalization"] = {"method": method_detail}

        logg.info(f"VSN 归一化完成 ({method_detail})")
        return adata

    def _run_r(self, adata: AnnData, X: np.ndarray) -> AnnData:
        """通过 rpy2 调用 limma::normalizeVSN"""
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr

        numpy2ri.activate()
        limma = importr("limma")
        X_norm = np.array(limma.normalizeVSN(X.T)).T

        adata.layers["normalized"] = X_norm.astype(np.float32)
        adata.uns["standardization"] = adata.uns.get("standardization", {})
        adata.uns["standardization"]["normalization"] = {"method": "vsn (rpy2/limma)"}

        logg.info("VSN 归一化完成 (vsn (rpy2/limma))")
        return adata

    def _run_arcsinh(self, X: np.ndarray) -> tuple[np.ndarray | None, str]:
        """arcsinh 方差稳定化: arcsinh(a + b*X)

        自动调优 a, b 以最小化各特征均值与方差之间的相关性。
        若调优失败返回 (None, "")。
        """
        try:
            from scipy.optimize import minimize_scalar

            # a: 小偏移，处理零值；使用正值数据的 5% 分位数
            X_pos = X[X > 0] if np.any(X > 0) else X.ravel()
            a_init = float(np.percentile(X_pos, 5)) if len(X_pos) > 0 else 0.01
            a_init = max(a_init, 0.001)

            def _corr_mean_var(log_b: float, a_val: float, X_ref: np.ndarray) -> float:
                b_val = np.exp(log_b)
                X_t = np.arcsinh(a_val + b_val * X_ref)
                means = np.mean(X_t, axis=0)
                vars_ = np.var(X_t, axis=0)
                mask = vars_ > 1e-10
                if mask.sum() < 10:
                    return 1.0
                corr = np.corrcoef(means[mask], vars_[mask])[0, 1]
                return abs(float(corr))

            # 优化 b（log 空间保证正值）
            res = minimize_scalar(
                lambda log_b: _corr_mean_var(log_b, a_init, X),
                bounds=(-5, 5),
                method="bounded",
            )
            b_opt = np.exp(res.x)

            X_norm = np.arcsinh(a_init + b_opt * X)
            return X_norm, "vsn (arcsinh fallback)"

        except Exception:
            return None, ""
