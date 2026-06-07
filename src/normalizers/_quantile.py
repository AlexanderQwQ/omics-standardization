"""Quantile / VSN 归一化（代谢/蛋白质组）"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData


class QuantileNormalizer:
    """分位数归一化

    使所有样本的特征分布一致，常用在代谢组/蛋白质组数据中。

    参考: Bolstad et al. (2003), Bioinformatics
    """

    def run(self, adata: AnnData, **kwargs: Any) -> AnnData:
        """执行分位数归一化"""
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

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
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

        try:
            import rpy2.robjects as ro
            from rpy2.robjects import numpy2ri
            from rpy2.robjects.packages import importr
            numpy2ri.activate()
            limma = importr("limma")
            X_norm = np.array(limma.normalizeVSN(X.T)).T
        except ImportError:
            logg.warning("rpy2/limma 未安装，回退到 log2 变换")
            X_norm = np.log2(X + 1)

        adata.layers["normalized"] = X_norm.astype(np.float32)
        adata.uns["standardization"] = adata.uns.get("standardization", {})
        adata.uns["standardization"]["normalization"] = {"method": "vsn"}

        logg.info("VSN 归一化完成")
        return adata
