"""TMM / DESeq2 归一化实现（bulk RNA-seq）"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .. import logging as logg

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
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import numpy2ri
            from rpy2.robjects.packages import importr
            numpy2ri.activate()
            edgeR = importr("edgeR")
        except ImportError:
            logg.warning("rpy2 未安装，使用简化 TMM 实现")
            return self._run_simple(adata)

        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

        # 调用 edgeR::calcNormFactors → TMM
        dge = edgeR.DGEList(counts=X.T)
        dge = edgeR.calcNormFactors(dge, method="TMM")
        norm_factors = np.array(dge.rx2("samples").rx2("norm.factors"))

        X_norm = X / (norm_factors.reshape(1, -1) + 1e-8)

        _save_normalization(adata, X_norm, "tmm")
        return adata

    def _run_simple(self, adata: AnnData) -> AnnData:
        """简化版 TMM（不依赖 R）"""
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        lib_sizes = X.sum(axis=1, keepdims=True)
        X_norm = X / (lib_sizes + 1e-8) * 1e6

        _save_normalization(adata, X_norm, "tmm_simple")
        logg.warning("使用了简化版 TMM（非标准实现），建议安装 rpy2 和 edgeR")
        return adata


class DESeq2Normalizer:
    """DESeq2 归一化

    通过 rpy2 调用 DESeq2::estimateSizeFactors()。

    参考: Love, Huber & Anders (2014), Genome Biology
    """

    def run(self, adata: AnnData, **kwargs: Any) -> AnnData:
        """执行 DESeq2 归一化"""
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import numpy2ri
            from rpy2.robjects.packages import importr
            numpy2ri.activate()
            deseq2 = importr("DESeq2")
        except ImportError:
            logg.warning("rpy2/DESeq2 未安装，使用简化实现")
            return self._run_simple(adata)

        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

        dds = deseq2.DESeqDataSetFromMatrix(
            countData=X.T.astype(int),
            colData=ro.DataFrame({"condition": ro.StrVector(["A"] * X.shape[0])}),
        )
        dds = deseq2.estimateSizeFactors(dds)
        size_factors = np.array(dds.rx2("sizeFactors"))

        X_norm = X / (size_factors.reshape(1, -1) + 1e-8)

        _save_normalization(adata, X_norm, "deseq2")
        return adata

    def _run_simple(self, adata: AnnData) -> AnnData:
        """简化版 DESeq2: median-of-ratios"""
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        geo_means = np.exp(np.mean(np.log(X + 1), axis=0))
        ratios = X / (geo_means + 1e-8)
        size_factors = np.median(ratios, axis=1)
        X_norm = X / (size_factors.reshape(1, -1) + 1e-8)

        _save_normalization(adata, X_norm, "deseq2_simple")
        return adata


def _save_normalization(adata: AnnData, X_norm: np.ndarray, method: str) -> None:
    """将归一化结果保存到 .layers 和 .uns"""
    adata.layers["normalized"] = X_norm.astype(np.float32)
    adata.uns["standardization"] = adata.uns.get("standardization", {})
    adata.uns["standardization"]["normalization"] = {"method": method}
    logg.info(f"{method} 归一化完成")
