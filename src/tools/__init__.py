"""
工具命名空间（tl）

降维、嵌入、聚类、评估工具集。

用法:
    from omics_standardization import tl

    tl.pca(adata)
    tl.umap(adata)
    tl.evaluate(adata)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anndata import AnnData


def pca(adata: AnnData, n_comps: int = 50, **kwargs: Any) -> None:
    """PCA 降维

    Args:
        adata: 输入 AnnData
        n_comps: 主成分数量
        **kwargs: 传递给 scanpy.pp.pca() 的参数
    """
    import scanpy as sc

    # 如果有归一化层，使用它
    if "normalized" in adata.layers:
        # 临时交换 .X
        import numpy as np
        X_orig = adata.X.copy() if hasattr(adata.X, "toarray") else adata.X
        adata.X = adata.layers["normalized"].copy()

    sc.pp.pca(adata, n_comps=n_comps, **kwargs)


def umap(adata: AnnData, **kwargs: Any) -> None:
    """UMAP 嵌入

    Args:
        adata: 输入 AnnData（需要 .obsm["X_pca"]）
        **kwargs: 传递给 scanpy.tl.umap() 的参数
    """
    import scanpy as sc

    sc.tl.umap(adata, **kwargs)


def evaluate(adata: AnnData) -> dict[str, float]:
    """标准化效果评估

    返回质量指标字典（RMSE、相关性保留率、批次混合度等）。

    Returns:
        {"rmse": float, "correlation_preserved": float, "batch_mixing": float, ...}
    """
    from ._evaluation import run_evaluation
    return run_evaluation(adata)


__all__ = [
    "pca",
    "umap",
    "evaluate",
]
