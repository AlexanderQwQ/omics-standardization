"""
可视化命名空间（pl）

标准化前后对比可视化。

用法:
    from omics_standardization import pl

    pl.qc_before_after(adata_before, adata_after)
    pl.batch_heatmap(adata)
    pl.embedding(adata, color="batch")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


def qc_before_after(
    adata_before: AnnData,
    adata_after: AnnData,
    save: str | None = None,
) -> None:
    """标准化前后 QC 对比图

    Args:
        adata_before: 处理前数据
        adata_after: 处理后数据
        save: 保存路径（可选）
    """
    from ._qc import plot_qc_comparison
    plot_qc_comparison(adata_before, adata_after, save=save)


def batch_heatmap(adata: AnnData, batch_key: str = "batch", save: str | None = None) -> None:
    """批次效应热图

    Args:
        adata: AnnData 数据
        batch_key: 批次标签列名
        save: 保存路径（可选）
    """
    from ._heatmap import plot_batch_heatmap
    plot_batch_heatmap(adata, batch_key=batch_key, save=save)


def embedding(
    adata: AnnData,
    basis: str = "umap",
    color: str | list[str] | None = None,
    save: str | None = None,
) -> None:
    """降维嵌入可视化

    Args:
        adata: AnnData 数据
        basis: 嵌入键名（"pca" | "umap" | "X_umap"）
        color: 着色依据
        save: 保存路径（可选）
    """
    from ._embedding import plot_embedding
    plot_embedding(adata, basis=basis, color=color, save=save)


__all__ = [
    "qc_before_after",
    "batch_heatmap",
    "embedding",
]
