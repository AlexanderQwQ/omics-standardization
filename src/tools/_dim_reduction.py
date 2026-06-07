"""降维工具：PCA / UMAP / t-SNE 封装"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anndata import AnnData


def run_pca(adata: AnnData, n_comps: int = 50, layer: str | None = "normalized", **kwargs: Any) -> None:
    """在指定 layer 上运行 PCA"""
    import scanpy as sc

    if layer and layer in adata.layers:
        X_orig = adata.X
        adata.X = adata.layers[layer]
        sc.pp.pca(adata, n_comps=n_comps, **kwargs)
        adata.X = X_orig
    else:
        sc.pp.pca(adata, n_comps=n_comps, **kwargs)


def run_umap(adata: AnnData, use_rep: str = "X_pca", **kwargs: Any) -> None:
    """运行 UMAP"""
    import scanpy as sc
    sc.tl.umap(adata, **kwargs)
