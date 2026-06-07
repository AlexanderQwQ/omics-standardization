"""降维嵌入可视化"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData


def plot_embedding(
    adata: AnnData,
    basis: str = "umap",
    color: str | list[str] | None = None,
    save: str | None = None,
) -> None:
    """绘制降维嵌入散点图

    Args:
        adata: AnnData 数据
        basis: 嵌入键名
        color: 着色依据（obs 列名）
        save: 保存路径
    """
    import scanpy as sc

    if color is None:
        if "batch" in adata.obs.columns:
            color = "batch"
        else:
            color = None

    sc.pl.embedding(adata, basis=basis, color=color, show=(save is None))

    if save:
        import matplotlib.pyplot as plt
        plt.savefig(save, dpi=150)
        logg.info(f"嵌入图已保存至 {save}")
