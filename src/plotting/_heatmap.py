"""批次效应热图"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData


def plot_batch_heatmap(
    adata: AnnData,
    batch_key: str = "batch",
    n_top_genes: int = 50,
    save: str | None = None,
) -> None:
    """绘制批次效应热图

    展示各批次中高变基因的表达模式。
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    if batch_key not in adata.obs.columns:
        logg.warning(f"未找到批次列 '{batch_key}'，无法绘制热图")
        return

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

    # 选择最高变的基因
    variances = np.var(X, axis=0)
    top_idx = np.argsort(variances)[-n_top_genes:]

    # 按批次聚合
    batches = adata.obs[batch_key].values
    unique_batches = sorted(np.unique(batches))

    batch_means = np.zeros((len(unique_batches), n_top_genes))
    for i, b in enumerate(unique_batches):
        mask = batches == b
        batch_means[i] = X[mask][:, top_idx].mean(axis=0)

    # 绘制
    fig, ax = plt.subplots(figsize=(12, max(4, len(unique_batches) * 0.5)))
    sns.heatmap(
        batch_means,
        xticklabels=False,
        yticklabels=unique_batches,
        cmap="RdBu_r",
        center=0,
        ax=ax,
    )
    ax.set_title(f"Batch Effect Heatmap (top {n_top_genes} variable features)")
    ax.set_ylabel("Batch")
    ax.set_xlabel("Features (sorted by variance)")

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150)
        logg.info(f"批次热图已保存至 {save}")
    else:
        plt.show()
