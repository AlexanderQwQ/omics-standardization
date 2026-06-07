"""标准化前后 QC 对比图"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData


def plot_qc_comparison(
    adata_before: AnnData,
    adata_after: AnnData,
    save: str | None = None,
) -> None:
    """绘制标准化前后对比图"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 缺失率分布
    X_before = adata_before.X.toarray() if hasattr(adata_before.X, "toarray") else adata_before.X
    X_after = adata_after.X.toarray() if hasattr(adata_after.X, "toarray") else adata_after.X

    axes[0, 0].hist((X_before == 0).mean(axis=1), bins=50, alpha=0.7, label="Before")
    axes[0, 0].hist((X_after == 0).mean(axis=1), bins=50, alpha=0.7, label="After")
    axes[0, 0].set_title("Zero Rate per Cell")
    axes[0, 0].legend()

    # 均值-方差关系
    mean_before = X_before.mean(axis=0)
    var_before = X_before.var(axis=0)
    mean_after = X_after.mean(axis=0)
    var_after = X_after.var(axis=0)

    axes[0, 1].scatter(mean_before, var_before, alpha=0.3, s=1, label="Before")
    axes[0, 1].scatter(mean_after, var_after, alpha=0.3, s=1, label="After")
    axes[0, 1].set_xlabel("Mean")
    axes[0, 1].set_ylabel("Variance")
    axes[0, 1].set_title("Mean-Variance Relationship")
    axes[0, 1].legend()

    # 表达分布
    axes[0, 2].hist(X_before.flatten(), bins=100, alpha=0.5, density=True, label="Before")
    axes[0, 2].hist(X_after.flatten(), bins=100, alpha=0.5, density=True, label="After")
    axes[0, 2].set_title("Expression Distribution")
    axes[0, 2].legend()

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150)
        logg.info(f"QC 对比图已保存至 {save}")
    else:
        plt.show()
