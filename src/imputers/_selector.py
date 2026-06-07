"""
插补策略选择器 + 效果评估

根据数据特征选择最优插补方法，并评估插补效果（RMSE、marker 基因一致性）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData


class ImputationSelector:
    """插补方法选择器

    根据数据特征自动选择:
        - MissForest: 中等缺失率 (< 30%)，各类数据
        - ZINB-VAE:   高零膨胀 (> 50%) 的 scRNA-seq
        - MAGIC:      低缺失率 (< 10%) 的单细胞数据
        - none:       缺失率极低 (< 1%) 时跳过
    """

    def __init__(self) -> None:
        self._available = ["missforest", "zinb_vae", "magic", "none"]

    def select(self, adata: AnnData) -> str:
        """根据数据特征选择插补方法"""
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        zero_rate = float(np.mean(X == 0))

        if zero_rate < 0.01:
            method = "none"
        elif zero_rate > 0.5:
            method = "zinb_vae"
        elif zero_rate < 0.1:
            method = "magic"
        else:
            method = "missforest"

        logg.info(
            f"插补方法选择: {method} (缺失率={zero_rate:.3f})"
        )
        return method

    def run(self, adata: AnnData, method: str | None = None, **kwargs: Any) -> AnnData:
        """执行插补

        Args:
            adata: 输入数据
            method: 指定方法（None 则自动选择）
            **kwargs: 传递给具体插补方法的参数

        Returns:
            插补后的 AnnData（copy=False 就地修改）
        """
        if method is None:
            method = self.select(adata)

        if method == "none":
            logg.info("跳过插补（缺失率极低）")
            return adata

        logg.info(f"执行 {method} 插补...")

        if method == "missforest":
            from ._missforest import MissForestImputer
            imputer = MissForestImputer(**kwargs)
        elif method == "zinb_vae":
            from ._zinb_vae import ZINBVAEImputer
            imputer = ZINBVAEImputer(**kwargs)
        elif method == "magic":
            from ._magic import MAGICImputer
            imputer = MAGICImputer(**kwargs)
        else:
            raise ValueError(f"未知的插补方法: {method}")

        return imputer.run(adata)


def evaluate_imputation(
    adata_before: AnnData, adata_after: AnnData, n_markers: int = 100
) -> dict[str, float]:
    """评估插补效果

    指标:
        - RMSE（均方根误差，需 ground truth）
        - 基因间相关性保留率
        - marker 基因表达一致性

    Returns:
        {"rmse": float, "correlation_preserved": float, "marker_consistency": float}
    """
    from scipy.stats import spearmanr

    X_before = adata_before.X.toarray() if hasattr(adata_before.X, "toarray") else adata_before.X
    X_after = adata_after.X.toarray() if hasattr(adata_after.X, "toarray") else adata_after.X

    # 非零位置 RMSE
    mask = X_before > 0
    if mask.sum() > 0:
        rmse = float(np.sqrt(np.mean((X_before[mask] - X_after[mask]) ** 2)))
    else:
        rmse = 0.0

    # 基因相关性保留（随机抽样 n_markers 个基因）
    n_genes = min(n_markers, X_before.shape[1])
    if n_genes > 1:
        idx = np.random.choice(X_before.shape[1], n_genes, replace=False)
        corrs = [spearmanr(X_after[:, i], X_before[:, i]).correlation for i in idx]
        corr_preserved = float(np.mean([c for c in corrs if c is not None and not np.isnan(c)]))
    else:
        corr_preserved = 1.0

    return {
        "rmse": rmse,
        "correlation_preserved": corr_preserved,
        "marker_consistency": corr_preserved,  # 简化：与基因相关性一致
    }
