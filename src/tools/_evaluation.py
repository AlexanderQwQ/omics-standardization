"""标准化效果评估"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData


def run_evaluation(adata: AnnData) -> dict[str, float]:
    """评估标准化处理效果

    指标:
        - rmse: 均方根误差（需原始数据）
        - n_features: 处理后的特征数
        - batch_mixing: 批次混合度（越高越好）

    Returns:
        评估指标字典
    """
    metrics: dict[str, float] = {}

    # 特征数
    metrics["n_features"] = float(adata.n_vars)

    # RMSE（如果存在原始数据层）
    if "raw" in adata.layers:
        X_raw = adata.layers["raw"]
        X_cur = adata.X
        if hasattr(X_raw, "toarray"):
            X_raw = X_raw.toarray()
        if hasattr(X_cur, "toarray"):
            X_cur = X_cur.toarray()
        mask = X_raw > 0
        if mask.sum() > 0:
            metrics["rmse"] = float(np.sqrt(np.mean((X_raw[mask] - X_cur[mask]) ** 2)))

    # 批次混合度（简化：用校正前后批次标签的分布均匀度）
    if "batch" in adata.obs.columns:
        batch_counts = adata.obs["batch"].value_counts()
        batch_mixing = float(batch_counts.min() / batch_counts.max()) if len(batch_counts) > 1 else 1.0
        metrics["batch_mixing"] = batch_mixing

    logg.info(f"评估完成: {metrics}")
    return metrics
