"""ComBat 经验贝叶斯批次校正"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData


class ComBatCorrector:
    """ComBat 批次效应校正

    经验贝叶斯方法，调整每个基因/特征的均值和方差以消除批次效应。

    参考: Johnson, Li & Rabinovic (2007), Biostatistics

    Parameters:
        parametric: 是否使用参数化方法
    """

    def __init__(self, parametric: bool = True) -> None:
        self.parametric = parametric

    def run(self, adata: AnnData, batch_key: str = "batch", **kwargs: Any) -> AnnData:
        """执行 ComBat 校正

        Args:
            adata: 输入 AnnData
            batch_key: obs 中的批次标签列名
            **kwargs: 覆盖默认参数

        Returns:
            校正后的 AnnData
        """
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        batches = adata.obs[batch_key].values

        try:
            # 尝试使用 scanpy 内置的 combat
            import scanpy as sc
            adata_copy = adata.copy()
            sc.pp.combat(adata_copy, key=batch_key, parametric=self.parametric)
            X_corrected = adata_copy.X.toarray() if hasattr(adata_copy.X, "toarray") else adata_copy.X
        except Exception:
            logg.warning("scanpy combat 失败，使用简化实现")
            X_corrected = self._simple_combat(X, batches)

        # 保存到 .obsm
        adata.obsm["X_corrected"] = X_corrected.astype(np.float32)
        adata.uns["standardization"] = adata.uns.get("standardization", {})
        adata.uns["standardization"]["batch_correction"] = {
            "method": "combat",
            "batch_key": batch_key,
        }

        logg.info("ComBat 批次校正完成")
        return adata

    def _simple_combat(self, X: np.ndarray, batches: np.ndarray) -> np.ndarray:
        """简化 ComBat：减去批次均值 + 全局均值"""
        unique_batches = np.unique(batches)
        global_mean = X.mean(axis=0)

        X_corrected = X.copy()
        for batch in unique_batches:
            mask = batches == batch
            batch_mean = X[mask].mean(axis=0)
            X_corrected[mask] = X[mask] - batch_mean + global_mean

        return X_corrected
