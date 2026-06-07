"""批次校正策略选择器"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData


class BatchCorrectionSelector:
    """批次校正方法选择器

    根据数据特征自动选择:
        - ComBat:   小批次 (< 5 批次)，bulk 数据
        - Harmony:  大批次 (≥ 5)，单细胞数据
        - DANN:     复杂批次效应（推荐 PyTorch 环境）
        - none:     单批次或大批次效应不明显时跳过
    """

    def __init__(self) -> None:
        self._available = ["combat", "harmony", "dann", "none"]

    def select(self, adata: AnnData, batch_key: str = "batch") -> str:
        """根据数据特征选择批次校正方法

        Args:
            adata: 输入数据
            batch_key: obs 中的批次标签列名

        Returns:
            方法名称: "combat" | "harmony" | "dann" | "none"
        """
        if batch_key not in adata.obs.columns:
            logg.warning(f"未找到批次列 '{batch_key}'，跳过批次校正")
            return "none"

        n_batches = adata.obs[batch_key].nunique()
        n_obs = adata.n_obs

        if n_batches <= 1:
            method = "none"
        elif n_batches < 5:
            method = "combat"
        elif n_obs > 50000:
            method = "dann"
        else:
            method = "harmony"

        logg.info(
            f"批次校正方法选择: {method} "
            f"(批次={n_batches}, 细胞数={n_obs})"
        )
        return method

    def run(
        self, adata: AnnData, method: str | None = None, batch_key: str = "batch", **kwargs: Any
    ) -> AnnData:
        """执行批次校正

        Args:
            adata: 输入数据
            method: 指定方法（None 则自动选择）
            batch_key: 批次标签列名
            **kwargs: 传递给具体校正方法的参数

        Returns:
            校正后的 AnnData
        """
        if method is None:
            method = self.select(adata, batch_key)

        if method == "none":
            logg.info("跳过批次校正")
            return adata

        logg.info(f"执行 {method} 批次校正...")

        if method == "combat":
            from ._combat import ComBatCorrector
            corrector = ComBatCorrector(**kwargs)
        elif method == "harmony":
            from ._harmony import HarmonyCorrector
            corrector = HarmonyCorrector(**kwargs)
        elif method == "dann":
            from ._dann import DANCorrector
            corrector = DANCorrector(**kwargs)
        else:
            raise ValueError(f"未知的批次校正方法: {method}")

        return corrector.run(adata, batch_key=batch_key)
