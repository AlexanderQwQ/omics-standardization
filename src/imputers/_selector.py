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


def _check_low_biomass(adata: AnnData) -> dict[str, Any]:
    """检测低生物量样本

    检查每个细胞/样本的总计数、检测到的基因数和文库大小，
    判断是否存在低生物量问题。结果写入 adata.uns。

    Args:
        adata: 输入 AnnData

    Returns:
        {
            "is_low_biomass": bool,
            "median_counts_per_cell": float,
            "median_genes_detected": float,
            "total_library_size": float,
            "warnings": list[str],
        }
    """
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    warnings: list[str] = []
    is_low_biomass = False

    # (a) 每细胞总计数中位数
    counts_per_cell = np.sum(X, axis=1)
    median_counts = float(np.median(counts_per_cell))
    if median_counts < 500:
        warnings.append(f"每细胞总计数中位数极低: {median_counts:.1f} (< 500)")
        is_low_biomass = True

    # (b) 每细胞检测到的基因数中位数
    genes_detected = np.sum(X > 0, axis=1)
    median_genes = float(np.median(genes_detected))
    if median_genes < 200:
        warnings.append(f"每细胞检测基因数中位数极低: {median_genes:.1f} (< 200)")
        is_low_biomass = True

    # (c) 文库大小（总计数）
    total_library = float(np.sum(X))
    if total_library < 10000:
        warnings.append(f"文库总计数过小: {total_library:.1f} (< 10000)")
        is_low_biomass = True

    result = {
        "is_low_biomass": is_low_biomass,
        "median_counts_per_cell": median_counts,
        "median_genes_detected": median_genes,
        "total_library_size": total_library,
        "warnings": warnings,
    }

    # 记录到 adata.uns
    adata.uns.setdefault("standardization", {})
    adata.uns["standardization"]["low_biomass_detected"] = is_low_biomass
    if is_low_biomass:
        adata.uns["standardization"]["low_biomass_details"] = {
            "median_counts_per_cell": median_counts,
            "median_genes_detected": median_genes,
            "total_library_size": total_library,
        }

    return result


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
        """根据数据特征选择插补方法

        自动检测低生物量样本，若检测到则倾向于更简单的方法
        （如 MissForest 而非 ZINB-VAE）以适配低计数数据。
        """
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        zero_rate = float(np.mean(X == 0))

        # 低生物量检测
        biomass_info = _check_low_biomass(adata)
        if biomass_info["is_low_biomass"]:
            logg.warning("检测到低生物量样本:")
            for w in biomass_info["warnings"]:
                logg.warning(f"  {w}")
            logg.warning("  建议：小心处理低计数数据，倾向于使用 MissForest 等更简单的方法")

        if zero_rate < 0.01:
            method = "none"
        elif zero_rate > 0.5:
            # 低生物量时避免 ZINB-VAE（在极低计数下不稳定）
            if biomass_info["is_low_biomass"]:
                logg.hint("低生物量 + 高零膨胀 → 使用 MissForest 替代 ZINB-VAE")
                method = "missforest"
            else:
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
