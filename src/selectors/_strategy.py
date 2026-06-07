"""
策略推荐器：基于 RandomForest 推荐最优处理方法

输入特征:
    - 数据模态
    - 缺失率
    - 样本量
    - 特征量
    - 批次数量

输出推荐:
    - 插补方法: missforest | zinb_vae | magic | none
    - 归一化方法: tmm | deseq2 | scran | quantile | vsn
    - 批次校正方法: combat | harmony | dann | none
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData

# 预定义的策略映射（无训练模型时的 fallback）
_FALLBACK_STRATEGIES: dict[str, dict[str, str]] = {
    "scrna": {"imputation": "magic", "normalization": "scran", "batch": "harmony"},
    "bulk_rna": {"imputation": "none", "normalization": "tmm", "batch": "combat"},
    "proteomics": {"imputation": "missforest", "normalization": "quantile", "batch": "combat"},
    "metabolomics": {"imputation": "missforest", "normalization": "quantile", "batch": "combat"},
    "atac": {"imputation": "none", "normalization": "scran", "batch": "harmony"},
}


def _build_feature_vector(
    modality: str, adata: AnnData, batch_key: str = "batch"
) -> np.ndarray:
    """构建用于策略推荐的归一化特征向量"""
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    n_obs, n_vars = adata.shape
    missing_rate = float(np.mean(X == 0))
    n_batches = len(np.unique(adata.obs[batch_key])) if batch_key in adata.obs else 1

    # 对分类变量做简单编码
    modality_map = {"scrna": 0, "bulk_rna": 1, "proteomics": 2, "metabolomics": 3, "atac": 4}
    modality_code = modality_map.get(modality, 0)

    return np.array([
        modality_code,
        missing_rate,
        np.log1p(n_obs),
        np.log1p(n_vars),
        n_batches,
    ]).reshape(1, -1)


class StrategySelector:
    """基于 RandomForest 的策略推荐器

    根据数据特征推荐:
        - 插补方法
        - 归一化方法
        - 批次校正方法
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._models: dict[str, Any] = {}

    def fit(self, X: np.ndarray, y_impute: np.ndarray, y_norm: np.ndarray, y_batch: np.ndarray) -> StrategySelector:
        """训练三个独立的 RandomForest 分类器

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y_impute: 插补方法标签
            y_norm: 归一化方法标签
            y_batch: 批次校正方法标签
        """
        from sklearn.ensemble import RandomForestClassifier

        self._models["imputation"] = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        ).fit(X, y_impute)

        self._models["normalization"] = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        ).fit(X, y_norm)

        self._models["batch"] = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        ).fit(X, y_batch)

        return self

    def predict(self, modality: str, adata: AnnData, batch_key: str = "batch") -> dict[str, str]:
        """预测最优处理策略

        Returns:
            {"imputation": str, "normalization": str, "batch": str}
        """
        features = _build_feature_vector(modality, adata, batch_key)

        if not self._models:
            logg.warning("StrategySelector 未训练，使用 fallback 策略")
            return self._fallback(modality)

        return {
            "imputation": str(self._models["imputation"].predict(features)[0]),
            "normalization": str(self._models["normalization"].predict(features)[0]),
            "batch": str(self._models["batch"].predict(features)[0]),
        }

    def _fallback(self, modality: str) -> dict[str, str]:
        """无训练模型时的 fallback 策略"""
        strategy = _FALLBACK_STRATEGIES.get(modality, _FALLBACK_STRATEGIES["scrna"])
        logg.info(f"Fallback 策略 [{modality}]: impute={strategy['imputation']}, "
                  f"norm={strategy['normalization']}, batch={strategy['batch']}")
        return strategy


def recommend_strategy(adata: AnnData, modality: str | None = None) -> dict[str, str]:
    """便捷函数：推荐单个 AnnData 的最优处理策略

    优先使用已训练的 RF 模型，否则使用 fallback 策略表。

    Args:
        adata: 输入数据
        modality: 已知模态（None 则自动检测）

    Returns:
        {"imputation": str, "normalization": str, "batch": str}
    """
    if modality is None:
        from ._modality import detect_modality
        modality = detect_modality(adata)

    selector = StrategySelector()

    # 尝试加载持久化模型
    from ._persistence import load_strategy_models

    persisted = load_strategy_models()
    if all(m is not None for m in persisted.values()):
        selector._models = persisted
        logg.info("使用已训练的 RF 模型进行策略推荐")
        return selector.predict(modality, adata)

    # 未训练模型时使用 fallback
    return selector._fallback(modality)
