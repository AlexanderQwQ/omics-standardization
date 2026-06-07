"""插补模块测试"""

import numpy as np
import pytest
from imputers._selector import ImputationSelector, evaluate_imputation
from imputers._missforest import MissForestImputer


class TestImputationSelector:
    """插补选择器测试"""

    def test_select_high_missing(self, high_missing_adata) -> None:
        selector = ImputationSelector()
        method = selector.select(high_missing_adata)
        # 高缺失率 (>50%) 应推荐 zinb_vae
        assert method in ["zinb_vae", "missforest", "magic", "none"]

    def test_select_low_missing(self, small_adata) -> None:
        selector = ImputationSelector()
        method = selector.select(small_adata)
        # 中等缺失率 → 推荐 MAGIC 或 missforest
        assert method in ["magic", "missforest", "none"]


class TestMissForest:
    """MissForest 插补器测试"""

    def test_impute_no_nan(self, small_adata) -> None:
        imputer = MissForestImputer(n_estimators=10, max_iter=2, random_state=42)
        result = imputer.run(small_adata)
        assert "imputed" in result.layers
        assert result.layers["imputed"].shape == small_adata.shape
        assert not np.any(np.isnan(result.layers["imputed"]))


class TestEvaluateImputation:
    """插补效果评估测试"""

    def test_evaluate(self, small_adata) -> None:
        metrics = evaluate_imputation(small_adata, small_adata)
        assert "rmse" in metrics
        assert "correlation_preserved" in metrics
        assert metrics["rmse"] >= 0
