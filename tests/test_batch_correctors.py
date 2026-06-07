"""批次校正模块测试"""

import numpy as np
import pytest
from batch_correctors._combat import ComBatCorrector
from batch_correctors._harmony import HarmonyCorrector
from batch_correctors._selector import BatchCorrectionSelector


class TestBatchCorrectionSelector:
    """批次校正选择器测试"""

    def test_select_single_batch(self, small_adata_no_batch) -> None:
        selector = BatchCorrectionSelector()
        # 无批次列 → 应返回 none
        result = selector.select(small_adata_no_batch, batch_key="nonexistent")
        assert result == "none"

    def test_select_two_batches(self, small_adata) -> None:
        selector = BatchCorrectionSelector()
        result = selector.select(small_adata, batch_key="batch")
        # 2 个批次 → combat
        assert result in ["combat", "none"]

    def test_select_methods_are_valid(self, small_adata) -> None:
        selector = BatchCorrectionSelector()
        result = selector.select(small_adata, batch_key="batch")
        assert result in selector._available


class TestComBatCorrector:
    """ComBat 批次校正测试"""

    def test_run_simple_combat(self, small_adata) -> None:
        X = small_adata.X.toarray() if hasattr(small_adata.X, "toarray") else small_adata.X
        batches = small_adata.obs["batch"].values

        corrector = ComBatCorrector()
        X_corrected = corrector._simple_combat(X, batches)

        assert X_corrected.shape == X.shape
        assert not np.any(np.isnan(X_corrected))

    def test_run_scanpy_combat(self, small_adata) -> None:
        """调用 scanpy combat（如果可用）"""
        corrector = ComBatCorrector()
        result = corrector.run(small_adata, batch_key="batch")
        assert "X_corrected" in result.obsm
        assert result.obsm["X_corrected"].shape == small_adata.shape

    def test_batch_correction_reduces_bias(self, small_adata) -> None:
        """校正后批次间均值差异应减小"""
        X = small_adata.X.toarray() if hasattr(small_adata.X, "toarray") else small_adata.X
        batches = small_adata.obs["batch"].values

        corrector = ComBatCorrector()
        X_corrected = corrector._simple_combat(X, batches)

        # 计算校正前后批次间差异
        unique_batches = np.unique(batches)
        def batch_diff(X_arr):
            means = [X_arr[batches == b].mean(axis=0) for b in unique_batches]
            return float(np.mean(np.abs(means[0] - means[1])))

        diff_before = batch_diff(X)
        diff_after = batch_diff(X_corrected)
        assert diff_after <= diff_before + 1e-6  # 应减小或持平


class TestHarmonyCorrector:
    """Harmony 批次校正测试"""

    def test_run_with_fallback(self, small_adata) -> None:
        corrector = HarmonyCorrector()
        result = corrector.run(small_adata, batch_key="batch")
        # 无论 scanpy external 是否可用，都应成功
        assert result is not None
        assert "standardization" in result.uns
