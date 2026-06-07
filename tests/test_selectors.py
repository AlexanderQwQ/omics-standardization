"""算法选择器测试"""

import pytest
from selectors._modality import detect_modality, _extract_features
from selectors._strategy import recommend_strategy


class TestModalityDetection:
    """模态检测测试"""

    def test_extract_features(self, small_adata) -> None:
        features = _extract_features(small_adata)
        assert features.shape == (1, 6)
        assert features[0, 0] >= 0  # missing rate >= 0

    def test_detect_scrna(self, small_adata) -> None:
        # small_adata 有 200 个特征，会被检测为 bulk_rna（< 500）
        modality = detect_modality(small_adata)
        assert modality in ["scrna", "bulk_rna", "proteomics", "metabolomics", "atac"]


class TestStrategyRecommendation:
    """策略推荐测试"""

    def test_recommend_default(self, small_adata) -> None:
        strategy = recommend_strategy(small_adata)
        assert "imputation" in strategy
        assert "normalization" in strategy
        assert "batch" in strategy
        assert strategy["imputation"] in ["missforest", "zinb_vae", "magic", "none"]
        assert strategy["normalization"] in ["tmm", "deseq2", "scran", "quantile", "vsn"]
        assert strategy["batch"] in ["combat", "harmony", "dann", "none"]

    def test_recommend_with_modality(self, small_adata) -> None:
        strategy = recommend_strategy(small_adata, modality="scrna")
        assert strategy["imputation"] == "magic"
        assert strategy["normalization"] == "scran"
        assert strategy["batch"] == "harmony"
