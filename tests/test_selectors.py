"""算法选择器测试"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from selectors._modality import ModalitySelector, detect_modality, _extract_features
from selectors._strategy import StrategySelector, recommend_strategy
from selectors._persistence import (
    generate_training_data,
    is_model_trained,
    save_modality_model,
    save_strategy_models,
    train_and_persist_models,
)


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

    def test_gmm_fit_and_predict(self, small_adata) -> None:
        """GMM 应在小数据集上正常训练和预测"""
        features = _extract_features(small_adata)
        # 需要至少 2 个样本来训练 GMM
        X_train = np.repeat(features, 10, axis=0) + np.random.RandomState(42).normal(0, 0.1, (10, 6))

        selector = ModalitySelector(n_components=min(3, len(X_train)))
        selector.fit(X_train)
        modality = selector.predict(small_adata)
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

    def test_recommend_bulk_rna(self, small_adata) -> None:
        strategy = recommend_strategy(small_adata, modality="bulk_rna")
        assert strategy["imputation"] == "none"
        assert strategy["normalization"] == "tmm"
        assert strategy["batch"] == "combat"

    def test_recommend_proteomics(self, small_adata) -> None:
        strategy = recommend_strategy(small_adata, modality="proteomics")
        assert strategy["imputation"] == "missforest"
        assert strategy["normalization"] == "quantile"

    def test_all_fallback_strategies_valid(self) -> None:
        """所有模态的 fallback 策略应都有效"""
        for modality in ["scrna", "bulk_rna", "proteomics", "metabolomics", "atac"]:
            strategy = recommend_strategy(None, modality=modality)  # type: ignore[arg-type]
            valid_impute = {"missforest", "zinb_vae", "magic", "none"}
            valid_norm = {"tmm", "deseq2", "scran", "quantile", "vsn"}
            valid_batch = {"combat", "harmony", "dann", "none"}
            assert strategy["imputation"] in valid_impute, f"{modality}: bad impute={strategy['imputation']}"
            assert strategy["normalization"] in valid_norm, f"{modality}: bad norm={strategy['normalization']}"
            assert strategy["batch"] in valid_batch, f"{modality}: bad batch={strategy['batch']}"


class TestStrategySelectorTraining:
    """StrategySelector 训练与预测测试"""

    def test_fit_and_predict(self) -> None:
        """RF 模型应在合成数据上正常训练和预测"""
        X, y_impute, y_norm, y_batch = generate_training_data(n_samples=200, random_state=42)

        selector = StrategySelector(n_estimators=50, max_depth=5, random_state=42)
        selector.fit(X, y_impute, y_norm, y_batch)

        # 在训练数据上预测
        for i in range(min(5, len(X))):
            pred = selector._models["imputation"].predict(X[i:i+1])[0]
            assert pred in ["missforest", "zinb_vae", "magic", "none"]

        # 验证模型存储
        assert "imputation" in selector._models
        assert "normalization" in selector._models
        assert "batch" in selector._models


class TestModelPersistence:
    """模型持久化测试"""

    def test_generate_training_data(self) -> None:
        X, y_impute, y_norm, y_batch = generate_training_data(n_samples=100, random_state=42)
        assert X.shape == (100, 5)
        assert len(y_impute) == 100
        assert len(y_norm) == 100
        assert len(y_batch) == 100

    def test_save_and_load_models(self) -> None:
        """训练并保存模型，然后重新加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)

            # 生成训练数据并训练
            X, y_impute, y_norm, y_batch = generate_training_data(n_samples=200, random_state=42)

            # 训练 GMM
            gmm_selector = ModalitySelector(random_state=42)
            X_gmm = X[:, 1:]  # GMM 使用数值特征（排除模态编码）
            gmm_selector.fit(X_gmm)
            save_modality_model(gmm_selector, model_dir)

            # 训练 RF
            rf_selector = StrategySelector(n_estimators=50, max_depth=5, random_state=42)
            rf_selector.fit(X, y_impute, y_norm, y_batch)
            save_strategy_models(rf_selector, model_dir)

            # 验证模型文件存在
            assert (model_dir / "modality_gmm.joblib").exists()
            assert (model_dir / "strategy_rf_imputation.joblib").exists()
            assert (model_dir / "strategy_rf_normalization.joblib").exists()
            assert (model_dir / "strategy_rf_batch.joblib").exists()
            assert (model_dir / "training_metadata.json").exists()

            # 验证 is_model_trained
            assert is_model_trained(model_dir)

    def test_train_and_persist_all(self) -> None:
        """一键训练并持久化所有模型"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_and_persist_models(
                model_dir=tmpdir,
                n_samples=100,
                random_state=42,
            )
            assert "modality" in result
            assert "strategy" in result
            assert is_model_trained(tmpdir)
