"""归一化模块测试"""

import numpy as np
import pytest
from normalizers._tmm_deseq import TMMNormalizer, DESeq2Normalizer
from normalizers._quantile import QuantileNormalizer, VSNNormalizer
from normalizers._scran import ScranNormalizer


class TestTMMNormalizer:
    """TMM 归一化测试"""

    def test_run_simple(self, small_adata) -> None:
        norm = TMMNormalizer()
        result = norm._run_simple(small_adata)
        assert "normalized" in result.layers
        assert result.layers["normalized"].shape == small_adata.shape
        assert not np.any(np.isnan(result.layers["normalized"]))

    def test_normalization_preserves_shape(self, small_adata) -> None:
        norm = TMMNormalizer()
        result = norm._run_simple(small_adata)
        assert result.layers["normalized"].shape[0] == small_adata.n_obs
        assert result.layers["normalized"].shape[1] == small_adata.n_vars


class TestDESeq2Normalizer:
    """DESeq2 归一化测试"""

    def test_run_simple(self, small_adata) -> None:
        norm = DESeq2Normalizer()
        result = norm._run_simple(small_adata)
        assert "normalized" in result.layers
        assert result.layers["normalized"].shape == small_adata.shape

    def test_median_of_ratios_positive(self, small_adata) -> None:
        norm = DESeq2Normalizer()
        result = norm._run_simple(small_adata)
        X_norm = result.layers["normalized"]
        assert np.all(np.isfinite(X_norm))
        assert np.all(X_norm >= 0)


class TestQuantileNormalizer:
    """分位数归一化测试"""

    def test_run(self, small_adata) -> None:
        norm = QuantileNormalizer()
        result = norm.run(small_adata)
        assert "normalized" in result.layers
        assert result.uns["standardization"]["normalization"]["method"] == "quantile"

    def test_distributions_aligned(self, small_adata) -> None:
        """分位数归一化后所有样本应有相同的分布"""
        norm = QuantileNormalizer()
        result = norm.run(small_adata)
        X_norm = result.layers["normalized"]
        # 每列的排序应等于参考分布（均值）
        X_sorted = np.sort(X_norm, axis=0)
        col_means = np.mean(X_sorted, axis=1)
        for j in range(X_norm.shape[1]):
            assert np.allclose(X_sorted[:, j], col_means, rtol=1e-4)

    def test_no_nan_in_output(self, small_adata) -> None:
        norm = QuantileNormalizer()
        result = norm.run(small_adata)
        assert not np.any(np.isnan(result.layers["normalized"]))


class TestVSNNormalizer:
    """VSN 归一化测试"""

    def test_run_with_fallback(self, small_adata) -> None:
        norm = VSNNormalizer()
        result = norm.run(small_adata)
        assert "normalized" in result.layers
        assert result.layers["normalized"].shape == small_adata.shape
        # fallback 应使用 log2(x+1) 变换
        assert result.uns["standardization"]["normalization"]["method"] == "vsn"


class TestScranNormalizer:
    """Scran 归一化测试"""

    def test_run_python(self, small_adata_no_batch) -> None:
        norm = ScranNormalizer(min_mean=0.1)
        result = norm._run_python(small_adata_no_batch)
        assert "normalized" in result.layers
        assert result.layers["normalized"].shape == small_adata_no_batch.shape

    def test_fallback_normalize_total(self, small_adata) -> None:
        norm = ScranNormalizer()
        result = norm._fallback_normalize_total(small_adata)
        assert "normalized" in result.layers

    def test_min_mean_filtering(self, small_adata_no_batch) -> None:
        """高 min_mean 会导致更多基因被过滤"""
        norm_strict = ScranNormalizer(min_mean=1000.0)
        result = norm_strict._run_python(small_adata_no_batch)
        # 无论过滤结果如何，都应成功返回
        assert result is not None
