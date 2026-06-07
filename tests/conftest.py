"""
测试共享 fixtures

提供小型 AnnData 对象用于各模块的单元测试。
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix


@pytest.fixture
def small_adata() -> AnnData:
    """生成小型 scRNA-seq AnnData（100 细胞 × 200 基因）"""
    n_obs, n_vars = 100, 200
    rng = np.random.RandomState(42)

    # 模拟零膨胀计数矩阵
    X = rng.negative_binomial(5, 0.5, size=(n_obs, n_vars)).astype(np.float32)
    mask = rng.random((n_obs, n_vars)) < 0.3
    X[mask] = 0

    return AnnData(
        X=csr_matrix(X),
        obs=pd.DataFrame(
            {"batch": np.repeat(["A", "B"], 50)},
            index=[f"cell_{i}" for i in range(n_obs)],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)]),
    )


@pytest.fixture
def small_adata_no_batch() -> AnnData:
    """生成无批次的 AnnData"""
    n_obs, n_vars = 50, 100
    rng = np.random.RandomState(123)

    X = rng.poisson(10, size=(n_obs, n_vars)).astype(np.float32)

    return AnnData(
        X=csr_matrix(X),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)]),
    )


@pytest.fixture
def high_missing_adata() -> AnnData:
    """生成高缺失率 AnnData（80% 零值）"""
    n_obs, n_vars = 50, 100
    rng = np.random.RandomState(77)

    X = rng.negative_binomial(2, 0.9, size=(n_obs, n_vars)).astype(np.float32)

    return AnnData(
        X=csr_matrix(X),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)]),
    )
