"""
测试共享 fixtures

提供小型 AnnData 对象用于各模块的单元测试。
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from pathlib import Path
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


@pytest.fixture
def multi_modality_adata_dict(small_adata, small_adata_no_batch, high_missing_adata) -> dict[str, AnnData]:
    """返回模拟多模态数据的 {modality: AnnData} 字典"""
    return {
        "scrna": small_adata,
        "bulk_rna": small_adata_no_batch,
        "atac": high_missing_adata,
    }


@pytest.fixture
def mock_data_dir(tmp_path, small_adata) -> Path:
    """创建模拟的 data/raw/ 目录结构用于批量处理测试

    生成:
        tmp_path/
        ├── scrna/
        │   ├── sample.h5ad
        │   └── sample.csv          ← 同一数据，应被过滤
        └── bulk_rna/
            └── counts.csv
    """
    import numpy as np

    data_root = tmp_path / "data" / "raw"
    data_root.mkdir(parents=True)

    # scrna: .h5ad + .csv（同一数据不同格式）
    scrna_dir = data_root / "scrna"
    scrna_dir.mkdir()
    small_adata.write(scrna_dir / "sample.h5ad")

    # 生成 CSV 格式的同样数据
    import pandas as pd
    X = small_adata.X.toarray() if hasattr(small_adata.X, "toarray") else small_adata.X
    df = pd.DataFrame(
        X,
        index=small_adata.obs_names,
        columns=small_adata.var_names,
    )
    df.to_csv(scrna_dir / "sample.csv")

    # bulk_rna: 仅 CSV
    bulk_dir = data_root / "bulk_rna"
    bulk_dir.mkdir()
    rng = np.random.RandomState(42)
    X_bulk = rng.poisson(50, size=(20, 500)).astype(np.float32)
    df_bulk = pd.DataFrame(
        X_bulk,
        index=[f"sample_{i}" for i in range(20)],
        columns=[f"gene_{i}" for i in range(500)],
    )
    df_bulk.to_csv(bulk_dir / "counts.csv")

    return data_root
