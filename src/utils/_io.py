"""AnnData / MuData 保存、加载、格式转换"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData
    from mudata import MuData


def load_data(path: str | Path) -> AnnData:
    """加载 AnnData 或 MuData 文件

    支持: .h5ad, .h5mu, .h5ad.gz
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".h5mu":
        from mudata import read
        return read(str(path))
    elif suffix in (".h5ad", ".h5ad.gz"):
        from anndata import read_h5ad
        return read_h5ad(str(path))
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")


def save_data(data: AnnData | MuData, path: str | Path, **kwargs: Any) -> None:
    """保存 AnnData 或 MuData

    根据扩展名自动选择保存格式。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.write(str(path), **kwargs)
    logg.info(f"数据已保存至 {path}")


def to_parquet(data: AnnData, output_dir: str | Path) -> None:
    """将 AnnData 导出为 Parquet 格式（适合超大规模数据）

    .X → parquet/ 目录下的功能矩阵
    .obs → parquet/metadata.parquet
    .var → parquet/features.parquet
    """
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = data.X.toarray() if hasattr(data.X, "toarray") else data.X
    X_df = pd.DataFrame(X, index=data.obs_names, columns=data.var_names)
    X_df.to_parquet(output_dir / "matrix.parquet")

    data.obs.to_parquet(output_dir / "metadata.parquet")
    data.var.to_parquet(output_dir / "features.parquet")

    logg.info(f"数据已导出为 Parquet: {output_dir}")
