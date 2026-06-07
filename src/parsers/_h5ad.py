""".h5ad / .loom / .mtx 文件解析器"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anndata
import mudata

from ._base import BaseParser

if TYPE_CHECKING:
    from anndata import AnnData


class H5ADParser(BaseParser):
    """单细胞 / 多模态 AnnData/MuData 解析器

    支持格式: .h5ad, .h5mu, .loom, .mtx(.gz)
    """

    SUPPORTED_SUFFIXES = {".h5ad", ".h5mu", ".loom", ".mtx", ".mtx.gz"}

    def _parse(self) -> AnnData:
        suffix = self.file_path.suffix.lower()
        name_lower = self.file_path.name.lower()

        # .h5mu (MuData)
        if suffix == ".h5mu":
            return mudata.read(str(self.file_path))

        # .loom
        if suffix == ".loom":
            return anndata.read_loom(str(self.file_path))

        # .mtx / .mtx.gz
        if suffix == ".mtx" or name_lower.endswith(".mtx.gz"):
            mtx_dir = self.file_path.parent
            return anndata.read_mtx(str(mtx_dir / "matrix.mtx.gz" if name_lower.endswith(".mtx.gz") else self.file_path))

        # .h5ad (默认)
        return anndata.read_h5ad(str(self.file_path))
