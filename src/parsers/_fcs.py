"""流式细胞术 .fcs 文件解析器"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from ._base import BaseParser

if TYPE_CHECKING:
    pass


class FCSParser(BaseParser):
    """.fcs 流式细胞术数据解析器

    将 .fcs 文件转换为 AnnData:
        - .X: 细胞 × 通道计数矩阵（sparse CSR）
        - .var: 通道名称和参数
        - .obs: 细胞元数据
    """

    SUPPORTED_SUFFIXES = {".fcs"}

    def _parse(self) -> AnnData:
        try:
            import fcsparser
        except ImportError:
            raise ImportError(
                "解析 .fcs 文件需要 fcsparser 包。请运行: pip install fcsparser"
            )

        meta, data = fcsparser.parse(str(self.file_path))

        # 构建 AnnData
        adata = AnnData(
            X=csr_matrix(data.values),
            var=pd.DataFrame(index=data.columns),
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(data.shape[0])]),
        )

        # 保存通道元数据
        adata.var["channel"] = data.columns.tolist()
        adata.uns["fcs_metadata"] = meta

        return adata
