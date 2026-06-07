"""质谱 mzML 文件解析器"""

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


class MzMLParser(BaseParser):
    """mzML 质谱数据解析器

    将 mzML 文件转换为 AnnData:
        - .X: 样本 × 特征（质荷比 m/z 强度矩阵）
        - .var: m/z 值、保留时间等
        - .obs: 样本名称
    """

    SUPPORTED_SUFFIXES = {".mzml", ".mzml.gz"}

    def _parse(self) -> AnnData:
        try:
            import pymzml
        except ImportError:
            raise ImportError(
                "解析 mzML 文件需要 pymzML 包。请运行: pip install pymzml"
            )

        run = pymzml.run.Reader(str(self.file_path))

        # 提取所有谱图
        spectra = []
        mz_union: set[float] = set()

        for spec in run:
            peaks = spec.peaks("raw")
            if peaks is not None and len(peaks) > 0:
                spectra.append({"id": spec.ID, "rt": spec.scan_time_in_minutes()})
                mz_union.update(peaks[:, 0])

        # 构建特征矩阵
        mz_list = sorted(mz_union)
        mz_to_idx = {mz: i for i, mz in enumerate(mz_list)}

        X = np.zeros((len(spectra), len(mz_list)), dtype=np.float32)
        for i, spec in enumerate(run):
            peaks = spec.peaks("raw")
            if peaks is not None:
                for mz_val, intensity in peaks:
                    if mz_val in mz_to_idx:
                        X[i, mz_to_idx[mz_val]] = intensity

        return AnnData(
            X=csr_matrix(X),
            var=pd.DataFrame({"mz": mz_list}, index=[f"{mz:.4f}" for mz in mz_list]),
            obs=pd.DataFrame(spectra).set_index("id") if spectra else None,
        )
