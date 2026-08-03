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

        # 第一遍遍历：收集所有谱图数据和 m/z 值
        # 注意：pymzml 的 Reader 迭代器只能遍历一次，必须在单次遍历中完成所有数据收集
        spectra_peaks: list[dict] = []  # 保存每个谱图的完整 peaks 数据
        mz_union: set[float] = set()

        for spec in run:
            peaks = spec.peaks("raw")
            if peaks is not None and len(peaks) > 0:
                spectra_peaks.append({
                    "id": spec.ID,
                    "rt": spec.scan_time_in_minutes(),
                    "peaks": peaks,  # 保存完整 peaks，避免二次遍历迭代器
                })
                mz_union.update(peaks[:, 0])

        # 构建特征矩阵：m/z → 列索引
        mz_list = sorted(mz_union)
        mz_to_idx = {mz: i for i, mz in enumerate(mz_list)}

        X = np.zeros((len(spectra_peaks), len(mz_list)), dtype=np.float32)
        for i, entry in enumerate(spectra_peaks):
            for mz_val, intensity in entry["peaks"]:
                if mz_val in mz_to_idx:
                    X[i, mz_to_idx[mz_val]] = intensity

        # 构建 .obs 元数据（谱图 ID 和保留时间）
        spectra_meta = [{"id": e["id"], "rt": e["rt"]} for e in spectra_peaks]

        return AnnData(
            X=csr_matrix(X),
            var=pd.DataFrame({"mz": mz_list}, index=[f"{mz:.4f}" for mz in mz_list]),
            obs=pd.DataFrame(spectra_meta).set_index("id") if spectra_meta else None,
        )
