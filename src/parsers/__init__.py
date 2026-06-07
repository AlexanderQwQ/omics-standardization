"""多模态数据解析模块

支持的文件格式:
    - .h5ad / .loom / .mtx → AnnData
    - .fcs                   → AnnData (流式细胞术)
    - mzML                   → AnnData (质谱)
    - FASTQ / BAM            → AnnData (测序)
"""

from ._base import BaseParser, detect_file_type, parse_file
from ._h5ad import H5ADParser
from ._fcs import FCSParser
from ._mzml import MzMLParser
from ._fastq import FASTQParser

__all__ = [
    "BaseParser",
    "detect_file_type",
    "parse_file",
    "H5ADParser",
    "FCSParser",
    "MzMLParser",
    "FASTQParser",
]
