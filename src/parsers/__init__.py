"""多模态数据解析模块

支持的文件格式:
    - .h5ad / .loom / .mtx → AnnData
    - .csv / .tsv / .txt  → AnnData (表达矩阵)
    - .fcs                   → AnnData (流式细胞术)
    - mzML                   → AnnData (质谱)
    - FASTQ / BAM            → AnnData (测序)
    - .biom                  → AnnData (宏基因组/微生物组)
"""

from ._base import BaseParser, detect_file_type, parse_file
from ._h5ad import H5ADParser
from ._csv import CSVParser, TSVParser
from ._fcs import FCSParser
from ._mzml import MzMLParser
from ._fastq import FASTQParser
from ._biom import BIOMParser
from ._utils import list_supported_files

__all__ = [
    "BaseParser",
    "detect_file_type",
    "parse_file",
    "list_supported_files",
    "H5ADParser",
    "CSVParser",
    "TSVParser",
    "FCSParser",
    "MzMLParser",
    "FASTQParser",
    "BIOMParser",
]
