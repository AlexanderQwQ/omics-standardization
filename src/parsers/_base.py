"""
解析引擎基类 + 自动文件类型识别路由

设计模式:
    - 每个具体解析器继承 BaseParser，实现 _parse() 方法
    - parse_file() 函数自动检测文件类型并路由到对应解析器
    - 返回 AnnData 或 MuData 对象
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData


# 文件扩展名 → 解析器映射表
_EXTENSION_MAP: dict[str, str] = {
    ".h5ad": "h5ad",
    ".h5mu": "h5ad",
    ".loom": "h5ad",
    ".mtx": "h5ad",
    ".mtx.gz": "h5ad",
    ".fcs": "fcs",
    ".mzml": "mzml",
    ".mzml.gz": "mzml",
    ".fastq": "fastq",
    ".fastq.gz": "fastq",
    ".fq": "fastq",
    ".fq.gz": "fastq",
    ".bam": "fastq",
    ".sam": "fastq",
}


class BaseParser(ABC):
    """解析器抽象基类"""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def parse(self) -> AnnData:
        """解析文件，返回 AnnData"""
        logg.info(f"Parsing {self.file_path} with {self.__class__.__name__}")
        adata = self._parse()
        # 在 .uns 中记录解析信息
        adata.uns.setdefault("standardization", {})
        adata.uns["standardization"]["parser"] = {
            "source": str(self.file_path),
            "parser": self.__class__.__name__,
        }
        return adata

    @abstractmethod
    def _parse(self) -> AnnData:
        """子类实现：实际解析逻辑"""
        ...


def detect_file_type(path: str | Path) -> str:
    """根据文件扩展名检测数据类型

    Returns:
        解析器名称: "h5ad" | "fcs" | "mzml" | "fastq"
    """
    path = Path(path)
    # 处理双重后缀（如 .mtx.gz, .fastq.gz）
    name = path.name.lower()
    if name.endswith((".mtx.gz", ".fastq.gz", ".fq.gz", ".mzml.gz")):
        suffix = "." + ".".join(name.split(".")[-2:])
    else:
        suffix = path.suffix.lower()

    parser_name = _EXTENSION_MAP.get(suffix)
    if parser_name is None:
        msg = f"不支持的文件类型: {suffix} (文件: {path})"
        raise ValueError(msg)

    logg.hint(f"检测到文件类型: {parser_name} (扩展名: {suffix})")
    return parser_name


def parse_file(path: str | Path) -> AnnData:
    """自动检测文件类型并解析

    根据文件扩展名路由到合适的解析器。

    Args:
        path: 文件路径（单个文件或包含多文件的目录）

    Returns:
        AnnData 对象（多模态时为 MuData）
    """
    parser_type = detect_file_type(path)

    parser_map: dict[str, type[BaseParser]] = {
        "h5ad": H5ADParser,
        "fcs": FCSParser,
        "mzml": MzMLParser,
        "fastq": FASTQParser,
    }

    parser_cls = parser_map[parser_type]
    parser = parser_cls(path)
    return parser.parse()


# 延迟导入以避免循环依赖
from ._h5ad import H5ADParser  # noqa: E402, F811
from ._fcs import FCSParser  # noqa: E402
from ._mzml import MzMLParser  # noqa: E402
from ._fastq import FASTQParser  # noqa: E402
