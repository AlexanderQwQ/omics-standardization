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
    ".csv": "csv",
    ".csv.gz": "csv",
    ".tsv": "csv",
    ".tsv.gz": "csv",
    ".txt": "csv",
    ".txt.gz": "csv",
    ".biom": "biom",
    ".biom.gz": "biom",
    ".json": "biom",  # BIOM 1.0 JSON 格式
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


def validate_file_content(path: str | Path, parser_type: str) -> tuple[bool, list[str]]:
    """验证文件内容是否与声明的类型匹配

    在解析前进行轻量级内容检查，非阻塞 — 无效时仅记录警告。

    Args:
        path: 文件路径
        parser_type: 由 detect_file_type() 返回的解析器类型

    Returns:
        (is_valid, warnings): 有效性标志和警告信息列表
    """
    path = Path(path)
    warnings: list[str] = []

    if not path.exists():
        return False, [f"文件不存在: {path}"]

    if not path.is_file():
        return False, [f"路径不是文件: {path}"]

    try:
        if parser_type == "csv":
            # CSV/TSV: 检查是否有标题行、数值数据列、至少 2 行
            is_valid, w = _validate_csv_content(path)
            warnings.extend(w)
            return is_valid, warnings

        elif parser_type == "mzml":
            # mzML: 检查是否以有效 XML 开头并包含 <mzML 或 <indexedmzML 标签
            is_valid, w = _validate_mzml_content(path)
            warnings.extend(w)
            return is_valid, warnings

        elif parser_type == "biom":
            # BIOM: 检查是否为有效 JSON 并包含预期 BIOM 键
            is_valid, w = _validate_biom_content(path)
            warnings.extend(w)
            return is_valid, warnings

        elif parser_type == "fcs":
            # FCS: 检查文件是否以 "FCS" 魔术字节开头
            is_valid, w = _validate_fcs_content(path)
            warnings.extend(w)
            return is_valid, warnings

        elif parser_type == "h5ad":
            # H5AD/HDF5 文件: 检查 HDF5 magic bytes
            is_valid, w = _validate_h5ad_content(path)
            warnings.extend(w)
            return is_valid, warnings

        elif parser_type == "fastq":
            # FASTQ: 检查是否以 '@' 开头并包含质量行
            is_valid, w = _validate_fastq_content(path)
            warnings.extend(w)
            return is_valid, warnings

        else:
            return True, []  # 未知类型不做内容验证

    except Exception as exc:
        warnings.append(f"内容验证异常: {exc}")
        return True, warnings  # 验证失败不阻塞解析


def _validate_csv_content(path: Path) -> tuple[bool, list[str]]:
    """验证 CSV/TSV 文件内容"""
    warnings: list[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()
            third_line = f.readline().strip()

        if not first_line:
            return False, ["CSV/TSV 文件为空"]

        # 检测分隔符
        delimiter = "\t" if "\t" in first_line else ","
        headers = first_line.split(delimiter)

        if len(headers) < 2:
            warnings.append("CSV/TSV 标题行列数过少 (< 2)")

        if not second_line:
            warnings.append("CSV/TSV 文件仅有一行（缺少数据行）")
            return True, warnings  # 不阻塞，可能只有 header

        # 检查第二行是否包含数值数据
        values = second_line.split(delimiter)
        numeric_count = 0
        for v in values:
            try:
                float(v)
                numeric_count += 1
            except (ValueError, TypeError):
                pass

        if numeric_count == 0:
            warnings.append("CSV/TSV 数据行中未检测到数值列")

        if not third_line:
            warnings.append("CSV/TSV 数据行数过少 (< 2 行数据)")

        return True, warnings

    except UnicodeDecodeError:
        warnings.append("CSV/TSV 文件编码不是 UTF-8，尝试其他编码")
        return True, warnings


def _validate_mzml_content(path: Path) -> tuple[bool, list[str]]:
    """验证 mzML 文件内容"""
    warnings: list[str] = []
    # 处理 .gz 文件
    if path.suffix == ".gz":
        import gzip
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                content = f.read(2048)
        except Exception as exc:
            warnings.append(f"无法读取 gzip 压缩的 mzML 文件: {exc}")
            return True, warnings
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(2048)
        except UnicodeDecodeError:
            # 二进制文件也尝试读取
            with open(path, "rb") as f:
                raw = f.read(2048)
                content = raw.decode("utf-8", errors="replace")

    # 检查 XML 声明或 mzML 标签
    if not content.strip().startswith("<"):
        warnings.append("mzML 文件未以 XML 标签开头")
    elif "<mzML" not in content and "<indexedmzML" not in content and "mzML" not in content.lower():
        warnings.append("mzML 文件中未检测到 <mzML 或 <indexedmzML 标签")

    return True, warnings


def _validate_biom_content(path: Path) -> tuple[bool, list[str]]:
    """验证 BIOM 文件内容"""
    warnings: list[str] = []
    # 处理 .gz 文件
    if path.suffix == ".gz":
        import gzip
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                content = f.read(4096)
        except Exception as exc:
            warnings.append(f"无法读取 gzip 压缩的 BIOM 文件: {exc}")
            return True, warnings
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(4096)
        except UnicodeDecodeError:
            warnings.append("BIOM 文件不是文本格式（可能是 HDF5 BIOM 2.0）")
            return True, warnings  # BIOM 2.0 HDF5 格式不需要文本验证

    # 检查 JSON 有效性
    try:
        import json
        data = json.loads(content) if content.strip() else {}
    except json.JSONDecodeError:
        warnings.append("BIOM 文件内容不是有效 JSON")
        return True, warnings

    # 检查预期 BIOM 键
    expected_keys = {"id", "format", "type", "data", "rows", "columns"}
    if isinstance(data, dict):
        missing = expected_keys - set(data.keys())
        if len(missing) >= 4:  # 缺少大部分关键键
            actual_keys = list(data.keys())[:5]
            warnings.append(f"BIOM JSON 缺少预期键 (前5个实际键: {actual_keys})")

    return True, warnings


def _validate_fcs_content(path: Path) -> tuple[bool, list[str]]:
    """验证 FCS 文件内容"""
    warnings: list[str] = []
    try:
        with open(path, "rb") as f:
            magic = f.read(6)
        # FCS magic bytes: "FCS" followed by version (e.g., "FCS3.1")
        if len(magic) < 3:
            return False, ["FCS 文件过小，无法读取魔术字节"]
        magic_str = magic[:3].decode("ascii", errors="replace")
        if magic_str != "FCS":
            warnings.append(f"FCS 文件魔术字节不正确: 期望 'FCS'，实际 '{magic_str}'")
    except Exception as exc:
        warnings.append(f"读取 FCS 文件失败: {exc}")

    return True, warnings


def _validate_h5ad_content(path: Path) -> tuple[bool, list[str]]:
    """验证 H5AD/HDF5 文件内容"""
    warnings: list[str] = []
    try:
        with open(path, "rb") as f:
            magic = f.read(8)
        # HDF5 magic: \x89HDF\r\n\x1a\n
        if len(magic) < 8:
            return False, ["H5AD 文件过小"]
        if magic[:3] != b"\x89HD":
            warnings.append("H5AD 文件缺少 HDF5 魔术字节")
    except Exception as exc:
        warnings.append(f"读取 H5AD 文件失败: {exc}")

    return True, warnings


def _validate_fastq_content(path: Path) -> tuple[bool, list[str]]:
    """验证 FASTQ 文件内容"""
    warnings: list[str] = []
    # 处理 .gz 文件
    if path.suffix == ".gz":
        import gzip
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                first_char = f.read(1)
        except Exception as exc:
            warnings.append(f"无法读取 gzip 压缩的 FASTQ 文件: {exc}")
            return True, warnings
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                first_char = f.read(1)
        except UnicodeDecodeError:
            return True, warnings  # 可能是 BAM/SAM 二进制格式

    if first_char and first_char != "@":
        warnings.append("FASTQ 文件未以 '@' 开头（非标准 FASTQ 格式）")

    return True, warnings


def parse_file(path: str | Path) -> AnnData:
    """自动检测文件类型并解析

    根据文件扩展名路由到合适的解析器。

    Args:
        path: 文件路径（单个文件或包含多文件的目录）

    Returns:
        AnnData 对象（多模态时为 MuData）
    """
    parser_type = detect_file_type(path)

    # 内容验证（非阻塞 — 仅日志记录警告）
    is_valid, warnings = validate_file_content(path, parser_type)
    if not is_valid:
        logg.error(f"文件内容验证失败: {path}")
        for w in warnings:
            logg.error(f"  {w}")
    elif warnings:
        logg.warning(f"文件内容验证警告 ({path}):")
        for w in warnings:
            logg.warning(f"  {w}")

    parser_map: dict[str, type[BaseParser]] = {
        "h5ad": H5ADParser,
        "fcs": FCSParser,
        "mzml": MzMLParser,
        "fastq": FASTQParser,
        "csv": CSVParser,
        "biom": BIOMParser,
    }

    parser_cls = parser_map[parser_type]
    parser = parser_cls(path)
    return parser.parse()


# 延迟导入以避免循环依赖
from ._h5ad import H5ADParser  # noqa: E402, F811
from ._fcs import FCSParser  # noqa: E402
from ._mzml import MzMLParser  # noqa: E402
from ._fastq import FASTQParser  # noqa: E402
from ._csv import CSVParser  # noqa: E402
from ._biom import BIOMParser  # noqa: E402
