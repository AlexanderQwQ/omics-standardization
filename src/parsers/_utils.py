"""解析通用工具函数

- 文件类型判断
- 元数据提取
- 批量目录解析
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def list_supported_files(
    directory: str | Path, patterns: tuple[str, ...] | None = None
) -> list[Path]:
    """列出目录下所有支持的数据文件

    Args:
        directory: 数据目录路径
        patterns: 限定文件扩展名，None 表示全部支持格式

    Returns:
        排序后的文件路径列表
    """
    if patterns is None:
        patterns = (
            ".h5ad", ".h5mu", ".loom", ".mtx", ".mtx.gz",
            ".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt", ".txt.gz",
            ".fcs", ".mzml", ".mzml.gz",
            ".fastq", ".fastq.gz", ".fq", ".fq.gz",
            ".bam", ".sam",
            ".biom", ".biom.gz",
        )

    directory = Path(directory)
    if not directory.is_dir():
        return []

    files: list[Path] = []
    for pattern in patterns:
        files.extend(directory.rglob(f"*{pattern}"))

    return sorted(files, key=lambda p: p.name)
