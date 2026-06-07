"""FASTQ / BAM 测序文件解析器"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from anndata import AnnData

from ._base import BaseParser

if TYPE_CHECKING:
    pass


class FASTQParser(BaseParser):
    """FASTQ / BAM 测序数据解析器（占位实现）

    生产环境中应通过 kallisto / STAR / salmon 等工具进行定量，
    再将定量结果导入为 AnnData。

    当前实现：
        - .fastq(.gz): 读取并统计 reads 数量，返回骨架 AnnData
        - .bam / .sam: 同上
    """

    SUPPORTED_SUFFIXES = {".fastq", ".fastq.gz", ".fq", ".fq.gz", ".bam", ".sam"}

    def _parse(self) -> AnnData:
        name_lower = self.file_path.name.lower()

        # FASTQ
        if name_lower.endswith((".fastq", ".fastq.gz", ".fq", ".fq.gz")):
            return self._parse_fastq()
        # BAM/SAM
        return self._parse_bam()

    # ---------------------------------------------------------------
    # 内部方法
    # ---------------------------------------------------------------

    def _parse_fastq(self) -> AnnData:
        """统计 FASTQ 中 reads 数量"""
        import gzip

        n_reads = 0
        open_fn = gzip.open if self.file_path.name.endswith(".gz") else open

        with open_fn(str(self.file_path), "rt") as fh:
            for _ in fh:
                n_reads += 1

        n_reads //= 4  # FASTQ 每条 read 占 4 行

        # 返回骨架 AnnData（实际项目需替换为定量结果）
        return AnnData(
            X=np.zeros((1, 1)),
            obs=pd.DataFrame({"n_reads": [n_reads]}),
            uns={"warning": "FASTQ 解析为占位实现，请使用定量工具（kallisto/STAR）生成计数矩阵后导入"},
        )

    def _parse_bam(self) -> AnnData:
        """统计 BAM/SAM 中 reads 数量（需要 pysam）"""
        try:
            import pysam
        except ImportError:
            raise ImportError("解析 BAM/SAM 文件需要 pysam 包。请运行: pip install pysam")

        samfile = pysam.AlignmentFile(str(self.file_path), "rb" if self.file_path.suffix == ".bam" else "r")
        n_reads = samfile.count()
        samfile.close()

        return AnnData(
            X=np.zeros((1, 1)),
            obs=pd.DataFrame({"n_reads": [n_reads]}),
            uns={"warning": "FASTQ/BAM 解析为占位实现，请使用定量工具生成计数矩阵后导入"},
        )
