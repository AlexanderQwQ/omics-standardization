"""FASTQ / BAM 测序文件解析器

将测序原始文件量化为表达矩阵:
    - FASTQ(.gz): kallisto 伪比对定量 → 基因×样本计数矩阵
    - BAM/SAM:    通过 pysam 统计 + 已知注释定量

生产环境应通过 kallisto / STAR / salmon 等工具进行定量。
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from anndata import AnnData

from ._base import BaseParser
from .. import logging as logg

if TYPE_CHECKING:
    pass


class FASTQParser(BaseParser):
    """FASTQ / BAM 测序数据解析器

    支持格式: .fastq, .fastq.gz, .fq, .fq.gz, .bam, .sam

    定量策略:
        - FASTQ: 使用 kallisto 进行伪比对定量
        - BAM/SAM: 通过 pysam 读取比对结果并统计

    用法:
        parser = FASTQParser("data/reads.fastq.gz",
                             transcriptome_index="ref/transcriptome.idx",
                             tech="rna")
        adata = parser.parse()  # → AnnData (obs=样本, var=基因/转录本)

    Parameters:
        file_path: FASTQ/BAM 文件路径
        transcriptome_index: kallisto 索引文件路径（FASTQ 定量必需）
        tech: 测序技术 ("rna" | "atac" | "chip")
        paired_end: 是否为双端测序
        output_dir: 定量结果输出目录
    """

    SUPPORTED_SUFFIXES = {".fastq", ".fastq.gz", ".fq", ".fq.gz", ".bam", ".sam"}

    def __init__(
        self,
        file_path: str | Path,
        transcriptome_index: str | Path | None = None,
        tech: str = "rna",
        paired_end: bool = False,
        output_dir: str | Path | None = None,
    ) -> None:
        super().__init__(file_path)
        self.transcriptome_index = Path(transcriptome_index) if transcriptome_index else None
        self.tech = tech
        self.paired_end = paired_end
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="omics_qc_"))

    def _parse(self) -> AnnData:
        name_lower = self.file_path.name.lower()

        # FASTQ
        if name_lower.endswith((".fastq", ".fastq.gz", ".fq", ".fq.gz")):
            return self._parse_fastq()
        # BAM/SAM
        return self._parse_bam()

    # ------------------------------------------------------------------
    # FASTQ → kallisto 定量
    # ------------------------------------------------------------------

    def _parse_fastq(self) -> AnnData:
        """通过 kallisto 将 FASTQ 定量为表达矩阵"""
        # 检查 kallisto 是否可用
        if not self._check_kallisto():
            logg.warning("kallisto 未安装，返回 reads 统计")
            return self._parse_fastq_basic()

        if self.transcriptome_index is None or not self.transcriptome_index.exists():
            logg.warning("未提供有效的转录组索引，返回 reads 统计")
            return self._parse_fastq_basic()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            return self._run_kallisto()
        except Exception as exc:
            logg.warning(f"kallisto 定量失败 ({exc})，返回 reads 统计")
            return self._parse_fastq_basic()

    def _run_kallisto(self) -> AnnData:
        """运行 kallisto 定量"""
        sample_name = self.file_path.stem.replace(".fastq", "").replace(".fq", "")
        output_path = self.output_dir / sample_name
        output_path.mkdir(parents=True, exist_ok=True)

        # 构建 kallisto 命令
        cmd = [
            "kallisto", "quant",
            "-i", str(self.transcriptome_index),
            "-o", str(output_path),
            "-t", str(max(1, os.cpu_count() or 4 - 2)),
        ]

        if self.paired_end:
            # 双端文件：reads_1.fastq.gz 和 reads_2.fastq.gz
            mate2 = self._find_mate_pair()
            cmd.extend([str(self.file_path), str(mate2)])
        else:
            # 单端
            cmd.extend(["--single", "-l", "200", "-s", "20"])
            cmd.append(str(self.file_path))

        logg.info(f"运行 kallisto: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            raise RuntimeError(f"kallisto 失败: {result.stderr}")

        # 读取 kallisto 输出
        abundance_file = output_path / "abundance.tsv"
        if not abundance_file.exists():
            raise FileNotFoundError(f"kallisto 输出文件未找到: {abundance_file}")

        # 解析 abundence.tsv → AnnData
        df = pd.read_csv(abundance_file, sep="\t")

        # 使用 TPM 或 estimated counts
        if "est_counts" in df.columns:
            counts = df["est_counts"].values
        else:
            counts = df["tpm"].values

        return AnnData(
            X=np.array([counts], dtype=np.float32),
            obs=pd.DataFrame({"sample": [sample_name], "n_reads_quantified": [int(counts.sum())]}),
            var=pd.DataFrame({
                "transcript_id": df["target_id"].values,
                "length": df.get("length", [0] * len(df)).values,
                "effective_length": df.get("eff_length", [0] * len(df)).values,
                "tpm": df.get("tpm", [0] * len(df)).values,
            }, index=df["target_id"].values),
            uns={
                "quantification_tool": "kallisto",
                "transcriptome_index": str(self.transcriptome_index),
                "tech": self.tech,
                "paired_end": self.paired_end,
            },
        )

    # ------------------------------------------------------------------
    # FASTQ 基础统计（无定量工具时）
    # ------------------------------------------------------------------

    def _parse_fastq_basic(self) -> AnnData:
        """统计 FASTQ 中 reads 数量和质量（占位实现）"""
        import gzip

        n_reads = 0
        total_bases = 0
        quality_scores: list[float] = []

        open_fn = gzip.open if self.file_path.name.endswith(".gz") else open

        with open_fn(str(self.file_path), "rt") as fh:
            line_idx = 0
            for line in fh:
                if line_idx % 4 == 1:  # sequence line
                    total_bases += len(line.strip())
                elif line_idx % 4 == 3:  # quality line
                    # Phred scores
                    scores = [ord(c) - 33 for c in line.strip()]
                    if scores:
                        quality_scores.append(np.mean(scores))
                line_idx += 1

        n_reads = line_idx // 4
        avg_quality = float(np.mean(quality_scores)) if quality_scores else 0.0

        logg.info(f"FASTQ 统计: {n_reads} reads, {total_bases} bases, avg Q={avg_quality:.1f}")

        return AnnData(
            X=np.zeros((1, 1)),
            obs=pd.DataFrame({
                "n_reads": [n_reads],
                "total_bases": [total_bases],
                "avg_phred_quality": [avg_quality],
            }),
            uns={
                "warning": (
                    "FASTQ 解析为占位实现（reads 统计）。"
                    "如需定量，请安装 kallisto 并提供 --transcriptome-index 参数。"
                ),
                "quantification_tool": "none",
                "tech": self.tech,
            },
        )

    # ------------------------------------------------------------------
    # BAM/SAM 解析
    # ------------------------------------------------------------------

    def _parse_bam(self) -> AnnData:
        """解析 BAM/SAM 文件"""
        try:
            import pysam
        except ImportError:
            raise ImportError("解析 BAM/SAM 文件需要 pysam 包。请运行: pip install pysam")

        file_mode = "rb" if self.file_path.suffix == ".bam" else "r"
        samfile = pysam.AlignmentFile(str(self.file_path), file_mode)

        n_reads = 0
        n_mapped = 0
        n_unique = 0
        gene_counts: dict[str, int] = {}

        for read in samfile:
            n_reads += 1
            if not read.is_unmapped:
                n_mapped += 1
                if read.mapping_quality >= 30:
                    n_unique += 1
                # 尝试从标签中提取基因名
                if read.has_tag("XT"):
                    gene_name = read.get_tag("XT")
                    gene_counts[gene_name] = gene_counts.get(gene_name, 0) + 1
                elif read.reference_name:
                    gene_counts[read.reference_name] = gene_counts.get(read.reference_name, 0) + 1

        samfile.close()

        # 如果有基因计数，构建 AnnData
        if gene_counts:
            genes = list(gene_counts.keys())
            counts = np.array([[gene_counts[g] for g in genes]], dtype=np.float32)
            return AnnData(
                X=counts,
                obs=pd.DataFrame({
                    "n_reads": [n_reads],
                    "n_mapped": [n_mapped],
                    "n_unique": [n_unique],
                    "mapping_rate": [n_mapped / max(n_reads, 1)],
                }),
                var=pd.DataFrame(index=genes),
                uns={"quantification_tool": "pysam", "file_format": self.file_path.suffix},
            )

        # 无法提取基因计数时返回汇总统计
        return AnnData(
            X=np.zeros((1, 1)),
            obs=pd.DataFrame({
                "n_reads": [n_reads],
                "n_mapped": [n_mapped],
                "n_unique": [n_unique],
                "mapping_rate": [n_mapped / max(n_reads, 1)],
            }),
            uns={
                "warning": "BAM/SAM 解析为汇总统计，无法提取基因级计数。请考虑使用 featureCounts 或其他定量工具。",
                "quantification_tool": "pysam",
            },
        )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _check_kallisto(self) -> bool:
        """检查 kallisto 是否在 PATH 中"""
        try:
            result = subprocess.run(
                ["kallisto", "version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _find_mate_pair(self) -> Path:
        """查找双端测序的 mate pair 文件

        例如: sample_R1.fastq.gz → sample_R2.fastq.gz
        """
        name = str(self.file_path)
        # 常见的双端命名约定
        for r1_tag, r2_tag in [
            ("_R1", "_R2"),
            ("_r1", "_r2"),
            ("_1", "_2"),
            ("_read1", "_read2"),
        ]:
            if r1_tag in name:
                mate_path = name.replace(r1_tag, r2_tag)
                mate = Path(mate_path)
                if mate.exists():
                    return mate

        # 如果找不到，尝试最常见的命名
        raise FileNotFoundError(
            f"未找到双端测序 mate pair 文件。请确保配对文件命名一致 "
            f"(如 sample_R1.fastq.gz 和 sample_R2.fastq.gz)"
        )
