"""CSV/TSV 表达矩阵解析器

将逗号/制表符分隔的表达矩阵转换为 AnnData:
    - 支持 .csv, .tsv, .txt 格式
    - 自动检测矩阵布局:
        - 基因×样本: 第一列为基因名，其余列为样本
        - 样本×基因: 第一行为基因名，其余行为样本
    - 处理整数计数 (raw counts) 和浮点丰度 (normalized abundance)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from ._base import BaseParser
import _logging as logg

if TYPE_CHECKING:
    pass


def _detect_layout(df: pd.DataFrame) -> str:
    """自动检测矩阵布局

    启发式规则:
        - 如果第一列名称是基因/特征类关键词 → 基因×样本布局
        - 如果行数远大于列数且列名非数值 → 样本×基因布局
        - 如果索引值看起来像样本 ID → 样本×基因布局

    Returns:
        "genes_x_samples" | "samples_x_genes"
    """
    first_col_name = str(df.columns[0]).lower()

    # 检查第一列名
    gene_keywords = ["gene", "symbol", "feature", "id", "gene_id", "geneid", "ensembl", "entrez", "probe"]
    if any(kw in first_col_name for kw in gene_keywords):
        return "genes_x_samples"

    # 检查数值列数量
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= len(df.columns) * 0.8:
        # 大多是数值列 → 可能是样本×基因（列是基因）
        if df.shape[0] < df.shape[1]:
            return "samples_x_genes"

    # 启发式：行数多 → 基因×样本；行数少列数多 → 样本×基因
    if df.shape[0] > df.shape[1] * 2:
        return "genes_x_samples"

    return "samples_x_genes"  # 默认：样本行×基因列


class CSVParser(BaseParser):
    """CSV/TSV 表达矩阵解析器

    支持格式: .csv, .tsv, .txt (逗号/制表符分隔)
    支持布局: 基因×样本 或 样本×基因（自动检测）

    用法:
        parser = CSVParser("data/expression_matrix.csv")
        adata = parser.parse()
    """

    SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz"}

    def __init__(
        self,
        file_path: str | Path,
        separator: str | None = None,
        index_col: int = 0,
        layout: str | None = None,
    ) -> None:
        """初始化 CSV 解析器

        Args:
            file_path: 文件路径
            separator: 分隔符 (None=自动检测: 逗号/制表符)
            index_col: 索引列位置 (默认第 0 列)
            layout: 矩阵布局 ("genes_x_samples" | "samples_x_genes" | None=自动检测)
        """
        super().__init__(file_path)
        self.separator = separator
        self.index_col = index_col
        self._layout = layout

    def _parse(self) -> AnnData:
        # 自动检测分隔符
        sep = self.separator or self._detect_separator()

        # 读取数据
        df = pd.read_csv(
            str(self.file_path),
            sep=sep,
            index_col=self.index_col,
            low_memory=False,
        )

        # 提取元数据列（time, condition, batch 等 → .obs）— 必须在转置前
        _META_KEYWORDS = {"time", "condition", "batch", "time_unit", "group", "sample_id", "donor", "treatment"}
        meta_cols = []
        for c in df.columns:
            c_lower = str(c).lower().strip()
            if c_lower in _META_KEYWORDS:
                meta_cols.append(c)

        obs_df = pd.DataFrame(index=df.index.tolist())
        if meta_cols:
            for mc in meta_cols:
                obs_df[mc] = df[mc].values
            df = df.drop(columns=meta_cols)
            logg.hint(f"检测到元数据列 {meta_cols}，移至 .obs")

        # 检测布局
        layout = self._layout or _detect_layout(df)

        # AnnData 期望 X 为 (n_obs, n_vars) = (样本, 基因)
        # samples_x_genes: 样本行×基因列 → 不需要转置
        # genes_x_samples: 基因行×样本列 → 需要转置
        if layout == "genes_x_samples":
            df = df.T

        # 确保数值类型
        df = df.apply(pd.to_numeric, errors="coerce")

        # 构建 AnnData
        X = df.values
        nan_mask = np.isnan(X)
        if nan_mask.any():
            X[nan_mask] = 0.0
            logg.info(f"CSV 中 {nan_mask.sum()} 个 NaN 值已替换为 0")

        # 判断是否为整数计数（如 RNA-seq raw counts）
        is_integer = np.all(X == X.astype(np.int64))

        return AnnData(
            X=csr_matrix(X.astype(np.float32)),
            obs=obs_df,
            var=pd.DataFrame(index=df.columns.tolist()),
            uns={
                "source_format": self.file_path.suffix.lower(),
                "layout": layout,
                "is_integer_counts": is_integer,
                "delimiter": "comma" if sep == "," else "tab",
            },
        )

    def _detect_separator(self) -> str:
        """自动检测分隔符（逗号 vs 制表符）"""
        with open(str(self.file_path), "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()
        if first_line.count("\t") > first_line.count(","):
            return "\t"
        return ","


class TSVParser(CSVParser):
    """TSV 表达矩阵解析器（制表符分隔的便捷别名）"""

    SUPPORTED_SUFFIXES = {".tsv", ".tsv.gz", ".txt", ".txt.gz"}

    def __init__(self, file_path: str | Path, **kwargs) -> None:
        super().__init__(file_path, separator="\t", **kwargs)
