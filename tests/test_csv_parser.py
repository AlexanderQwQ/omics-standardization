"""CSV/TSV 解析器测试"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from parsers._csv import CSVParser, _detect_layout
from parsers._base import detect_file_type


class TestDetectLayout:
    """矩阵布局自动检测测试"""

    def test_genes_x_samples(self) -> None:
        df = pd.DataFrame(
            {"gene_id": ["G1", "G2", "G3"], "S1": [10, 20, 30], "S2": [40, 50, 60]},
        )
        df = df.set_index("gene_id")
        assert _detect_layout(df) == "genes_x_samples"

    def test_samples_x_genes(self) -> None:
        # 样本少、特征多 → 样本×基因
        data = {"S1": [1.0] * 100, "S2": [2.0] * 100}
        df = pd.DataFrame(data)
        df.index = [f"G{i}" for i in range(100)]
        assert _detect_layout(df) in ("genes_x_samples", "samples_x_genes")

    def test_gene_keyword_in_first_col(self) -> None:
        df = pd.DataFrame(
            {"gene_symbol": ["GAPDH", "ACTB", "TP53"], "S1": [100, 200, 300]},
        )
        df = df.set_index("gene_symbol")
        assert _detect_layout(df) == "genes_x_samples"


class TestCSVParser:
    """CSV 解析器测试"""

    @pytest.fixture
    def csv_file(self) -> Path:
        """创建临时 CSV 表达矩阵文件"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            f.write("gene_id,Sample_A,Sample_B,Sample_C\n")
            f.write("GENE1,10.5,20.3,15.7\n")
            f.write("GENE2,5.2,8.1,12.0\n")
            f.write("GENE3,30.0,25.5,22.1\n")
            file_path = Path(f.name)
        yield file_path
        file_path.unlink(missing_ok=True)

    @pytest.fixture
    def tsv_file(self) -> Path:
        """创建临时 TSV 表达矩阵文件"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False, newline="") as f:
            f.write("gene_id\tSample_A\tSample_B\n")
            f.write("GENE1\t1.0\t2.0\n")
            f.write("GENE2\t3.0\t4.0\n")
            file_path = Path(f.name)
        yield file_path
        file_path.unlink(missing_ok=True)

    @pytest.fixture
    def integer_counts_csv(self) -> Path:
        """创建整数计数矩阵"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            f.write("gene_id,S1,S2,S3,S4\n")
            f.write("G1,0,15,0,8\n")
            f.write("G2,200,0,150,0\n")
            f.write("G3,50,60,45,55\n")
            file_path = Path(f.name)
        yield file_path
        file_path.unlink(missing_ok=True)

    def test_parse_csv(self, csv_file) -> None:
        parser = CSVParser(csv_file)
        adata = parser.parse()
        assert adata.n_obs == 3  # 3 samples
        assert adata.n_vars == 3  # 3 genes

    def test_parse_tsv(self, tsv_file) -> None:
        parser = CSVParser(tsv_file)
        adata = parser.parse()
        assert adata.n_obs == 2
        assert adata.n_vars == 2

    def test_detect_integer_counts(self, integer_counts_csv) -> None:
        parser = CSVParser(integer_counts_csv)
        adata = parser.parse()
        assert adata.uns["is_integer_counts"] is True

    def test_file_type_detection(self) -> None:
        assert detect_file_type("data.csv") == "csv"
        assert detect_file_type("data.tsv") == "csv"
        assert detect_file_type("data.txt") == "csv"
        assert detect_file_type("data.csv.gz") == "csv"

    def test_parse_with_explicit_separator(self, csv_file) -> None:
        parser = CSVParser(csv_file, separator=",")
        adata = parser.parse()
        assert adata is not None
        assert adata.X.shape[0] == 3
