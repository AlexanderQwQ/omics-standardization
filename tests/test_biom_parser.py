"""BIOM 解析器测试"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from parsers._biom import BIOMParser
from parsers._base import detect_file_type


class TestBIOMParser:
    """BIOM 解析器测试"""

    @pytest.fixture
    def biom_json_file(self) -> Path:
        """创建 BIOM 1.0 JSON 格式测试文件"""
        biom_data = {
            "id": "test_otu_table",
            "format": "Biological Observation Matrix 1.0.0",
            "format_url": "http://biom-format.org",
            "type": "OTU table",
            "generated_by": "test",
            "date": "2024-01-01T00:00:00",
            "rows": [
                {"id": "OTU_1", "metadata": {"taxonomy": ["Bacteria", "Firmicutes"]}},
                {"id": "OTU_2", "metadata": {"taxonomy": ["Bacteria", "Proteobacteria"]}},
                {"id": "OTU_3", "metadata": {"taxonomy": ["Bacteria", "Actinobacteria"]}},
            ],
            "columns": [
                {"id": "Sample_A", "metadata": {"condition": "control"}},
                {"id": "Sample_B", "metadata": {"condition": "treatment"}},
                {"id": "Sample_C", "metadata": {"condition": "control"}},
                {"id": "Sample_D", "metadata": {"condition": "treatment"}},
            ],
            "matrix_type": "sparse",
            "matrix_element_type": "int",
            "shape": [3, 4],
            "data": [
                [0, 0, 150],  # OTU_1, Sample_A → 150
                [1, 1, 200],  # OTU_2, Sample_B → 200
                [2, 2, 80],   # OTU_3, Sample_C → 80
                [0, 3, 50],   # OTU_1, Sample_D → 50
                [1, 0, 120],  # OTU_2, Sample_A → 120
                [2, 1, 300],  # OTU_3, Sample_B → 300
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".biom", delete=False) as f:
            json.dump(biom_data, f)
            file_path = Path(f.name)
        yield file_path
        file_path.unlink(missing_ok=True)

    @pytest.fixture
    def biom_dense_file(self) -> Path:
        """创建 BIOM dense 格式测试文件"""
        biom_data = {
            "id": "test_dense",
            "format": "Biological Observation Matrix 1.0.0",
            "format_url": "http://biom-format.org",
            "type": "OTU table",
            "generated_by": "test",
            "date": "2024-01-01T00:00:00",
            "rows": [
                {"id": "OTU_A", "metadata": {}},
                {"id": "OTU_B", "metadata": {}},
            ],
            "columns": [
                {"id": "S1", "metadata": {}},
                {"id": "S2", "metadata": {}},
            ],
            "matrix_type": "dense",
            "matrix_element_type": "int",
            "shape": [2, 2],
            "data": [[10, 20], [30, 40]],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".biom", delete=False) as f:
            json.dump(biom_data, f)
            file_path = Path(f.name)
        yield file_path
        file_path.unlink(missing_ok=True)

    def test_parse_sparse_biom(self, biom_json_file) -> None:
        parser = BIOMParser(biom_json_file)
        adata = parser.parse()
        assert adata.n_obs == 4  # 4 samples
        assert adata.n_vars == 3  # 3 OTUs

    def test_parse_dense_biom(self, biom_dense_file) -> None:
        parser = BIOMParser(biom_dense_file)
        adata = parser.parse()
        assert adata.n_obs == 2
        assert adata.n_vars == 2

    def test_biom_taxonomy_columns(self, biom_json_file) -> None:
        parser = BIOMParser(biom_json_file)
        adata = parser.parse()
        assert "tax_0" in adata.var.columns or adata.uns.get("biom_metadata") is not None

    def test_file_type_detection(self) -> None:
        assert detect_file_type("table.biom") == "biom"
        assert detect_file_type("table.biom.gz") == "biom"
