"""解析器模块测试"""

import pytest
from parsers._base import detect_file_type


class TestDetectFileType:
    """文件类型自动检测测试"""

    def test_h5ad(self) -> None:
        assert detect_file_type("data.h5ad") == "h5ad"

    def test_h5mu(self) -> None:
        assert detect_file_type("data.h5mu") == "h5ad"

    def test_fcs(self) -> None:
        assert detect_file_type("sample.fcs") == "fcs"

    def test_mzml(self) -> None:
        assert detect_file_type("run.mzml") == "mzml"

    def test_fastq(self) -> None:
        assert detect_file_type("reads.fastq") == "fastq"
        assert detect_file_type("reads.fastq.gz") == "fastq"

    def test_unknown_extension(self) -> None:
        with pytest.raises(ValueError, match="不支持的文件类型"):
            detect_file_type("data.xyz")
