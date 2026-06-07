"""流水线集成测试"""

import pytest
from pipeline import StandardizationPipeline


class TestStandardizationPipeline:
    """端到端流水线测试"""

    def test_pipeline_creation(self) -> None:
        pipeline = StandardizationPipeline()
        assert pipeline is not None
        assert len(pipeline._steps) == 0

    def test_pipeline_with_data(self, small_adata) -> None:
        pipeline = StandardizationPipeline()
        result = pipeline.run(data=small_adata)
        assert result is not None
        # 验证所有处理步骤
        assert "parse" not in pipeline._steps  # 跳过了解析
        assert "select" in pipeline._steps
        assert "impute" in pipeline._steps
        assert "normalize" in pipeline._steps
        assert "batch_correct" in pipeline._steps
        assert "evaluate" in pipeline._steps

    def test_pipeline_no_input(self) -> None:
        pipeline = StandardizationPipeline()
        with pytest.raises(ValueError, match="必须提供"):
            pipeline.run()
