"""
端到端标准化流水线主类

串联整个处理流程:
    Parse → Select Strategy → Impute → Normalize → Batch Correct → Evaluate
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import logging as logg
from .._settings import settings

if TYPE_CHECKING:
    from anndata import AnnData
    from mudata import MuData


class StandardizationPipeline:
    """多模态组学数据标准化流水线

    用法:
        pipeline = StandardizationPipeline(config="config/default.yaml")
        result = pipeline.run("data/raw/")
    """

    def __init__(
        self, config: str | Path | None = None, use_storage: bool = False
    ) -> None:
        if config is not None:
            settings.load_config(Path(config))

        self._steps: list[str] = []
        self._results: dict[str, Any] = {}
        self._use_storage = use_storage

    def run(
        self,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        data: AnnData | MuData | None = None,
        use_storage: bool = False,
    ) -> AnnData:
        """运行完整标准化流水线

        Args:
            input_path: 输入数据路径（或目录）
            output_path: 输出路径
            data: 已解析的 AnnData/MuData（跳过解析步骤）

        Returns:
            标准化后的 AnnData 或 MuData
        """
        logg.info("=" * 60)
        logg.info("开始标准化流水线")
        logg.info("=" * 60)

        # Step 1: 解析
        if data is None and input_path is not None:
            data = self._step_parse(input_path)
        elif data is None:
            raise ValueError("必须提供 input_path 或 data 参数")

        # Step 2: 选择策略
        self._step_select_strategy(data)

        # Step 3: 插补
        data = self._step_impute(data)

        # Step 4: 归一化
        data = self._step_normalize(data)

        # Step 5: 批次校正
        data = self._step_batch_correct(data)

        # Step 6: 评估
        self._step_evaluate(data)

        # 保存
        if output_path is not None:
            self._save(data, output_path, use_storage=use_storage)

        logg.info("=" * 60)
        logg.info("标准化流水线完成")
        logg.info(f"处理步骤: {' → '.join(self._steps)}")
        logg.info("=" * 60)

        return data

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _step_parse(self, input_path: str | Path) -> AnnData:
        """Step 1: 解析数据"""
        logg.info("\n[Step 1/6] 解析数据...")
        self._steps.append("parse")

        from ..parsers import parse_file
        return parse_file(input_path)

    def _step_select_strategy(self, data: AnnData) -> dict[str, str]:
        """Step 2: 选择处理策略"""
        logg.info("\n[Step 2/6] 选择处理策略...")
        self._steps.append("select")

        from ..selectors import recommend_strategy, detect_modality
        modality = detect_modality(data)
        strategy = recommend_strategy(data, modality=modality)
        self._results["strategy"] = strategy

        # 记录到 trace（修复溯源链断裂）
        data.uns["standardization"] = data.uns.get("standardization", {})
        data.uns["standardization"]["strategy"] = {
            "modality": modality,
            "strategy": strategy,
            "timestamp": str(datetime.now(timezone.utc)),
        }

        logg.info(f"  推荐策略: {strategy}")
        return strategy

    def _step_impute(self, data: AnnData) -> AnnData:
        """Step 3: 缺失值插补"""
        logg.info("\n[Step 3/6] 缺失值插补...")
        self._steps.append("impute")

        from ..preprocessing import impute

        method = settings.imputation.get("method", "auto")
        if method == "auto":
            method = None  # 由 selector 自动决定

        return impute(data, method=method)

    def _step_normalize(self, data: AnnData) -> AnnData:
        """Step 4: 归一化"""
        logg.info("\n[Step 4/6] 尺度归一化...")
        self._steps.append("normalize")

        from ..preprocessing import normalize

        method = settings.normalization.get("method", "auto")
        if method == "auto":
            method = None

        return normalize(data, method=method)

    def _step_batch_correct(self, data: AnnData) -> AnnData:
        """Step 5: 批次校正"""
        logg.info("\n[Step 5/6] 批次校正...")
        self._steps.append("batch_correct")

        from ..preprocessing import batch_correct

        method = settings.batch_correction.get("method", "auto")
        batch_key = settings.batch_correction.get("batch_key", "batch")
        if method == "auto":
            method = None

        return batch_correct(data, method=method, batch_key=batch_key)

    def _step_evaluate(self, data: AnnData) -> dict[str, float]:
        """Step 6: 效果评估"""
        logg.info("\n[Step 6/6] 效果评估...")
        self._steps.append("evaluate")

        from ..tools._evaluation import run_evaluation
        metrics = run_evaluation(data)
        self._results["metrics"] = metrics

        # 记录评估结果到 trace（修复溯源链断裂）
        data.uns["standardization"] = data.uns.get("standardization", {})
        data.uns["standardization"]["evaluation"] = {
            "metrics": metrics,
            "timestamp": str(datetime.now(timezone.utc)),
        }

        logg.info(f"  评估指标: {metrics}")
        return metrics

    def _save(
        self, data: AnnData, output_path: str | Path, use_storage: bool = False
    ) -> None:
        """保存结果到文件系统或混合存储后端

        Args:
            data: 处理后的 AnnData
            output_path: 输出路径
            use_storage: 是否同时保存到 StorageManager 混合存储后端
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # --- 文件系统保存（始终执行）---
        fmt = settings.output.get("format", "h5mu")

        if fmt == "h5mu":
            from mudata import MuData

            if isinstance(data, MuData):
                data.write(str(output_path.with_suffix(".h5mu")))
            else:
                MuData({"data": data}).write(str(output_path.with_suffix(".h5mu")))
        elif fmt == "h5ad":
            data.write(str(output_path.with_suffix(".h5ad")))
        else:
            data.write(str(output_path.with_suffix(f".{fmt}")))

        logg.info(f"结果已保存至 {output_path}")

        # --- StorageManager 混合存储集成 ---
        if not use_storage:
            return

        try:
            from ..storage import StorageManager

            store = StorageManager.from_settings()
            store.connect()

            try:
                experiment_id = output_path.stem

                # a. 存储处理后的 AnnData 到对象存储
                store.put_anndata(experiment_id, data)
                logg.info(f"  AnnData 已存储到对象存储: {experiment_id}")

                # b. 保存样本元数据
                sample_id = experiment_id
                modality = None
                strategy_info = (
                    data.uns.get("standardization", {}).get("strategy", {})
                )
                if isinstance(strategy_info, dict):
                    modality = strategy_info.get("modality")

                store.save_sample(
                    sample_id=sample_id,
                    experiment_id=experiment_id,
                    modality=modality,
                )
                logg.info(f"  样本元数据已保存: {sample_id}")

                # c. 从 adata.uns["standardization"] 提取方法详情并记录流水线运行
                std_info = data.uns.get("standardization", {})

                imputation_method = None
                normalization_method = None
                batch_correction_method = None

                for step_name, var_name in [
                    ("imputation", "imputation_method"),
                    ("normalization", "normalization_method"),
                    ("batch_correction", "batch_correction_method"),
                ]:
                    step_info = std_info.get(step_name, {})
                    if isinstance(step_info, dict):
                        val = step_info.get("method")
                        if var_name == "imputation_method":
                            imputation_method = val
                        elif var_name == "normalization_method":
                            normalization_method = val
                        elif var_name == "batch_correction_method":
                            batch_correction_method = val

                n_batches = None
                if "batch" in data.obs.columns:
                    n_batches = int(data.obs["batch"].nunique())

                run_id = store.record_pipeline_run(
                    experiment_id=experiment_id,
                    sample_id=sample_id,
                    imputation_method=imputation_method,
                    normalization_method=normalization_method,
                    batch_correction_method=batch_correction_method,
                    n_batches=n_batches,
                )
                logg.info(f"  流水线运行已记录: {run_id}")

                # d. 保存质量指标
                metrics = self._results.get("metrics", {})
                if metrics:
                    store.save_quality_metrics(run_id, metrics)
                    logg.info(f"  质量指标已保存: {run_id}")

                # e. 构建知识图谱
                store.build_knowledge_graph(data, experiment_id)

            finally:
                store.disconnect()

        except Exception as e:
            logg.warning(f"Storage 操作失败（非致命）: {e}")
