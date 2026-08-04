"""
端到端标准化流水线主类

串联整个处理流程:
    Parse → Select Strategy → Impute → Normalize → Batch Correct → Evaluate
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import _logging as logg
from _settings import settings

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
    ) -> AnnData | MuData:
        """运行完整标准化流水线

        Args:
            input_path: 输入数据路径（文件或目录）。
                       目录时自动批量处理所有子目录下的数据文件。
            output_path: 输出路径（文件或目录）
            data: 已解析的 AnnData/MuData（跳过解析步骤）
            use_storage: 是否同时保存到 StorageManager 混合存储后端

        Returns:
            标准化后的 AnnData 或 MuData（批量模式返回合并的 MuData）
        """
        logg.info("=" * 60)
        logg.info("开始标准化流水线")
        logg.info("=" * 60)

        # --- 批量模式：input_path 是目录 ---
        if data is None and input_path is not None and Path(input_path).is_dir():
            return self._run_batch(
                Path(input_path), Path(output_path) if output_path else Path("data/processed"), use_storage
            )

        # --- 单文件模式 ---
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

        from parsers import parse_file
        return parse_file(input_path)

    def _step_select_strategy(self, data: AnnData) -> dict[str, str]:
        """Step 2: 选择处理策略"""
        logg.info("\n[Step 2/6] 选择处理策略...")
        self._steps.append("select")

        from _selectors import recommend_strategy, detect_modality
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

        from preprocessing import impute

        method = settings.imputation.get("method", "auto")
        if method == "auto":
            method = None  # 由 selector 自动决定

        return impute(data, method=method)

    def _step_normalize(self, data: AnnData) -> AnnData:
        """Step 4: 归一化"""
        logg.info("\n[Step 4/6] 尺度归一化...")
        self._steps.append("normalize")

        from preprocessing import normalize

        method = settings.normalization.get("method", "auto")
        if method == "auto":
            method = None

        return normalize(data, method=method)

    def _step_batch_correct(self, data: AnnData) -> AnnData:
        """Step 5: 批次校正"""
        logg.info("\n[Step 5/6] 批次校正...")
        self._steps.append("batch_correct")

        from preprocessing import batch_correct

        method = settings.batch_correction.get("method", "auto")
        batch_key = settings.batch_correction.get("batch_key", "batch")
        if method == "auto":
            method = None

        return batch_correct(data, method=method, batch_key=batch_key)

    def _step_evaluate(self, data: AnnData) -> dict[str, float]:
        """Step 6: 效果评估"""
        logg.info("\n[Step 6/6] 效果评估...")
        self._steps.append("evaluate")

        from tools._evaluation import run_evaluation
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

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_best_file(files: list[Path]) -> Path:
        """从同一模态的多个文件中选取最优格式

        优先级（原生仪器/标准格式优先于通用导出格式）:
            .h5ad > .fcs > .mzml > .biom > .csv > .tsv > .txt

        Args:
            files: 同一模态目录下的文件路径列表

        Returns:
            选定的最优文件路径
        """
        if len(files) == 1:
            return files[0]

        _FORMAT_PRIORITY: dict[str, int] = {
            ".h5ad": 0, ".h5mu": 0,
            ".fcs": 1,
            ".mzml": 2, ".mzml.gz": 2,
            ".biom": 3, ".biom.gz": 3,
            ".csv": 4, ".csv.gz": 4, ".tsv": 5, ".tsv.gz": 5, ".txt": 6, ".txt.gz": 6,
            ".fastq": 7, ".fastq.gz": 7, ".fq": 7, ".fq.gz": 7,
        }

        def _score(f: Path) -> int:
            name = f.name.lower()
            if name.endswith((".mtx.gz", ".fastq.gz", ".fq.gz", ".mzml.gz", ".csv.gz", ".tsv.gz", ".txt.gz", ".biom.gz")):
                suffix = "." + ".".join(name.split(".")[-2:])
            else:
                suffix = f.suffix.lower()
            return _FORMAT_PRIORITY.get(suffix, 99)

        best = min(files, key=_score)
        logg.hint(f"从 {len(files)} 个文件中选定最优格式: {best.name} (优先级 {_score(best)})")
        return best

    @staticmethod
    def _merge_to_combined_mudata(results: dict[str, AnnData]) -> MuData:
        """将多模态处理结果合并为一个 MuData 容器

        Args:
            results: {modality_name: AnnData} 字典

        Returns:
            包含所有模态的 MuData 对象
        """
        from mudata import MuData

        combined = MuData(results)
        logg.info(f"合并多模态 MuData: {list(results.keys())}")
        return combined

    def _run_batch(
        self,
        input_dir: Path,
        output_dir: Path,
        use_storage: bool = False,
    ) -> MuData:
        """批量处理目录下所有子目录的数据文件

        流程:
            1. 扫描目录发现全部支持文件
            2. 按父目录名分组（每组对应一种模态）
            3. 每组选取最优格式文件
            4. 逐个处理（6 步流水线）
            5. 合并为汇总 MuData

        Args:
            input_dir: 输入数据根目录（如 data/raw/）
            output_dir: 输出根目录（如 data/processed/）
            use_storage: 是否保存到 StorageManager

        Returns:
            包含所有模态处理结果的合并 MuData
        """
        from parsers._utils import list_supported_files

        # 1. 发现全部支持文件
        all_files = list_supported_files(input_dir)
        if not all_files:
            raise ValueError(
                f"目录 {input_dir} 中未找到支持的数据文件。"
                f" 支持的扩展名: .h5ad, .csv, .fcs, .mzml, .biom, .fastq, ..."
            )

        # 2. 按父目录（模态）分组
        grouped: dict[str, list[Path]] = defaultdict(list)
        for f in all_files:
            try:
                relative = f.relative_to(input_dir)
            except ValueError:
                continue
            modality = relative.parts[0] if len(relative.parts) > 1 else "unknown"
            grouped[modality].append(f)

        # 3. 每模态选最优格式
        selected: dict[str, Path] = {}
        for modality, files in grouped.items():
            selected[modality] = self._pick_best_file(files)

        batch_config = settings.output.get("batch", {})
        combined_filename = batch_config.get("combined_filename", "combined")
        per_modality_subdir = batch_config.get("per_modality_subdir", True)

        logg.info(
            f"批量处理: {len(selected)} 种模态, "
            f"共 {len(all_files)} 个文件, "
            f"选定 {len(selected)} 个最优文件"
        )

        # 4. 逐个处理
        results: dict[str, AnnData] = {}
        n_total = len(selected)

        for idx, (modality, file_path) in enumerate(selected.items(), 1):
            logg.info(f"\n{'─' * 50}")
            logg.info(f"[{idx}/{n_total}] 处理 {modality}: {file_path.name}")
            logg.info(f"{'─' * 50}")

            try:
                data = self._step_parse(file_path)
                self._step_select_strategy(data)
                data = self._step_impute(data)
                data = self._step_normalize(data)
                data = self._step_batch_correct(data)
                self._step_evaluate(data)

                results[modality] = data

                # 保存单模态结果
                if per_modality_subdir:
                    modality_out = output_dir / modality / file_path.stem
                else:
                    modality_out = output_dir / modality
                self._save(data, modality_out, use_storage=use_storage)

            except Exception as exc:
                logg.error(f"  [{modality}] 处理失败: {type(exc).__name__}: {exc}")
                continue

        if not results:
            raise RuntimeError(
                f"批量处理失败：目录 {input_dir} 中所有 {n_total} 种模态均处理失败"
            )

        # 5. 合并为汇总 MuData
        combined = self._merge_to_combined_mudata(results)
        combined_out = output_dir / combined_filename
        combined_out.parent.mkdir(parents=True, exist_ok=True)

        from mudata import MuData

        if isinstance(combined, MuData):
            combined.write(str(combined_out.with_suffix(".h5mu")))

        logg.info(f"\n{'=' * 60}")
        logg.info(f"批量处理完成: {len(results)}/{n_total} 种模态成功")
        logg.info(f"汇总 MuData 已保存至 {combined_out}.h5mu")
        logg.info(f"{'=' * 60}")

        return combined

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
            from storage import StorageManager

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
