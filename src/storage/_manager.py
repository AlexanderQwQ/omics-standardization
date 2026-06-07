"""
StorageManager — 混合存储协调器

统一管理三种存储后端:
    - MinIO / S3: 大型文件对象存储
    - SQLite / DM8: 关系型元数据
    - Neo4j: 图数据库

生命周期:
    store = StorageManager(config_path_or_dict)
    store.connect()                        # 连接所有后端
    store.put_anndata("exp001", adata)     # 存储数据
    store.save_sample_metadata(...)         # 记录元数据
    store.build_knowledge_graph(adata)      # 构建知识图谱
    store.disconnect()                     # 断开所有连接
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import logging as logg
from .._settings import settings

if TYPE_CHECKING:
    from anndata import AnnData


class StorageManager:
    """混合存储协调器

    用法:
        store = StorageManager()

        # 方式 1: 自动从 settings 中加载配置
        store.connect()

        # 方式 2: 手动指定配置
        store = StorageManager.from_config("config/default.yaml")
        store.connect()

        with store:  # context manager 自动 connect/disconnect
            store.save_sample("S001", experiment_id="E001", modality="scrna")
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._minio: Any = None
        self._relational: Any = None
        self._graph: Any = None
        self._connected = False

    @classmethod
    def from_config(cls, path: str | Path) -> StorageManager:
        """从 YAML 配置文件加载存储配置"""
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)
        storage_config = full_config.get("storage", {})
        return cls(storage_config)

    @classmethod
    def from_settings(cls) -> StorageManager:
        """从全局 settings 对象加载存储配置"""
        try:
            storage_config = getattr(settings, "storage", {})
            return cls(storage_config)
        except Exception:
            return cls()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> StorageManager:
        """连接所有存储后端"""
        from ._minio import MinIOClient
        from ._relational import RelationalDBClient
        from ._graph import GraphDBClient

        # MinIO (对象存储)
        minio_config = self._config.get("minio", {})
        self._minio = MinIOClient(minio_config)
        self._minio.connect()

        # 关系型数据库
        relational_config = self._config.get("relational", {})
        self._relational = RelationalDBClient(relational_config)
        self._relational.connect()

        # 图数据库
        graph_config = self._config.get("graph", {})
        self._graph = GraphDBClient(graph_config)
        self._graph.connect()

        self._connected = True
        logg.info("混合存储管理器已就绪 (MinIO + RelationalDB + GraphDB)")
        return self

    def disconnect(self) -> None:
        """断开所有存储后端"""
        for client in [self._minio, self._relational, self._graph]:
            if client is not None and hasattr(client, "disconnect"):
                client.disconnect()
        self._connected = False
        logg.info("混合存储连接已关闭")

    def __enter__(self) -> StorageManager:
        return self.connect()

    def __exit__(self, *args: Any) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def minio(self) -> Any:
        return self._minio

    @property
    def db(self) -> Any:
        """关系型数据库客户端"""
        return self._relational

    @property
    def graph(self) -> Any:
        """图数据库客户端"""
        return self._graph

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # High-level workflow methods
    # ------------------------------------------------------------------

    def put_anndata(self, experiment_id: str, adata_or_path: AnnData | str | Path) -> str:
        """上传 AnnData 到对象存储

        Args:
            experiment_id: 实验 ID
            adata_or_path: AnnData 对象或文件路径

        Returns:
            存储 key
        """
        if self._minio is None:
            raise RuntimeError("存储管理器未连接，请先调用 connect()")

        if isinstance(adata_or_path, (str, Path)):
            return self._minio.put_anndata(experiment_id, adata_or_path)
        else:
            # 写入临时文件然后上传
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
                adata_or_path.write(tmp.name)
                return self._minio.put_anndata(experiment_id, tmp.name)

    def save_sample(
        self,
        sample_id: str,
        experiment_id: str,
        adata_or_path: AnnData | str | Path | None = None,
        *,
        modality: str | None = None,
        condition: str | None = None,
        organism: str | None = None,
        tissue: str | None = None,
        platform: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """一键保存样本：AnnData → MinIO + 元数据 → 关系型数据库

        Returns:
            sample_id
        """
        # 1. 上传 AnnData
        if adata_or_path is not None:
            if isinstance(adata_or_path, (str, Path)):
                self.put_anndata(experiment_id, adata_or_path)
            else:
                n_cells = adata_or_path.n_obs if hasattr(adata_or_path, "n_obs") else None
                n_features = adata_or_path.n_vars if hasattr(adata_or_path, "n_vars") else None
                self.put_anndata(experiment_id, adata_or_path)
            metadata = metadata or {}
            if "n_cells" not in metadata and n_cells is not None:
                metadata["n_cells"] = n_cells
            if "n_features" not in metadata and n_features is not None:
                metadata["n_features"] = n_features

        # 2. 保存元数据到关系型数据库
        if self._relational is not None:
            self._relational.save_sample(
                sample_id=sample_id,
                experiment_id=experiment_id,
                modality=modality,
                condition=condition,
                organism=organism,
                tissue=tissue,
                platform=platform,
                metadata=metadata,
            )

        return sample_id

    def record_pipeline_run(
        self,
        experiment_id: str,
        sample_id: str | None = None,
        *,
        imputation_method: str | None = None,
        normalization_method: str | None = None,
        batch_correction_method: str | None = None,
        n_batches: int | None = None,
        config_snapshot: dict[str, Any] | None = None,
    ) -> str:
        """记录一次流水线运行"""
        run_id = str(uuid.uuid4())[:8]

        if self._relational is not None:
            self._relational.save_pipeline_run(
                run_id=run_id,
                experiment_id=experiment_id,
                sample_id=sample_id,
                imputation_method=imputation_method,
                normalization_method=normalization_method,
                batch_correction_method=batch_correction_method,
                n_batches=n_batches,
                config_snapshot=config_snapshot,
            )

        return run_id

    def save_quality_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        """保存流水线质量指标"""
        if self._relational is not None:
            self._relational.save_metrics_batch(run_id, metrics)

    def build_knowledge_graph(self, adata: AnnData, experiment_id: str) -> None:
        """从 AnnData 构建知识图谱

        创建样本节点、批次节点及它们之间的关系。
        """
        if self._graph is not None and "batch" in adata.obs.columns:
            self._graph.build_batch_knowledge_graph(adata, batch_key="batch")

        # 为实验中的每个样本在图中创建节点
        if self._graph is not None and self._relational is not None:
            samples = self._relational.query_samples(experiment_id=experiment_id)
            for sample in samples:
                self._graph.create_sample_node(
                    sample_id=sample["sample_id"],
                    modality=sample.get("modality"),
                    condition=sample.get("condition"),
                    organism=sample.get("organism"),
                    batch=sample.get("batch"),
                )

        logg.info(f"实验 {experiment_id} 的知识图谱已构建")

    def export_experiment(self, experiment_id: str) -> dict[str, Any]:
        """导出实验的完整数据快照"""
        result: dict[str, Any] = {
            "experiment_id": experiment_id,
        }

        # 从关系型数据库导出
        if self._relational is not None:
            result["relational"] = self._relational.export_experiment(experiment_id)

        # 列出对象存储中的文件
        if self._minio is not None:
            result["objects"] = self._minio.list(f"experiments/{experiment_id}/")

        return result

    def health_check(self) -> dict[str, bool]:
        """检查所有存储后端的健康状态"""
        return {
            "minio": self._minio.is_healthy() if self._minio else False,
            "relational": self._relational.is_healthy() if self._relational else False,
            "graph": self._graph.is_healthy() if self._graph else False,
        }

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def batch_save_samples(
        self,
        samples: list[dict[str, Any]],
        experiment_id: str,
    ) -> list[str]:
        """批量保存样本

        Args:
            samples: 样本字典列表，每个包含 sample_id, modality, condition 等字段
            experiment_id: 实验 ID

        Returns:
            已保存的 sample_id 列表
        """
        saved: list[str] = []
        for sample in samples:
            sid = self.save_sample(
                sample_id=sample["sample_id"],
                experiment_id=experiment_id,
                modality=sample.get("modality"),
                condition=sample.get("condition"),
                organism=sample.get("organism"),
                tissue=sample.get("tissue"),
                platform=sample.get("platform"),
                metadata=sample.get("metadata"),
            )
            saved.append(sid)
        logg.info(f"批量保存完成: {len(saved)} 个样本")
        return saved
