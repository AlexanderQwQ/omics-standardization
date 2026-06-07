"""混合存储架构模块

节点分离的混合存储层:
    - MinIO / S3 兼容对象存储: 大型 AnnData / HDF5 / 原始文件
    - 关系型数据库 (SQLite / DM8): 样本元数据、处理参数、质量指标
    - 图数据库 (Neo4j): 样本间关系、细胞相似性图、知识图谱

用法:
    from storage import StorageManager

    store = StorageManager(config="config/default.yaml")
    store.put_anndata("experiment_001", adata)
    store.save_metadata(sample_id="S001", metadata={"condition": "microgravity"})
    store.link_samples("S001", "S002", relation="same_batch")
"""

from __future__ import annotations

from ._base import BaseStorageClient
from ._minio import MinIOClient
from ._relational import RelationalDBClient
from ._graph import GraphDBClient
from ._manager import StorageManager

__all__ = [
    "BaseStorageClient",
    "MinIOClient",
    "RelationalDBClient",
    "GraphDBClient",
    "StorageManager",
]
