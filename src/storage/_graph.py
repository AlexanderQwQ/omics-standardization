"""
图数据库客户端 (Neo4j)

存储关系型数据:
    - 样本间相似性关系 (sample_A) -[:CORRELATED {score: 0.95}]-> (sample_B)
    - 细胞谱系轨迹 (cell lineage trajectory)
    - 批次效应知识图谱 (batch → instrument → date → operator)
    - 多模态关联图 (transcriptomic ↔ proteomic ↔ metabolomic)

未安装 Neo4j 驱动时使用 JSON-LD 文件作为 fallback。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import logging as logg
from ._base import BaseStorageClient

if TYPE_CHECKING:
    pass


class GraphDBClient(BaseStorageClient):
    """Neo4j 图数据库客户端

    存储节点与关系，支持:
        - Cypher 查询
        - 批量节点/关系创建
        - 路径查询 (shortest path, BFS/DFS)
        - 知识图谱构建

    未安装 neo4j 驱动时使用 JSON-LD 文件 fallback。

    用法:
        graph = GraphDBClient(config={
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password",
            "database": "omics",
        })
        with graph:
            graph.create_sample_node("S001", modality="scrna", condition="microgravity")
            graph.link_samples("S001", "S002", relation="SAME_BATCH", score=0.87)
            path = graph.find_path("S001", "S005", max_depth=5)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._driver: Any = None
        self._database: str = self.config.get("database", "omics")
        self._local_fallback: Path | None = None
        self._graph_data: dict[str, Any] = {"nodes": [], "edges": []}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """建立 Neo4j 连接，失败则使用本地 JSON 文件 fallback"""
        try:
            from neo4j import GraphDatabase

            uri = self.config.get("uri", "bolt://localhost:7687")
            user = self.config.get("user", "neo4j")
            password = self.config.get("password", "neo4j")

            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            # 验证连接
            with self._driver.session(database=self._database) as session:
                session.run("RETURN 1")
            self._connected = True
            logg.info(f"Neo4j 已连接 (uri={uri}, db={self._database})")

        except ImportError:
            logg.warning("neo4j 驱动未安装，使用 JSON-LD 文件 fallback")
            self._setup_local_fallback()
        except Exception as exc:
            logg.warning(f"Neo4j 连接失败 ({exc})，使用 JSON-LD 文件 fallback")
            self._setup_local_fallback()

    def _setup_local_fallback(self) -> None:
        """设置本地 JSON-LD 文件 fallback"""
        fallback_dir = self.config.get("fallback_dir", "data/storage/graph")
        self._local_fallback = Path(fallback_dir)
        self._local_fallback.mkdir(parents=True, exist_ok=True)
        # 加载已有图数据
        graph_file = self._local_fallback / "graph.json"
        if graph_file.exists():
            try:
                self._graph_data = json.loads(graph_file.read_text("utf-8"))
            except json.JSONDecodeError:
                self._graph_data = {"nodes": [], "edges": []}
        self._connected = True
        logg.info(f"本地图存储 fallback: {self._local_fallback}")

    def disconnect(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
        # 持久化本地 fallback 数据
        self._flush_local()
        self._connected = False

    def _flush_local(self) -> None:
        """将内存中的图数据写入本地 JSON 文件"""
        if self._local_fallback is not None and self._graph_data:
            graph_file = self._local_fallback / "graph.json"
            graph_file.write_text(
                json.dumps(self._graph_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def is_healthy(self) -> bool:
        if not self._connected:
            return False
        if self._driver:
            try:
                with self._driver.session(database=self._database) as session:
                    session.run("RETURN 1")
                return True
            except Exception:
                return False
        return self._local_fallback is not None and self._local_fallback.is_dir()

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def put(self, key: str, data: bytes | str | Path, metadata: dict[str, str] | None = None) -> str:
        return key

    def get(self, key: str) -> bytes | None:
        return None

    def list(self, prefix: str = "") -> list[str]:
        labels: list[str] = []
        for node in self._graph_data["nodes"]:
            for label in node.get("labels", []):
                if label.startswith(prefix) and label not in labels:
                    labels.append(label)
        return labels

    def delete(self, key: str) -> bool:
        if self._driver:
            with self._driver.session(database=self._database) as session:
                session.run(f"MATCH (n:{key}) DETACH DELETE n")
        self._graph_data["nodes"] = [n for n in self._graph_data["nodes"] if n.get("id") != key]
        self._graph_data["edges"] = [e for e in self._graph_data["edges"] if e["from"] != key and e["to"] != key]
        return True

    def exists(self, key: str) -> bool:
        return any(n.get("id") == key for n in self._graph_data["nodes"])

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def create_sample_node(
        self,
        sample_id: str,
        modality: str | None = None,
        condition: str | None = None,
        organism: str | None = None,
        batch: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """创建样本节点

        Args:
            sample_id: 样本唯一 ID
            modality: 模态 (scrna, bulk_rna, proteomics, ...)
            condition: 实验条件
            organism: 物种
            batch: 批次标签
            properties: 额外属性

        Returns:
            sample_id
        """
        props = {
            "sample_id": sample_id,
            "modality": modality or "unknown",
            "condition": condition or "unknown",
            "organism": organism or "unknown",
            "batch": batch or "unknown",
        }
        if properties:
            props.update(properties)

        if self._driver:
            self._run_cypher(
                """
                MERGE (s:Sample {sample_id: $sample_id})
                SET s = $props
                RETURN s
                """,
                {"sample_id": sample_id, "props": props},
            )
        else:
            # Local fallback
            existing = next((n for n in self._graph_data["nodes"] if n.get("id") == sample_id), None)
            if existing:
                existing["properties"] = props
            else:
                self._graph_data["nodes"].append({
                    "id": sample_id,
                    "type": "Sample",
                    "labels": ["Sample"],
                    "properties": props,
                })

        logg.info(f"图节点已创建: Sample({sample_id})")
        return sample_id

    def create_batch_node(self, batch_id: str, properties: dict[str, Any] | None = None) -> str:
        """创建批次节点"""
        props = {"batch_id": batch_id}
        if properties:
            props.update(properties)

        if self._driver:
            self._run_cypher(
                "MERGE (b:Batch {batch_id: $batch_id}) SET b = $props",
                {"batch_id": batch_id, "props": props},
            )
        else:
            existing = next((n for n in self._graph_data["nodes"] if n.get("id") == batch_id), None)
            if existing:
                existing["properties"] = props
            else:
                self._graph_data["nodes"].append({
                    "id": batch_id,
                    "type": "Batch",
                    "labels": ["Batch"],
                    "properties": props,
                })

        return batch_id

    # ------------------------------------------------------------------
    # Relationship operations
    # ------------------------------------------------------------------

    def link_samples(
        self,
        source_id: str,
        target_id: str,
        relation: str = "CORRELATED",
        score: float | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """创建样本间关系

        Args:
            source_id: 源样本 ID
            target_id: 目标样本 ID
            relation: 关系类型 (SAME_BATCH, CORRELATED, DIFFERENTIAL, SAME_TRAJECTORY, ...)
            score: 关系权重/置信度
            properties: 额外属性
        """
        props = properties or {}
        if score is not None:
            props["score"] = score

        if self._driver:
            rel_upper = relation.upper()
            self._run_cypher(
                f"""
                MATCH (a:Sample {{sample_id: $source_id}})
                MATCH (b:Sample {{sample_id: $target_id}})
                MERGE (a)-[r:{rel_upper}]->(b)
                SET r = $props
                """,
                {"source_id": source_id, "target_id": target_id, "props": props},
            )
        else:
            self._graph_data["edges"].append({
                "from": source_id,
                "to": target_id,
                "type": relation,
                "properties": props,
            })
            self._flush_local()

        logg.info(f"图关系已创建: ({source_id})-[:{relation}]->({target_id})")

    def link_sample_to_batch(self, sample_id: str, batch_id: str, properties: dict[str, Any] | None = None) -> None:
        """将样本关联到批次"""
        self.link_samples(sample_id, batch_id, relation="BELONGS_TO_BATCH", properties=properties)

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def find_path(self, source_id: str, target_id: str, max_depth: int = 5) -> list[dict[str, Any]]:
        """查找两个样本之间的最短路径

        Returns:
            路径列表，每个元素为 {nodes: [...], relationships: [...]}
        """
        if self._driver:
            result = self._run_cypher(
                """
                MATCH path = shortestPath(
                    (a:Sample {sample_id: $source_id})-[*1..$max_depth]-(b:Sample {sample_id: $target_id})
                )
                RETURN path LIMIT 1
                """,
                {"source_id": source_id, "target_id": target_id, "max_depth": max_depth},
            )
            return [dict(r) for r in result]
        else:
            # 本地 BFS
            return self._local_bfs(source_id, target_id, max_depth)

    def get_neighbors(self, sample_id: str, relation: str | None = None, depth: int = 1) -> list[dict[str, Any]]:
        """获取样本的邻居节点

        Args:
            sample_id: 样本 ID
            relation: 限定关系类型（None 表示所有类型）
            depth: 邻居深度（1 = 直接邻居）
        """
        if self._driver:
            rel_clause = f":{relation.upper()}" if relation else ""
            result = self._run_cypher(
                f"""
                MATCH (s:Sample {{sample_id: $sample_id}})-[r{rel_clause}*1..$depth]-(neighbor)
                RETURN DISTINCT neighbor, r
                """,
                {"sample_id": sample_id, "depth": depth},
            )
            return [dict(r) for r in result]
        else:
            neighbors: list[dict[str, Any]] = []
            for edge in self._graph_data["edges"]:
                if relation and edge["type"] != relation:
                    continue
                if edge["from"] == sample_id:
                    target = next((n for n in self._graph_data["nodes"] if n["id"] == edge["to"]), None)
                    if target:
                        neighbors.append({"node": target, "relationship": edge})
                elif edge["to"] == sample_id:
                    source = next((n for n in self._graph_data["nodes"] if n["id"] == edge["from"]), None)
                    if source:
                        neighbors.append({"node": source, "relationship": edge})
            return neighbors[:depth * 10]

    def get_correlated_samples(self, sample_id: str, min_score: float = 0.8) -> list[dict[str, Any]]:
        """获取与给定样本高度相关的其他样本"""
        neighbors = self.get_neighbors(sample_id, relation="CORRELATED")
        return [
            n for n in neighbors
            if n.get("relationship", {}).get("properties", {}).get("score", 0) >= min_score
        ]

    def query_nodes(
        self,
        label: str,
        properties: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """按标签和属性查询节点"""
        if self._driver:
            where_clauses = []
            params: dict[str, Any] = {"limit": limit}
            if properties:
                for i, (k, v) in enumerate(properties.items()):
                    param_key = f"prop_{i}"
                    where_clauses.append(f"n.{k} = ${param_key}")
                    params[param_key] = v

            where_str = " AND ".join(where_clauses)
            if where_str:
                where_str = "WHERE " + where_str

            result = self._run_cypher(
                f"MATCH (n:{label}) {where_str} RETURN n LIMIT $limit",
                params,
            )
            return [dict(r) for r in result]
        else:
            matches = []
            for node in self._graph_data["nodes"]:
                if label not in node.get("labels", []):
                    continue
                if properties:
                    node_props = node.get("properties", {})
                    if all(node_props.get(k) == v for k, v in properties.items()):
                        matches.append(node)
                else:
                    matches.append(node)
            return matches[:limit]

    # ------------------------------------------------------------------
    # Knowledge graph construction
    # ------------------------------------------------------------------

    def build_batch_knowledge_graph(self, adata: Any, batch_key: str = "batch") -> None:
        """从 AnnData 构建批次效应知识图谱

        节点: Sample, Batch, Condition, Platform
        关系: BELONGS_TO_BATCH, HAS_CONDITION, RUN_ON_PLATFORM
        """
        if batch_key not in adata.obs.columns:
            logg.warning(f"批次列 '{batch_key}' 不在 obs 中")
            return

        batches = adata.obs[batch_key].unique()
        for batch in batches:
            batch_id = f"batch_{batch}"
            self.create_batch_node(batch_id, properties={
                "batch_label": str(batch),
                "n_samples": int((adata.obs[batch_key] == batch).sum()),
            })

        # 创建样本节点（每批取一个代表）
        for batch in batches:
            mask = adata.obs[batch_key] == batch
            sample_indices = adata.obs.index[mask][:5]  # 最多 5 个
            for idx in sample_indices:
                sample_id = f"sample_{idx}"
                self.create_sample_node(
                    sample_id,
                    batch=str(batch),
                    properties={"index": str(idx)},
                )
                self.link_sample_to_batch(sample_id, f"batch_{batch}")

        logg.info(f"批次知识图谱已构建: {len(batches)} 个批次")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_cypher(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """执行 Cypher 查询"""
        if self._driver is None:
            logg.warning("图数据库未连接")
            return []
        with self._driver.session(database=self._database) as session:
            return list(session.run(query, params or {}))

    def _local_bfs(self, source_id: str, target_id: str, max_depth: int) -> list[dict[str, Any]]:
        """本地 BFS 路径搜索（fallback 模式）"""
        from collections import deque

        # 构建邻接表
        adjacency: dict[str, list[tuple[str, dict]]] = {}
        for edge in self._graph_data["edges"]:
            frm, to = edge["from"], edge["to"]
            adjacency.setdefault(frm, []).append((to, edge))
            adjacency.setdefault(to, []).append((frm, edge))

        if source_id not in adjacency:
            return []

        queue: deque[tuple[str, list[str], list[dict]]] = deque()
        queue.append((source_id, [source_id], []))
        visited: set[str] = {source_id}

        while queue:
            current, path_nodes, path_edges = queue.popleft()
            if len(path_nodes) - 1 > max_depth:
                continue

            if current == target_id:
                return [{"nodes": path_nodes, "relationships": path_edges}]

            for neighbor, edge in adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((
                        neighbor,
                        path_nodes + [neighbor],
                        path_edges + [edge],
                    ))

        return []
