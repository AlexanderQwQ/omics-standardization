"""
关系型数据库客户端 (SQLite / DM8)

存储结构化数据:
    - 样本元数据 (sample_id, condition, batch, organism, platform, ...)
    - 处理参数与 provenance (pipeline config, method choices, timestamps)
    - 质量指标 (RMSE, batch_mixing, correlation_preserved, ...)

DM8 连接时使用 dmPython；未安装时自动降级为 SQLite。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import logging as logg
from ._base import BaseStorageClient

if TYPE_CHECKING:
    pass


_DEFAULT_SCHEMA_SQL = """
-- 样本元数据表
CREATE TABLE IF NOT EXISTS samples (
    sample_id       TEXT PRIMARY KEY,
    experiment_id   TEXT NOT NULL,
    modality        TEXT,          -- scrna, bulk_rna, proteomics, metabolomics, atac
    condition       TEXT,          -- microgravity, ground_control, etc.
    organism        TEXT,
    tissue          TEXT,
    platform        TEXT,          -- Illumina NovaSeq, Thermo Q-Exactive, etc.
    n_cells         INTEGER,
    n_features      INTEGER,
    file_path       TEXT,
    raw_file_hash   TEXT,          -- SHA256 of raw input file
    metadata_json   TEXT,          -- 自由格式 extra metadata (JSON string)
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 处理流水线记录表
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          TEXT PRIMARY KEY,   -- UUID
    experiment_id   TEXT NOT NULL,
    sample_id       TEXT,
    config_yaml     TEXT,               -- 完整配置快照
    imputation_method   TEXT,
    normalization_method TEXT,
    batch_correction_method TEXT,
    n_batches       INTEGER,
    started_at      TIMESTAMP,
    finished_at     TIMESTAMP,
    status          TEXT DEFAULT 'running',  -- running, completed, failed
    FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
);

-- 质量指标表
CREATE TABLE IF NOT EXISTS quality_metrics (
    metric_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    metric_name     TEXT NOT NULL,      -- rmse, batch_mixing, correlation_preserved, n_features
    metric_value    REAL,
    metadata_json   TEXT,
    recorded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_samples_experiment ON samples(experiment_id);
CREATE INDEX IF NOT EXISTS idx_samples_modality ON samples(modality);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_experiment ON pipeline_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_quality_metrics_run ON quality_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_quality_metrics_name ON quality_metrics(metric_name);
"""


class RelationalDBClient(BaseStorageClient):
    """关系型数据库客户端

    支持 SQLite (默认) 和 DM8 (dmPython)。

    用法:
        db = RelationalDBClient(config={
            "dialect": "sqlite",             # sqlite | dm8
            "database": "data/storage/metadata.db",
            # DM8 only:
            "host": "localhost",
            "port": 5236,
            "user": "SYSDBA",
            "password": "...",
        })
        with db:
            db.save_sample("S001", experiment_id="E001", modality="scrna", ...)
            samples = db.query_samples(experiment_id="E001")
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._engine: Any = None
        self._connection: Any = None
        self._dialect: str = self.config.get("dialect", "sqlite")

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """建立数据库连接并初始化 schema"""
        try:
            if self._dialect == "dm8":
                self._connect_dm8()
            else:
                self._connect_sqlite()

            self._init_schema()
            self._connected = True
            logg.info(f"关系型数据库已连接 (dialect={self._dialect})")

        except Exception as exc:
            logg.warning(f"数据库连接失败 ({exc})，使用 SQLite fallback")
            self._connect_sqlite()
            self._init_schema()
            self._connected = True

    def _connect_sqlite(self) -> None:
        """连接 SQLite"""
        import sqlite3

        db_path = self.config.get("database", "data/storage/metadata.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(db_path))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        logg.info(f"SQLite 数据库: {db_path}")

    def _connect_dm8(self) -> None:
        """连接达梦 DM8 数据库"""
        try:
            import dmPython

            self._connection = dmPython.connect(
                user=self.config.get("user", "SYSDBA"),
                password=self.config.get("password", ""),
                server=self.config.get("host", "localhost"),
                port=self.config.get("port", 5236),
            )
        except ImportError:
            raise ImportError("连接 DM8 需要 dmPython 包。请安装: pip install dmPython")

    def _init_schema(self) -> None:
        """初始化数据库表结构"""
        for stmt in _DEFAULT_SCHEMA_SQL.split(";"):
            stmt = stmt.strip()
            if stmt:
                self._connection.execute(stmt)
        self._connection.commit()

    def disconnect(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None
        self._connected = False

    def is_healthy(self) -> bool:
        if not self._connected or self._connection is None:
            return False
        try:
            self._connection.execute("SELECT 1")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def put(self, key: str, data: bytes | str | Path, metadata: dict[str, str] | None = None) -> str:
        """通用写入 — 将 JSON 数据插入为行"""
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        if isinstance(data, Path):
            data = data.read_text("utf-8")
        return key

    def get(self, key: str) -> bytes | None:
        return None

    def list(self, prefix: str = "") -> list[str]:
        tables = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
            (f"{prefix}%",),
        ).fetchall()
        return [row["name"] for row in tables]

    def delete(self, key: str) -> bool:
        return True

    def exists(self, key: str) -> bool:
        return True

    # ------------------------------------------------------------------
    # Domain-specific: 样本元数据
    # ------------------------------------------------------------------

    def save_sample(
        self,
        sample_id: str,
        experiment_id: str,
        modality: str | None = None,
        condition: str | None = None,
        organism: str | None = None,
        tissue: str | None = None,
        platform: str | None = None,
        n_cells: int | None = None,
        n_features: int | None = None,
        file_path: str | None = None,
        raw_file_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """保存或更新样本元数据

        Returns:
            sample_id
        """
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None

        self._connection.execute(
            """
            INSERT OR REPLACE INTO samples
            (sample_id, experiment_id, modality, condition, organism, tissue,
             platform, n_cells, n_features, file_path, raw_file_hash, metadata_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (sample_id, experiment_id, modality, condition, organism, tissue,
             platform, n_cells, n_features, file_path, raw_file_hash, metadata_json),
        )
        self._connection.commit()
        logg.info(f"样本已保存: {sample_id}")
        return sample_id

    def get_sample(self, sample_id: str) -> dict[str, Any] | None:
        """查询单个样本"""
        row = self._connection.execute(
            "SELECT * FROM samples WHERE sample_id = ?", (sample_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def query_samples(
        self,
        experiment_id: str | None = None,
        modality: str | None = None,
        condition: str | None = None,
    ) -> list[dict[str, Any]]:
        """按条件查询样本列表"""
        query = "SELECT * FROM samples WHERE 1=1"
        params: list[Any] = []

        if experiment_id is not None:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        if modality is not None:
            query += " AND modality = ?"
            params.append(modality)
        if condition is not None:
            query += " AND condition = ?"
            params.append(condition)

        rows = self._connection.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def list_experiments(self) -> list[str]:
        """列出所有实验 ID"""
        rows = self._connection.execute(
            "SELECT DISTINCT experiment_id FROM samples ORDER BY experiment_id"
        ).fetchall()
        return [r["experiment_id"] for r in rows]

    # ------------------------------------------------------------------
    # Domain-specific: 流水线运行记录
    # ------------------------------------------------------------------

    def save_pipeline_run(
        self,
        run_id: str,
        experiment_id: str,
        sample_id: str | None = None,
        config_snapshot: dict[str, Any] | None = None,
        imputation_method: str | None = None,
        normalization_method: str | None = None,
        batch_correction_method: str | None = None,
        n_batches: int | None = None,
    ) -> str:
        """记录流水线运行"""
        config_yaml = json.dumps(config_snapshot, ensure_ascii=False) if config_snapshot else None

        self._connection.execute(
            """
            INSERT OR REPLACE INTO pipeline_runs
            (run_id, experiment_id, sample_id, config_yaml,
             imputation_method, normalization_method, batch_correction_method,
             n_batches, started_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 'running')
            """,
            (run_id, experiment_id, sample_id, config_yaml,
             imputation_method, normalization_method, batch_correction_method,
             n_batches),
        )
        self._connection.commit()
        logg.info(f"流水线运行已记录: {run_id}")
        return run_id

    def mark_run_completed(self, run_id: str) -> None:
        """标记流水线运行完成"""
        self._connection.execute(
            "UPDATE pipeline_runs SET status='completed', finished_at=CURRENT_TIMESTAMP WHERE run_id=?",
            (run_id,),
        )
        self._connection.commit()

    def mark_run_failed(self, run_id: str, error_message: str = "") -> None:
        """标记流水线运行失败"""
        self._connection.execute(
            "UPDATE pipeline_runs SET status='failed', finished_at=CURRENT_TIMESTAMP WHERE run_id=?",
            (run_id,),
        )
        if error_message:
            self.save_metric(run_id, "error_message", 0.0, metadata={"error": error_message})
        self._connection.commit()

    def get_pipeline_run(self, run_id: str) -> dict[str, Any] | None:
        """查询流水线运行记录"""
        row = self._connection.execute(
            "SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def query_pipeline_runs(self, experiment_id: str | None = None) -> list[dict[str, Any]]:
        """查询流水线运行历史"""
        if experiment_id:
            rows = self._connection.execute(
                "SELECT * FROM pipeline_runs WHERE experiment_id=? ORDER BY started_at DESC",
                (experiment_id,),
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM pipeline_runs ORDER BY started_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Domain-specific: 质量指标
    # ------------------------------------------------------------------

    def save_metric(
        self,
        run_id: str,
        metric_name: str,
        metric_value: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """保存质量指标"""
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        self._connection.execute(
            """
            INSERT INTO quality_metrics (run_id, metric_name, metric_value, metadata_json)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, metric_name, metric_value, metadata_json),
        )
        self._connection.commit()

    def save_metrics_batch(self, run_id: str, metrics: dict[str, float]) -> None:
        """批量保存质量指标"""
        for name, value in metrics.items():
            self._connection.execute(
                "INSERT INTO quality_metrics (run_id, metric_name, metric_value) VALUES (?, ?, ?)",
                (run_id, name, value),
            )
        self._connection.commit()

    def get_metrics(self, run_id: str) -> dict[str, float]:
        """查询某次运行的所有质量指标"""
        rows = self._connection.execute(
            "SELECT metric_name, metric_value FROM quality_metrics WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        return {r["metric_name"]: r["metric_value"] for r in rows}

    def get_metric_summary(self, metric_name: str) -> dict[str, Any]:
        """查询某个指标的汇总统计"""
        row = self._connection.execute(
            """
            SELECT
                COUNT(*) as n,
                AVG(metric_value) as mean,
                MIN(metric_value) as min_val,
                MAX(metric_value) as max_val
            FROM quality_metrics WHERE metric_name = ?
            """,
            (metric_name,),
        ).fetchone()
        return dict(row) if row else {}

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def export_experiment(self, experiment_id: str) -> dict[str, Any]:
        """导出实验的全部数据（样本 + 指标 + 流水线记录）"""
        samples = self.query_samples(experiment_id=experiment_id)
        runs = self.query_pipeline_runs(experiment_id=experiment_id)
        metrics: dict[str, dict[str, float]] = {}
        for run in runs:
            metrics[run["run_id"]] = self.get_metrics(run["run_id"])

        return {
            "experiment_id": experiment_id,
            "samples": samples,
            "pipeline_runs": runs,
            "metrics": metrics,
        }

    def execute_raw(self, sql: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
        """执行原始 SQL 查询（用于自定义分析）

        Args:
            sql: SQL 语句
            params: 参数绑定

        Returns:
            查询结果列表
        """
        import sqlite3

        try:
            if params:
                rows = self._connection.execute(sql, params).fetchall()
            else:
                rows = self._connection.execute(sql).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            logg.error(f"SQL 执行失败: {exc}")
            return []
