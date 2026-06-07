"""
Settings 配置管理类（scanpy 风格）

读取 config/default.yaml 并暴露为 Python 属性，
可通过环境变量或代码覆盖。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ._verbosity import Verbosity

if TYPE_CHECKING:
    from types import TracebackType


# 默认配置（与 config/default.yaml 保持一致）
_DEFAULT_CONFIG: dict[str, Any] = {
    "modalities": {
        "enabled": ["scrna", "bulk_rna", "proteomics", "metabolomics", "atac"],
        "default": "scrna",
    },
    "imputation": {
        "method": "auto",  # auto | missforest | zinb_vae | magic | none
        "missing_threshold": 0.1,
        "n_estimators": 100,
    },
    "normalization": {
        "method": "auto",  # auto | tmm | deseq2 | scran | quantile | vsn
        "log_transform": True,
        "scale": True,
    },
    "batch_correction": {
        "method": "auto",  # auto | combat | harmony | dann | none
        "batch_key": "batch",
    },
    "selector": {
        "modality_model": "gmm",
        "strategy_model": "random_forest",
        "model_dir": "config/models",
    },
    "logging": {
        "level": "info",
        "to_file": True,
        "log_dir": "logs",
    },
    "output": {
        "format": "h5mu",  # h5mu | h5ad | parquet
        "compress": True,
        "save_reports": True,
    },
}


class Settings:
    """全局配置管理器

    属性：
        modalities: 模态配置
        imputation: 插补策略配置
        normalization: 归一化配置
        batch_correction: 批次校正配置
        selector: 算法选择器配置
        logging: 日志配置
        output: 输出配置
    """

    def __init__(self) -> None:
        self._config = _DEFAULT_CONFIG.copy()
        self._verbosity = Verbosity.info
        self._root_logger = logging.getLogger("omics_std")
        self._root_logger.setLevel(logging.INFO)
        self._setup_logger()

    # ------------------------------------------------------------------
    # 属性访问
    # ------------------------------------------------------------------

    @property
    def modalities(self) -> dict:
        return self._config.get("modalities", {})

    @property
    def imputation(self) -> dict:
        return self._config.get("imputation", {})

    @property
    def normalization(self) -> dict:
        return self._config.get("normalization", {})

    @property
    def batch_correction(self) -> dict:
        return self._config.get("batch_correction", {})

    @property
    def selector(self) -> dict:
        return self._config.get("selector", {})

    @property
    def output(self) -> dict:
        return self._config.get("output", {})

    @property
    def verbosity(self) -> Verbosity:
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value: Verbosity | str | int) -> None:
        if isinstance(value, str):
            value = Verbosity[value]
        elif isinstance(value, int):
            value = Verbosity(value)
        self._verbosity = value
        # 更新根日志级别
        level_map = {
            Verbosity.error: logging.ERROR,
            Verbosity.warning: logging.WARNING,
            Verbosity.info: logging.INFO,
            Verbosity.hint: logging.DEBUG,
            Verbosity.debug: logging.DEBUG,
        }
        self._root_logger.setLevel(level_map.get(value, logging.INFO))

    @property
    def _root_logger(self) -> logging.Logger:
        return logging.getLogger("omics_std")

    # ------------------------------------------------------------------
    # 配置加载
    # ------------------------------------------------------------------

    def load_config(self, path: str | Path) -> None:
        """从 YAML 文件加载配置，合并到现有配置上"""
        path = Path(path)
        if not path.exists():
            import warnings
            warnings.warn(f"配置文件 {path} 不存在，使用默认配置", stacklevel=2)
            return

        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}

        # 递归合并
        self._config = self._deep_merge(self._config, loaded)

        # 应用日志配置
        self._apply_logging_config()

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """递归合并两个字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _apply_logging_config(self) -> None:
        """应用日志配置到 root logger"""
        log_cfg = self._config.get("logging", {})
        level_name = log_cfg.get("level", "info").upper()
        self._root_logger.setLevel(getattr(logging, level_name, logging.INFO))

        if log_cfg.get("to_file", True):
            log_dir = Path(log_cfg.get("log_dir", "logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            if not any(isinstance(h, logging.FileHandler) for h in self._root_logger.handlers):
                from logging import FileHandler
                handler = FileHandler(log_dir / "pipeline.log", encoding="utf-8")
                handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
                self._root_logger.addHandler(handler)

    def _setup_logger(self) -> None:
        """初始化日志处理器"""
        self._root_logger.handlers.clear()
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self._root_logger.addHandler(handler)

    def __repr__(self) -> str:
        return f"Settings(verbosity={self._verbosity.name})"
