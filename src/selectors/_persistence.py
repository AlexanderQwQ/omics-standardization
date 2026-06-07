"""选择器模型持久化

保存/加载训练好的 GMM 模态识别模型和 RF 策略推荐模型。
模型文件存储在 config/models/ 目录下。

文件结构:
    config/models/
    ├── modality_gmm.joblib
    ├── strategy_rf_imputation.joblib
    ├── strategy_rf_normalization.joblib
    ├── strategy_rf_batch.joblib
    └── training_metadata.json
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import logging as logg

if TYPE_CHECKING:
    from sklearn.mixture import GaussianMixture
    from sklearn.ensemble import RandomForestClassifier

_DEFAULT_MODEL_DIR = Path("config/models")


def save_model(model: Any, name: str, model_dir: str | Path = _DEFAULT_MODEL_DIR) -> Path:
    """保存模型到文件

    Args:
        model: sklearn 模型对象
        name: 模型名称（不含扩展名）
        model_dir: 模型目录

    Returns:
        保存的文件路径
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    path = model_dir / f"{name}.joblib"

    try:
        from joblib import dump
        dump(model, str(path))
    except ImportError:
        # fallback to pickle
        with open(path, "wb") as f:
            pickle.dump(model, f)

    # 更新时间戳
    _update_metadata(name, model_dir)
    logg.info(f"模型已保存: {path}")
    return path


def load_model(name: str, model_dir: str | Path = _DEFAULT_MODEL_DIR) -> Any | None:
    """从文件加载模型

    Args:
        name: 模型名称（不含扩展名）
        model_dir: 模型目录

    Returns:
        模型对象，或 None（文件不存在时）
    """
    path = Path(model_dir) / f"{name}.joblib"
    if not path.exists():
        return None

    try:
        from joblib import load
        model = load(str(path))
        logg.info(f"模型已加载: {path}")
        return model
    except ImportError:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        logg.warning(f"模型加载失败 ({name}): {exc}")
        return None


def save_strategy_models(
    selector: Any,
    model_dir: str | Path = _DEFAULT_MODEL_DIR,
) -> dict[str, Path]:
    """保存 StrategySelector 的三个 RF 模型

    Args:
        selector: StrategySelector 实例（已训练）
        model_dir: 模型目录

    Returns:
        {"imputation": Path, "normalization": Path, "batch": Path}
    """
    paths: dict[str, Path] = {}
    for task in ["imputation", "normalization", "batch"]:
        if task in selector._models:
            paths[task] = save_model(
                selector._models[task],
                f"strategy_rf_{task}",
                model_dir,
            )
    return paths


def load_strategy_models(
    model_dir: str | Path = _DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """加载 StrategySelector 的三个 RF 模型

    Returns:
        {"imputation": RandomForest | None, "normalization": ..., "batch": ...}
    """
    models: dict[str, Any] = {}
    for task in ["imputation", "normalization", "batch"]:
        models[task] = load_model(f"strategy_rf_{task}", model_dir)
    return models


def save_modality_model(
    selector: Any,
    model_dir: str | Path = _DEFAULT_MODEL_DIR,
) -> Path:
    """保存 ModalitySelector 的 GMM 模型"""
    return save_model(selector.model, "modality_gmm", model_dir)


def load_modality_model(
    model_dir: str | Path = _DEFAULT_MODEL_DIR,
) -> Any | None:
    """加载 ModalitySelector 的 GMM 模型"""
    return load_model("modality_gmm", model_dir)


def is_model_trained(model_dir: str | Path = _DEFAULT_MODEL_DIR) -> bool:
    """检查模型是否已训练"""
    model_dir = Path(model_dir)
    required = [
        "modality_gmm.joblib",
        "strategy_rf_imputation.joblib",
        "strategy_rf_normalization.joblib",
        "strategy_rf_batch.joblib",
    ]
    return all((model_dir / f).exists() for f in required)


# ------------------------------------------------------------------
# 元数据管理
# ------------------------------------------------------------------

def _update_metadata(name: str, model_dir: Path) -> None:
    """更新模型元数据文件"""
    meta_path = model_dir / "training_metadata.json"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text("utf-8"))
    else:
        meta = {"models": {}, "last_trained": None}

    meta["models"][name] = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    meta["last_trained"] = datetime.now(timezone.utc).isoformat()

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def get_model_metadata(model_dir: str | Path = _DEFAULT_MODEL_DIR) -> dict[str, Any]:
    """获取模型训练元数据"""
    meta_path = Path(model_dir) / "training_metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text("utf-8"))
    return {"models": {}, "last_trained": None}


# ------------------------------------------------------------------
# 训练数据生成器
# ------------------------------------------------------------------

def generate_training_data(
    n_samples: int = 500,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成用于训练选择器的合成数据集

    模拟五种模态在不同条件下的特征分布:
        - scrna: 高维 (5000-30000 基因), 高零膨胀 (50-90%)
        - bulk_rna: 中维 (500-5000 基因), 低零膨胀 (0-10%)
        - proteomics: 低维 (20-500 特征), 中等零膨胀 (20-50%)
        - metabolomics: 低维 (50-1000 特征), 中等零膨胀 (20-50%)
        - atac: 高维 (10000-100000 区域), 极高零膨胀 (>90%)

    Returns:
        X: 特征矩阵 (n_samples, 5)
        y_impute: 插补方法标签
        y_norm: 归一化方法标签
        y_batch: 批次校正方法标签
    """
    rng = np.random.RandomState(random_state)

    # 模态分布
    modalities = ["scrna", "bulk_rna", "proteomics", "metabolomics", "atac"]
    modality_weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    modality_map = {"scrna": 0, "bulk_rna": 1, "proteomics": 2, "metabolomics": 3, "atac": 4}

    X_list: list[np.ndarray] = []
    y_impute_list: list[str] = []
    y_norm_list: list[str] = []
    y_batch_list: list[str] = []

    for _ in range(n_samples):
        # 随机选择模态
        modality = rng.choice(modalities, p=modality_weights)
        modality_code = modality_map[modality]

        # 模态特定的特征分布
        if modality == "scrna":
            n_vars = rng.randint(5000, 30000)
            missing_rate = rng.uniform(0.5, 0.9)
            n_obs = rng.randint(500, 10000)
            n_batches = rng.choice([1, 2, 3, 4, 5, 8, 10, 15], p=[0.1, 0.2, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05])
            impute = rng.choice(["zinb_vae", "magic"], p=[0.7, 0.3])
            norm = "scran"
            batch = rng.choice(["harmony", "dann"], p=[0.6, 0.4]) if n_batches >= 5 else "harmony"

        elif modality == "bulk_rna":
            n_vars = rng.randint(500, 5000)
            missing_rate = rng.uniform(0.0, 0.1)
            n_obs = rng.randint(10, 200)
            n_batches = rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            impute = "none"
            norm = rng.choice(["tmm", "deseq2"], p=[0.6, 0.4])
            batch = "combat" if n_batches > 1 else "none"

        elif modality == "proteomics":
            n_vars = rng.randint(20, 500)
            missing_rate = rng.uniform(0.2, 0.5)
            n_obs = rng.randint(10, 100)
            n_batches = rng.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1])
            impute = "missforest"
            norm = rng.choice(["quantile", "vsn"], p=[0.7, 0.3])
            batch = "combat" if n_batches > 1 else "none"

        elif modality == "metabolomics":
            n_vars = rng.randint(50, 1000)
            missing_rate = rng.uniform(0.2, 0.5)
            n_obs = rng.randint(10, 200)
            n_batches = rng.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1])
            impute = "missforest"
            norm = rng.choice(["quantile", "vsn"], p=[0.8, 0.2])
            batch = "combat" if n_batches > 1 else "none"

        else:  # atac
            n_vars = rng.randint(10000, 100000)
            missing_rate = rng.uniform(0.8, 0.99)
            n_obs = rng.randint(500, 5000)
            n_batches = rng.choice([1, 2, 3, 5], p=[0.1, 0.3, 0.4, 0.2])
            impute = "none"
            norm = "scran"
            batch = "harmony" if n_batches > 1 else "none"

        # 构建特征向量: [modality_code, missing_rate, log(n_obs), log(n_vars), n_batches]
        features = np.array([
            modality_code,
            missing_rate,
            np.log1p(n_obs),
            np.log1p(n_vars),
            n_batches,
        ])

        X_list.append(features)
        y_impute_list.append(impute)
        y_norm_list.append(norm)
        y_batch_list.append(batch)

    X = np.array(X_list)
    return X, np.array(y_impute_list), np.array(y_norm_list), np.array(y_batch_list)


def train_and_persist_models(
    model_dir: str | Path = _DEFAULT_MODEL_DIR,
    n_samples: int = 500,
    random_state: int = 42,
) -> dict[str, Any]:
    """一键训练并持久化所有选择器模型

    Returns:
        包含所有已训练模型的字典
    """
    from ._modality import ModalitySelector
    from ._strategy import StrategySelector
    from .._settings import settings

    logg.info("开始训练选择器模型...")

    # 1. 生成训练数据
    X, y_impute, y_norm, y_batch = generate_training_data(
        n_samples=n_samples,
        random_state=random_state,
    )

    # 2. 训练 GMM 模态识别器
    gmm_config = settings.selector.get("gmm", {})
    modality_selector = ModalitySelector(
        n_components=gmm_config.get("n_components", 5),
        covariance_type=gmm_config.get("covariance_type", "full"),
        random_state=gmm_config.get("random_state", 42),
    )
    # GMM 只在特征向量的数值部分上训练（排除第一列的模态编码）
    modality_selector.fit(X[:, 1:])  # 使用 [missing_rate, log1p(n_obs), log1p(n_vars), n_batches]
    save_modality_model(modality_selector, model_dir)

    # 3. 训练 RF 策略推荐器
    rf_config = settings.selector.get("random_forest", {})
    strategy_selector = StrategySelector(
        n_estimators=rf_config.get("n_estimators", 200),
        max_depth=rf_config.get("max_depth", 10),
        random_state=rf_config.get("random_state", 42),
    )
    strategy_selector.fit(X, y_impute, y_norm, y_batch)
    save_strategy_models(strategy_selector, model_dir)

    logg.info(f"选择器模型训练完成 (n_samples={n_samples})")
    return {
        "modality": {"gmm": modality_selector.model},
        "strategy": strategy_selector._models,
    }
