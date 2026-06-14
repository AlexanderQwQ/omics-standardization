"""
模态识别器：基于 GMM 聚类判定数据模态类型

特征向量与 generate_training_data() 保持一致:
    - missing_rate: 缺失率（零值比例）
    - log1p(n_obs): 样本量（对数）
    - log1p(n_vars): 特征数（对数）
    - n_batches: 批次数
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.mixture import GaussianMixture

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData

# 已知模态标签
MODALITY_LABELS = ["scrna", "bulk_rna", "proteomics", "metabolomics", "atac"]


def _extract_features(adata: AnnData) -> np.ndarray:
    """从 AnnData 提取用于模态识别的特征向量

    特征与 generate_training_data() 训练 GMM 时使用的 X[:, 1:] 对齐：
        [missing_rate, log1p(n_obs), log1p(n_vars), n_batches]

    Returns:
        shape (1, 4) 的特征向量
    """
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    n_obs, n_vars = adata.shape
    missing_rate = float(np.mean(X == 0)) if np.any(X >= 0) else 0.0
    n_batches = len(np.unique(adata.obs["batch"])) if "batch" in adata.obs else 1

    return np.array([[missing_rate, np.log1p(n_obs), np.log1p(n_vars), n_batches]])


class ModalitySelector:
    """基于高斯混合模型（GMM）的模态识别器"""

    def __init__(
        self,
        n_components: int = 5,
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> None:
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
        )

    def fit(self, features: np.ndarray) -> ModalitySelector:
        """训练 GMM 模型（用于在线学习场景）"""
        if features.shape[0] < self.model.n_components:
            logg.warning(
                f"样本数 ({features.shape[0]}) 小于 GMM 组件数 "
                f"({self.model.n_components})，将使用较少组件"
            )
            self.model.n_components = max(1, features.shape[0])
        self.model.fit(features)
        return self

    def predict(self, adata: AnnData) -> str:
        """预测单个 AnnData 的模态类型

        Returns:
            "scrna" | "bulk_rna" | "proteomics" | "metabolomics" | "atac"
        """
        features = _extract_features(adata)
        cluster = self.model.predict(features)[0]
        # 将 cluster 映射到模态标签（简单按索引映射）
        idx = cluster % len(MODALITY_LABELS)
        modality = MODALITY_LABELS[idx]
        logg.info(f"GMM 检测到模态: {modality} (cluster={cluster})")
        return modality


def detect_modality(adata: AnnData) -> str:
    """便捷函数：自动检测 AnnData 的模态类型

    优先使用已训练的 GMM 模型（如 config/models/ 下存在），
    否则使用启发式规则:
        - 高维 + 极高零膨胀 (>85%) → atac（染色质可及性）
        - 高维 + 中等零膨胀 → scrna（单细胞转录组）
        - 低维 + 低零膨胀 → proteomics（蛋白质组）
        - 低维 + 中等零膨胀 → metabolomics（代谢组）
        - 中维 + 低零膨胀 → bulk_rna（散装转录组）
    """
    # 尝试使用持久化的 GMM 模型
    from ._persistence import load_modality_model

    gmm = load_modality_model()
    if gmm is not None:
        # 传递 AnnData 对象，由 ModalitySelector.predict() 内部提取特征
        return gmm.predict(adata)

    # 启发式 fallback：使用 adata 原始维度信息进行判断
    n_vars = adata.shape[1]
    missing_rate = _extract_features(adata)[0, 0]

    if n_vars > 10000:
        # ATAC-seq 峰值/区域数据：特征数极高且极度稀疏（>85% 零值）
        # scRNA-seq：特征数高但零膨胀相对较低（50-80%）
        if missing_rate > 0.85:
            modality = "atac"
        else:
            modality = "scrna"
    elif n_vars < 500:
        if missing_rate < 0.5:
            modality = "proteomics"
        else:
            modality = "metabolomics"
    elif n_vars < 5000:
        # 中等特征数、低缺失率 → bulk RNA-seq
        modality = "bulk_rna"
    else:
        # 5000-10000 特征：可能是低维 scRNA 或高维 bulk RNA
        modality = "scrna" if missing_rate > 0.3 else "bulk_rna"

    logg.info(f"启发式模态检测: {modality} (特征数={int(n_vars)}, 缺失率={missing_rate:.3f})")
    return modality
