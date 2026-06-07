"""
模态识别器：基于 GMM 聚类判定数据模态类型

特征向量包括:
    - 基因/特征数量
    - 缺失率
    - 计数分布统计量（均值、方差、零膨胀率）
    - 样本量
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

    Returns:
        shape (1, n_features) 的特征向量
    """
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    n_obs, n_vars = adata.shape
    missing_rate = np.mean(X == 0) if np.any(X >= 0) else 0.0
    mean_val = float(np.mean(X))
    var_val = float(np.var(X))
    zero_inflation = float(np.mean(X == 0))
    cell_count_log = np.log1p(n_obs)

    return np.array([[missing_rate, mean_val, var_val, zero_inflation, cell_count_log, n_vars]])


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
        logg.info(f"检测到模态: {modality} (cluster={cluster})")
        return modality


def detect_modality(adata: AnnData) -> str:
    """便捷函数：自动检测 AnnData 的模态类型

    使用启发式规则（无训练模型时）:
        - 特征数 > 10000 → scrna
        - 特征数 < 1000 且零膨胀率高 → proteomics
        - 特征数适中 → bulk_rna
    """
    features = _extract_features(adata)
    n_vars = features[0, 5]
    zero_inf = features[0, 3]

    if n_vars > 10000:
        modality = "scrna"
    elif n_vars < 500 and zero_inf < 0.5:
        modality = "proteomics"
    elif n_vars < 500:
        modality = "metabolomics"
    else:
        modality = "bulk_rna"

    logg.info(f"启发式模态检测: {modality} (特征数={int(n_vars)}, 零膨胀率={zero_inf:.3f})")
    return modality
