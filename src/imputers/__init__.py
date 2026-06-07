"""缺失值分类插补模块

支持的插补方法:
    - MissForest:   随机森林迭代插补（通用型，适合各种数据）
    - ZINB-VAE:     深度变分自编码器（适合高零膨胀的 scRNA-seq 数据）
    - MAGIC:        图扩散平滑插补（适合单细胞数据，保留细胞间关系）
"""

from ._selector import ImputationSelector, evaluate_imputation

__all__ = [
    "ImputationSelector",
    "evaluate_imputation",
]
