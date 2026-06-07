"""算法智能选择引擎

基于数据特征（缺失率、数据类型、样本量等）自动推荐最优处理策略。

模态识别:    GMM 聚类 → 判定数据属于哪种组学模态
策略推荐:    RandomForest 分类器 → 推荐插补/归一化/批次校正方法
"""

from ._modality import ModalitySelector, detect_modality
from ._strategy import StrategySelector, recommend_strategy

__all__ = [
    "ModalitySelector",
    "detect_modality",
    "StrategySelector",
    "recommend_strategy",
]
