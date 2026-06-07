"""深度批次解耦模块

支持的批次校正方法:
    - ComBat:    经验贝叶斯批次校正（通用型）
    - Harmony:   迭代软聚类批次校正（单细胞）
    - DANN:      深度对抗网络批次校正（PyTorch）
"""

from ._selector import BatchCorrectionSelector
from ._combat import ComBatCorrector
from ._harmony import HarmonyCorrector
from ._dann import DANCorrector

__all__ = [
    "BatchCorrectionSelector",
    "ComBatCorrector",
    "HarmonyCorrector",
    "DANCorrector",
]
