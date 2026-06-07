"""尺度归一化与统一模块

支持的归一化方法:
    - TMM / DESeq2:  散装转录组归一化（通过 rpy2 调用 R 包）
    - Scran:          单细胞池化标准化
    - Quantile / VSN: 代谢/蛋白质谱归一化
"""

from ._tmm_deseq import TMMNormalizer, DESeq2Normalizer
from ._scran import ScranNormalizer
from ._quantile import QuantileNormalizer, VSNNormalizer

__all__ = [
    "TMMNormalizer",
    "DESeq2Normalizer",
    "ScranNormalizer",
    "QuantileNormalizer",
    "VSNNormalizer",
]
