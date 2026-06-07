"""通用工具函数"""

from ._io import load_data, save_data
from ._decorators import deprecated

__all__ = [
    "load_data",
    "save_data",
    "deprecated",
]
