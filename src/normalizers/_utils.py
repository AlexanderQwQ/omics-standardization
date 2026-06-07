"""归一化通用工具

- R 包调用封装
- 稀疏矩阵辅助函数
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


def ensure_dense(X, max_memory_gb: float = 8.0) -> np.ndarray:
    """将稀疏矩阵安全转换为密集数组

    Args:
        X: 稀疏/密集矩阵
        max_memory_gb: 内存上限 (GB)，超出则报错

    Returns:
        numpy ndarray
    """
    if hasattr(X, "toarray"):
        n_elements = X.shape[0] * X.shape[1]
        estimated_gb = n_elements * 4 / (1024**3)  # float32
        if estimated_gb > max_memory_gb:
            raise MemoryError(
                f"矩阵过大 ({estimated_gb:.1f} GB > {max_memory_gb} GB 上限)，请分批处理"
            )
        return X.toarray()
    return np.asarray(X)


def check_r_available(package: str) -> bool:
    """检查 R 包是否可用"""
    try:
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects.packages import importr
        importr(package)
        return True
    except Exception:
        return False
