"""通用装饰器

- @deprecated: 标记废弃函数
- @doc: docstring 装饰器
"""

from __future__ import annotations

import warnings
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import ParamSpec, TypeVar

    P = ParamSpec("P")
    R = TypeVar("R")


def deprecated(msg: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """标记函数/类为废弃

    Args:
        msg: 废弃说明（含替代方案）

    Usage:
        @deprecated("Use new_function() instead")
        def old_function():
            ...
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warnings.warn(
                f"{fn.__name__} is deprecated: {msg}",
                FutureWarning,
                stacklevel=2,
            )
            return fn(*args, **kwargs)

        return wrapper

    return decorator
