"""空间环境免疫多组学数据标准化处理模块"""

from ._version import __version__

# Settings must be imported first
from ._settings import Verbosity, settings

# Core namespaces (scanpy/muon convention)
from . import preprocessing as pp
from . import tools as tl
from . import plotting as pl

# Pipeline
from .pipeline import StandardizationPipeline

# Parsers
from . import parsers

# Selectors
from . import selectors

# Storage (lazy — drivers loaded on demand)
from . import storage

__all__ = [
    "__version__",
    "Verbosity",
    "settings",
    "pp",
    "tl",
    "pl",
    "parsers",
    "selectors",
    "storage",
    "StandardizationPipeline",
]
