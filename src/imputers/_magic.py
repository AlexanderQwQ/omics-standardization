"""MAGIC 图扩散平滑插补器"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData


class MAGICImputer:
    """MAGIC（Markov Affinity-based Graph Imputation of Cells）插补

    通过图扩散平滑基因表达，保留细胞间关系的同时填补缺失值。

    参考: van Dijk et al. (2018), Cell

    Parameters:
        t: 扩散步数（越高越平滑）
        k: 近邻数
        k_a: 自适应近邻数
    """

    def __init__(self, t: int = 3, k: int = 30, k_a: int = 10) -> None:
        self.t = t
        self.k = k
        self.k_a = k_a

    def run(self, adata: AnnData, **kwargs: Any) -> AnnData:
        """执行 MAGIC 插补

        Args:
            adata: 输入 AnnData（就地修改 .layers['imputed']）
            **kwargs: 覆盖默认参数

        Returns:
            插补后的 AnnData
        """
        try:
            import magic
        except ImportError:
            raise ImportError(
                "MAGIC 需要 magic-impute 包。请运行: pip install magic-impute"
            )

        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

        magic_op = magic.MAGIC(
            t=self.t,
            k=self.k,
            k_a=self.k_a,
        )

        X_imputed = magic_op.fit_transform(X, genes=adata.var_names.tolist())

        adata.layers["imputed"] = X_imputed
        adata.uns["standardization"] = adata.uns.get("standardization", {})
        adata.uns["standardization"]["imputation"] = {
            "method": "magic",
            "t": self.t,
            "k": self.k,
            "k_a": self.k_a,
        }

        logg.info("MAGIC 插补完成")
        return adata
