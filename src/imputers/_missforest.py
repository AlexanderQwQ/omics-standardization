"""MissForest 随机森林迭代插补器"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData


class MissForestImputer:
    """基于 RandomForest 的迭代缺失值插补

    参考: Stekhoven & Buhlmann (2012), Bioinformatics

    Parameters:
        n_estimators: 随机森林树数量
        max_iter: 最大迭代轮数
        random_state: 随机种子
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_iter: int = 10,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state

    def run(self, adata: AnnData, **kwargs: Any) -> AnnData:
        """执行 MissForest 插补

        Args:
            adata: 输入 AnnData（就地修改 .X / .layers['imputed']）
            **kwargs: 覆盖默认参数

        Returns:
            插补后的 AnnData
        """
        from sklearn.ensemble import RandomForestRegressor

        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X.copy()

        # 找到缺失位置
        missing_mask = X == 0
        if not np.any(missing_mask):
            logg.info("无缺失值，跳过 MissForest")
            return adata

        # 用列均值初始化缺失值
        X_imputed = X.copy()
        col_means = np.nanmean(np.where(X == 0, np.nan, X), axis=0)
        for j in range(X.shape[1]):
            X_imputed[missing_mask[:, j], j] = col_means[j]

        # 迭代插补
        for iteration in range(self.max_iter):
            changed = False
            for j in range(X.shape[1]):
                if not np.any(missing_mask[:, j]):
                    continue

                # 用其他列预测第 j 列的缺失值
                train_mask = ~missing_mask[:, j]
                if train_mask.sum() < 10:
                    continue

                X_train = X_imputed[train_mask, :]
                X_train = np.delete(X_train, j, axis=1)
                y_train = X[train_mask, j]

                X_pred = X_imputed[missing_mask[:, j], :]
                X_pred = np.delete(X_pred, j, axis=1)

                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state + j,
                    n_jobs=-1,
                )
                rf.fit(X_train, y_train)
                X_imputed[missing_mask[:, j], j] = np.maximum(0, rf.predict(X_pred))
                changed = True

            if not changed:
                logg.info(f"MissForest 收敛于第 {iteration + 1} 轮")
                break

        # 保存结果到 layer
        adata.layers["imputed"] = X_imputed
        adata.uns["standardization"] = adata.uns.get("standardization", {})
        adata.uns["standardization"]["imputation"] = {
            "method": "missforest",
            "n_estimators": self.n_estimators,
            "max_iter": self.max_iter,
        }

        logg.info("MissForest 插补完成")
        return adata
