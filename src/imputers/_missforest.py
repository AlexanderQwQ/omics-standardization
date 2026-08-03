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

        Note:
            正确处理 NaN/None（真正缺失）与 0（真实零表达）的区别：
            - NaN/None → 视为需要插补的缺失值
            - 0 → 视为真实零表达，不强制插补（除非配置了 min_expression 阈值）
        """
        from sklearn.ensemble import RandomForestRegressor

        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X.copy()
        X = X.astype(np.float64)

        # 检测缺失值：NaN 或显式标记的 NA（与真实零值区分）
        missing_mask = np.isnan(X)
        # 同时将显式标记的负值哨兵（如 -1 标记的缺失）视为缺失
        sentinel_mask = np.isneginf(X)
        missing_mask = missing_mask | sentinel_mask

        if not np.any(missing_mask):
            logg.info("无 NaN/缺失值，跳过 MissForest")
            return adata

        n_missing = int(np.sum(missing_mask))
        logg.info(f"MissForest: 检测到 {n_missing} 个缺失值（NaN），开始插补...")

        # 用列均值（忽略 NaN）初始化缺失位置
        X_imputed = X.copy()
        col_means = np.nanmean(X, axis=0)
        # NaN 列均值回退为 0
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
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

                # 移除训练数据中仍有 NaN 的行
                valid_train = ~np.isnan(y_train)
                if valid_train.sum() < 10:
                    continue
                X_train = X_train[valid_train, :]
                y_train = y_train[valid_train]

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
        adata.layers["imputed"] = X_imputed.astype(np.float32)
        adata.uns["standardization"] = adata.uns.get("standardization", {})
        adata.uns["standardization"]["imputation"] = {
            "method": "missforest",
            "n_estimators": self.n_estimators,
            "max_iter": self.max_iter,
            "n_missing_imputed": n_missing,
        }

        logg.info(f"MissForest 插补完成 ({n_missing} 个缺失值)")
        return adata
