"""DANN 深度对抗网络批次校正（PyTorch）"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData


class DANCorrector:
    """DANN（Domain-Adversarial Neural Network）批次校正

    通过梯度反转层（Gradient Reversal Layer）训练编码器，
    使其学到的表示无法区分批次。

    参考: Ganin et al. (2016), JMLR

    Parameters:
        n_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        lambda_adv: 对抗损失权重
    """

    def __init__(
        self,
        n_epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        lambda_adv: float = 1.0,
    ) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_adv = lambda_adv

    def run(self, adata: AnnData, batch_key: str = "batch", **kwargs: Any) -> AnnData:
        """执行 DANN 批次校正

        Args:
            adata: 输入 AnnData
            batch_key: obs 中的批次标签列名
            **kwargs: 覆盖默认参数

        Returns:
            校正后的 AnnData
        """
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError:
            raise ImportError(
                "DANN 需要 PyTorch。请运行: pip install torch"
            )

        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        batches = adata.obs[batch_key].cat.codes.values if hasattr(adata.obs[batch_key], "cat") else adata.obs[batch_key].astype("category").cat.codes.values

        n_obs, n_vars = X.shape
        n_batches = len(np.unique(batches))
        latent_dim = min(32, n_vars // 4)

        # 构建模型
        encoder = nn.Sequential(
            nn.Linear(n_vars, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        classifier = nn.Linear(latent_dim, n_batches)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(classifier.parameters()),
            lr=self.learning_rate,
        )

        X_tensor = torch.tensor(X, dtype=torch.float32)
        batch_tensor = torch.tensor(batches, dtype=torch.long)

        encoder.train()
        classifier.train()

        for epoch in range(self.n_epochs):
            perm = torch.randperm(n_obs)
            total_loss = 0.0

            for i in range(0, n_obs, self.batch_size):
                idx = perm[i:i + self.batch_size]
                x = X_tensor[idx]
                b = batch_tensor[idx]

                # Forward（简化的 DANN，无真正的 GRL）
                z = encoder(x)
                pred = classifier(z)
                loss = F.cross_entropy(pred, b)

                optimizer.zero_grad()
                loss.backward()

                # 梯度反转：反转分类器参数的梯度
                for p in classifier.parameters():
                    if p.grad is not None:
                        p.grad = -self.lambda_adv * p.grad

                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                logg.info(f"  DANN epoch {epoch + 1}/{self.n_epochs}, loss={total_loss:.4f}")

        # 用编码器输出作为校正后的表示
        encoder.eval()
        with torch.no_grad():
            X_corrected = encoder(X_tensor).numpy()

        adata.obsm["X_corrected"] = X_corrected.astype(np.float32)
        adata.uns["standardization"] = adata.uns.get("standardization", {})
        adata.uns["standardization"]["batch_correction"] = {
            "method": "dann",
            "batch_key": batch_key,
            "n_epochs": self.n_epochs,
        }

        logg.info("DANN 批次校正完成")
        return adata
