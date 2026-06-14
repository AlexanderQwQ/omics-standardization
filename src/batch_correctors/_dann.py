"""DANN 深度对抗网络批次校正（PyTorch）

通过梯度反转层（Gradient Reversal Layer）训练编码器，
使其学到的域不变表示（domain-invariant features）。

标准实现:
    - GradientReversalLayer(torch.autograd.Function): forward 恒等，backward 反转梯度
    - 联合训练: 特征提取器 + 标签分类器 + 域判别器
    - 对抗目标: 最大化域分类器误差 = 最小化 batch 可区分性

参考: Ganin et al. (2016), JMLR
      "Domain-Adversarial Training of Neural Networks"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData

# 延迟导入 torch（模块顶层不应强制导入）
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    torch: Any = None  # type: ignore[no-redef]
    nn: Any = None  # type: ignore[no-redef]
    F: Any = None  # type: ignore[no-redef]
    _TORCH_AVAILABLE = False


# GRL 类仅在 torch 可用时定义
if _TORCH_AVAILABLE:

    class GradientReversalLayer(nn.Module):
        """标准梯度反转层 (Gradient Reversal Layer)

        基于 torch.autograd.Function 实现:
            - forward: 恒等映射 (x → x)
            - backward: 梯度乘以 -lambda (∂L/∂x → -λ * ∂L/∂x)

        用法:
            grl = GradientReversalLayer(lambda_=1.0)
            reversed_features = grl(features)  # forward 不变，backward 时梯度反转
        """

        def __init__(self, lambda_: float = 1.0) -> None:
            super().__init__()
            self.lambda_ = lambda_

        def forward(self, x) -> Any:
            return _GRLFunction.apply(x, self.lambda_)

    class _GRLFunction(torch.autograd.Function):
        """GRL 的 autograd Function 实现

        必须定义为顶层类（非嵌套），autograd 通过类名追踪。
        """

        @staticmethod
        def forward(ctx, x, lambda_: float) -> Any:
            ctx.lambda_ = lambda_
            return x

        @staticmethod
        def backward(ctx, grad_output) -> tuple:
            # 梯度反转
            return -ctx.lambda_ * grad_output, None


class DANCorrector:
    """DANN（Domain-Adversarial Neural Network）批次校正

    通过对抗训练学得域不变特征表示:
        1. 特征提取器 E: 高维表达 → 低维隐空间
        2. 标签预测器 C: 隐空间 → 基因表达重建（保留生物学信息）
        3. 域判别器 D: 隐空间 → 批次标签（对抗目标）

    损失: L = L_recon - λ * L_domain + L_reg

    Parameters:
        n_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        lambda_adv: 对抗损失权重（越大越强制域不变）
        latent_dim: 隐空间维度
    """

    def __init__(
        self,
        n_epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        lambda_adv: float = 1.0,
        latent_dim: int = 32,
    ) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_adv = lambda_adv
        self.latent_dim = latent_dim

    def run(self, adata: AnnData, batch_key: str = "batch", **kwargs: Any) -> AnnData:
        """执行 DANN 批次校正

        Args:
            adata: 输入 AnnData
            batch_key: obs 中的批次标签列名
            **kwargs: 覆盖默认参数

        Returns:
            校正后的 AnnData (.obsm["X_corrected"] = 域不变特征)
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "DANN 需要 PyTorch。请运行: pip install torch"
            )

        # 覆盖参数
        self.n_epochs = kwargs.get("n_epochs", self.n_epochs)
        self.batch_size = kwargs.get("batch_size", self.batch_size)
        self.learning_rate = kwargs.get("learning_rate", self.learning_rate)
        self.lambda_adv = kwargs.get("lambda_adv", self.lambda_adv)

        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

        # 编码批次标签
        if batch_key not in adata.obs.columns:
            logg.warning(f"批次列 '{batch_key}' 不存在，默认跳过批次校正")
            return adata

        batch_labels = adata.obs[batch_key].values
        unique_batches = sorted(np.unique(batch_labels))
        batch_to_idx = {b: i for i, b in enumerate(unique_batches)}
        batch_indices = np.array([batch_to_idx[b] for b in batch_labels])

        n_obs, n_vars = X.shape
        n_batches = len(unique_batches)
        latent_dim = min(self.latent_dim, max(4, n_vars // 8))
        hidden_dim = min(256, n_vars)

        # ------------------------------------------------------------------
        # 构建网络
        # ------------------------------------------------------------------
        # 特征提取器 E(x) → z
        encoder = nn.Sequential(
            nn.Linear(n_vars, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

        # 标签预测器（重建）C(z) → x'
        decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_vars),
        )

        # 域判别器 D(z) → batch
        domain_classifier = nn.Sequential(
            GradientReversalLayer(lambda_=self.lambda_adv),
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_batches),
        )

        # ------------------------------------------------------------------
        # 训练
        # ------------------------------------------------------------------
        X_tensor = torch.tensor(X, dtype=torch.float32)
        batch_tensor = torch.tensor(batch_indices, dtype=torch.long)

        # 优化器：encoder + decoder + domain_classifier 联合优化
        all_params = (
            list(encoder.parameters())
            + list(decoder.parameters())
            + list(domain_classifier.parameters())
        )
        optimizer = torch.optim.Adam(all_params, lr=self.learning_rate)

        encoder.train()
        decoder.train()
        domain_classifier.train()

        for epoch in range(self.n_epochs):
            perm = torch.randperm(n_obs)
            total_recon_loss = 0.0
            total_domain_loss = 0.0

            for i in range(0, n_obs, self.batch_size):
                idx = perm[i:i + self.batch_size]
                x = X_tensor[idx]
                b = batch_tensor[idx]

                # Forward pass
                z = encoder(x)                      # 提取特征
                x_recon = decoder(z)                 # 重建原始表达
                domain_pred = domain_classifier(z)   # 域分类（梯度已通过 GRL 反转）

                # 损失
                recon_loss = F.mse_loss(x_recon, x)
                domain_loss = F.cross_entropy(domain_pred, b)

                # 总损失：重建损失 + 域分类损失（GRL 已处理符号反转）
                loss = recon_loss + domain_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_recon_loss += recon_loss.item()
                total_domain_loss += domain_loss.item()

            if (epoch + 1) % 20 == 0:
                logg.info(
                    f"  DANN epoch {epoch + 1}/{self.n_epochs}, "
                    f"recon={total_recon_loss:.4f}, domain={total_domain_loss:.4f}"
                )

        # ------------------------------------------------------------------
        # 提取域不变特征
        # ------------------------------------------------------------------
        encoder.eval()
        with torch.no_grad():
            X_corrected = encoder(X_tensor).numpy()

        adata.obsm["X_corrected"] = X_corrected.astype(np.float32)
        adata.uns["standardization"] = adata.uns.get("standardization", {})
        adata.uns["standardization"]["batch_correction"] = {
            "method": "dann",
            "batch_key": batch_key,
            "n_epochs": self.n_epochs,
            "latent_dim": latent_dim,
            "lambda_adv": self.lambda_adv,
            "n_batches": n_batches,
        }

        logg.info(f"DANN 批次校正完成 (domain-invariant dim={latent_dim})")
        return adata

    def _validate_domain_invariance(self, X_corrected: np.ndarray, batch_labels: np.ndarray) -> float:
        """验证域不变性：用简单分类器的 batch 预测准确率

        准确率越低，说明域不变性越好。
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        scores = cross_val_score(clf, X_corrected, batch_labels, cv=3)
        invariance_score = 1.0 - float(np.mean(scores))
        logg.info(f"域不变性分数: {invariance_score:.4f} (越高越好)")
        return invariance_score
