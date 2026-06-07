"""ZINB-VAE 深度变分自编码器插补器（PyTorch）

基于零膨胀负二项分布 (Zero-Inflated Negative Binomial) 的变分自编码器。
通过编码器-解码器在无监督状态下学习高维细胞流形的共表达关系，
从逻辑上精准区分表观遗传调控引发的真实基因沉默与测序深度不足诱发的技术性 Dropout 假阴性。

参考: Lopez et al. (2018), Nature Methods
      Eraslan et al. (2019), Nature Communications
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData


class ZINBVAEImputer:
    """ZINB-VAE（Zero-Inflated Negative Binomial VAE）插补器

    Decoder 输出三个参数:
        - pi (dropout probability): 零膨胀概率 — 区分技术 Dropout vs 真实零
        - mu (mean): 负二项分布均值 — 真实表达水平
        - theta (dispersion): 负二项分布离散度 — 基因特异性噪声

    Parameters:
        n_epochs: 训练轮数
        batch_size: 批次大小
        latent_dim: 隐空间维度
        learning_rate: 学习率
        use_scvi: 是否优先使用 scvi-tools（更稳定）
    """

    def __init__(
        self,
        n_epochs: int = 200,
        batch_size: int = 128,
        latent_dim: int = 32,
        learning_rate: float = 0.001,
        use_scvi: bool = False,
    ) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.use_scvi = use_scvi

    def run(self, adata: AnnData, **kwargs: Any) -> AnnData:
        """执行 ZINB-VAE 插补

        Args:
            adata: 输入 AnnData（就地修改 .layers['imputed']）
            **kwargs: 覆盖默认参数

        Returns:
            插补后的 AnnData
        """
        # 尝试 scvi-tools 集成（优先）
        if self.use_scvi:
            try:
                return self._run_scvi(adata, **kwargs)
            except ImportError:
                logg.warning("scvi-tools 未安装，使用内置 ZINB-VAE 实现")
            except Exception as exc:
                logg.warning(f"scvi-tools 运行失败 ({exc})，使用内置实现")

        return self._run_torch(adata, **kwargs)

    # ------------------------------------------------------------------
    # scvi-tools 集成
    # ------------------------------------------------------------------

    def _run_scvi(self, adata: AnnData, **kwargs: Any) -> AnnData:
        """通过 scvi-tools SCVI 模型进行插补"""
        import scvi

        # 确保原始计数是整数
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        adata.raw = adata.copy()

        scvi.model.SCVI.setup_anndata(adata, batch_key=kwargs.get("batch_key", None))

        model = scvi.model.SCVI(
            adata,
            n_latent=self.latent_dim,
            n_layers=2,
        )

        model.train(
            max_epochs=getattr(self, "n_epochs", 200),
            batch_size=getattr(self, "batch_size", 128),
            early_stopping=True,
            plan_kwargs={"lr": self.learning_rate},
        )

        # 获取插补值（负二项均值）
        imputed = model.get_normalized_expression(transform_batch=kwargs.get("transform_batch", None))

        adata.layers["imputed"] = imputed
        adata.uns["standardization"] = adata.uns.get("standardization", {})
        adata.uns["standardization"]["imputation"] = {
            "method": "zinb_vae_scvi",
            "n_epochs": self.n_epochs,
            "latent_dim": self.latent_dim,
        }

        logg.info("ZINB-VAE (scvi-tools) 插补完成")
        return adata

    # ------------------------------------------------------------------
    # 内置 PyTorch 实现（真 ZINB 分布）
    # ------------------------------------------------------------------

    def _run_torch(self, adata: AnnData, **kwargs: Any) -> AnnData:
        """内置 PyTorch ZINB-VAE 实现"""
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError:
            raise ImportError(
                "ZINB-VAE 需要 PyTorch。请运行: pip install torch"
            )

        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        X_tensor = torch.tensor(X, dtype=torch.float32)
        n_obs, n_vars = X.shape

        latent_dim = min(self.latent_dim, max(2, n_vars // 4))
        hidden_dim = min(256, n_vars)

        # 构建编码器
        encoder = ZINBEncoder(n_vars, hidden_dim, latent_dim)
        # 构建解码器 — 输出 ZINB 三参数
        decoder = ZINBDecoder(latent_dim, hidden_dim, n_vars)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=self.learning_rate,
        )

        encoder.train()
        decoder.train()

        for epoch in range(self.n_epochs):
            perm = torch.randperm(n_obs)
            total_loss = 0.0

            for i in range(0, n_obs, self.batch_size):
                batch_idx = perm[i:i + self.batch_size]
                batch = X_tensor[batch_idx]

                # Encode
                mu_z, logvar_z = encoder(batch)
                z = mu_z + torch.randn_like(mu_z) * torch.exp(0.5 * logvar_z)

                # Decode → ZINB 参数
                pi, mu, theta = decoder(z)

                # ZINB 负对数似然损失
                zinb_loss = _zinb_nll_loss(batch, pi, mu, theta)

                # KL 散度
                kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp()) / batch.size(0)

                loss = zinb_loss + 0.001 * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                logg.info(f"  ZINB-VAE epoch {epoch + 1}/{self.n_epochs}, loss={total_loss:.4f}")

        # 用解码器均值（mu）填充零值位置
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            mu_z, _ = encoder(X_tensor)
            pi_pred, mu_pred, _ = decoder(mu_z)
            X_recon = mu_pred.numpy()  # 使用负二项均值作为插补值

        # 只替换技术 Dropout（非真实生物零）
        # 使用预测的 dropout 概率：pi < 0.5 的位置更可能是技术效应
        pi_np = pi_pred.numpy()
        missing_mask = (X == 0) & (pi_np < 0.5)  # pi 高 = 真实 Dropout

        X_imputed = X.copy()
        X_imputed[missing_mask] = np.maximum(0, X_recon[missing_mask])

        # 同时用重建值填充所有零值（保守但常用）
        all_zero_mask = X == 0
        X_imputed_all = X.copy()
        X_imputed_all[all_zero_mask] = np.maximum(0, X_recon[all_zero_mask])

        adata.layers["imputed"] = X_imputed_all
        adata.layers["imputed_zinb"] = X_imputed  # 更保守的版本
        adata.uns["standardization"] = adata.uns.get("standardization", {})
        adata.uns["standardization"]["imputation"] = {
            "method": "zinb_vae",
            "n_epochs": self.n_epochs,
            "latent_dim": latent_dim,
            "n_zero_filled": int(all_zero_mask.sum()),
            "n_technical_dropout": int(missing_mask.sum()),
        }

        logg.info("ZINB-VAE 插补完成")
        return adata


# =============================================================================
# 网络模块
# =============================================================================

class ZINBEncoder(nn.Module):
    """ZINB-VAE 编码器

    输出隐空间的均值 (mu) 和对数方差 (logvar)。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.mu_layer = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar


class ZINBDecoder(nn.Module):
    """ZINB 解码器

    输出 ZINB 分布的三参数:
        - pi:    dropout 概率 (经过 sigmoid, ∈ [0,1])
        - mu:    负二项均值 (经过 softplus, > 0)
        - theta: 负二项离散度 (经过 softplus, > 0)
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256, output_dim: int = 20000) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.pi_layer = nn.Linear(hidden_dim, output_dim)       # dropout probability
        self.mu_layer = nn.Linear(hidden_dim, output_dim)       # NB mean
        self.theta_layer = nn.Linear(hidden_dim, output_dim)    # NB dispersion

    def forward(self, z):
        h = self.decoder(z)
        pi = F.sigmoid(self.pi_layer(h))                    # ∈ (0, 1)
        mu = F.softplus(self.mu_layer(h)) + 1e-6            # > 0
        theta = F.softplus(self.theta_layer(h)) + 1e-6      # > 0
        return pi, mu, theta


# =============================================================================
# ZINB 损失函数
# =============================================================================

def _zinb_nll_loss(
    x,
    pi,
    mu,
    theta,
    eps: float = 1e-10,
) -> torch.Tensor:
    """ZINB 负对数似然损失

    ZINB 概率质量函数:
        P(X=0) = pi + (1-pi) * (theta/(theta+mu))^theta
        P(X=k) = (1-pi) * Γ(k+theta)/(Γ(theta)*k!) * (theta/(theta+mu))^theta * (mu/(theta+mu))^k

    Args:
        x: 原始计数 (N, G)
        pi: dropout 概率 (N, G), ∈ (0,1)
        mu: NB 均值 (N, G), > 0
        theta: NB 离散度 (N, G), > 0

    Returns:
        scalar loss (平均每个元素的负对数似然)
    """
    # 数值稳定: clamp 参数
    pi = torch.clamp(pi, eps, 1 - eps)
    mu = torch.clamp(mu, eps, 1e6)
    theta = torch.clamp(theta, eps, 1e6)

    # 负二项分布的 log-prob
    # log P_NB(x | mu, theta):
    #   = log Γ(x + theta) - log Γ(theta) - log(x!)
    #     + theta * log(theta/(theta+mu)) + x * log(mu/(theta+mu))
    eps_float = 1e-10

    # log Gamma 项: lgamma 比直接计算 Gamma 更数值稳定
    # torch.lgamma 对 (x+theta) 和 theta 分别计算

    theta_ratio = theta / (theta + mu + eps_float)
    mu_ratio = mu / (theta + mu + eps_float)

    # NB 的对数概率
    nb_log_prob = (
        torch.lgamma(x + theta + eps_float)
        - torch.lgamma(theta + eps_float)
        - torch.lgamma(x + 1 + eps_float)  # log(x!) = lgamma(x+1)
        + theta * torch.log(theta_ratio + eps_float)
        + x * torch.log(mu_ratio + eps_float)
    )

    # 零膨胀项
    # P(X=0) = pi + (1-pi) * NB(0|mu,theta)
    nb_zero_log_prob = theta * torch.log(theta_ratio + eps_float)
    zero_log_prob = torch.log(pi + (1 - pi) * torch.exp(nb_zero_log_prob) + eps_float)

    # 非零项
    # P(X=k) = log(1-pi) + log NB(k|mu,theta)
    nonzero_log_prob = torch.log(1 - pi + eps_float) + nb_log_prob

    # 按零值/非零值选择
    is_zero = (x < eps_float).float()

    log_prob = is_zero * zero_log_prob + (1 - is_zero) * nonzero_log_prob

    # 返回平均负对数似然
    return -torch.mean(log_prob)
