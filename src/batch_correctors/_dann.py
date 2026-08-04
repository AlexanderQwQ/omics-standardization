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

import _logging as logg

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
        device: str = "auto",
    ) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_adv = lambda_adv
        self.latent_dim = latent_dim
        self.device = device

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
        # 设备选择（auto: CUDA > MPS > CPU）
        # ------------------------------------------------------------------
        if self.device == "auto":
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = self.device
        device = torch.device(resolved_device)
        logg.info(f"  DANN using device: {resolved_device}")

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
        # 移动模型到设备
        # ------------------------------------------------------------------
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        domain_classifier = domain_classifier.to(device)

        # ------------------------------------------------------------------
        # 训练
        # ------------------------------------------------------------------
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        batch_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device)

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

        recon_losses = []
        domain_losses = []

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

            recon_losses.append(total_recon_loss)
            domain_losses.append(total_domain_loss)

            if (epoch + 1) % 20 == 0:
                logg.info(
                    f"  DANN epoch {epoch + 1}/{self.n_epochs}, "
                    f"recon={total_recon_loss:.4f}, domain={total_domain_loss:.4f}"
                )

        # ------------------------------------------------------------------
        # 提取域不变特征 + 模式崩塌检测
        # ------------------------------------------------------------------
        encoder.eval()
        with torch.no_grad():
            X_corrected = encoder(X_tensor).cpu().numpy()

        # 先做域不变性验证（供模式崩塌检测使用）
        invariance_score = None
        try:
            invariance_score = self._validate_domain_invariance(X_corrected, batch_labels)
        except Exception:
            logg.warning("域不变性验证跳过")

        # 模式崩塌检测
        collapse_info = self._detect_mode_collapse(
            recon_losses=recon_losses,
            batch_indices=batch_indices,
            encoder=encoder,
            domain_classifier=domain_classifier,
            X_tensor=X_tensor,
            invariance_score=invariance_score,
        )

        # 若检测到模式崩塌，用 ×2 lambda_adv 重试一次
        if collapse_info.get("collapsed", False):
            retry_lambda = self.lambda_adv * 2.0
            logg.warning(
                f"DANN 模式崩塌检测到: {collapse_info.get('reasons', [])}，"
                f"尝试增加 lambda_adv (×2) 重新训练"
            )
            logg.info(f"  DANN 重试: lambda_adv={self.lambda_adv} → {retry_lambda}")

            # 重建域判别器（新 GRL 使用更高的 lambda_adv）
            domain_classifier = nn.Sequential(
                GradientReversalLayer(lambda_=retry_lambda),
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, n_batches),
            ).to(device)

            all_params_retry = (
                list(encoder.parameters())
                + list(decoder.parameters())
                + list(domain_classifier.parameters())
            )
            optimizer_retry = torch.optim.Adam(all_params_retry, lr=self.learning_rate)

            encoder.train()
            decoder.train()
            domain_classifier.train()

            recon_losses_retry = []
            domain_losses_retry = []

            for epoch in range(self.n_epochs):
                perm = torch.randperm(n_obs)
                total_recon_loss = 0.0
                total_domain_loss = 0.0

                for i in range(0, n_obs, self.batch_size):
                    idx = perm[i:i + self.batch_size]
                    x_batch = X_tensor[idx]
                    b_batch = batch_tensor[idx]

                    z = encoder(x_batch)
                    x_recon = decoder(z)
                    domain_pred = domain_classifier(z)

                    recon_loss = F.mse_loss(x_recon, x_batch)
                    domain_loss = F.cross_entropy(domain_pred, b_batch)
                    loss = recon_loss + domain_loss

                    optimizer_retry.zero_grad()
                    loss.backward()
                    optimizer_retry.step()

                    total_recon_loss += recon_loss.item()
                    total_domain_loss += domain_loss.item()

                recon_losses_retry.append(total_recon_loss)
                domain_losses_retry.append(total_domain_loss)

                if (epoch + 1) % 20 == 0:
                    logg.info(
                        f"  DANN retry epoch {epoch + 1}/{self.n_epochs}, "
                        f"recon={total_recon_loss:.4f}, domain={total_domain_loss:.4f}"
                    )

            # 重试后重新提取特征并评估
            encoder.eval()
            with torch.no_grad():
                X_corrected = encoder(X_tensor).cpu().numpy()

            invariance_score = None
            try:
                invariance_score = self._validate_domain_invariance(X_corrected, batch_labels)
            except Exception:
                pass

            collapse_info = self._detect_mode_collapse(
                recon_losses=recon_losses_retry,
                batch_indices=batch_indices,
                encoder=encoder,
                domain_classifier=domain_classifier,
                X_tensor=X_tensor,
                invariance_score=invariance_score,
            )

            if collapse_info.get("collapsed", False):
                logg.error(
                    f"DANN 重试后仍检测到模式崩塌: {collapse_info.get('reasons', [])}。"
                    f"建议使用 Harmony 作为替代批次校正方法。"
                )
            else:
                logg.info("DANN 重试成功，模式崩塌已缓解")
                self.lambda_adv = retry_lambda

        adata.obsm["X_corrected"] = X_corrected.astype(np.float32)
        adata.uns["standardization"] = adata.uns.get("standardization", {})
        adata.uns["standardization"]["batch_correction"] = {
            "method": "dann",
            "batch_key": batch_key,
            "n_epochs": self.n_epochs,
            "latent_dim": latent_dim,
            "lambda_adv": self.lambda_adv,
            "n_batches": n_batches,
            "device": resolved_device,
            "domain_invariance": float(invariance_score) if invariance_score is not None else None,
            "mode_collapse_risk": {
                "collapsed": collapse_info.get("collapsed", False),
                "reasons": collapse_info.get("reasons", []),
                "domain_accuracy": collapse_info.get("domain_accuracy"),
                "recon_loss_cv": collapse_info.get("recon_loss_cv"),
            },
        }

        logg.info(f"DANN 批次校正完成 (domain-invariant dim={latent_dim})")
        return adata

    def _detect_mode_collapse(
        self,
        recon_losses: list,
        batch_indices: np.ndarray,
        encoder,
        domain_classifier,
        X_tensor,
        invariance_score: float | None = None,
    ) -> dict:
        """检测 DANN 训练中的模式崩塌（mode collapse）

        三个检测维度:
            1. 域判别器准确率过高 (>95%) — GRL 失效，模型过拟合批次标签
            2. 重建损失振荡过大 (CV > 0.5) — 训练不稳定
            3. 域不变性分数过低 (<0.3) — 校正后仍可区分批次

        Args:
            recon_losses: 每 epoch 的重建损失列表
            batch_indices: 真实批次标签
            encoder: 特征提取器（需在 eval 模式）
            domain_classifier: 域判别器
            X_tensor: 输入数据张量
            invariance_score: 预计算的域不变性分数（可选）

        Returns:
            dict with keys: collapsed, reasons, domain_accuracy, recon_loss_cv
        """
        reasons = []
        domain_accuracy = None
        recon_loss_cv = None

        n_batches = len(np.unique(batch_indices))

        # 检查 1: 域判别器准确率（仅当 n_batches > 1 时有意义）
        if n_batches > 1 and torch is not None:
            try:
                encoder.eval()
                domain_classifier.eval()
                with torch.no_grad():
                    z = encoder(X_tensor)
                    domain_pred = domain_classifier(z)
                    domain_pred_labels = domain_pred.argmax(dim=1).cpu().numpy()
                    domain_accuracy = float((domain_pred_labels == batch_indices).mean())
                if domain_accuracy > 0.95:
                    reasons.append(
                        f"domain classifier accuracy too high ({domain_accuracy:.3f} > 0.95)"
                    )
            except Exception:
                pass

        # 检查 2: 重建损失振荡幅度（最后 20 个 epoch 或全部）
        if recon_losses:
            n_recent = min(20, len(recon_losses))
            recent = recon_losses[-n_recent:]
            mean_val = float(np.mean(recent))
            std_val = float(np.std(recent))
            recon_loss_cv = std_val / (mean_val + 1e-8)
            if recon_loss_cv > 0.5:
                reasons.append(
                    f"recon loss oscillating wildly (CV={recon_loss_cv:.3f} > 0.5)"
                )

        # 检查 3: 域不变性分数
        if invariance_score is not None and invariance_score < 0.3:
            reasons.append(
                f"domain invariance too low ({invariance_score:.3f} < 0.3)"
            )

        collapsed = len(reasons) > 0

        if collapsed:
            logg.warning(f"DANN 模式崩塌风险: {'; '.join(reasons)}")

        return {
            "collapsed": collapsed,
            "reasons": reasons,
            "domain_accuracy": domain_accuracy,
            "recon_loss_cv": recon_loss_cv,
        }

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
