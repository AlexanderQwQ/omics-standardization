"""标准化效果评估"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData


def run_evaluation(adata: AnnData) -> dict[str, float]:
    """Evaluate standardization pipeline results.

    Metrics computed:
        - n_features: Number of features after processing
        - rmse: Root mean square error (requires raw data layer in adata.layers)
        - batch_mixing: Batch label distribution uniformity (higher = better)
        - mmd: Maximum Mean Discrepancy between batch distributions using RBF kernel
          (lower = better batch mixing). Computed on corrected data if available.
        - wasserstein: Average pairwise Wasserstein distance between batch
          distributions (lower = better). Computed on per-feature basis.
        - batch_silhouette: 1 - silhouette_score on batch labels. Values near 1
          indicate good batch mixing; values near 0 indicate poor mixing.

    Metrics mmd, wasserstein, and batch_silhouette are computed on
    ``adata.obsm["X_corrected"]`` when available, falling back to ``adata.X``.
    Each is wrapped defensively so a single-batch dataset or other edge case
    does not crash evaluation.

    Returns:
        Dictionary of evaluation metrics.
    """
    metrics: dict[str, float] = {}

    # -- 特征数 --
    metrics["n_features"] = float(adata.n_vars)

    # -- RMSE（如果存在原始数据层）--
    if "raw" in adata.layers:
        X_raw = adata.layers["raw"]
        X_cur = adata.X
        if hasattr(X_raw, "toarray"):
            X_raw = X_raw.toarray()
        if hasattr(X_cur, "toarray"):
            X_cur = X_cur.toarray()
        mask = X_raw > 0
        if mask.sum() > 0:
            metrics["rmse"] = float(
                np.sqrt(np.mean((X_raw[mask] - X_cur[mask]) ** 2))
            )

    # -- 批次混合度（简化：批次标签分布均匀度）--
    if "batch" in adata.obs.columns:
        batch_counts = adata.obs["batch"].value_counts()
        batch_mixing = (
            float(batch_counts.min() / batch_counts.max())
            if len(batch_counts) > 1
            else 1.0
        )
        metrics["batch_mixing"] = batch_mixing

    # -- 高级批次混合指标（在 X_corrected 上计算，回退到 adata.X）--
    if "batch" in adata.obs.columns:
        # 获取用于评估的数据矩阵
        if "X_corrected" in adata.obsm:
            X_eval = adata.obsm["X_corrected"]
        else:
            X_eval = adata.X

        # 转为稠密数组
        if hasattr(X_eval, "toarray"):
            X_eval = X_eval.toarray()
        X_eval = np.asarray(X_eval, dtype=np.float64)

        batch_labels = adata.obs["batch"].values

        # MMD
        try:
            metrics["mmd"] = _compute_mmd(X_eval, batch_labels)
        except Exception:
            pass

        # Wasserstein
        try:
            metrics["wasserstein"] = _compute_wasserstein(X_eval, batch_labels)
        except Exception:
            pass

        # Batch silhouette
        try:
            metrics["batch_silhouette"] = _compute_batch_silhouette(
                X_eval, batch_labels
            )
        except Exception:
            pass

    logg.info(f"Evaluation completed: {metrics}")
    return metrics


def _compute_mmd(X: np.ndarray, batch_labels: np.ndarray) -> float:
    """Compute Maximum Mean Discrepancy between batch distributions.

    Uses an RBF kernel with the median-distance heuristic for bandwidth
    selection.  Lower MMD values indicate better batch mixing (the
    distributions are more similar).

    Args:
        X: Data matrix of shape (n_samples, n_features).
        batch_labels: Batch label for each sample.

    Returns:
        MMD value (>= 0).  Returns 0.0 when only one batch is present.
    """
    from scipy.spatial.distance import cdist

    unique_batches = np.unique(batch_labels)
    if len(unique_batches) < 2:
        return 0.0

    # Subsample for efficiency on large datasets
    n_max = min(X.shape[0], 2000)
    if X.shape[0] > n_max:
        rng = np.random.RandomState(42)
        indices = rng.choice(X.shape[0], n_max, replace=False)
        X = X[indices]
        batch_labels = batch_labels[indices]

    # RBF kernel with median heuristic
    pairwise_dist = cdist(X, X, metric="euclidean")
    sigma = np.median(pairwise_dist[pairwise_dist > 0])
    if sigma == 0 or np.isnan(sigma):
        sigma = 1.0
    gamma = 1.0 / (2.0 * sigma**2)
    K = np.exp(-gamma * pairwise_dist**2)

    # Average pairwise MMD^2 over all batch pairs
    total_mmd2 = 0.0
    n_pairs = 0
    for i in range(len(unique_batches)):
        for j in range(i + 1, len(unique_batches)):
            idx_i = np.where(batch_labels == unique_batches[i])[0]
            idx_j = np.where(batch_labels == unique_batches[j])[0]

            K_ii = K[np.ix_(idx_i, idx_i)]
            K_jj = K[np.ix_(idx_j, idx_j)]
            K_ij = K[np.ix_(idx_i, idx_j)]

            mmd2_ij = np.mean(K_ii) + np.mean(K_jj) - 2.0 * np.mean(K_ij)
            total_mmd2 += mmd2_ij
            n_pairs += 1

    if n_pairs == 0:
        return 0.0

    mmd2 = total_mmd2 / n_pairs
    return float(max(0.0, mmd2))


def _compute_wasserstein(X: np.ndarray, batch_labels: np.ndarray) -> float:
    """Compute average pairwise Wasserstein distance between batch distributions.

    Computes the 1-D Wasserstein distance per feature and averages across
    feature dimensions (capped at 100 randomly-sampled features for
    efficiency).  Lower values indicate better batch mixing.

    Args:
        X: Data matrix of shape (n_samples, n_features).
        batch_labels: Batch label for each sample.

    Returns:
        Average Wasserstein distance (>= 0).  Returns 0.0 for a single batch.
    """
    from scipy.stats import wasserstein_distance

    unique_batches = np.unique(batch_labels)
    if len(unique_batches) < 2:
        return 0.0

    # Cap features for efficiency
    n_features = X.shape[1]
    max_features = min(n_features, 100)
    if n_features > max_features:
        rng = np.random.RandomState(42)
        feature_indices = rng.choice(n_features, max_features, replace=False)
        X = X[:, feature_indices]

    total_dist = 0.0
    n_comparisons = 0
    for i in range(len(unique_batches)):
        for j in range(i + 1, len(unique_batches)):
            mask_i = batch_labels == unique_batches[i]
            mask_j = batch_labels == unique_batches[j]

            for k in range(X.shape[1]):
                d = wasserstein_distance(X[mask_i, k], X[mask_j, k])
                total_dist += d
                n_comparisons += 1

    if n_comparisons == 0:
        return 0.0

    return float(total_dist / n_comparisons)


def _compute_batch_silhouette(X: np.ndarray, batch_labels: np.ndarray) -> float:
    """Compute batch mixing score based on the silhouette coefficient.

    Returns ``1 - silhouette_score(batch_labels)``.  Values near 1 indicate
    good batch mixing (batches overlap in the feature space); values near 0
    indicate poor mixing (batches form well-separated clusters).

    Args:
        X: Data matrix of shape (n_samples, n_features).
        batch_labels: Batch label for each sample.

    Returns:
        Batch mixing score in [0, 2] where higher = better mixing.
        Returns 1.0 when only one batch is present.
    """
    from sklearn.metrics import silhouette_score

    unique_batches = np.unique(batch_labels)
    if len(unique_batches) < 2:
        return 1.0

    # Subsample for efficiency on large datasets
    n_max = min(X.shape[0], 2000)
    if X.shape[0] > n_max:
        rng = np.random.RandomState(42)
        indices = rng.choice(X.shape[0], n_max, replace=False)
        X = X[indices]
        batch_labels = batch_labels[indices]

    score = silhouette_score(X, batch_labels)
    # silhouette in [-1, 1] — clamp to [0, 1] and invert
    return float(1.0 - max(0.0, min(1.0, score)))
