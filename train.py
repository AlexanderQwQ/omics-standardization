"""Train selector models — placed at project root to avoid import conflicts.

Usage:
    D:/Anaconda3/envs/omics-std/python.exe train.py
"""
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "config" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---- Training data generator (inline to avoid imports) ----
def generate_training_data(n_samples=500, random_state=42):
    rng = np.random.RandomState(random_state)
    modalities = ["scrna", "bulk_rna", "proteomics", "metabolomics", "atac"]
    modality_weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    modality_map = {m: i for i, m in enumerate(modalities)}

    X_list, y_impute, y_norm, y_batch = [], [], [], []

    for _ in range(n_samples):
        modality = rng.choice(modalities, p=modality_weights)
        modality_code = modality_map[modality]

        if modality == "scrna":
            n_vars = rng.randint(5000, 30000)
            missing_rate = rng.uniform(0.5, 0.9)
            n_obs = rng.randint(500, 10000)
            n_batches = rng.choice([1, 2, 3, 4, 5, 8, 10, 15], p=[0.1, 0.2, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05])
            impute = rng.choice(["zinb_vae", "magic"], p=[0.7, 0.3])
            norm = "scran"
            batch = rng.choice(["harmony", "dann"], p=[0.6, 0.4]) if n_batches >= 5 else "harmony"
        elif modality == "bulk_rna":
            n_vars = rng.randint(500, 5000)
            missing_rate = rng.uniform(0.0, 0.1)
            n_obs = rng.randint(10, 200)
            n_batches = rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            impute = "none"
            norm = rng.choice(["tmm", "deseq2"], p=[0.6, 0.4])
            batch = "combat" if n_batches > 1 else "none"
        elif modality == "proteomics":
            n_vars = rng.randint(20, 500)
            missing_rate = rng.uniform(0.2, 0.5)
            n_obs = rng.randint(10, 100)
            n_batches = rng.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1])
            impute = "missforest"
            norm = rng.choice(["quantile", "vsn"], p=[0.7, 0.3])
            batch = "combat" if n_batches > 1 else "none"
        elif modality == "metabolomics":
            n_vars = rng.randint(50, 1000)
            missing_rate = rng.uniform(0.2, 0.5)
            n_obs = rng.randint(10, 200)
            n_batches = rng.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1])
            impute = "missforest"
            norm = rng.choice(["quantile", "vsn"], p=[0.8, 0.2])
            batch = "combat" if n_batches > 1 else "none"
        else:  # atac
            n_vars = rng.randint(10000, 100000)
            missing_rate = rng.uniform(0.8, 0.99)
            n_obs = rng.randint(500, 5000)
            n_batches = rng.choice([1, 2, 3, 5], p=[0.1, 0.3, 0.4, 0.2])
            impute = "none"
            norm = "scran"
            batch = "harmony" if n_batches > 1 else "none"

        if modality == "scrna":
            file_ext_code = rng.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
        elif modality == "bulk_rna":
            file_ext_code = rng.choice([0, 1], p=[0.7, 0.3])
        elif modality == "proteomics":
            file_ext_code = rng.choice([0, 3], p=[0.5, 0.5])
        elif modality == "metabolomics":
            file_ext_code = rng.choice([0, 4], p=[0.6, 0.4])
        else:
            file_ext_code = rng.choice([0, 1], p=[0.3, 0.7])

        features = [modality_code, missing_rate, np.log1p(n_obs), np.log1p(n_vars), n_batches, file_ext_code]
        X_list.append(features)
        y_impute.append(impute)
        y_norm.append(norm)
        y_batch.append(batch)

    return np.array(X_list), np.array(y_impute), np.array(y_norm), np.array(y_batch)


# ---- Training ----
print("Generating synthetic training data (n=500)...")
X, y_impute, y_norm, y_batch = generate_training_data(500)
print(f"  X shape: {X.shape}")

# GMM on [missing_rate, log1p(n_obs), log1p(n_vars), n_batches] (cols 1-4)
from sklearn.mixture import GaussianMixture
print("\nTraining GMM modality detector...")
gmm = GaussianMixture(n_components=5, covariance_type="full", random_state=42)
gmm.fit(X[:, 1:5])

# Calibrate labels: sort clusters by log1p(n_vars) -> proteomics < metabolomics < bulk_rna < scrna < atac
centers = gmm.means_
n_vars_rank = np.argsort(centers[:, 2])  # log1p(n_vars) ascending
cluster_to_label = {}
ordered = ["proteomics", "metabolomics", "bulk_rna", "scrna", "atac"]
for rank, cluster_idx in enumerate(n_vars_rank):
    cluster_to_label[cluster_idx] = ordered[rank]
print(f"  Cluster -> modality mapping: {cluster_to_label}")

# Save GMM (with calibration mapping)
gmm_path = MODEL_DIR / "modality_gmm.joblib"
from joblib import dump
dump({"model": gmm, "cluster_to_label": cluster_to_label}, str(gmm_path))
print(f"  Saved: {gmm_path}")

# RF models for strategy recommendation
from sklearn.ensemble import RandomForestClassifier

for task, y in [("imputation", y_impute), ("normalization", y_norm), ("batch", y_batch)]:
    print(f"\nTraining RF for {task}...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X, y)
    path = MODEL_DIR / f"strategy_rf_{task}.joblib"
    dump(rf, str(path))
    print(f"  Saved: {path}")

# Metadata
def count_dist(labels):
    return dict(Counter(str(l) for l in labels))

metadata = {
    "models": {
        "modality_gmm": {"last_updated": datetime.now(timezone.utc).isoformat()},
        "strategy_rf_imputation": {"last_updated": datetime.now(timezone.utc).isoformat()},
        "strategy_rf_normalization": {"last_updated": datetime.now(timezone.utc).isoformat()},
        "strategy_rf_batch": {"last_updated": datetime.now(timezone.utc).isoformat()},
    },
    "last_trained": datetime.now(timezone.utc).isoformat(),
    "training_stats": {
        "n_samples": 500,
        "label_distribution": {
            "imputation": count_dist(y_impute),
            "normalization": count_dist(y_norm),
            "batch_correction": count_dist(y_batch),
        },
    },
}
meta_path = MODEL_DIR / "training_metadata.json"
meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n  Saved metadata: {meta_path}")

print("\n" + "=" * 50)
print("Training complete! 5 files saved to config/models/")
print("=" * 50)
