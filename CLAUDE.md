# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Recommended: conda (one-shot, includes R + kallisto + CUDA PyTorch)**

```bash
conda env create -f environment.yml     # create full environment
conda activate omics-std                # activate
pip install -e ".[test,dev]"            # install project in editable mode
```

**Alternative: pip + venv (Python packages only, no R/kallisto)**

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt                      # install all Python deps
pip install -e ".[test,dev]"                         # editable install
```

System-level prerequisites for pip users (see `requirements.txt` header for details):
- R + Bioconductor packages (`edgeR`, `DESeq2`, `limma`, `scran`) for TMM/DESeq2/VSN/Scran
- `kallisto` on PATH for FASTQ paired-end quantification
- DM8 database client (Windows only, optional — SQLite is the default)

### Windows 注意事项

- **Conda**：必须在 Anaconda Prompt 或 PowerShell（`conda init powershell` 后）中运行，Git Bash 通常无法使用 conda
- **conda env create 耗时较长**：需下载 R、Bioconductor、PyTorch、kallisto 等 ~2-5 GB 包，建议网络良好时执行
- **Python 版本**：项目要求 `>=3.10,<3.14`，Windows Store 版 Python 可能版本过新
- **路径分隔符**：所有配置文件中统一使用正斜杠 `/`

## Demo Data

项目提供两个脚本来生成和使用合成 demo 数据，无需外部数据源即可验证全流程：

```bash
# 1. 生成所有 5 种模态的合成数据文件 → data/raw/
python scripts/generate_demo_data.py

# 2. 在 demo 数据上跑全流程验证
python scripts/run_demo_pipeline.py

# 快速模式（仅解析 + 模态检测，跳过完整 pipeline）
python scripts/run_demo_pipeline.py --quick

# 单文件验证
python scripts/run_demo_pipeline.py --file data/raw/scrna/scrna_expression.h5ad
```

生成的目录结构：
```
data/raw/
├── scrna/          # scRNA-seq: .h5ad + .csv (500 cells × 2000 genes)
├── bulk_rna/       # Bulk RNA-seq: .csv + .tsv (50 samples × 1000 genes)
├── proteomics/     # 流式细胞术: .fcs + .csv (100 events × 50 channels)
├── metabolomics/   # 质谱: .mzML + .csv (80 spectra × 200 peaks)
├── atac/           # ATAC-seq: .h5ad + .csv (300 cells × 15000 peaks)
└── microbiome/     # 宏基因组: .biom (30 samples × 500 OTUs)
```

脚本仅依赖 `numpy`, `pandas`, `scipy`, `anndata`, `h5py`，可在项目包安装前独立运行。

## Build, Lint, and Test

```bash
# Lint
ruff check src/ tests/
ruff format src/ tests/              # auto-format
ruff format --check src/ tests/      # check only

# Run all tests
pytest tests/

# Run a single test file or specific test
pytest tests/test_parsers.py
pytest tests/test_pipeline.py::TestStandardizationPipeline::test_pipeline_with_data

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Build for distribution
python -m build
```

## Architecture

### Package Layout

This project uses an **unusual `src/`-as-package** layout: `src/` itself is the Python package root (configured via `packages = ["src"]` in `pyproject.toml`). Import paths are **flat** — no `omics_standardization.` prefix:

```python
from parsers import parse_file          # correct
from pipeline import StandardizationPipeline
from preprocessing import impute        # pp.* API
from storage import StorageManager      # hybrid storage
```

This means all internal imports use relative paths (`from .. import logging`, `from ._base import BaseParser`).

### Core Pipeline (6 Steps)

The central abstraction is `StandardizationPipeline` in [src/pipeline/_pipeline.py](src/pipeline/_pipeline.py). It executes six sequential steps:

```
Parse → Select Strategy → Impute → Normalize → Batch Correct → Evaluate
```

Each step records its outcome in `adata.uns["standardization"][step_name]`. Results are persisted across steps for provenance tracking.

### Three Public Namespaces (scanpy/muon convention)

- **`pp`** ([src/preprocessing/__init__.py](src/preprocessing/__init__.py)) — unified entry for `impute()`, `normalize()`, `batch_correct()`. These functions accept `method=None` to auto-select, or a specific method string.
- **`tl`** ([src/tools/__init__.py](src/tools/__init__.py)) — `pca()`, `umap()`, `evaluate()`
- **`pl`** ([src/plotting/__init__.py](src/plotting/__init__.py)) — `qc_before_after()`, `batch_heatmap()`, `embedding()`

### Hybrid Storage Layer (`src/storage/`)

A node-separated hybrid storage architecture with three backends, coordinated by `StorageManager` ([src/storage/_manager.py](src/storage/_manager.py)):

| Backend | File | Fallback |
|---------|------|----------|
| **MinIO / S3** (object storage) | `_minio.py` | Local filesystem under `data/storage/objects/` |
| **SQLite / DM8** (relational) | `_relational.py` | SQLite auto-selected when DM8 unavailable |
| **Neo4j** (graph) | `_graph.py` | JSON-LD file under `data/storage/graph/` |

All clients share `BaseStorageClient` ([src/storage/_base.py](src/storage/_base.py)) with `put/get/list/delete/exists/is_healthy` + context-manager support. **Every driver is lazy-imported** — the package imports without any of them installed.

The relational schema includes three tables: `samples`, `pipeline_runs`, `quality_metrics`. The graph schema models `Sample` and `Batch` nodes with `CORRELATED` / `BELONGS_TO_BATCH` / `SAME_BATCH` relationships.

Storage config lives in `config/default.yaml` under the `storage:` key.

### Settings System

A global `Settings` singleton lives at [src/_settings/_settings.py](src/_settings/_settings.py). It loads YAML from `config/default.yaml` and exposes nested dicts: `settings.imputation`, `settings.normalization`, `settings.batch_correction`, `settings.selector`, `settings.storage`, etc. Import as `from _settings import settings`.

The `Verbosity` IntEnum ([src/_settings/_verbosity.py](src/_settings/_verbosity.py)) mirrors scanpy: `error(0) < warning(1) < info(2) < hint(3) < debug(4)`.

### Logging

Custom `_RootLogger` ([src/logging.py](src/logging.py)) with convenience functions `info()`, `warning()`, `error()`, `hint()`, `debug()`. All accept optional `time=` and `deep=` keyword arguments. Imported internally as `from .. import logging as logg`.

### Selector Pattern (Method Auto-Selection)

When `config/default.yaml` specifies `method: auto`, the pipeline delegates to the selectors module:

- `selectors/_modality.py`: GMM clustering on 4-feature vector `[missing_rate, log1p(n_obs), log1p(n_vars), n_batches]` — **must stay aligned with `generate_training_data()[:, 1:]`** in `_persistence.py`. With trained models, GMM distinguishes all 5 modalities (`scrna`, `bulk_rna`, `proteomics`, `metabolomics`, `atac`). Without models, heuristic rules cover all 5: ATAC is detected when `n_vars > 10000` and `missing_rate > 0.85` (peak/region data with extreme sparsity), scRNA when `n_vars > 10000` and `missing_rate ≤ 0.85`.
- `selectors/_strategy.py`: Per-modality fallback table mapping modality → `{imputation, normalization, batch}` method. RF model uses 5 features: `[modality_code, missing_rate, log1p(n_obs), log1p(n_vars), n_batches]`.
- `selectors/_persistence.py`: Model training (`train_and_persist_models()`), save/load via joblib, synthetic training data generator (500 samples × 5 modalities). Models stored at `config/models/`. GMM is trained on `X[:, 1:]` (excluding modality_code).

Both `detect_modality()` and `recommend_strategy()` auto-load persisted models when available; they fall back to heuristics/strategy-table otherwise. Run `train_and_persist_models()` once to bootstrap the model directory.

Each processing module also has its own lightweight selector (e.g., `imputers/_selector.py` uses zero-rate thresholds). These are used when called directly, bypassing the global strategy selector.

### Module-by-Module Organization

Each functional module follows the same internal pattern:
- Public `__init__.py` re-exports key classes/functions
- Implementation files prefixed with `_` (e.g., `imputers/_missforest.py`)
- Most normalizers/correctors are classes with a `.run(adata, **kwargs) -> AnnData` method
- Imputers store results in `adata.layers["imputed"]`, normalizers in `adata.layers["normalized"]`, batch correctors in `adata.obsm["X_corrected"]`

### Algorithm Details Worth Knowing

- **ZINB-VAE** ([src/imputers/_zinb_vae.py](src/imputers/_zinb_vae.py)): Decoder outputs three ZINB parameters (`pi`/dropout, `mu`/mean, `theta`/dispersion) with proper ZINB negative log-likelihood loss. Two modes: scvi-tools (`use_scvi=True`) or built-in PyTorch. Distinguishes technical dropout (pi < 0.5) from biological zeros.
- **MAGIC** ([src/imputers/_magic.py](src/imputers/_magic.py)): Wraps `magic-impute` package for KNN-graph Markov diffusion. Includes a heterogeneity guard: warns about over-smoothing risk when `n_vars < 200` and `zero_rate > 0.3` (typical of low-dimensional heterogeneous data like .fcs flow cytometry). For such data, prefer MissForest or ZINB-VAE.
- **DANN** ([src/batch_correctors/_dann.py](src/batch_correctors/_dann.py)): Uses `torch.autograd.Function`-based `GradientReversalLayer` (standard GRL, not manual gradient flip). Joint training: encoder + decoder (reconstruction) + domain classifier behind GRL. The GRL classes are conditionally defined only when torch is available — the module remains importable without it.
- **Scran** ([src/normalizers/_scran.py](src/normalizers/_scran.py)): Three-tier fallback — R `scran::computeSumFactors()` via rpy2 → Python-native k-means pooling + linear deconvolution via `scipy.linalg.lstsq` → `scanpy.pp.normalize_total`.
- **FASTQ** ([src/parsers/_fastq.py](src/parsers/_fastq.py)): Kallisto pseudoalignment via subprocess (`kallisto quant` → parse `abundance.tsv`). Auto-detects paired-end mate files (R1/R2, _1/_2 conventions). Falls back to basic read counting when kallisto is absent.
- **CSV/TSV** ([src/parsers/_csv.py](src/parsers/_csv.py)): Auto-detects delimiter (comma vs tab) and matrix layout (genes×samples vs samples×genes).
- **BIOM** ([src/parsers/_biom.py](src/parsers/_biom.py)): Supports BIOM 1.0 (JSON) and 2.0 (HDF5). Falls back from `biom-format` library to manual JSON parsing.

### Test Fixtures

Shared fixtures in [tests/conftest.py](tests/conftest.py):
- `small_adata` — 100 cells × 200 genes with two batches ("A", "B"), ~30% zeros, sparse CSR
- `small_adata_no_batch` — 50 × 100, no batch column
- `high_missing_adata` — 50 × 100 with ~90% zeros

Test files (10 total):
- `test_parsers.py` — file type detection for all supported extensions
- `test_csv_parser.py` — CSV/TSV layout detection and parsing
- `test_biom_parser.py` — BIOM sparse/dense parsing and taxonomy extraction
- `test_imputers.py` — imputation selection and MissForest
- `test_normalizers.py` — TMM, DESeq2, Quantile, VSN, Scran
- `test_batch_correctors.py` — ComBat, Harmony, correction selector
- `test_selectors.py` — modality detection (GMM + heuristic for all 5 modalities), strategy recommendation (RF + fallback table), model training and persistence. Note: `_extract_features()` returns shape `(1, 4)`.
- `test_pipeline.py` — end-to-end pipeline, storage integration, metrics
- `test_storage.py` — MinIO, RelationalDB (SQLite), GraphDB, StorageManager integration

### Lazy Imports for Optional Dependencies

Torch (ZINB-VAE, DANN), rpy2 (TMM/DESeq2, VSN, Scran), minio, neo4j, dmPython, fcsparser, pymzml, magic-impute, pysam, and biom-format are **all optional**. Every module that needs them imports lazily inside the `.run()` method (or equivalent) inside a `try/except ImportError` with a simplified fallback. Never import any of these at module level — the package must remain importable without them.

The exception is DANN's `GradientReversalLayer` class definition, which is guarded by `if _TORCH_AVAILABLE:` to avoid `NameError` at definition time.

### Python 3.10 Compatibility

`pyproject.toml` declares `requires-python = ">=3.10"` and CI tests Python 3.10–3.12. Avoid Python 3.11+–only APIs:
- ❌ `datetime.UTC` — use `datetime.timezone.utc` instead
- ✅ `str | None` syntax is safe (PEP 604, present since 3.10)
- ✅ `from __future__ import annotations` is already in every source file (makes annotations lazy strings)

Also watch for version-constrained dependencies:
- `numba>=0.57` is required (numba < 0.57 doesn't support Python 3.11; < 0.59 doesn't support 3.12). Without this pin, `pip` may resolve an incompatible numba on Python ≥ 3.11.

### Known Gotchas

- **Environment files**: `requirements.txt` (pip) and `environment.yml` (conda) are the authoritative dependency lists. `pyproject.toml` mirrors them but the env files include system-level notes (R, kallisto, DM8). When adding a new dependency, update all three.
- **`scvi-tools` is in the `torch` optional group**: ZINB-VAE's scvi mode imports `scvi`. It was missing from early `pyproject.toml` — added now. If you see `ImportError: scvi`, install with `pip install -e ".[torch]"`.
- **`src/__init__.py` imports all submodules eagerly** at package load time — including `storage`, `parsers`, and `selectors`. Adding a new submodule import there means it runs on every `import` of the package. Keep top-level imports lean. Note that `storage._minio`, `storage._graph`, etc. are NOT imported at package level — only `storage` (the namespace) is.
- **`pp.normalize()` bypasses its own selector**: unlike `pp.impute()` and `pp.batch_correct()` which use their own module-level selectors, `normalize()` calls the global `recommend_strategy()` from selectors. This is intentional — the normalization selector depends on the detected modality.
- **Method map in `pp.normalize()`**: the function maps method strings to class names via a hardcoded `method_map` dict. Adding a new normalizer requires updating both `normalizers/__init__.py` (export) and this dict (string→class mapping).
- **Storage backends auto-fallback**: `MinIOClient`, `RelationalDBClient`, and `GraphDBClient` all silently fall back to local storage when their drivers are missing. Tests should target the fallback paths (tmpdir-based), not expect live servers.
- **Selector models directory**: `config/models/` is gitignored except for `.gitkeep`. Models must be regenerated in each environment by calling `train_and_persist_models()`. The `recommend_strategy()` and `detect_modality()` functions check for persisted models and use fallbacks when absent — they never fail on missing model files.
- **kallisto is called via subprocess** in `FASTQParser._run_kallisto()`. This means it must be on the system PATH, not installed via pip. The parser also creates temp directories (`omics_qc_*`) that must be cleaned up.
- **GMM feature contract**: `_extract_features()` in `_modality.py` returns 4 features `[missing_rate, log1p(n_obs), log1p(n_vars), n_batches]` — this order and count must match `generate_training_data()[:, 1:]` in `_persistence.py`. Changing either requires updating the other, `detect_modality()`, and `tests/test_selectors.py` shape assertions.
- **MAGIC over-smoothing**: The `MAGICImputer.run()` method now detects low-dimensional heterogeneous data (`n_vars < 200` and `zero_rate > 0.3`) and emits a warning recommending MissForest or ZINB-VAE instead. This guard targets .fcs flow cytometry data specifically.
- **Harmony vs ComBat fallback quality gap**: Harmony's fallback (no scanpy) is a **no-op passthrough** — batch effects are silently uncorrected. ComBat's fallback does mean-centering only (no variance adjustment). DANN has no fallback at all (hard-requires PyTorch). When scanpy/torch are unavailable, the corrector silently degrades; check `adata.uns["standardization"]["batch_correction"]["method"]` to confirm what actually ran.

### Reference Projects

Three scverse reference projects live at `d:/Database/_reference/` (outside this repo):
- `muon/` — multimodal omics framework
- `scanpy/` — single-cell analysis toolkit  
- `muon-tutorials/` — muon tutorial notebooks
