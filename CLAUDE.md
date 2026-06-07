# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, Lint, and Test

```bash
# Install in editable mode with all extras
pip install -e ".[dev,test,torch,rpy2]"

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

### Settings System

A global `Settings` singleton lives at [src/_settings/_settings.py](src/_settings/_settings.py). It loads YAML from `config/default.yaml` and exposes nested dicts: `settings.imputation`, `settings.normalization`, `settings.batch_correction`, etc. Import as `from _settings import settings`.

The `Verbosity` IntEnum ([src/_settings/_verbosity.py](src/_settings/_verbosity.py)) mirrors scanpy: `error(0) < warning(1) < info(2) < hint(3) < debug(4)`.

### Logging

Custom `_RootLogger` ([src/logging.py](src/logging.py)) with convenience functions `info()`, `warning()`, `error()`, `hint()`, `debug()`. All accept optional `time=` and `deep=` keyword arguments. Imported internally as `from .. import logging as logg`.

### Selector Pattern (Method Auto-Selection)

When `config/default.yaml` specifies `method: auto`, the pipeline delegates to the selectors module:

- `selectors/_modality.py`: Heuristic feature extraction → GMM clustering → labels like `"scrna"`, `"proteomics"`, `"bulk_rna"`
- `selectors/_strategy.py`: Per-modality fallback table mapping modality → `{imputation, normalization, batch}` method

Each processing module also has its own lightweight selector (e.g., `imputers/_selector.py` uses zero-rate thresholds). These are used when called directly, bypassing the global strategy selector.

### Module-by-Module Organization

Each functional module follows the same internal pattern:
- Public `__init__.py` re-exports key classes/functions
- Implementation files prefixed with `_` (e.g., `imputers/_missforest.py`)
- Most normalizers/correctors are classes with a `.run(adata, **kwargs) -> AnnData` method
- Imputers store results in `adata.layers["imputed"]`, normalizers in `adata.layers["normalized"]`, batch correctors in `adata.obsm["X_corrected"]`

### Test Fixtures

Shared fixtures in [tests/conftest.py](tests/conftest.py):
- `small_adata` — 100 cells × 200 genes with two batches ("A", "B"), ~30% zeros, sparse CSR
- `small_adata_no_batch` — 50 × 100, no batch column
- `high_missing_adata` — 50 × 100 with ~90% zeros

### Reference Projects

Three scverse reference projects live at `d:/Database/_reference/` (outside this repo):
- `muon/` — multimodal omics framework
- `scanpy/` — single-cell analysis toolkit  
- `muon-tutorials/` — muon tutorial notebooks
