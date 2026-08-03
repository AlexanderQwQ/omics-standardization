# CLAUDE.md — 项目开发指南（AI 助手上线文档）

本文件为 Claude Code (claude.ai/code) 提供项目开发指引。中文注释标注 `//` 内容为面向人类开发者的解释说明。

## 快速命令参考（Quick Reference）

| 任务（Task） | 命令（Command） |
|------|---------|
| 完整环境安装 | `conda env create -f environment.yml && conda activate omics-std && pip install -e ".[test,dev]"` |
| 生成 demo 数据 | `python scripts/generate_demo_data.py` |
| 运行全部测试 | `pytest tests/` |
| 覆盖率测试 | `pytest tests/ --cov=src --cov-report=term-missing` |
| 代码检查 | `ruff check src/ tests/` |
| 自动格式化 | `ruff format src/ tests/` |
| 构建发布包 | `python -m build` |
| CLI 运行 | `omics-std config/default.yaml -i data/raw/ -o data/processed/` |

## 环境搭建（Environment Setup）

**推荐方式：Conda（一键安装，含 R + kallisto + CUDA PyTorch）**

```bash
conda env create -f environment.yml     # 创建完整环境（~2-5 GB）
conda activate omics-std                # 激活环境
pip install -e ".[test,dev]"            # 可编辑安装
```

**备选：pip + venv（仅 Python 包，不含 R/kallisto）**

```bash
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate                            # Windows
pip install -r requirements.txt                      # 安装全部 Python 依赖
pip install -e ".[test,dev]"                         # 可编辑安装
```

pip 用户的系统级前置依赖（详见 `requirements.txt` 头部）：
- R + Bioconductor 包（`edgeR`, `DESeq2`, `limma`, `scran`）→ TMM/DESeq2/VSN/Scran 归一化
- `kallisto` 需在 PATH 中 → FASTQ 双端定量
- DM8 数据库客户端（仅 Windows，可选 → SQLite 为默认后端）

### Windows 注意事项

- **Conda**：必须在 Anaconda Prompt 或 PowerShell（`conda init powershell` 后）中运行，Git Bash 通常无法使用 conda
- **conda env create 耗时较长**：需下载 R、Bioconductor、PyTorch、kallisto 等 ~2-5 GB 包，建议网络良好时执行
- **Python 版本**：项目要求 `>=3.10,<3.14`，Windows Store 版 Python 可能版本过新
- **路径分隔符**：所有配置文件中统一使用正斜杠 `/`

## Demo 数据（示例数据验证）

项目提供两个脚本，无需外部数据源即可验证全流程：

```bash
# 1. 生成所有 6 种模态的合成数据文件 → data/raw/
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
├── scrna/          # scRNA-seq: .h5ad + .csv（500 cells × 2000 genes）
├── bulk_rna/       # Bulk RNA-seq: .csv + .tsv（50 samples × 1000 genes）
├── proteomics/     # 流式细胞术: .fcs + .csv（100 events × 50 channels）
├── metabolomics/   # 质谱: .mzML + .csv（80 spectra × 200 peaks）
├── atac/           # ATAC-seq: .h5ad + .csv（300 cells × 15000 peaks）
└── microbiome/     # 宏基因组: .biom（30 samples × 500 OTUs）
```

脚本仅依赖 `numpy`, `pandas`, `scipy`, `anndata`, `h5py`，可在项目包安装前独立运行。

## 构建、检查与测试（Build, Lint, and Test）

```bash
# 代码检查
ruff check src/ tests/
ruff format src/ tests/              # 自动格式化
ruff format --check src/ tests/      # 仅检查不修改

# 运行全部测试
pytest tests/

# 运行单个测试文件或特定测试
pytest tests/test_parsers.py
pytest tests/test_pipeline.py::TestStandardizationPipeline::test_pipeline_with_data

# 覆盖率
pytest tests/ --cov=src --cov-report=term-missing

# 构建发布包
python -m build
```

## 架构（Architecture）

### 包布局（Package Layout）

本项目采用 **`src/`-as-package 布局**：`src/` 本身就是 Python 包的根目录（通过 `pyproject.toml` 中的 `packages = ["src"]` 配置）。导入路径是**扁平**的 — 没有 `omics_standardization.` 前缀：

```python
from parsers import parse_file          # 正确导入方式
from pipeline import StandardizationPipeline
from preprocessing import impute        # pp.* API
from storage import StorageManager      # 混合存储
```

这意味着所有内部导入使用相对路径（`from .. import logging`, `from ._base import BaseParser`）。

### 核心流水线（6 步处理流程）

核心抽象是 [src/pipeline/_pipeline.py](src/pipeline/_pipeline.py) 中的 `StandardizationPipeline`，依次执行六个步骤：

```
① Parse（解析）  →  ② Select Strategy（策略选择）  →  ③ Impute（插补）
→  ④ Normalize（归一化）  →  ⑤ Batch Correct（批次校正）  →  ⑥ Evaluate（评估）
```

每步结果记录在 `adata.uns["standardization"][step_name]` 中，用于溯源追踪。

### 三个公共命名空间（Three Public Namespaces，遵循 scanpy/muon 约定）

- **`pp`** ([src/preprocessing/__init__.py](src/preprocessing/__init__.py)) — 统一入口：`impute()`, `normalize()`, `batch_correct()`, `standardize()`。这些函数接受 `method=None` 自动选择，或指定方法字符串。
- **`tl`** ([src/tools/__init__.py](src/tools/__init__.py)) — 降维/评估：`pca()`, `umap()`, `evaluate()` — 返回 rmse、batch_mixing、mmd、wasserstein、batch_silhouette 等指标
- **`pl`** ([src/plotting/__init__.py](src/plotting/__init__.py)) — 可视化：`qc_before_after()`, `batch_heatmap()`, `embedding()`

### 混合存储层（Hybrid Storage Layer, `src/storage/`）

节点分离的混合存储架构，由 `StorageManager` ([src/storage/_manager.py](src/storage/_manager.py)) 协调三个后端：

| 后端（Backend） | 文件 | Fallback 机制 |
|---------|------|----------|
| **MinIO / S3**（对象存储） | `_minio.py` | 本地文件系统 `data/storage/objects/` |
| **SQLite / DM8**（关系型数据库） | `_relational.py` | DM8 不可用时自动选择 SQLite |
| **Neo4j**（图数据库） | `_graph.py` | JSON-LD 文件 `data/storage/graph/` |

所有客户端共享 `BaseStorageClient` ([src/storage/_base.py](src/storage/_base.py))，提供 `put/get/list/delete/exists/is_healthy` + context-manager 支持。**所有驱动均为延迟导入（lazy import）** — 包在无任何存储后端安装时仍可导入。

关系型模式包括三个表：`samples`, `pipeline_runs`, `quality_metrics`。图模式建模 `Sample`、`Batch`、`Gene`、`Pathway`、`Disease` 五种节点类型，以及 `CORRELATED` / `BELONGS_TO_BATCH` / `SAME_BATCH` / `EXPRESSES` / `INVOLVED_IN` / `ASSOCIATED_WITH` 八种关系类型。

存储配置位于 `config/default.yaml` 的 `storage:` 键下。

### 设置系统（Settings System）

全局 `Settings` 单例位于 [src/_settings/_settings.py](src/_settings/_settings.py)。它从 `config/default.yaml` 加载 YAML 并暴露嵌套字典：`settings.imputation`, `settings.normalization`, `settings.batch_correction`, `settings.selector`, `settings.storage` 等。通过 `from _settings import settings` 导入。

`Verbosity` IntEnum ([src/_settings/_verbosity.py](src/_settings/_verbosity.py)) 模仿 scanpy 的日志级别：`error(0) < warning(1) < info(2) < hint(3) < debug(4)`.

### 日志系统（Logging）

自定义 `_RootLogger` ([src/logging.py](src/logging.py)) 提供便捷函数 `info()`, `warning()`, `error()`, `hint()`, `debug()`。均支持可选 `time=` 和 `deep=` 关键字参数。内部导入为 `from .. import logging as logg`。

### 选择器模式（Selector Pattern — 算法自动选择）

当 `config/default.yaml` 指定 `method: auto` 时，流水线委托给 selectors 模块：

- `selectors/_modality.py`：GMM 聚类在 4 特征向量 `[missing_rate, log1p(n_obs), log1p(n_vars), n_batches]` 上 — **必须与 `_persistence.py` 中的 `generate_training_data()[:, 1:5]` 对齐**。训练后 GMM 区分全部 5 种模态。无模型时启发式规则覆盖全部 5 种：ATAC 在 `n_vars > 10000` 且 `missing_rate > 0.85` 时检测，scRNA 在 `n_vars > 10000` 且 `missing_rate ≤ 0.85` 时检测。
- `selectors/_strategy.py`：每模态 fallback 表映射 modality → `{imputation, normalization, batch}` 方法。RF 模型使用 6 个特征：`[modality_code, missing_rate, log1p(n_obs), log1p(n_vars), n_batches, file_ext_code]`。
- `selectors/_persistence.py`：模型训练（`train_and_persist_models()`）、joblib 保存/加载、合成训练数据生成器（500 样本 × 5 模态）。模型存储在 `config/models/`。GMM 训练在 `X[:, 1:5]` 上（排除 modality_code 和 file_ext_code）。`assess_annotation_quality()` 评估训练数据质量（样本量 >100、类别平衡、模型新鲜度）。`is_high_quality_available()` 返回 True 仅当模型存在且质量评分 ≥0.7，为需求中"动态切换 GMM↔RF"提供判断依据。

`detect_modality()` 和 `recommend_strategy()` 在有持久化模型时自动加载；否则回退到启发式/策略表。首次使用需运行 `train_and_persist_models()` 初始化模型目录。

每个处理模块也有自己的轻量级选择器（如 `imputers/_selector.py` 使用 zero_rate 阈值）。在直接调用时使用，绕过全局策略选择器。

### 模块组织结构（Module-by-Module Organization）

每个功能模块遵循相同的内部模式：
- 公开的 `__init__.py` 重导出关键类/函数
- 实现文件以 `_` 前缀命名（如 `imputers/_missforest.py`）
- 大多数归一化器/校正器是实现 `.run(adata, **kwargs) -> AnnData` 方法的类
- 插补器将结果存储在 `adata.layers["imputed"]`，归一化器存储在 `adata.layers["normalized"]`，批次校正器存储在 `adata.obsm["X_corrected"]`

### 关键算法细节（Algorithm Details）

- **ZINB-VAE** ([src/imputers/_zinb_vae.py](src/imputers/_zinb_vae.py))：Decoder 输出三个 ZINB 参数（`pi`/dropout, `mu`/mean, `theta`/dispersion），使用正确的 ZINB 负对数似然损失。两种模式：scvi-tools (`use_scvi=True`) 或内置 PyTorch。区分技术性 dropout（pi < 0.5）和生物学零值。
- **MAGIC** ([src/imputers/_magic.py](src/imputers/_magic.py))：包装 `magic-impute` 包用于 KNN 图马尔可夫扩散。包含异质性 guard：当 `n_vars < 200` 且 `zero_rate > 0.3` 时发出过度平滑风险警告（典型的低维异质性数据如 .fcs 流式细胞术）。对此类数据，推荐使用 MissForest 或 ZINB-VAE。
- **DANN** ([src/batch_correctors/_dann.py](src/batch_correctors/_dann.py))：使用 `torch.autograd.Function` 基的 `GradientReversalLayer`（标准 GRL，非手动梯度翻转）。联合训练：编码器 + 解码器（重构）+ GRL 后的域分类器。GRL 类仅在 torch 可用时定义 — 模块无 torch 仍可导入。支持 GPU 加速（CUDA/MPS/CPU 自动检测，`device='auto'`）。含模式崩溃检测：训练后验证域分类器准确率、recon loss 振荡 CV 和域不变性分数，检测到崩溃时以 λ×2 重训练。
- **TMM Python** ([src/normalizers/_tmm_deseq.py](src/normalizers/_tmm_deseq.py))：Python 原生 TMM 实现：几何均值伪参考 → M-value (log2 ratio) + A-value (mean log2 intensity) → 30% M / 5% A 双尾裁剪 → 逆方差加权均值 → 归一化因子。CPM 降为最终 fallback。
- **VSN arcsinh** ([src/normalizers/_quantile.py](src/normalizers/_quantile.py))：arcsinh 方差稳定化 fallback：`arcsinh(a + b*X)`，用 `scipy.optimize.minimize_scalar` 自动调节 b 最小化特征均值-方差相关性。log2 降为最终 fallback。
- **Scran** ([src/normalizers/_scran.py](src/normalizers/_scran.py))：三级 fallback — R `scran::computeSumFactors()` via rpy2 → Python 原生 k-means 池化 + 线性反卷积 via `scipy.linalg.lstsq` → `scanpy.pp.normalize_total`。
- **FASTQ** ([src/parsers/_fastq.py](src/parsers/_fastq.py))：Kallisto 伪比对 via subprocess (`kallisto quant` → 解析 `abundance.tsv`)。自动检测双端配对文件（R1/R2, _1/_2 约定）。kallisto 不可用时回退到基础 reads 计数。
- **CSV/TSV** ([src/parsers/_csv.py](src/parsers/_csv.py))：自动检测分隔符（逗号 vs 制表符）和矩阵布局（基因×样本 vs 样本×基因）。
- **BIOM** ([src/parsers/_biom.py](src/parsers/_biom.py))：支持 BIOM 1.0 (JSON) 和 2.0 (HDF5)。从 `biom-format` 库回退到手动 JSON 解析。
- **mzML** ([src/parsers/_mzml.py](src/parsers/_mzml.py))：**注意！** pymzml 的 `Reader` 迭代器只能遍历一次。必须在单次遍历中完成所有数据收集（见 `spectra_peaks` 结构）。已修复：不再二次遍历迭代器。

### 测试 Fixtures（Test Fixtures）

[tests/conftest.py](tests/conftest.py) 中的共享 fixtures：
- `small_adata` — 100 cells × 200 genes，两个批次（"A", "B"），~30% 零值，CSR 稀疏
- `small_adata_no_batch` — 50 × 100，无批次列
- `high_missing_adata` — 50 × 100，~90% 零值

测试文件（11 个）：
- `test_parsers.py` — 所有支持扩展名的文件类型检测
- `test_csv_parser.py` — CSV/TSV 分隔符和布局检测
- `test_biom_parser.py` — BIOM 稀疏/密集解析和分类学提取
- `test_imputers.py` — 插补选择和 MissForest
- `test_normalizers.py` — TMM, DESeq2, Quantile, VSN, Scran
- `test_batch_correctors.py` — ComBat, Harmony, 校正选择器
- `test_selectors.py` — 模态检测（GMM + 启发式，覆盖全部 5 种模态）、策略推荐（RF + fallback 表）、模型训练和持久化。注意：`_extract_features()` 返回 shape `(1, 4)`。
- `test_pipeline.py` — 端到端流水线、存储集成、指标评估
- `test_storage.py` — MinIO, RelationalDB (SQLite), GraphDB, StorageManager 集成

### 可选依赖的延迟导入（Lazy Imports）

torch（ZINB-VAE, DANN）、rpy2（TMM/DESeq2, VSN, Scran）、minio、neo4j、dmPython、fcsparser、pymzml、magic-impute、pysam 和 biom-format **全部为可选依赖**。每个需要它们的模块在 `.run()` 方法（或等效方法）内部通过 `try/except ImportError` 延迟导入，并配有简化 fallback。绝不在模块级别导入任何可选依赖 — 包必须在无它们时仍可导入。

例外：DANN 的 `GradientReversalLayer` 类定义由 `if _TORCH_AVAILABLE:` 守卫，以避免定义时 `NameError`。

### Python 3.10 兼容性

`pyproject.toml` 声明 `requires-python = ">=3.10"`，CI 测试 Python 3.10–3.12。避免 Python 3.11+ 独有 API：
- ❌ `datetime.UTC` — 改用 `datetime.timezone.utc`
- ✅ `str | None` 语法安全（PEP 604，3.10 已有）
- ✅ `from __future__ import annotations` 已存在于所有源文件中（使注解变为惰性字符串）

版本约束依赖注意事项：
- `numba>=0.57` 是必需的（numba < 0.57 不支持 Python 3.11；< 0.59 不支持 3.12）。无此约束时，`pip` 可能在 Python ≥ 3.11 上解析到不兼容的 numba。

### 已知问题与陷阱（Known Gotchas）

- **环境文件**：`requirements.txt`（pip）和 `environment.yml`（conda）是权威依赖列表。`pyproject.toml` 镜像它们但环境文件包含系统级说明（R, kallisto, DM8）。添加新依赖时，必须同时更新三者。
- **`scvi-tools` 位于 `torch` 可选组**：ZINB-VAE 的 scvi 模式导入 `scvi`。早期 `pyproject.toml` 缺失此项 — 现已添加。如果看到 `ImportError: scvi`，请用 `pip install -e ".[torch]"` 安装。
- **`src/__init__.py` 在包加载时急切导入所有子模块** — 包括 `storage`, `parsers` 和 `selectors`。在其中添加新子模块导入意味着每次 `import` 包时都会执行它。保持顶级导入精简。注意 `storage._minio`, `storage._graph` 等不在包级别导入 — 仅导入 `storage`（命名空间）。
- **`pp.normalize()` 绕过自己的选择器**：与使用自己模块级选择器的 `pp.impute()` 和 `pp.batch_correct()` 不同，`normalize()` 调用 selectors 中的全局 `recommend_strategy()`。这是故意的 — 归一化选择器依赖检测到的模态。
- **`pp.normalize()` 中的方法映射**：函数通过硬编码的 `method_map` 字典将方法字符串映射到类名。添加新归一化器需要同时更新 `normalizers/__init__.py`（导出）和此字典（字符串→类映射）。
- **存储后端自动 fallback**：`MinIOClient`, `RelationalDBClient` 和 `GraphDBClient` 在驱动缺失时均静默回退到本地存储。测试应针对 fallback 路径（基于 tmpdir），而非期望运行中的服务器。
- **选择器模型目录**：`config/models/` 除 `.gitkeep` 外均被 gitignore。模型必须在每个环境中通过调用 `train_and_persist_models()` 重新生成。`recommend_strategy()` 和 `detect_modality()` 函数在无持久化模型时检查并使用 fallback — 它们从不会因缺失模型文件而失败。
- **kallisto 通过 subprocess 调用**：`FASTQParser._run_kallisto()` 中。这意味着它必须在系统 PATH 上，而非通过 pip 安装。解析器还会创建必须清理的临时目录（`omics_qc_*`）。
- **GMM 特征契约**：`_modality.py` 中的 `_extract_features()` 返回 4 个特征 `[missing_rate, log1p(n_obs), log1p(n_vars), n_batches]` — 此顺序和数量必须与 `_persistence.py` 中的 `generate_training_data()[:, 1:5]` 匹配。更改任一方都需要更新另一方、`detect_modality()` 以及 `tests/test_selectors.py` 中的 shape 断言。
- **GMM 聚类标签校准**：`train_and_persist_models()` 训练 GMM 后调用 `_calibrate_labels()` 将聚类中心按 `log1p(n_vars)` 升序排列，映射到正确模态顺序 `proteomics → metabolomics → bulk_rna → scrna → atac`。不要使用模运算映射！
- **mzML 解析器**：pymzml 的 `Reader` 迭代器只能遍历一次。已在单次遍历中将所有 peaks 数据收集到 `spectra_peaks` 列表中。添加新特征时不要引入第二次遍历。
- **MissForest 正确处理 NA⇔0 区分**：`MissForestImputer.run()` 现在使用 `np.isnan(X)` 检测缺失值，而非将 0 视为缺失。NaN/None 是需要插补的缺失值；0 是真实零表达。
- **MAGIC 过度平滑**：`MAGICImputer.run()` 检测低维异质性数据（`n_vars < 200` 且 `zero_rate > 0.3`）并发出警告，建议改用 MissForest 或 ZINB-VAE。此 guard 专门针对 .fcs 流式细胞术数据。
- **Harmony vs ComBat fallback 质量差距**：Harmony 的 fallback（无 scanpy）是 **no-op 直通** — 批次效应静默地未被校正。ComBat 的 fallback 仅做均值中心化（无方差调整）。DANN 完全无 fallback（硬性要求 PyTorch）。当 scanpy/torch 不可用时，校正器静默降级；检查 `adata.uns["standardization"]["batch_correction"]["method"]` 以确认实际运行了什么。
- **DANN GPU 加速与模式崩溃检测**：`DANCorrector` 新增 `device="auto"` 参数，自动检测 CUDA/MPS/CPU 并将模型和数据移至设备。`_detect_mode_collapse()` 在训练后检测域分类器准确率(>95%)、重建 loss 振荡(CV>0.5)和域不变性分数(<0.3)来判断模式崩溃。检测到崩溃时以 λ×2 自动重训练，仍失败则建议 Harmony。记录在 `mode_collapse_risk` 字段中。
- **TMM 三级 fallback 体系**：R (edgeR) → Python 原生实现（M/A-value + 30%/5% trimming + 逆方差加权） → CPM。原 fallback 仅为 CPM，现 Python 路径实现了标准 TMM 的截尾均值 M 值计算。
- **VSN 三级 fallback 体系**：R (limma) → arcsinh 方差稳定化（auto-tune 最小化 mean-variance correlation） → log2。原 fallback 仅为 `log2(x+1)`，现 arcsinh 路径提供真正的方差稳定化效果。
- **DESeq2 可配置设计公式**：`DESeq2Normalizer` 新增 `design` 参数。`None` 时使用截距模型 `~1`（正确用于 size factor 估计）。传入 `["condition", "batch"]` 时从 `adata.obs` 提取真实生物学设计列构建 colData。移除了硬编码的 `["A"]*n` 占位符。
- **`ensure_dense()` / `check_r_available()` 已激活**：`normalizers/_utils.py` 中的工具函数现已在 TMM、DESeq2、VSN、Scran 等归一化器中实际调用。`ensure_dense()` 含内存安全检查，`check_r_available()` 替代原始 try/except ImportError。
- **Pipeline ↔ StorageManager 集成**：`StandardizationPipeline.run()` 和 `__init__()` 新增 `use_storage=True` 参数。启用时 `_save()` 调用 StorageManager 的 `put_anndata()`、`save_sample()`、`record_pipeline_run()`、`save_quality_metrics()` 和 `build_knowledge_graph()`。所有存储操作包裹在 try/except 中，失败不阻塞流水线。
- **TMM/DESeq2 reshape bug 修复**：`size_factors.reshape(1, -1)` 修正为 `.reshape(-1, 1)`（3 处）。原方向广播错误导致归一化因子维度不匹配。

### CLI 命令行入口

[src/main.py](src/main.py) 提供 `omics-std` 控制台命令（在 `pyproject.toml` 的 `[project.scripts]` 中注册）：

```bash
omics-std config/default.yaml -i data/raw/ -o data/processed/ --verbose
# 等价于：
python -m src config/default.yaml --input data/raw/ --output data/processed/
```

参数：位置参数 `config`（默认 `config/default.yaml`）、`--input/-i`、`--output/-o`（默认 `data/processed`）、`--verbose/-v`、`--version`。

### CI/CD（`.github/workflows/`）

- `ci.yml` — Python 3.10–3.12 上运行 ruff 检查 + pytest 测试，覆盖 Ubuntu 和 Windows
- `publish.yml` — 标签推送时构建并发布到 PyPI

### 工具模块（`src/utils/`）

- `src/_compat.py` — Python 版本兼容性适配
- `src/utils/_io.py` — 文件 I/O 辅助（临时目录、路径解析）
- `src/utils/_decorators.py` — 共享装饰器（`@timeit`、`@deprecated` 等）

### 参考项目

以下 scverse 生态参考项目位于 `d:/Database/_reference/`（仓库外）：
- `muon/` — 多模态组学框架
- `scanpy/` — 单细胞分析工具包
- `muon-tutorials/` — muon 官方教程
