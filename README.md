# omics_standardization

空间环境免疫多组学数据标准化处理模块

[![Python](https://img.shields.io/pypi/pyversions/omics_standardization)](https://pypi.org/project/omics_standardization)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)
[![CI](https://github.com/AlexanderQwQ/omics-standardization/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexanderQwQ/omics-standardization/actions/workflows/ci.yml)

## 概述

`omics_standardization` 是一个面向**空间环境免疫学**研究的多模态组学数据标准化处理流水线。针对航天免疫学研究中的数据异质性问题（不同实验平台、不同数据格式、不同批次），提供从原始文件解析到标准化输出的端到端解决方案。

### 核心能力

- **多模态统一解析** — 9 种文件格式 → AnnData/MuData，覆盖 6 大组学数据类型
- **智能算法选择** — 基于 GMM + RandomForest 自动识别数据模态并推荐最优处理策略
- **缺失值分类插补** — 3 种算法适配不同零膨胀模式（技术缺失 vs 生物零），含低生物量样本检测
- **尺度归一化** — 5 种方法覆盖从 bulk 到 single-cell 的归一化需求
- **批次效应校正** — 3 种方法从经典经验贝叶斯到深度对抗网络（GPU 加速 + 模式崩溃检测）
- **混合存储架构** — MinIO（对象）+ SQLite/DM8（关系型）+ Neo4j（图，扩展 Gene/Pathway/Disease 节点），全部支持本地 fallback + Pipeline 集成
- **效果评估体系** — MMD / Wasserstein / Batch Silhouette 跨域对齐量化指标

### 支持的组学模态与文件格式

| 模态 | 描述 | 支持格式 | 默认插补 | 默认归一化 | 默认批次校正 |
|------|------|----------|----------|------------|--------------|
| **scRNA-seq** | 单细胞转录组 | `.h5ad`, `.loom`, `.mtx` | ZINB-VAE | Scran | Harmony |
| **Bulk RNA-seq** | 散装转录组 | `.csv`, `.tsv`, `.txt` | MissForest | TMM/DESeq2 | ComBat |
| **Proteomics** | 流式细胞术 / 质谱蛋白 | `.fcs`, `.csv` | MissForest | Quantile | ComBat |
| **Metabolomics** | 质谱代谢组 | `.mzML`, `.csv` | MissForest | VSN | ComBat |
| **ATAC-seq** | 染色质可及性 | `.h5ad`, `.csv` | MissForest | TMM | Harmony |
| **Microbiome** | 宏基因组 / 微生物组 | `.biom`, `.csv` | MissForest | TMM | ComBat |
| **FASTQ/BAM** | 原始测序数据 | `.fastq`, `.bam` | — | — | — |

## 流水线架构

```
                          ┌──────────────────────────────────────────┐
                          │         StandardizationPipeline          │
                          └──────────────────────────────────────────┘
                                              │
  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
  │ ① Parse  │───→│ ② Select     │───→│ ③ Impute     │───→│ ④ Normalize  │───→│ ⑤ Batch      │───→│ ⑥ Eval   │
  │   解析   │    │   策略选择    │    │   缺失值插补  │    │   尺度归一化  │    │   Correct    │    │   评估   │
  └──────────┘    └──────────────┘    └──────────────┘    └──────────────┘    │   批次校正    │    └──────────┘
       │                │                    │                    │           └──────────────┘          │
       ▼                ▼                    ▼                    ▼                  ▼                   ▼
  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
  │ 9 种格式 │    │ GMM 模态识别 │    │ MissForest   │    │ TMM / DESeq2 │    │ ComBat       │    │ MMD/WS   │
  │ →AnnData │    │ RF  策略推荐 │    │ ZINB-VAE     │    │ Scran        │    │ Harmony      │    │ Silhouette│
  │          │    │              │    │ MAGIC        │    │ Quantile/VSN │    │ DANN (GPU)   │    │          │
  └──────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘
```

每步处理结果写入 `adata.uns["standardization"]` 用于溯源追踪。

## 安装

### 推荐方式：Conda（一键安装，含 R + kallisto + CUDA PyTorch）

```bash
conda env create -f environment.yml     # 创建完整环境（~2-5 GB）
conda activate omics-std                # 激活
pip install -e ".[test,dev]"            # 可编辑安装
```

### 备选：pip + venv（仅 Python 包，不含 R/kallisto）

```bash
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate                            # Windows
pip install -r requirements.txt
pip install -e ".[test,dev]"
```

### 可选依赖组

```bash
pip install -e ".[torch]"         # PyTorch + scvi-tools（ZINB-VAE, DANN）
pip install -e ".[rpy2]"          # R 桥接（TMM, DESeq2, VSN, Scran）
pip install -e ".[fcs]"           # 流式细胞术 .fcs 解析
pip install -e ".[ms]"            # 质谱 mzML 解析
pip install -e ".[impute]"        # MAGIC 图扩散插补
pip install -e ".[ngs]"           # NGS 测序数据解析（FASTQ/BAM）
pip install -e ".[microbiome]"    # BIOM 微生物组格式
pip install -e ".[storage]"       # MinIO + Neo4j + DM8 混合存储
pip install -e ".[docs]"          # 文档构建
pip install -e ".[dev,test]"      # 开发工具 + 测试
```

### 系统级依赖（手动安装）

部分功能需要额外安装系统级软件：

| 依赖 | 用途 | 安装方式 |
|------|------|----------|
| R + Bioconductor | TMM / DESeq2 / VSN / Scran | `install.packages("BiocManager"); BiocManager::install(c("edgeR","DESeq2","limma","scran"))` |
| kallisto | FASTQ 双端定量 | `conda install -c bioconda kallisto` 或 [官网下载](https://pachterlab.github.io/kallisto/download) |
| DM8 客户端 | 国产数据库存储（仅 Windows，可选） | 达梦官网，SQLite 为默认后端 |

## 快速开始

### Demo 数据验证（无需外部数据）

```bash
# 1. 生成合成 demo 数据 → data/raw/（6 种模态 × 多种格式）
python scripts/generate_demo_data.py

# 2. 全流程验证
python scripts/run_demo_pipeline.py

# 快速模式（仅解析 + 模态检测）
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

### Python API

```python
from parsers import parse_file
from pipeline import StandardizationPipeline
from preprocessing import impute, normalize, batch_correct  # pp.* 命名空间

# ---- 方式 1：端到端流水线 ----

pipeline = StandardizationPipeline(config="config/default.yaml")
result = pipeline.run(input_path="data/raw/", output_path="data/processed/result.h5mu")

# ---- 方式 2：分步调用 ----

# 1. 解析多模态数据
adata = parse_file("data/raw/batch1/")

# 2. 手动控制每个步骤
from preprocessing import impute, normalize, batch_correct

adata = impute(adata, method="missforest")           # 指定插补方法
adata = normalize(adata)                             # 自动选择归一化
adata = batch_correct(adata, batch_key="batch")      # 自动选择批次校正

# ---- 方式 3：训练选择器模型 ----

from selectors import train_and_persist_models
train_and_persist_models()  # 训练并保存 GMM + RF 模型到 config/models/

# ---- 方式 4：混合存储 ----

from storage import StorageManager

store = StorageManager.from_config("config/default.yaml")
with store:
    store.save_sample("S001", experiment_id="E001", modality="scrna", condition="microgravity")
    store.build_knowledge_graph(adata, experiment_id="E001")
    store.record_pipeline_run("E001", imputation_method="zinb_vae", normalization_method="scran")

# ---- 方式 5：Pipeline 直接存入存储 ----
pipeline = StandardizationPipeline(use_storage=True)
result = pipeline.run(input_path="data/raw/", output_path="data/processed/result.h5mu")
```

### 命令行

```bash
# 使用默认配置
omics-std

# 指定输入输出
omics-std config/default.yaml -i data/raw/ -o data/processed/

# 详细日志
omics-std config/default.yaml -i data/raw/ -o data/processed/ -v
```

## 算法目录

### 缺失值插补（`imputers/`）

| 算法 | 实现文件 | 原理 | 适用场景 |
|------|----------|------|----------|
| **MissForest** | `_missforest.py` | 随机森林迭代插补 | 通用型，适合各类组学数据 |
| **ZINB-VAE** | `_zinb_vae.py` | 零膨胀负二项变分自编码器 | 高零膨胀的 scRNA-seq，区分技术 dropout 与生物零 |
| **MAGIC** | `_magic.py` | KNN 图马尔可夫扩散平滑 | 单细胞数据，保留细胞间关系 |

### 尺度归一化（`normalizers/`）

| 算法 | 实现文件 | 原理 | 适用场景 |
|------|----------|------|----------|
| **TMM** | `_tmm_deseq.py` | 加权截断均值 M 值（R + Python 原生 + CPM 三级 fallback） | Bulk RNA-seq |
| **DESeq2** | `_tmm_deseq.py` | 中位数比率法 | Bulk RNA-seq |
| **Scran** | `_scran.py` | 单细胞池化去卷积标准化 | scRNA-seq（三级 fallback） |
| **Quantile** | `_quantile.py` | 分位数归一化 | 蛋白质组 / 代谢组 |
| **VSN** | `_quantile.py` | 方差稳定（R limma + arcsinh auto-tune + log2 三级 fallback） | 质谱数据 |

### 批次效应校正（`batch_correctors/`）

| 算法 | 实现文件 | 原理 | 适用场景 |
|------|----------|------|----------|
| **ComBat** | `_combat.py` | 经验贝叶斯批次校正 | 通用型，适合各模态 |
| **Harmony** | `_harmony.py` | 迭代软聚类校正 | 单细胞数据 |
| **DANN** | `_dann.py` | 域对抗神经网络（GRL + GPU 加速 + 模式崩溃检测） | 深度学习，需要 PyTorch |

## 智能选择器引擎（`selectors/`）

根据数据特征自动推荐最优处理策略，无需手动选择算法：

```
数据 AnnData
     │
     ▼
┌─────────────────┐
│ 特征提取         │  → [missing_rate, n_obs, n_vars, n_batches, file_ext]
└────────┬────────┘
         ▼
┌─────────────────┐
│ GMM 模态识别     │  → scrna | bulk_rna | proteomics | metabolomics | atac
└────────┬────────┘
         ▼
┌─────────────────┐
│ RF 策略推荐      │  → {imputation, normalization, batch_correction}
└────────┬────────┘
         ▼
    推荐策略
```

- **无模型时**：自动 fallback 到启发式规则（覆盖全部 5 种模态）
- **首次使用建议**：运行 `train_and_persist_models()` 训练并持久化模型到 `config/models/`
- **标注质量评估**：`assess_annotation_quality()` 评估训练数据质量，`is_high_quality_available()` 判断是否满足"高质量标注多组学库"条件，驱动 GMM→RF 动态切换

## 混合存储架构（`storage/`）

节点分离的三层存储，由 `StorageManager` 统一协调：

| 存储层 | 后端 | Fallback | 用途 |
|--------|------|----------|------|
| **对象存储** | MinIO / S3 | 本地文件系统 `data/storage/objects/` | 大型 AnnData / HDF5 / 原始文件 |
| **关系型数据库** | SQLite / DM8 | 自动选择 SQLite | 样本元数据、处理参数、质量指标 |
| **图数据库** | Neo4j | JSON-LD 文件 `data/storage/graph/` | 样本关系、细胞相似性图、知识图谱 |

所有驱动均延迟导入（lazy import），无任何存储后端时包仍可正常导入。

### 关系型模式

```
samples         — 样本 ID、模态、条件、来源文件
pipeline_runs   — 运行 ID、时间戳、配置哈希、处理步骤
quality_metrics — 指标 ID、关联运行、指标名、值
```

### 图模式

```
(:Sample) -[:CORRELATED]-> (:Sample)          样本间相关性
(:Sample) -[:BELONGS_TO_BATCH]-> (:Batch)     样本属于批次
(:Sample) -[:SAME_BATCH]-> (:Sample)           同批次样本
(:Sample) -[:EXPRESSES]-> (:Gene)              样本表达基因
(:Gene) -[:INVOLVED_IN]-> (:Pathway)            基因参与通路
(:Gene) -[:ASSOCIATED_WITH]-> (:Disease)        基因关联疾病
```

## 配置系统

全局配置文件 [config/default.yaml](config/default.yaml) 包含所有可调参数：

```yaml
modalities:         # 启用的模态列表
imputation:         # 插补方法和参数
normalization:      # 归一化方法和参数
batch_correction:   # 批次校正方法和参数
selector:           # 选择器模型配置
logging:            # 日志级别和输出
storage:            # 混合存储后端连接参数
output:             # 输出格式（h5mu | h5ad | parquet）
```

Python 中访问配置：

```python
from _settings import settings

print(settings.imputation)         # 插补配置
print(settings.storage.minio)      # MinIO 连接配置
settings.verbosity = Verbosity.debug
```

## 评估体系

标准化处理效果通过多维指标量化评估：

| 指标 | 描述 | 计算方式 |
|------|------|----------|
| **RMSE** | 均方根误差 | 与原始数据层比较（需 `layers["raw"]`） |
| **batch_mixing** | 批次混合度 | 批次标签分布均匀度 |
| **mmd** | 最大均值差异 | RBF kernel MMD²，batch 间分布距离（越低越好） |
| **wasserstein** | Wasserstein 距离 | 每特征 1D Wasserstein 平均（越低越好） |
| **batch_silhouette** | 批次轮廓系数 | 1 - silhouette_score(batch)，接近 1 表示批次融合良好 |

三个高级指标在 `obsm["X_corrected"]` 上计算（如存在），回退到 `.X`。单批次边界情况安全处理。

## 项目结构

```
omics_standardization/
├── src/                        # 源代码（Python 包根目录）
│   ├── parsers/                # 多模态数据解析（9 种格式 → AnnData）
│   ├── selectors/              # 智能算法选择引擎（GMM + RF）
│   ├── imputers/               # 缺失值分类插补（MissForest, ZINB-VAE, MAGIC）
│   ├── normalizers/            # 尺度归一化（TMM, DESeq2, Scran, Quantile, VSN）
│   ├── batch_correctors/       # 批次效应校正（ComBat, Harmony, DANN）
│   ├── preprocessing/          # pp 统一命名空间（impute/normalize/batch_correct）
│   ├── tools/                  # tl 降维/评估工具（PCA, UMAP, evaluation）
│   ├── plotting/               # pl 可视化（QC, heatmap, embedding）
│   ├── pipeline/               # 端到端标准化流水线
│   ├── storage/                # 混合存储架构（MinIO + SQLite/DM8 + Neo4j）
│   ├── _settings/              # 全局配置单例 + 日志级别
│   ├── utils/                  # I/O 辅助 + 装饰器
│   ├── logging.py              # 自定义日志系统
│   ├── main.py                 # CLI 入口（omics-std 命令）
│   └── _compat.py              # Python 版本兼容适配
├── config/
│   ├── default.yaml            # 默认配置（所有可调参数）
│   ├── logging.yaml            # 日志配置
│   └── models/                 # 训练好的选择器模型（gitignored）
├── scripts/
│   ├── generate_demo_data.py   # 合成 demo 数据生成器
│   └── run_demo_pipeline.py    # demo 数据全流程验证
├── tests/                      # 单元测试（11 个文件，10 个测试模块）
├── docs/                       # Sphinx 文档
├── notebooks/                  # Jupyter 教程（6 个 notebook）
├── data/                       # 数据目录（raw / processed / metadata / storage）
├── environment.yml             # Conda 环境（推荐）
├── requirements.txt            # Pip 依赖（备选）
└── pyproject.toml              # 项目元数据 + 构建 + 工具配置
```

## 开发指南

```bash
# 代码检查
ruff check src/ tests/

# 自动格式化
ruff format src/ tests/

# 运行全部测试
pytest tests/

# 单个测试文件
pytest tests/test_pipeline.py

# 覆盖率报告
pytest tests/ --cov=src --cov-report=term-missing

# 构建发布包
python -m build
```

### 测试概览

| 测试文件 | 覆盖内容 |
|----------|----------|
| `test_parsers.py` | 文件类型检测（全部支持扩展名） |
| `test_csv_parser.py` | CSV/TSV 分隔符和布局检测 |
| `test_biom_parser.py` | BIOM 稀疏/密集解析和分类学提取 |
| `test_imputers.py` | 插补选择和 MissForest |
| `test_normalizers.py` | TMM, DESeq2, Quantile, VSN, Scran |
| `test_batch_correctors.py` | ComBat, Harmony, 校正选择器 |
| `test_selectors.py` | 模态检测（GMM+启发式），策略推荐（RF+fallback），模型训练持久化 |
| `test_pipeline.py` | 端到端流水线、存储集成、指标评估 |
| `test_storage.py` | MinIO, RelationalDB (SQLite), GraphDB, StorageManager |

### 共享 Fixtures（`tests/conftest.py`）

| Fixture | 规格 | 用途 |
|---------|------|------|
| `small_adata` | 100 cells × 200 genes, 2 batches, ~30% zeros, CSR 稀疏 | 通用测试 |
| `small_adata_no_batch` | 50 × 100, 无 batch 列 | 无批次场景 |
| `high_missing_adata` | 50 × 100, ~90% zeros | 高缺失率场景 |

## 依赖

### 核心依赖

`anndata>=0.10` · `scanpy>=1.9` · `muon>=0.1` · `mudata>=0.2` · `numpy>=1.21` · `pandas>=1.5` · `scipy>=1.9` · `scikit-learn>=1.1` · `matplotlib>=3.7` · `seaborn>=0.12` · `pyyaml>=6.0` · `h5py>=3.8` · `joblib` · `numba>=0.57` · `pydantic>=2.0` · `rich` · `tqdm`

### 可选依赖

| 依赖组 | 包 | 用途 |
|--------|-----|------|
| `torch` | `torch>=1.13`, `scvi-tools>=1.0` | ZINB-VAE 插补 + DANN 批次校正 |
| `rpy2` | `rpy2>=3.5` | R 桥接（TMM/DESeq2/VSN/Scran） |
| `fcs` | `fcsparser>=0.2` | 流式细胞术 .fcs 解析 |
| `ms` | `pymzml>=2.5` | 质谱 mzML 解析 |
| `impute` | `magic-impute>=3.0` | MAGIC 图扩散插补 |
| `ngs` | `pysam>=0.22` | BAM/SAM/FASTQ 解析 |
| `microbiome` | `biom-format>=2.1` | BIOM 微生物组格式 |
| `storage` | `minio>=7.2`, `neo4j>=5.20`, `dmPython>=8.0` | 混合存储后端 |

## Python 版本兼容性

- 要求 `>=3.10,<3.14`
- CI 测试覆盖 Python 3.10、3.11、3.12（Ubuntu + Windows）
- `numba>=0.57` 版本约束确保 Python 3.11+ 兼容性
- 全部源文件使用 `from __future__ import annotations` 延迟注解求值

## 引用

待发表。

## 许可证

BSD-3-Clause License — 详见 [LICENSE](LICENSE) 文件。
