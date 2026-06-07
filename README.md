# omics_standardization

空间环境免疫多组学数据标准化处理模块

[![Python](https://img.shields.io/pypi/pyversions/omics_standardization)](https://pypi.org/project/omics_standardization)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)

## 概述

`omics_standardization` 是一个面向空间环境免疫学研究的多模态组学数据标准化处理流水线。
支持以下数据模态的统一读取、缺失值插补、尺度归一化和批次效应校正：

- **单细胞转录组**（scRNA-seq）：.h5ad, .loom, .mtx
- **散装转录组**（bulk RNA-seq）：计数矩阵 + 元数据
- **流式细胞术**（flow cytometry）：.fcs
- **质谱代谢/蛋白质组**（mass spectrometry）：mzML
- **染色质可及性**（ATAC-seq）：.h5ad (MuData)

## 流水线

```
原始数据 → [解析 Parsers] → [插补 Imputers] → [归一化 Normalizers] → [批次校正 Batch Correctors] → 标准化数据
              ↑                    ↑                    ↑                        ↑
              └─ 模态识别         └─ 算法选择器        └─ 算法选择器             └─ 算法选择器
              (GMM)              (RandomForest)       (RandomForest)            (RandomForest)
```

## 安装

```bash
# 基础安装
pip install -e .

# 含 PyTorch 支持（ZINB-VAE, DANN）
pip install -e ".[torch]"

# 含 R 桥接支持（DESeq2, ComBat）
pip install -e ".[rpy2]"

# 开发安装
pip install -e ".[dev,test,docs]"
```

## 快速开始

```python
from parsers import parse_file
from pipeline import StandardizationPipeline

# 1. 解析多模态数据
mdata = parse_file("data/raw/batch1/")

# 2. 运行标准化流水线
pipeline = StandardizationPipeline(config="config/default.yaml")
processed = pipeline.run(mdata)

# 3. 保存结果
processed.write("data/processed/adata/result.h5mu")
```

## 项目结构

```
omics_standardization/
├── src/                    # 源代码（Python 包）
│   ├── parsers/            # 多模态数据解析
│   ├── selectors/          # 算法智能选择引擎
│   ├── imputers/           # 缺失值分类插补
│   ├── normalizers/        # 尺度归一化
│   ├── batch_correctors/   # 深度批次解耦
│   ├── preprocessing/      # 统一 pp 命名空间
│   ├── tools/              # 降维/评估工具
│   ├── plotting/           # 可视化
│   └── pipeline/           # 端到端流水线
├── config/                 # YAML 配置文件
├── notebooks/              # Jupyter 教程
├── tests/                  # 单元测试
└── docs/                   # 文档
```

## 依赖

核心依赖：`anndata`, `scanpy`, `muon`, `mudata`, `numpy`, `pandas`, `scikit-learn`, `pydantic`

可选依赖：`torch` (深度学习), `rpy2` (R 包桥接)

## 引用

待发表。

## 许可证

BSD-3-Clause License
