# 技术实现规格说明

## 概述

`omics_standardization` 是一个面向空间环境免疫学研究的多模态组学数据标准化处理流水线。

## 支持的模态

| 模态 | 文件格式 | 解析器 |
|------|---------|--------|
| scRNA-seq | .h5ad, .loom, .mtx | H5ADParser |
| ATAC-seq | .h5ad (MuData) | H5ADParser |
| 流式细胞术 | .fcs | FCSParser |
| 质谱 | mzML | MzMLParser |
| FASTQ/BAM | .fastq, .bam | FASTQParser |

## 流水线步骤

1. **解析** (Parsers): 自动识别文件类型，读取为 AnnData/MuData
2. **算法选择** (Selectors): GMM 识别模态 + RandomForest 推荐策略
3. **插补** (Imputers): MissForest / ZINB-VAE / MAGIC 填补缺失值
4. **归一化** (Normalizers): TMM / DESeq2 / Scran / Quantile / VSN
5. **批次校正** (Batch Correctors): ComBat / Harmony / DANN
6. **评估** (Evaluation): 生成质量报告

## 核心数据结构

- **AnnData**: 单模态数据（anndata 包）
- **MuData**: 多模态数据容器（mudata 包）

## 配置

所有参数通过 `config/default.yaml` 集中管理，可通过 `omics_standardization.settings` 在代码中访问和覆盖。

## 参考

- [scanpy](https://scanpy.readthedocs.io/) — 单细胞分析框架
- [muon](https://muon.readthedocs.io/) — 多模态组学框架
- [mudata](https://mudata.readthedocs.io/) — 多模态数据容器
