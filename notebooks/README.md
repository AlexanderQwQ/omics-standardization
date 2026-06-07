# omics_standardization 教程

本目录包含 omics_standardization 的 Jupyter Notebook 教程，编号按推荐学习顺序排列。

## 教程列表

1. **01_data_exploration.ipynb** — 数据探索与初步分析
2. **02_parser_test.ipynb** — 多模态数据解析模块测试
3. **03_imputation_test.ipynb** — 缺失值插补方法对比
4. **04_normalization_test.ipynb** — 归一化方法对比
5. **05_batch_correction_test.ipynb** — 批次校正效果演示
6. **06_full_pipeline.ipynb** — 端到端标准化流水线完整演示（推荐从这里开始）

## 环境准备

```bash
pip install -e ".[torch,rpy2,docs]"
jupyter notebook
```

## 参考

- [muon-tutorials](https://muon-tutorials.readthedocs.io/) — muon 官方教程
- [scanpy-tutorials](https://scanpy.readthedocs.io/en/stable/tutorials.html) — scanpy 官方教程
