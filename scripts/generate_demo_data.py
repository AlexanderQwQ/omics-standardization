"""
Demo 数据生成器 — 为所有 5 种组学模态生成合成数据文件

生成文件覆盖项目支持的 6 种解析格式:
    .h5ad  — H5ADParser (scRNA, ATAC)
    .csv   — CSVParser (scRNA, bulk RNA, proteomics, metabolomics, ATAC)
    .tsv   — TSVParser (bulk RNA)
    .fcs   — FCSParser (proteomics / 流式细胞术)
    .mzML  — MzMLParser (metabolomics / 质谱)
    .biom  — BIOMParser (microbiome / 宏基因组)

用法:
    python scripts/generate_demo_data.py              # 生成全部
    python scripts/generate_demo_data.py --modality scrna  # 仅生成 scrna

输出: data/raw/{modality}/
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ==============================================================================
# 路径配置
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

MODALITY_CONFIG = {
    "scrna": {
        "n_obs": 500,
        "n_vars": 8000,       # > 5000，启发式规则识别为 scrna
        "n_batches": 2,
        "zero_rate": 0.40,
        "label": "scRNA-seq",
        "batch_labels": ["donor_A", "donor_B"],
    },
    "bulk_rna": {
        "n_obs": 50,
        "n_vars": 2000,       # 500-5000，启发式规则识别为 bulk_rna
        "n_batches": 1,
        "zero_rate": 0.02,
        "label": "Bulk RNA-seq",
        "batch_labels": ["control"],
    },
    "proteomics": {
        "n_obs": 100,
        "n_vars": 50,         # < 500 + 低缺失率 → proteomics
        "n_batches": 2,
        "zero_rate": 0.30,
        "label": "Proteomics (FCS)",
        "batch_labels": ["panel_v1", "panel_v2"],
    },
    "metabolomics": {
        "n_obs": 80,
        "n_vars": 200,        # < 500 + 中等缺失率 → metabolomics
        "n_batches": 1,
        "zero_rate": 0.55,    # > 0.5 以区别于 proteomics (< 0.5)
        "label": "Metabolomics",
        "batch_labels": ["run_01"],
    },
    "atac": {
        "n_obs": 300,
        "n_vars": 20000,      # > 10000 + 极高缺失率 → atac
        "n_batches": 2,
        "zero_rate": 0.92,
        "label": "ATAC-seq",
        "batch_labels": ["tissue_lung", "tissue_liver"],
    },
    "microbiome": {
        "n_obs": 30,
        "n_vars": 450,        # < 500 + 高缺失率 → metabolomics
        "n_batches": 2,
        "zero_rate": 0.55,
        "label": "Microbiome (BIOM)",
        "batch_labels": ["stool", "oral"],
    },
}

# 已知微生物分类层级
TAXONOMY = [
    ["k__Bacteria", "p__Firmicutes", "c__Bacilli", "o__Lactobacillales", "f__Streptococcaceae", "g__Streptococcus", "s__pneumoniae"],
    ["k__Bacteria", "p__Bacteroidetes", "c__Bacteroidia", "o__Bacteroidales", "f__Bacteroidaceae", "g__Bacteroides", "s__fragilis"],
    ["k__Bacteria", "p__Proteobacteria", "c__Gammaproteobacteria", "o__Enterobacterales", "f__Enterobacteriaceae", "g__Escherichia", "s__coli"],
    ["k__Bacteria", "p__Actinobacteria", "c__Actinobacteria", "o__Bifidobacteriales", "f__Bifidobacteriaceae", "g__Bifidobacterium", "s__longum"],
    ["k__Bacteria", "p__Firmicutes", "c__Clostridia", "o__Clostridiales", "f__Lachnospiraceae", "g__Roseburia", "s__intestinalis"],
]


# ==============================================================================
# 合成 AnnData 生成
# ==============================================================================

def _make_adata(
    n_obs: int,
    n_vars: int,
    n_batches: int,
    zero_rate: float,
    batch_labels: list[str],
    modality_name: str,
    seed: int = 42,
) -> "AnnData":
    """生成合成 AnnData 对象

    Args:
        n_obs: 样本/细胞数
        n_vars: 特征/基因数
        n_batches: 批次数
        zero_rate: 目标零值比例
        batch_labels: 批次标签
        modality_name: 模态名称（用于 var 命名）
        seed: 随机种子

    Returns:
        AnnData 对象，包含:
            - .X: sparse CSR 计数矩阵
            - .obs["batch"]: 批次注释
            - .var 索引: 特征名
    """
    from anndata import AnnData
    from scipy.sparse import csr_matrix

    rng = np.random.RandomState(seed)

    # 模拟计数分布：负二项分布
    mean = 10 if modality_name in ("scrna", "bulk_rna") else 50
    dispersion = 0.3 if modality_name in ("scrna", "atac") else 0.7

    X = rng.negative_binomial(int(mean * dispersion), dispersion, size=(n_obs, n_vars)).astype(np.float32)

    # 施加零膨胀
    if zero_rate > 0:
        existing_zeros = np.mean(X == 0)
        extra_zero_rate = max(0, zero_rate - existing_zeros)
        if extra_zero_rate > 0:
            mask = rng.random((n_obs, n_vars)) < extra_zero_rate
            X[mask] = 0

    # 批次分配
    batch_col = np.repeat(batch_labels[:n_batches], np.ceil(n_obs / n_batches))[:n_obs]
    rng.shuffle(batch_col)

    # 特征名（避免包含 "gene"/"sample"/"cell" 等关键词以防 CSV 布局检测误判）
    if modality_name == "scrna":
        var_names = [f"ENSG{str(i).zfill(8)}" for i in range(n_vars)]
    elif modality_name == "atac":
        var_names = [f"chr{chr_num}:{start}-{end}" for chr_num, start, end in zip(
            rng.choice([f"chr{i}" for i in range(1, 23)], n_vars),
            rng.randint(1, 250_000_000, n_vars),
            rng.randint(1, 250_000_000, n_vars),
        )]
    elif modality_name == "proteomics":
        var_names = [f"CD{cd}_{marker}" for cd, marker in zip(
            rng.choice(["CD3", "CD4", "CD8", "CD19", "CD45", "CD56", "CD11b", "CD14", "CD16", "CD25"], n_vars),
            rng.choice(["FITC", "PE", "APC", "PerCP", "BV421", "BV510", "PE-Cy7", "APC-Cy7", "AF700", "PacificBlue"], n_vars),
        )]
    elif modality_name == "metabolomics":
        var_names = [f"M{mz:.4f}T{rt:.1f}" for mz, rt in zip(
            rng.uniform(50, 1000, n_vars),
            rng.uniform(0.5, 30, n_vars),
        )]
    elif modality_name == "bulk_rna":
        var_names = [f"BR{str(i).zfill(6)}" for i in range(n_vars)]
    elif modality_name == "microbiome":
        var_names = [f"OTU{str(i).zfill(6)}" for i in range(n_vars)]
    else:
        var_names = [f"{modality_name}_feat_{i}" for i in range(n_vars)]

    # 样本名
    if modality_name in ("scrna", "atac"):
        obs_names = [f"cell_{i}" for i in range(n_obs)]
    elif modality_name == "proteomics":
        obs_names = [f"event_{i}" for i in range(n_obs)]
    elif modality_name == "microbiome":
        obs_names = [f"sample_{i}" for i in range(n_obs)]
    else:
        obs_names = [f"sample_{i}" for i in range(n_obs)]

    return AnnData(
        X=csr_matrix(X),
        obs=pd.DataFrame({"batch": batch_col}, index=obs_names),
        var=pd.DataFrame(index=var_names),
    )


# ==============================================================================
# 文件写入器
# ==============================================================================

def write_h5ad(adata: "AnnData", path: Path) -> None:
    """写入 .h5ad 文件"""
    adata.write(path)
    print(f"  ✓ {path} ({adata.n_obs}×{adata.n_vars})")


def write_csv(
    adata: "AnnData",
    path: Path,
    layout: str = "samples_x_genes",
    sep: str = ",",
) -> None:
    """将 AnnData 导出为 CSV/TSV 表达矩阵

    默认使用 samples×genes 布局（样本为行，基因为列），
    与 AnnData 的内存布局 (n_obs, n_vars) 一致，避免解析时的转置歧义。

    Args:
        adata: AnnData 对象
        path: 输出路径
        layout: "samples_x_genes" (样本行×基因列) 或 "genes_x_samples" (基因行×样本列)
        sep: 分隔符
    """
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

    if layout == "genes_x_samples":
        df = pd.DataFrame(
            X.T,
            index=adata.var_names,
            columns=adata.obs_names,
        )
        df.index.name = "gene_id"
    else:
        df = pd.DataFrame(
            X,
            index=adata.obs_names,
            columns=adata.var_names,
        )

    df.to_csv(path, sep=sep)
    print(f"  ✓ {path} ({df.shape[0]}×{df.shape[1]}, layout={layout})")


def write_biom_json(
    adata: "AnnData",
    path: Path,
    taxonomy: Optional[list[list[str]]] = None,
) -> None:
    """将 AnnData 导出为 BIOM 1.0 JSON 格式

    BIOM JSON 结构:
        {
            "id": ...,
            "format": "Biological Observation Matrix 1.0.0",
            "type": "OTU table",
            "matrix_type": "sparse",
            "shape": [n_features, n_samples],
            "rows": [{"id": ..., "metadata": {...}}, ...],
            "columns": [{"id": ..., "metadata": {...}}, ...],
            "data": [[feature_idx, sample_idx, value], ...]
        }
    """
    from scipy.sparse import issparse

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    n_features, n_samples = X.shape[1], X.shape[0]  # BIOM: feature × sample

    # 构建行（特征/OTU）元数据
    rows = []
    for i in range(n_features):
        row_entry = {"id": str(adata.var_names[i]), "metadata": {}}
        if taxonomy and i < len(taxonomy):
            row_entry["metadata"]["taxonomy"] = taxonomy[i % len(taxonomy)]
        rows.append(row_entry)

    # 构建列（样本）元数据
    columns = []
    for j in range(n_samples):
        col_entry = {"id": str(adata.obs_names[j]), "metadata": {}}
        if "batch" in adata.obs.columns:
            col_entry["metadata"]["batch"] = str(adata.obs.iloc[j]["batch"])
        columns.append(col_entry)

    # 构建稀疏三元组数据
    data_triplets = []
    for j in range(n_samples):
        for i in range(n_features):
            val = X[j, i]
            if val > 0:
                data_triplets.append([i, j, float(val)])

    if not data_triplets:
        data_triplets = [[0, 0, 0.0]]

    biom = {
        "id": path.stem,
        "format": "Biological Observation Matrix 1.0.0",
        "format_url": "http://biom-format.org",
        "type": "OTU table",
        "generated_by": "omics_standardization demo generator",
        "date": pd.Timestamp.now().isoformat(),
        "matrix_type": "sparse",
        "matrix_element_type": "float",
        "shape": [n_features, n_samples],
        "rows": rows,
        "columns": columns,
        "data": data_triplets,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(biom, f, indent=2, ensure_ascii=False)

    print(f"  ✓ {path} ({n_features} features × {n_samples} samples, {len(data_triplets)} non-zero)")


def write_fcs(
    adata: "AnnData",
    path: Path,
) -> None:
    """将 AnnData 导出为最小合法 FCS 3.0 文件

    FCS 3.0 文件结构:
        - HEADER (256 bytes): 版本 + 各段偏移量
        - TEXT 段: 键值对描述参数和数据
        - DATA 段: 二进制 list-mode 数据 (每个事件一行)

    生成的文件可被 fcsparser.parse() 正确读取。
    """
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    n_events, n_params = X.shape

    # --- 构建 TEXT 段 ---
    # FCS 要求的分隔符（使用不常见的字符）
    delimiter = "/"
    text_entries = []

    # 必填关键字
    text_entries.append("$PAR" + delimiter + f"{n_params}" + delimiter)
    text_entries.append("$MODE" + delimiter + "L" + delimiter)  # List mode
    text_entries.append("$TOT" + delimiter + f"{n_events}" + delimiter)

    # 参数描述（$P1N=Name, $P1R=Range, $P1B=Bits, $P1E=Amp）
    for p in range(1, n_params + 1):
        param_name = str(adata.var_names[p - 1])
        # 截断过长的参数名（FCS 限制）
        if len(param_name) > 64:
            param_name = param_name[:60] + "..."
        text_entries.append(f"$P{p}N" + delimiter + param_name + delimiter)
        text_entries.append(f"$P{p}R" + delimiter + "262144" + delimiter)  # 18-bit range as max
        text_entries.append(f"$P{p}B" + delimiter + "32" + delimiter)
        text_entries.append(f"$P{p}E" + delimiter + "0,0" + delimiter)

    # 数据类型
    text_entries.append("$DATATYPE" + delimiter + "F" + delimiter)

    # BYTEORD (3.1+)
    text_entries.append("$BYTEORD" + delimiter + "1,2,3,4" + delimiter)

    # 段边界标记
    text_entries.append("$BEGINANALYSIS" + delimiter + "0" + delimiter)
    text_entries.append("$ENDANALYSIS" + delimiter + "0" + delimiter)
    text_entries.append("$NEXTDATA" + delimiter + "0" + delimiter)

    # 构建 TEXT 字符串（FCS 标准要求以分隔符开头）
    text_body = delimiter + "".join(text_entries)

    # --- 计算偏移量 ---
    HEADER_SIZE = 256
    DELIMITER_BYTE = ord(delimiter)

    # TEXT 紧接 HEADER
    text_start = HEADER_SIZE
    text_end = text_start + len(text_body)

    # DATA 紧接 TEXT（4 字节对齐）
    data_start = text_end
    if data_start % 4 != 0:
        data_start += 4 - (data_start % 4)

    # DATA: 每个参数 4 字节 (float32)，list-mode 每行 n_params × 4 字节
    data_end = data_start + n_events * n_params * 4

    # --- 构建 HEADER ---
    # FCS3.0 格式: 前 10 字节 = "FCS3.0   "
    # 接下来 8 字节 ASCII 偏移量: text_start, text_end, data_start, data_end
    # (每个 8 字符，空格补齐)
    header = bytearray(HEADER_SIZE)
    header[0:10] = b"FCS3.0    "

    for i, val in enumerate([text_start, text_end, data_start, data_end]):
        # 右对齐 8 字符
        offset = 10 + i * 8
        header[offset:offset + 8] = f"{val:>8d}".encode("ascii")

    # 剩余填充空格
    for i in range(10 + 4 * 8, HEADER_SIZE):
        header[i] = 32  # space

    # --- 写入文件 ---
    data_chunk = X.astype(np.float32).tobytes()

    with open(path, "wb") as f:
        f.write(bytes(header))
        f.write(text_body.encode("ascii"))
        # 对齐填充
        if data_start > text_end:
            f.write(b"\x00" * (data_start - text_end))
        f.write(data_chunk)

    expected_size = data_start + len(data_chunk)
    actual_size = path.stat().st_size
    print(f"  ✓ {path} ({n_events} events × {n_params} params, {actual_size:,} bytes)")


def write_mzml(
    adata: "AnnData",
    path: Path,
) -> None:
    """将 AnnData 导出为最小合法 mzML XML 文件 (PSI-MS CV)

    生成的文件可被 pymzml.run.Reader() 读取。

    mzML 结构:
        <mzML>
          <run>
            <spectrumList>
              <spectrum index="0" id="scan=1">
                <binaryDataArrayList>
                  <binaryDataArray encodedLength="..." arrayLength="...">
                    <cvParam ... name="m/z array"/>
                    <binary>...</binary>
                  </binaryDataArray>
                  <binaryDataArray encodedLength="..." arrayLength="...">
                    <cvParam ... name="intensity array"/>
                    <binary>...</binary>
                  </binaryDataArray>
                </binaryDataArrayList>
              </spectrum>
              ...
            </spectrumList>
          </run>
        </mzML>
    """
    import base64
    import struct

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    n_spectra, n_peaks = X.shape

    # 每个 spectrum = 一个样本行 → m/z + intensity 对
    # m/z 值从 var_names 推断或使用均匀分布
    mz_values = np.linspace(50.0, 1000.0, n_peaks, dtype=np.float64)

    spectra_xml = []
    for i in range(n_spectra):
        # 只保留非零峰值
        intensities = X[i].astype(np.float64)
        nonzero_mask = intensities > 0
        if not nonzero_mask.any():
            nonzero_mask[:5] = True  # 至少保留一些峰

        mz_arr = mz_values[nonzero_mask]
        int_arr = intensities[nonzero_mask]

        # Base64 编码
        mz_b64 = base64.b64encode(mz_arr.tobytes()).decode("ascii")
        int_b64 = base64.b64encode(int_arr.tobytes()).decode("ascii")

        rt_minutes = i * 0.5 + np.random.default_rng(i).uniform(0, 0.3)

        spectrum_xml = f"""\
        <spectrum index="{i}" id="scan={i + 1}" defaultArrayLength="{len(mz_arr)}">
          <cvParam cvRef="MS" accession="MS:1000579" name="MS1 spectrum"/>
          <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>
          <cvParam cvRef="MS" accession="MS:1000285" name="total ion current" value="{float(int_arr.sum()):.4f}"/>
          <scanList count="1">
            <scan>
              <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{rt_minutes:.4f}" unitCvRef="UO" unitAccession="UO:0000031" unitName="minute"/>
            </scan>
          </scanList>
          <binaryDataArrayList count="2">
            <binaryDataArray encodedLength="{len(mz_b64)}" dataProcessingRef="mzArray">
              <cvParam cvRef="MS" accession="MS:1000514" name="m/z array"/>
              <cvParam cvRef="MS" accession="MS:1000576" name="no compression"/>
              <cvParam cvRef="MS" accession="MS:1000521" name="32-bit float"/>
              <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value="false"/>
              <binary>{mz_b64}</binary>
            </binaryDataArray>
            <binaryDataArray encodedLength="{len(int_b64)}" dataProcessingRef="intensityArray">
              <cvParam cvRef="MS" accession="MS:1000515" name="intensity array"/>
              <cvParam cvRef="MS" accession="MS:1000576" name="no compression"/>
              <cvParam cvRef="MS" accession="MS:1000521" name="32-bit float"/>
              <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value="false"/>
              <binary>{int_b64}</binary>
            </binaryDataArray>
          </binaryDataArrayList>
        </spectrum>"""
        spectra_xml.append(spectrum_xml)

    mzml_content = f"""<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://psi.hupo.org/ms/mzml http://psidev.info/files/ms/mzML/xsd/mzML1.1.0.xsd"
       id="omics_demo"
       version="1.1.0">
  <cvList count="2">
    <cv id="MS" fullName="Proteomics Standards Initiative Mass Spectrometry Ontology"
        version="4.1.0" URI="http://psidev.info/ms"/>
    <cv id="UO" fullName="Unit Ontology" version="1.0"
        URI="http://purl.obolibrary.org/obo/UO_"/>
  </cvList>
  <fileDescription>
    <fileContent>
      <cvParam cvRef="MS" accession="MS:1000579" name="MS1 spectrum"/>
    </fileContent>
    <sourceFileList count="1">
      <sourceFile id="SF1" name="demo_metabolomics.mzML">
        <cvParam cvRef="MS" accession="MS:1000569" name="SHA-1" value="0000000000000000000000000000000000000000"/>
      </sourceFile>
    </sourceFileList>
  </fileDescription>
  <softwareList count="1">
    <software id="omics-std" version="1.0.0">
      <cvParam cvRef="MS" accession="MS:1000799" name="custom unreleased software tool" value="omics_standardization"/>
    </software>
  </softwareList>
  <run id="demo_run" defaultInstrumentConfigurationRef="IC1" sampleRef="sample1" startTimeStamp="2024-01-01T00:00:00Z">
    <defaultSourceFileRef ref="SF1"/>
    <spectrumList count="{n_spectra}" defaultDataProcessingRef="centroid">
{"".join(spectra_xml)}
    </spectrumList>
  </run>
</mzML>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(mzml_content)

    file_size = path.stat().st_size
    print(f"  ✓ {path} ({n_spectra} spectra × max {n_peaks} peaks, {file_size:,} bytes)")


# ==============================================================================
# 主生成逻辑
# ==============================================================================

def generate_all(output_root: Optional[Path] = None) -> None:
    """生成全部 6 种模态的 demo 数据"""
    if output_root is None:
        output_root = DATA_RAW

    print("=" * 60)
    print("omics_standardization — Demo 数据生成器")
    print("=" * 60)
    print(f"输出目录: {output_root}")
    print()

    all_adata = {}

    # ---- scRNA-seq ----
    print("[1/6] scRNA-seq (H5AD + CSV)")
    cfg = MODALITY_CONFIG["scrna"]
    adata = _make_adata(**{k: v for k, v in cfg.items() if k != "label"}, modality_name="scrna")
    out_dir = output_root / "scrna"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_h5ad(adata, out_dir / "scrna_expression.h5ad")
    write_csv(adata, out_dir / "scrna_expression.csv")
    all_adata["scrna"] = adata

    # ---- Bulk RNA-seq ----
    print("\n[2/6] Bulk RNA-seq (CSV + TSV)")
    cfg = MODALITY_CONFIG["bulk_rna"]
    adata = _make_adata(**{k: v for k, v in cfg.items() if k != "label"}, modality_name="bulk_rna")
    out_dir = output_root / "bulk_rna"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(adata, out_dir / "bulk_rna_counts.csv")
    write_csv(adata, out_dir / "bulk_rna_counts.tsv", sep="\t")
    all_adata["bulk_rna"] = adata

    # ---- Proteomics (FCS) ----
    print("\n[3/6] Proteomics (FCS + CSV)")
    cfg = MODALITY_CONFIG["proteomics"]
    adata = _make_adata(**{k: v for k, v in cfg.items() if k != "label"}, modality_name="proteomics")
    out_dir = output_root / "proteomics"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_fcs(adata, out_dir / "proteomics_sample.fcs")
    write_csv(adata, out_dir / "proteomics_sample.csv", layout="samples_x_genes")
    all_adata["proteomics"] = adata

    # ---- Metabolomics (mzML) ----
    print("\n[4/6] Metabolomics (mzML + CSV)")
    cfg = MODALITY_CONFIG["metabolomics"]
    adata = _make_adata(**{k: v for k, v in cfg.items() if k != "label"}, modality_name="metabolomics")
    out_dir = output_root / "metabolomics"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_mzml(adata, out_dir / "metabolomics_run.mzML")
    write_csv(adata, out_dir / "metabolomics_intensities.csv", layout="samples_x_genes")
    all_adata["metabolomics"] = adata

    # ---- ATAC-seq ----
    print("\n[5/6] ATAC-seq (H5AD + CSV)")
    cfg = MODALITY_CONFIG["atac"]
    adata = _make_adata(**{k: v for k, v in cfg.items() if k != "label"}, modality_name="atac")
    out_dir = output_root / "atac"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_h5ad(adata, out_dir / "atac_peaks.h5ad")
    # ATAC CSV 使用 samples_x_genes 布局（大宽表）
    write_csv(adata, out_dir / "atac_peaks.csv", layout="samples_x_genes")
    all_adata["atac"] = adata

    # ---- Microbiome (BIOM) ----
    print("\n[6/6] Microbiome (BIOM JSON)")
    cfg = MODALITY_CONFIG["microbiome"]
    adata = _make_adata(**{k: v for k, v in cfg.items() if k != "label"}, modality_name="microbiome")
    out_dir = output_root / "microbiome"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_biom_json(adata, out_dir / "otu_table.biom", taxonomy=TAXONOMY)
    all_adata["microbiome"] = adata

    # ---- Summary ----
    print()
    print("=" * 60)
    print("生成完成! 目录结构:")
    print("=" * 60)
    _print_tree(output_root)
    print()
    print("下一步: python scripts/run_demo_pipeline.py")


def generate_modality(modality: str, output_root: Optional[Path] = None) -> None:
    """生成指定模态的 demo 数据"""
    if output_root is None:
        output_root = DATA_RAW

    if modality not in MODALITY_CONFIG:
        print(f"未知模态: {modality}")
        print(f"可用模态: {list(MODALITY_CONFIG.keys())}")
        sys.exit(1)

    cfg = MODALITY_CONFIG[modality]
    adata = _make_adata(**{k: v for k, v in cfg.items() if k != "label"}, modality_name=modality)
    out_dir = output_root / modality
    out_dir.mkdir(parents=True, exist_ok=True)

    if modality == "scrna":
        write_h5ad(adata, out_dir / "scrna_expression.h5ad")
        write_csv(adata, out_dir / "scrna_expression.csv")
    elif modality == "bulk_rna":
        write_csv(adata, out_dir / "bulk_rna_counts.csv")
        write_csv(adata, out_dir / "bulk_rna_counts.tsv", sep="\t")
    elif modality == "proteomics":
        write_fcs(adata, out_dir / "proteomics_sample.fcs")
        write_csv(adata, out_dir / "proteomics_sample.csv")
    elif modality == "metabolomics":
        write_mzml(adata, out_dir / "metabolomics_run.mzML")
        write_csv(adata, out_dir / "metabolomics_intensities.csv")
    elif modality == "atac":
        write_h5ad(adata, out_dir / "atac_peaks.h5ad")
        write_csv(adata, out_dir / "atac_peaks.csv")
    elif modality == "microbiome":
        write_biom_json(adata, out_dir / "otu_table.biom", taxonomy=TAXONOMY)


def _print_tree(root: Path, prefix: str = "") -> None:
    """打印目录树"""
    items = sorted(root.iterdir())
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "
        if item.is_dir():
            print(f"{prefix}{connector}{item.name}/")
            _print_tree(item, prefix + ("    " if is_last else "│   "))
        else:
            size_kb = item.stat().st_size / 1024
            if size_kb >= 1024:
                size_str = f"{size_kb / 1024:.1f} MB"
            else:
                size_str = f"{size_kb:.1f} KB"
            print(f"{prefix}{connector}{item.name} ({size_str})")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="生成 omics_standardization demo 数据文件",
    )
    parser.add_argument(
        "--modality",
        "-m",
        type=str,
        default=None,
        choices=list(MODALITY_CONFIG.keys()),
        help="仅生成指定模态的数据",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出根目录 (默认: data/raw/)",
    )
    args = parser.parse_args()

    output_root = Path(args.output) if args.output else None

    if args.modality:
        generate_modality(args.modality, output_root)
    else:
        generate_all(output_root)


if __name__ == "__main__":
    main()
