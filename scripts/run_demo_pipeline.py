"""
Demo 数据全流程验证脚本（独立模式）

读取 generate_demo_data.py 生成的合成数据文件，直接解析并验证。
不依赖项目包导入，可独立运行。

用法:
    python scripts/run_demo_pipeline.py              # 跑全部
    python scripts/run_demo_pipeline.py --quick      # 快速模式（仅解析 + 模态检测）
    python scripts/run_demo_pipeline.py --file data/raw/scrna/scrna_expression.h5ad  # 单文件
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---- 路径配置 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

# ---- ASCII 安全输出 ----
def green(s: str) -> str:   return f"[OK] {s}"
def red(s: str) -> str:     return f"[FAIL] {s}"
def yellow(s: str) -> str:  return f"[WARN] {s}"
def bold(s: str) -> str:    return f"=== {s} ==="


# ==============================================================================
# 文件解析器（独立实现）
# ==============================================================================

def parse_h5ad(path: Path):
    """解析 .h5ad 文件"""
    import anndata
    return anndata.read_h5ad(path)

def parse_csv(path: Path):
    """解析 CSV/TSV 文件（独立实现，与项目 _csv.py 保持一致）

    自动检测布局:
        - genes_x_samples: 基因行 × 样本列 → 转置为 AnnData (obs=样本, var=基因)
        - samples_x_genes: 样本行 × 基因列 → 直接用于 AnnData
    """
    # 检测分隔符
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    sep = "\t" if first_line.count("\t") > first_line.count(",") else ","

    df = pd.read_csv(path, sep=sep, index_col=0, low_memory=False)

    # 判断布局
    first_col_name = str(df.columns[0]).lower()
    gene_keywords = ["gene", "symbol", "feature", "gene_id", "ensembl", "probe", "transcript"]
    sample_keywords = ["sample", "cell", "barcode", "donor", "patient"]

    if any(kw in first_col_name for kw in gene_keywords):
        layout = "genes_x_samples"
    elif any(kw in first_col_name for kw in sample_keywords):
        layout = "samples_x_genes"
    elif df.shape[0] > df.shape[1] * 2:
        layout = "genes_x_samples"
    else:
        layout = "samples_x_genes"

    # genes_x_samples → 需要转置为 samples_x_genes (AnnData 预期: obs=样本, var=基因)
    if layout == "genes_x_samples":
        df = df.T

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    import anndata
    from scipy.sparse import csr_matrix
    return anndata.AnnData(
        X=csr_matrix(df.values.astype(np.float32)),
        obs=pd.DataFrame(index=df.index.tolist()),
        var=pd.DataFrame(index=df.columns.tolist()),
    )

def parse_biom(path: Path):
    """解析 BIOM JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("rows", [])
    columns = data.get("columns", [])
    matrix_data = data.get("data", [])

    # 构建稀疏矩阵
    from scipy.sparse import coo_matrix, csr_matrix
    row_idx = [item[0] for item in matrix_data]
    col_idx = [item[1] for item in matrix_data]
    values = [item[2] for item in matrix_data]

    X = coo_matrix((values, (col_idx, row_idx)), shape=(len(columns), len(rows)))
    X = csr_matrix(X)

    import anndata
    return anndata.AnnData(
        X=X,
        obs=pd.DataFrame(index=[c["id"] for c in columns]),
        var=pd.DataFrame(index=[r["id"] for r in rows]),
    )

def parse_fcs(path: Path):
    """解析 .fcs 文件 (minimal FCS 3.0 reader)"""
    with open(path, "rb") as f:
        header = f.read(256)

    # 读取偏移量
    def _read_offset(hdr: bytes, start: int) -> int:
        return int(hdr[start:start+8].strip())

    text_start = _read_offset(header, 10)
    text_end = _read_offset(header, 18)
    data_start = _read_offset(header, 26)

    with open(path, "rb") as f:
        f.seek(text_start)
        text_body = f.read(text_end - text_start).decode("ascii", errors="replace")

    # 解析 TEXT 段提取参数信息
    delimiter = text_body[0]
    parts = text_body.split(delimiter)

    n_params = None
    n_events = None
    param_names = []

    for i, part in enumerate(parts):
        if part.startswith("$PAR"):
            n_params = int(parts[i + 1]) if i + 1 < len(parts) else None
        elif part.startswith("$TOT"):
            n_events = int(parts[i + 1]) if i + 1 < len(parts) else None
        elif part.startswith("$P") and part.endswith("N") and not part.startswith("$PAR"):
            param_names.append(parts[i + 1] if i + 1 < len(parts) else f"P{len(param_names)+1}")

    if n_params is None:
        n_params = len(param_names)
    if n_events is None:
        # 从文件大小推断
        import os
        file_size = os.path.getsize(path)
        n_events = (file_size - data_start) // (n_params * 4) if n_params else 0

    # 读取 DATA 段
    with open(path, "rb") as f:
        f.seek(data_start)
        raw_data = f.read(n_events * n_params * 4)

    X = np.frombuffer(raw_data, dtype=np.float32).reshape(n_events, n_params)

    import anndata
    from scipy.sparse import csr_matrix
    return anndata.AnnData(
        X=csr_matrix(X),
        obs=pd.DataFrame(index=[f"event_{i}" for i in range(n_events)]),
        var=pd.DataFrame(index=param_names if len(param_names) == n_params else [f"P{i}" for i in range(n_params)]),
    )

def parse_mzml(path: Path):
    """解析 mzML 文件 (minimal XML reader)"""
    import base64
    import xml.etree.ElementTree as ET

    # 命名空间
    ns = {"mzml": "http://psi.hupo.org/ms/mzml"}

    tree = ET.parse(path)
    root = tree.getroot()

    spectra_data = []
    for spectrum in root.findall(".//mzml:spectrum", ns):
        spec_id = spectrum.get("id", "unknown")
        rt = 0.0
        for cv_param in spectrum.findall(".//mzml:cvParam[@name='scan start time']", ns):
            rt = float(cv_param.get("value", "0"))

        # 提取 m/z 和 intensity arrays
        intensities = None

        binary_arrays = spectrum.findall(".//mzml:binaryDataArray", ns)
        current_is_mz = False
        for ba in binary_arrays:
            cv_params = ba.findall("mzml:cvParam", ns)
            for cp in cv_params:
                name = cp.get("name", "")
                if "m/z array" in name:
                    current_is_mz = True
                    break
                elif "intensity array" in name:
                    current_is_mz = False
                    break

            binary_elem = ba.find("mzml:binary", ns)
            if binary_elem is not None and binary_elem.text:
                data = base64.b64decode(binary_elem.text)
                arr = np.frombuffer(data, dtype=np.float32)
                if not current_is_mz:
                    intensities = arr

        if intensities is not None:
            spectra_data.append({"id": spec_id, "rt": rt, "intensities": intensities})

    if not spectra_data:
        raise ValueError("mzML 文件中未找到谱图数据")

    # 构建特征矩阵：以所有谱图中最长 intensity 数组为准
    max_len = max(len(s["intensities"]) for s in spectra_data)
    X = np.zeros((len(spectra_data), max_len), dtype=np.float32)
    for i, spec in enumerate(spectra_data):
        n = len(spec["intensities"])
        X[i, :n] = spec["intensities"]

    import anndata
    from scipy.sparse import csr_matrix
    return anndata.AnnData(
        X=csr_matrix(X),
        obs=pd.DataFrame([{"id": s["id"], "rt": s["rt"]} for s in spectra_data]).set_index("id"),
        var=pd.DataFrame(index=[f"peak_{i}" for i in range(max_len)]),
    )


# ==============================================================================
# 模态检测（独立实现，与 src/_selectors/_modality.py 等价）
# ==============================================================================

MODALITY_LABELS = ["scrna", "bulk_rna", "proteomics", "metabolomics", "atac"]

def detect_modality_heuristic(adata) -> str:
    """启发式模态检测（与 _modality.py 的 fallback 路径一致）"""
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    _, n_vars = adata.shape
    missing_rate = float(np.mean(X == 0)) if np.any(X >= 0) else 0.0

    if n_vars > 10000:
        if missing_rate > 0.85:
            modality = "atac"
        else:
            modality = "scrna"
    elif n_vars < 500:
        if missing_rate < 0.5:
            modality = "proteomics"
        else:
            modality = "metabolomics"
    elif n_vars < 5000:
        modality = "bulk_rna"
    else:
        modality = "scrna" if missing_rate > 0.3 else "bulk_rna"

    return modality


# ==============================================================================
# 验证器
# ==============================================================================

def run_verification(data_root: Path, files: list[tuple[str, str]], quick: bool = False):
    """验证所有 demo 数据文件"""
    parsed_count = 0
    skipped_count = 0
    failed_count = 0
    modality_matches = 0
    modality_mismatches = 0

    for rel_path, expected_modality in files:
        file_path = data_root / rel_path
        if not file_path.exists():
            print(f"  {yellow('SKIP')} {rel_path} — 文件不存在")
            skipped_count += 1
            continue

        suffix = file_path.suffix.lower()
        if suffix in (".gz",):
            # 获取真实后缀
            name = file_path.name.lower()
            if name.endswith(".csv.gz"):
                suffix = ".csv"
            elif name.endswith(".tsv.gz"):
                suffix = ".tsv"

        print(f"\n{'─' * 55}")
        print(f"  {bold(file_path.relative_to(PROJECT_ROOT))}")
        print(f"  预期模态: {expected_modality}  |  格式: {suffix}")

        # ---- Parse ----
        t0 = time.perf_counter()
        try:
            if suffix in (".h5ad", ".h5mu", ".loom"):
                adata = parse_h5ad(file_path)
            elif suffix in (".csv", ".tsv", ".txt"):
                adata = parse_csv(file_path)
            elif suffix == ".biom":
                adata = parse_biom(file_path)
            elif suffix == ".fcs":
                adata = parse_fcs(file_path)
            elif suffix in (".mzml",):
                adata = parse_mzml(file_path)
            else:
                print(f"  {yellow('SKIP')} 不支持的格式: {suffix}")
                skipped_count += 1
                continue
        except ImportError as e:
            print(f"  {yellow('SKIP')} 缺少依赖: {e}")
            skipped_count += 1
            continue
        except Exception as e:
            print(f"  {red('PARSE')} {type(e).__name__}: {e}")
            failed_count += 1
            continue

        elapsed = time.perf_counter() - t0
        print(f"  {green(f'Parse {adata.shape} ({elapsed:.2f}s)')}")

        parsed_count += 1

        # ---- Modality Detection ----
        try:
            detected = detect_modality_heuristic(adata)
            if detected == expected_modality:
                print(f"  {green(f'Modality: {detected}')}")
                modality_matches += 1
            else:
                print(f"  {yellow(f'Modality: {detected} (预期 {expected_modality})')}")
                modality_mismatches += 1
        except Exception as e:
            print(f"  {red(f'MODALITY')} {e}")

        if quick:
            print(f"  (快速模式 — 跳过后续步骤)")
            continue

        # ---- Basic statistics ----
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        zero_rate = float(np.mean(X == 0))
        print(f"  零值率: {zero_rate:.3f}  |  批次: {adata.obs['batch'].nunique() if 'batch' in adata.obs else 'N/A'}")

    # ---- Summary ----
    print(f"\n{'=' * 55}")
    print(f"  {bold('验证摘要')}")
    print(f"  解析成功: {parsed_count}  |  跳过: {skipped_count}  |  失败: {failed_count}")
    if modality_matches + modality_mismatches > 0:
        accuracy = modality_matches / (modality_matches + modality_mismatches) * 100
        print(f"  模态匹配: {modality_matches}/{modality_matches + modality_mismatches} ({accuracy:.0f}%)")
    print(f"{'=' * 55}")

    return failed_count == 0


# ==============================================================================
# CLI
# ==============================================================================

# 文件 → 预期模态映射（与 generate_demo_data.py 保持一致）
FILE_MODALITY_MAP: list[tuple[str, str]] = [
    ("scrna/scrna_expression.h5ad", "scrna"),
    ("scrna/scrna_expression.csv", "scrna"),
    ("bulk_rna/bulk_rna_counts.csv", "bulk_rna"),
    ("bulk_rna/bulk_rna_counts.tsv", "bulk_rna"),
    ("proteomics/proteomics_sample.fcs", "proteomics"),
    ("proteomics/proteomics_sample.csv", "proteomics"),
    ("metabolomics/metabolomics_run.mzML", "metabolomics"),
    ("metabolomics/metabolomics_intensities.csv", "metabolomics"),
    ("atac/atac_peaks.h5ad", "atac"),
    ("atac/atac_peaks.csv", "atac"),
    ("microbiome/otu_table.biom", "metabolomics"),
]


def main():
    parser = argparse.ArgumentParser(
        description="验证 omics_standardization pipeline 在 demo 数据上的运行 (独立模式)",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="快速模式：仅解析 + 模态检测，跳过统计输出",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="仅验证指定文件",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_RAW),
        help="demo 数据根目录",
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    if not data_root.exists():
        print(f"{red('数据目录不存在')}: {data_root}")
        print(f"  请先运行: python scripts/generate_demo_data.py")
        sys.exit(1)

    if args.file:
        # 单文件模式
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = data_root / file_path
        if not file_path.exists():
            print(f"{red('文件不存在')}: {file_path}")
            sys.exit(1)
        # 猜测模态
        modality = "scrna"
        for rel, mod in FILE_MODALITY_MAP:
            if rel in str(file_path).replace("\\", "/"):
                modality = mod
                break
        files = [(str(file_path.relative_to(data_root)), modality)]
    else:
        files = FILE_MODALITY_MAP
        print(f"{'=' * 55}")
        print(f"  omics_standardization — Demo Pipeline 验证 (独立模式)")
        print(f"  数据目录: {data_root}")
        if args.quick:
            print(f"  模式: 快速 (仅解析 + 模态检测)")

    ok = run_verification(data_root, files, quick=args.quick)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
