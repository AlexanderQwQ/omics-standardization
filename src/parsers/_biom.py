"""BIOM（Biological Observation Matrix）解析器

支持 BIOM 格式:
    - BIOM 1.0 (JSON): 文本格式，可直接解析
    - BIOM 2.0 (HDF5): 二进制格式，需 h5py 或 biom-format 库

参考: McDonald et al. (2012), GigaScience
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import coo_matrix, csr_matrix

from ._base import BaseParser
from .. import logging as logg

if TYPE_CHECKING:
    pass


class BIOMParser(BaseParser):
    """BIOM 文件解析器

    支持 BIOM 1.0 (JSON) 和 BIOM 2.0 (HDF5)。
    将观测矩阵（OTU/ASV table）转换为 AnnData:
        - .X: 样本 × 特征（观测计数矩阵）
        - .obs: 样本元数据
        - .var: 特征分类信息（界门纲目科属种）

    用法:
        parser = BIOMParser("data/otu_table.biom")
        adata = parser.parse()
    """

    SUPPORTED_SUFFIXES = {".biom", ".biom.gz", ".json"}

    def _parse(self) -> AnnData:
        """解析 BIOM 文件

        Returns:
            AnnData (obs=samples, var=features/OTUs)
        """
        suffix = self.file_path.suffix.lower()

        # 检测 BIOM 格式版本
        if self._is_hdf5():
            return self._parse_hdf5()
        else:
            return self._parse_json()

    def _is_hdf5(self) -> bool:
        """检测文件是否为 HDF5 格式（BIOM 2.0）"""
        try:
            import h5py
            with h5py.File(str(self.file_path), "r") as f:
                return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # BIOM 1.0 (JSON)
    # ------------------------------------------------------------------

    def _parse_json(self) -> AnnData:
        """解析 BIOM 1.0 JSON 格式"""
        try:
            from biom import load_table
            table = load_table(str(self.file_path))
            return self._table_to_anndata(table)
        except ImportError:
            logg.warning("biom-format 库未安装，使用手动 JSON 解析")
            return self._parse_json_manual()

    def _parse_json_manual(self) -> AnnData:
        """手动解析 BIOM JSON（不依赖 biom-format 库）"""
        import gzip

        open_fn = gzip.open if self.file_path.name.endswith(".gz") else open

        with open_fn(str(self.file_path), "rt", encoding="utf-8") as f:
            data = json.load(f)

        rows = data.get("rows", [])
        columns = data.get("columns", [])
        matrix_data = data.get("data", [])

        # 特征名和分类信息
        feature_ids = [r["id"] for r in rows]
        feature_metadata = {}
        for r in rows:
            meta = r.get("metadata", {})
            if meta:
                feature_metadata[r["id"]] = meta

        # 样本名和元数据
        sample_ids = [c["id"] for c in columns]
        sample_metadata = {}
        for c in columns:
            meta = c.get("metadata", {})
            if meta:
                sample_metadata[c["id"]] = meta

        # 构建稀疏矩阵
        matrix_type = data.get("matrix_type", "dense")
        matrix_shape = data.get("shape", [len(rows), len(columns)])

        if matrix_type == "sparse":
            # 稀疏格式：[feature_idx, sample_idx, value] triplets
            all_values = matrix_data
            row_indices = [item[0] for item in all_values]
            col_indices = [item[1] for item in all_values]
            values = [item[2] for item in all_values]
            X = coo_matrix((values, (col_indices, row_indices)), shape=(len(sample_ids), len(feature_ids)))
            X = csr_matrix(X)
        else:
            # 密集格式
            X = np.array(matrix_data, dtype=np.float32).T  # 转置：原始是 feature × sample

        # 构建 AnnData
        adata = AnnData(
            X=X,
            obs=pd.DataFrame(
                sample_metadata if sample_metadata else None,
                index=sample_ids if sample_metadata else sample_ids,
            ),
            var=pd.DataFrame(
                feature_metadata if feature_metadata else None,
                index=feature_ids if feature_metadata else feature_ids,
            ),
        )

        # 整理分类层级 (taxonomy)
        taxonomy = data.get("rows", [{}])
        tax_cols: dict[str, list[str]] = {}
        for i, row in enumerate(taxonomy):
            tax = row.get("metadata", {}).get("taxonomy", [])
            if tax:
                for level_idx, name in enumerate(tax):
                    col = f"tax_{level_idx}"
                    tax_cols.setdefault(col, [""] * len(feature_ids))[i] = name

        for col, values in tax_cols.items():
            adata.var[col] = values

        adata.uns["biom_metadata"] = {
            "format": data.get("format", "Biological Observation Matrix 1.0.0"),
            "generated_by": data.get("generated_by", "unknown"),
            "matrix_type": matrix_type,
            "format_url": data.get("format_url", ""),
            "type": data.get("type", "OTU table"),
        }

        logg.info(f"BIOM 手动解析完成: {adata.n_obs} 样本, {adata.n_vars} 特征")
        return adata

    # ------------------------------------------------------------------
    # BIOM 2.0 (HDF5)
    # ------------------------------------------------------------------

    def _parse_hdf5(self) -> AnnData:
        """解析 BIOM 2.0 HDF5 格式"""
        try:
            from biom import load_table
            table = load_table(str(self.file_path))
            return self._table_to_anndata(table)
        except ImportError:
            raise ImportError(
                "解析 BIOM 2.0 (HDF5) 需要 biom-format 库。请运行: pip install biom-format"
            )

    # ------------------------------------------------------------------
    # 通用转换
    # ------------------------------------------------------------------

    def _table_to_anndata(self, table) -> AnnData:
        """将 biom.Table 对象转换为 AnnData"""
        # 获取稠密或稀疏矩阵
        X_dense = table.matrix_data.toarray() if hasattr(table.matrix_data, "toarray") else np.array(table.matrix_data.todense())
        X = X_dense.T  # 转置为 sample × feature

        # 样本元数据
        sample_ids = table.ids(axis="sample")
        sample_meta = {}
        for sid in sample_ids:
            try:
                meta = table.metadata(id=sid, axis="sample")
                if meta:
                    sample_meta[sid] = meta
            except Exception:
                pass

        # 特征元数据
        feature_ids = table.ids(axis="observation")
        feature_meta = {}
        for fid in feature_ids:
            try:
                meta = table.metadata(id=fid, axis="observation")
                if meta:
                    feature_meta[fid] = meta
            except Exception:
                pass

        adata = AnnData(
            X=csr_matrix(X.astype(np.float32)),
            obs=pd.DataFrame(
                sample_meta if sample_meta else None,
                index=sample_ids if sample_meta else sample_ids,
            ),
            var=pd.DataFrame(
                feature_meta if feature_meta else None,
                index=feature_ids if feature_meta else feature_ids,
            ),
        )

        logg.info(f"BIOM 解析完成: {adata.n_obs} 样本, {adata.n_vars} OTU/特征")
        return adata
