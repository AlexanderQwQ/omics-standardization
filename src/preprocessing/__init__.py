"""
预处理命名空间（pp）

统一暴露插补、归一化、批次校正的 API，
与 scanpy 的 pp.* 约定保持一致。

用法:
    from omics_standardization import pp

    pp.impute(adata)          # 自动选择插补方法
    pp.normalize(adata)       # 自动选择归一化方法
    pp.batch_correct(adata)   # 自动选择批次校正方法
    pp.standardize(adata)     # 运行完整标准化流水线
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anndata import AnnData


def impute(adata: AnnData, method: str | None = None, **kwargs: Any) -> AnnData:
    """缺失值插补

    Args:
        adata: 输入 AnnData
        method: 指定方法（None 则自动选择）
        **kwargs: 传递给具体插补方法的参数

    Returns:
        插补后的 AnnData
    """
    from imputers._selector import ImputationSelector
    selector = ImputationSelector()
    return selector.run(adata, method=method, **kwargs)


def normalize(adata: AnnData, method: str | None = None, **kwargs: Any) -> AnnData:
    """尺度归一化

    Args:
        adata: 输入 AnnData
        method: 指定方法（None 则自动选择）
        **kwargs: 传递给具体归一化方法的参数

    Returns:
        归一化后的 AnnData
    """
    # 自动选择逻辑
    if method is None:
        from _selectors._strategy import recommend_strategy
        strategy = recommend_strategy(adata)
        method = strategy["normalization"]

    method_map = {
        "tmm": "TMMNormalizer",
        "deseq2": "DESeq2Normalizer",
        "scran": "ScranNormalizer",
        "quantile": "QuantileNormalizer",
        "vsn": "VSNNormalizer",
    }

    if method not in method_map:
        raise ValueError(f"未知的归一化方法: {method}，可选: {list(method_map.keys())}")

    import normalizers

    normalizer_cls = getattr(normalizers, method_map[method])
    return normalizer_cls(**kwargs).run(adata)


def batch_correct(adata: AnnData, method: str | None = None, batch_key: str = "batch", **kwargs: Any) -> AnnData:
    """批次效应校正

    Args:
        adata: 输入 AnnData
        method: 指定方法（None 则自动选择）
        batch_key: obs 中的批次标签列名
        **kwargs: 传递给具体校正方法的参数

    Returns:
        校正后的 AnnData
    """
    from batch_correctors._selector import BatchCorrectionSelector
    selector = BatchCorrectionSelector()
    return selector.run(adata, method=method, batch_key=batch_key, **kwargs)


def standardize(adata: AnnData, batch_key: str = "batch", **kwargs: Any) -> AnnData:
    """运行完整标准化流水线（impute → normalize → batch_correct）

    一站式的预处理入口，自动选择最优方法并串联执行。

    Args:
        adata: 输入 AnnData
        batch_key: obs 中的批次标签列名
        **kwargs: 传递给各步骤的额外参数

    Returns:
        标准化后的 AnnData
    """
    import _logging as logg

    logg.info("=" * 50)
    logg.info("开始标准化处理: impute → normalize → batch_correct")
    logg.info("=" * 50)

    # Step 1: 插补
    adata = impute(adata, **kwargs)

    # Step 2: 归一化
    adata = normalize(adata, **kwargs)

    # Step 3: 批次校正
    adata = batch_correct(adata, batch_key=batch_key, **kwargs)

    logg.info("标准化处理完成")
    return adata


__all__ = [
    "impute",
    "normalize",
    "batch_correct",
    "standardize",
]
