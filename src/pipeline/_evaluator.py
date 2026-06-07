"""处理效果评估与报告生成"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .. import logging as logg

if TYPE_CHECKING:
    from anndata import AnnData


def generate_report(
    adata_before: AnnData | None = None,
    adata_after: AnnData | None = None,
    output_dir: str | Path = "data/processed/reports",
) -> Path:
    """生成标准化效果 HTML 报告

    Args:
        adata_before: 处理前数据
        adata_after: 处理后数据
        output_dir: 报告输出目录

    Returns:
        生成的 HTML 报告路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "standardization_report.html"

    # 简单 HTML 报告模板
    html = "<html><head><meta charset='utf-8'><title>标准化报告</title></head><body>\n"
    html += "<h1>多模态组学数据标准化处理报告</h1>\n"

    if adata_after is not None:
        html += "<h2>处理概览</h2>\n"
        html += f"<p>样本数: {adata_after.n_obs}</p>\n"
        html += f"<p>特征数: {adata_after.n_vars}</p>\n"

        if "standardization" in adata_after.uns:
            info = adata_after.uns["standardization"]
            html += "<h3>处理参数</h3>\n<ul>\n"
            for step, params in info.items():
                html += f"<li><b>{step}</b>: {params}</li>\n"
            html += "</ul>\n"

    html += "</body></html>"

    report_path.write_text(html, encoding="utf-8")
    logg.info(f"报告已生成: {report_path}")

    return report_path
