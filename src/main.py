"""
CLI 入口：标准化流水线命令行调用

用法:
    python -m src config/default.yaml    # 按配置文件运行
    omics-std config/default.yaml        # 通过 console_scripts 入口运行
"""

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """命令行主入口"""
    parser = argparse.ArgumentParser(
        prog="omics-std",
        description="多模态空间环境免疫组学数据标准化处理",
    )
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default="config/default.yaml",
        help="流水线配置文件路径 (默认: config/default.yaml)",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="输入数据目录",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/processed"),
        help="输出目录 (默认: data/processed)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    args = parser.parse_args(argv)

    # 加载配置
    from ._settings import Verbosity, settings

    settings.load_config(args.config)

    if args.verbose:
        settings.verbosity = Verbosity.debug

    # 运行流水线
    from .pipeline import StandardizationPipeline

    pipeline = StandardizationPipeline(config=args.config)
    pipeline.run(input_path=args.input, output_path=args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
