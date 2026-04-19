#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hy_models.download import (  # noqa: E402
    DEFAULT_MLX_8BIT_DIR,
    DEFAULT_ORIGINAL_DIR,
    DEFAULT_REPORT_PATH,
    download_and_compare,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download HY-MT original and MLX 8bit snapshots, then generate a static "
            "comparison report."
        )
    )
    parser.add_argument(
        "--download-original",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the original tencent/HY-MT1.5-1.8B snapshot.",
    )
    parser.add_argument(
        "--download-mlx-8bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the mlx-community/HY-MT1.5-1.8B-8bit snapshot.",
    )
    parser.add_argument(
        "--original-dir",
        default=str(DEFAULT_ORIGINAL_DIR),
        help="Output directory for the original model snapshot.",
    )
    parser.add_argument(
        "--mlx-dir",
        default=str(DEFAULT_MLX_8BIT_DIR),
        help="Output directory for the MLX 8bit snapshot.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force snapshot_download for selected models even when already complete.",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="Path to write the JSON comparison report.",
    )
    return parser


def _print_summary(report: dict) -> None:
    original = report["original"]
    mlx = report["mlx_8bit"]
    comparison = report["comparison"]
    print(
        json.dumps(
            {
                "original_repo_id": original["repo_id"],
                "original_dir": original["directory"],
                "original_size_bytes": original["summary"]["total_size_bytes"],
                "mlx_8bit_repo_id": mlx["repo_id"],
                "mlx_8bit_dir": mlx["directory"],
                "mlx_8bit_size_bytes": mlx["summary"]["total_size_bytes"],
                "file_diff_summary": comparison["file_diff_summary"],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    report = download_and_compare(
        download_original=args.download_original,
        download_mlx_8bit=args.download_mlx_8bit,
        original_dir=Path(args.original_dir),
        mlx_dir=Path(args.mlx_dir),
        force=args.force,
        report_path=Path(args.report_path),
    )
    _print_summary(report)
    print(f"\nReport written to: {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
