from __future__ import annotations

from pathlib import Path

from shared.config import DEFAULT_CONFIG_PATH, load_config

from .benchmark import run_benchmark
from .catalog import generate_translation_catalog
from .package import package_translation_artifacts
from .prepare import prepare_translation
from .report import generate_report


def register_translation_group(subparsers) -> None:
    translation_parser = subparsers.add_parser("translation", help="Translation pipeline commands")
    translation_subparsers = translation_parser.add_subparsers(dest="translation_command", required=True)

    prepare_parser = translation_subparsers.add_parser("prepare", help="Download, export, and quantize translation artifacts")
    _add_config_arg(prepare_parser)
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.set_defaults(func=_run_prepare)

    benchmark_parser = translation_subparsers.add_parser("benchmark", help="Run translation benchmark inference")
    _add_config_arg(benchmark_parser)
    benchmark_parser.add_argument("--timestamp", default=None)
    benchmark_parser.set_defaults(func=_run_benchmark)

    report_parser = translation_subparsers.add_parser("report", help="Compute translation metrics and write reports")
    _add_config_arg(report_parser)
    report_parser.add_argument("--result-dir", type=Path, default=None)
    report_parser.set_defaults(func=_run_report)

    package_parser = translation_subparsers.add_parser("package", help="Package app-ready translation archives")
    _add_config_arg(package_parser)
    package_parser.set_defaults(func=_run_package)

    catalog_parser = translation_subparsers.add_parser("catalog", help="Generate translation catalog")
    _add_config_arg(catalog_parser)
    catalog_parser.add_argument("--output", type=Path, default=None)
    catalog_parser.add_argument("--version", type=int, default=None)
    catalog_parser.add_argument("--package-version", default=None)
    catalog_parser.add_argument("--min-app-version", default=None)
    catalog_parser.set_defaults(func=_run_catalog)

    all_parser = translation_subparsers.add_parser("all", help="Prepare, benchmark, and report for translation")
    _add_config_arg(all_parser)
    all_parser.add_argument("--force", action="store_true")
    all_parser.add_argument("--timestamp", default=None)
    all_parser.set_defaults(func=_run_all)


def _add_config_arg(parser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)


def _run_prepare(args) -> None:
    config = load_config(args.config)
    manifests = prepare_translation(config, force=args.force)
    print(f"[translation prepare] prepared {len(manifests)} artifacts")


def _run_benchmark(args) -> None:
    config = load_config(args.config)
    result_dir = run_benchmark(config, timestamp=args.timestamp)
    print(f"[translation benchmark] result dir: {result_dir}")


def _run_report(args) -> None:
    config = load_config(args.config)
    payload = generate_report(config, result_dir=args.result_dir)
    print(f"[translation report] wrote metrics for {len(payload['evaluations'])} evaluations")


def _run_package(args) -> None:
    config = load_config(args.config)
    archives = package_translation_artifacts(config)
    print(f"[translation package] packaged {len(archives)} archives")


def _run_catalog(args) -> None:
    config = load_config(args.config)
    output_path = generate_translation_catalog(
        config,
        output_path=args.output,
        requested_version=args.version,
        package_version=args.package_version,
        min_app_version=args.min_app_version,
    )
    print(f"[translation catalog] wrote {output_path}")


def _run_all(args) -> None:
    config = load_config(args.config)
    prepare_translation(config, force=args.force)
    result_dir = run_benchmark(config, timestamp=args.timestamp)
    payload = generate_report(config, result_dir=result_dir)
    print(f"[translation all] completed {len(payload['evaluations'])} evaluations in {result_dir}")
