from __future__ import annotations

from pathlib import Path

from shared.config import DEFAULT_CONFIG_PATH, load_config

from .catalog import generate_speech_catalog
from .package import package_speech_artifacts
from .prepare import prepare_speech


def register_speech_group(subparsers) -> None:
    speech_parser = subparsers.add_parser("speech", help="Speech pipeline commands")
    speech_subparsers = speech_parser.add_subparsers(dest="speech_command", required=True)

    prepare_parser = speech_subparsers.add_parser("prepare", help="Download speech artifacts")
    _add_config_arg(prepare_parser)
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.set_defaults(func=_run_prepare)

    package_parser = speech_subparsers.add_parser("package", help="Package speech archives")
    _add_config_arg(package_parser)
    package_parser.set_defaults(func=_run_package)

    catalog_parser = speech_subparsers.add_parser("catalog", help="Generate speech catalog")
    _add_config_arg(catalog_parser)
    catalog_parser.add_argument("--output", type=Path, default=None)
    catalog_parser.add_argument("--version", type=int, default=None)
    catalog_parser.add_argument("--package-version", default=None)
    catalog_parser.add_argument("--min-app-version", default=None)
    catalog_parser.set_defaults(func=_run_catalog)

    all_parser = speech_subparsers.add_parser("all", help="Prepare, package, and catalog speech assets")
    _add_config_arg(all_parser)
    all_parser.add_argument("--force", action="store_true")
    all_parser.set_defaults(func=_run_all)


def _add_config_arg(parser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)


def _run_prepare(args) -> None:
    config = load_config(args.config)
    downloaded = prepare_speech(config, force=args.force)
    print(f"[speech prepare] prepared {len(downloaded)} artifacts")


def _run_package(args) -> None:
    config = load_config(args.config)
    archives = package_speech_artifacts(config)
    print(f"[speech package] packaged {len(archives)} archives")


def _run_catalog(args) -> None:
    config = load_config(args.config)
    output_path = generate_speech_catalog(
        config,
        output_path=args.output,
        requested_version=args.version,
        package_version=args.package_version,
        min_app_version=args.min_app_version,
    )
    print(f"[speech catalog] wrote {output_path}")


def _run_all(args) -> None:
    config = load_config(args.config)
    prepare_speech(config, force=args.force)
    package_speech_artifacts(config)
    output_path = generate_speech_catalog(config)
    print(f"[speech all] wrote catalog {output_path}")
