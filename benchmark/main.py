from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    CURRENT_FILE = Path(__file__).resolve()
    PACKAGE_PARENT = CURRENT_FILE.parent.parent
    if PACKAGE_PARENT.as_posix() not in sys.path:
        sys.path.insert(0, PACKAGE_PARENT.as_posix())

    from benchmark.config import load_config
    from benchmark.corpus import load_corpus
    from benchmark.paths import (
        DEFAULT_CONFIG_PATH,
        artifact_stage_directory,
        artifact_manifest_path,
        ensure_directory,
        new_result_directory,
    )
    from benchmark.prepare import has_onnx_payload, load_artifact_manifest, prepare_benchmark
    from benchmark.registry import (
        load_executor,
        selected_routes,
        selected_systems,
        validate_config_selection,
    )
    from benchmark.report import generate_report, resolve_result_dir
else:
    from .config import load_config
    from .corpus import load_corpus
    from .paths import (
        DEFAULT_CONFIG_PATH,
        artifact_stage_directory,
        artifact_manifest_path,
        ensure_directory,
        new_result_directory,
    )
    from .prepare import has_onnx_payload, load_artifact_manifest, prepare_benchmark
    from .registry import load_executor, selected_routes, selected_systems, validate_config_selection
    from .report import generate_report, resolve_result_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline translation benchmark pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Download, export, quantize, and write artifact manifests")
    prepare_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    prepare_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run benchmark inference and write predictions")
    run_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    run_parser.add_argument("--timestamp", type=str, default=None)

    report_parser = subparsers.add_parser("report", help="Compute metrics and write benchmark report")
    report_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    report_parser.add_argument("--result-dir", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        manifests = prepare_benchmark(args.config, force=args.force)
        print(f"[prepare] prepared {len(manifests)} artifacts")
        return

    if args.command == "run":
        run_benchmark(args.config, timestamp=args.timestamp)
        return

    if args.command == "report":
        config = load_config(args.config)
        result_dir = resolve_result_dir(config.results_root, args.result_dir)
        payload = generate_report(result_dir, args.config)
        print(f"[report] wrote metrics for {len(payload['evaluations'])} evaluations to {result_dir}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


def run_benchmark(config_path: Path | None = None, *, timestamp: str | None = None) -> Path:
    config = load_config(config_path)
    validate_config_selection(config)

    corpus_entries = load_corpus()
    result_dir = new_result_directory(config.results_root, timestamp=timestamp)
    ensure_directory(result_dir)

    predictions = []
    runtime_summaries = []

    for system in selected_systems(config):
        readiness = _system_readiness(config, system)
        if readiness is not None:
            print(f"[run] skip system={system.system_id}: {readiness}")
            continue
        executor = load_executor(system)
        for route in selected_routes(config):
            print(f"[run] system={system.system_id} route={route.route_id}")
            system_predictions, runtime_summary = executor(config, system, route, corpus_entries)
            predictions.extend(system_predictions)
            runtime_summaries.append(runtime_summary)

    predictions_path = result_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as output_file:
        for prediction in predictions:
            output_file.write(json.dumps(prediction.to_json_dict(), ensure_ascii=False) + "\n")

    (result_dir / "runtime-summary.json").write_text(
        json.dumps([summary.to_json_dict() for summary in runtime_summaries], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (result_dir / "config-snapshot.json").write_text(
        json.dumps(config.to_json_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[run] wrote predictions to {predictions_path}")
    return result_dir


def _system_readiness(config, system) -> str | None:
    for artifact_id in system.artifact_ids:
        manifest_path = artifact_manifest_path(config.artifacts_root, artifact_id)
        if not manifest_path.exists():
            return f"missing artifact manifest for {artifact_id}; run prepare first"

        manifest = load_artifact_manifest(manifest_path)
        if not manifest.quantize_success:
            return f"artifact {artifact_id} is not ready ({manifest.error_message or 'prepare failed'})"

        quantized_dir = artifact_stage_directory(config.artifacts_root, "quantized", artifact_id)
        if not has_onnx_payload(quantized_dir):
            return f"artifact {artifact_id} is incomplete under {quantized_dir}; rerun prepare"

    return None


if __name__ == "__main__":
    main()
