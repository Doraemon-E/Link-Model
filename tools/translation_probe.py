from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from shared.config import DEFAULT_CONFIG_PATH, load_config
from translation.storage import translation_stage_directory


SUPPORTED_STAGES = ("downloaded", "exported", "quantized")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe one translation artifact across local pipeline stages.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--stage",
        dest="stages",
        action="append",
        choices=SUPPORTED_STAGES,
        help="Repeat to limit the probe to selected stages. Defaults to all stages.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir.as_posix(), local_files_only=True)


def load_model(model_dir: Path, stage: str):
    if stage == "downloaded":
        from transformers import AutoModelForSeq2SeqLM

        return AutoModelForSeq2SeqLM.from_pretrained(model_dir.as_posix(), local_files_only=True)

    from optimum.onnxruntime import ORTModelForSeq2SeqLM

    return ORTModelForSeq2SeqLM.from_pretrained(
        model_dir.as_posix(),
        local_files_only=True,
        provider="CPUExecutionProvider",
        use_merged=False,
    )


def probe_stage(*, artifact_id: str, stage: str, model_dir: Path, text: str, max_new_tokens: int) -> None:
    tokenizer = load_tokenizer(model_dir)
    model = load_model(model_dir, stage)

    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"][0].tolist()
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    outputs = model.generate(
        **encoded,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
    )
    output_ids = outputs[0].tolist()
    decoded_with_special_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
    decoded_without_special_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    print(f"=== {artifact_id} / {stage} ===")
    print(f"model_dir: {model_dir}")
    print(f"input_ids: {input_ids}")
    print(f"input_tokens: {input_tokens}")
    print(f"output_ids: {output_ids}")
    print(f"decoded_with_special_tokens: {decoded_with_special_tokens!r}")
    print(f"decoded_without_special_tokens: {decoded_without_special_tokens!r}")
    print(f"empty_after_decode: {not decoded_without_special_tokens.strip()}")
    print()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    artifact = config.translation.artifacts.get(args.artifact_id)
    if artifact is None:
        raise SystemExit(f"Unknown artifact_id: {args.artifact_id}")
    if artifact.runtime_backend != "onnxruntime":
        raise SystemExit(f"Artifact {args.artifact_id} uses runtime_backend={artifact.runtime_backend}; this probe only supports onnxruntime seq2seq artifacts.")

    stages = args.stages or list(SUPPORTED_STAGES)
    for stage in stages:
        model_dir = translation_stage_directory(config, stage, args.artifact_id)
        if not model_dir.exists():
            print(f"=== {args.artifact_id} / {stage} ===")
            print(f"missing: {model_dir}")
            print()
            continue
        probe_stage(
            artifact_id=args.artifact_id,
            stage=stage,
            model_dir=model_dir,
            text=args.text,
            max_new_tokens=args.max_new_tokens,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
