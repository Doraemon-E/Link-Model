from __future__ import annotations

import argparse

from speech import register_speech_group
from translation import register_translation_group


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Link-Model unified translation and speech pipeline")
    subparsers = parser.add_subparsers(dest="group")
    register_translation_group(subparsers)
    register_speech_group(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
