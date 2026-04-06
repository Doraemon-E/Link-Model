from __future__ import annotations

try:
    from .downloader import main as download_main
    from .legacy_migration import migrate_legacy_layout
    from .package_quantized_onnx import main as package_main
    from .quantize_onnx import main as quantize_main
    from .trans_to_onnx import main as export_main
except ImportError:
    from downloader import main as download_main
    from legacy_migration import migrate_legacy_layout
    from package_quantized_onnx import main as package_main
    from quantize_onnx import main as quantize_main
    from trans_to_onnx import main as export_main


def main() -> None:
    migrate_legacy_layout()
    download_main()
    export_main()
    quantize_main()
    package_main()


if __name__ == "__main__":
    main()
