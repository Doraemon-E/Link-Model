from __future__ import annotations

try:
    from .package_speech_model import main as package_main
    from .speech_downloader import main as download_main
    from .speech_legacy_migration import migrate_legacy_speech_layout
except ImportError:
    from package_speech_model import main as package_main
    from speech_downloader import main as download_main
    from speech_legacy_migration import migrate_legacy_speech_layout


def main() -> None:
    migrate_legacy_speech_layout()
    download_main()
    package_main()


if __name__ == "__main__":
    main()
