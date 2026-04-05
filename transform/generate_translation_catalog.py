from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from model_registry import MODEL_SPECS, model_directory_name

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_OUTPUT_PATH = REPO_ROOT.parent / "link" / "link" / "Resource" / "translation-catalog.json"
ARCHIVE_BASE_URL = "https://link.hackerapp.site/link/translation/packages"
LANGUAGE_CODE_BY_NAME = {
    "中文": "zho",
    "英文": "eng",
    "日文": "jpn",
    "韩文": "kor",
    "法文": "fra",
    "德文": "deu",
    "俄文": "rus",
    "西班牙文": "spa",
    "意大利文": "ita",
}


@dataclass(frozen=True)
class CatalogPackage:
    package_id: str
    version: str
    source: str
    target: str
    family: str
    archive_url: str
    sha256: str
    archive_size: int
    installed_size: int
    manifest_relative_path: str = "translation-manifest.json"
    min_app_version: str = "1.0.0"

    def to_dict(self) -> dict[str, object]:
        return {
            "packageId": self.package_id,
            "version": self.version,
            "source": self.source,
            "target": self.target,
            "family": self.family,
            "archiveURL": self.archive_url,
            "sha256": self.sha256,
            "archiveSize": self.archive_size,
            "installedSize": self.installed_size,
            "manifestRelativePath": self.manifest_relative_path,
            "minAppVersion": self.min_app_version,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate translation-catalog.json from local zip packages.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--package-version", default="1.0.0")
    parser.add_argument("--min-app-version", default="1.0.0")
    return parser.parse_args()


def sha256_for_file(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1_048_576)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def installed_size_for_archive(file_path: Path) -> int:
    with zipfile.ZipFile(file_path) as archive:
        return sum(info.file_size for info in archive.infolist())


def source_target_codes(language_pair: str) -> tuple[str, str]:
    source_name, target_name = [part.strip() for part in language_pair.split("->", maxsplit=1)]
    return LANGUAGE_CODE_BY_NAME[source_name], LANGUAGE_CODE_BY_NAME[target_name]


def resolve_catalog_version(output_path: Path, requested_version: int | None) -> int:
    if requested_version is not None:
        return requested_version

    if not output_path.exists():
        return 1

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return int(payload.get("version", 1))
    except Exception:
        return 1


def build_packages(models_dir: Path, package_version: str, min_app_version: str) -> list[CatalogPackage]:
    packages: list[CatalogPackage] = []

    for spec in MODEL_SPECS:
        local_name = model_directory_name(spec)
        archive_name = f"{local_name}-onnx-int8.zip"
        archive_path = models_dir / archive_name

        if not archive_path.exists():
            raise FileNotFoundError(f"Missing archive: {archive_path}")

        source_code, target_code = source_target_codes(spec["language_pair"])
        packages.append(
            CatalogPackage(
                package_id=f"{local_name}-onnx",
                version=package_version,
                source=source_code,
                target=target_code,
                family=spec["family"],
                archive_url=f"{ARCHIVE_BASE_URL}/{archive_name}",
                sha256=sha256_for_file(archive_path),
                archive_size=archive_path.stat().st_size,
                installed_size=installed_size_for_archive(archive_path),
                min_app_version=min_app_version,
            )
        )

    return packages


def main() -> None:
    args = parse_args()
    packages = build_packages(
        models_dir=args.models_dir,
        package_version=args.package_version,
        min_app_version=args.min_app_version,
    )

    payload = {
        "version": resolve_catalog_version(args.output, args.version),
        "generatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "packages": [package.to_dict() for package in packages],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已生成: {args.output}")


if __name__ == "__main__":
    main()
