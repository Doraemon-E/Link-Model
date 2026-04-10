from __future__ import annotations

from pathlib import Path

from shared.catalog import write_catalog_payload
from shared.config import RootConfig
from shared.files import installed_size_for_archive, sha256_for_file

from .package import package_enabled_artifacts
from .storage import translation_archive_path


def generate_translation_catalog(
    config: RootConfig,
    *,
    output_path: Path | None = None,
    requested_version: int | None = None,
    package_version: str | None = None,
    min_app_version: str | None = None,
) -> Path:
    package_cfg = config.translation.package
    resolved_output = output_path or config.shared_paths.translation_catalog_output
    resolved_package_version = package_version or package_cfg.package_version
    resolved_min_app_version = min_app_version or package_cfg.min_app_version

    packages: list[dict[str, object]] = []
    for artifact in package_enabled_artifacts(config):
        archive_path = translation_archive_path(config, artifact)
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing translation archive: {archive_path}")
        packages.append(
            {
                "packageId": artifact.package_id,
                "version": resolved_package_version,
                "family": artifact.family,
                "supportedLanguages": _supported_languages_for_artifact(artifact),
                "archiveURL": f"{package_cfg.archive_base_url}/{archive_path.name}",
                "sha256": sha256_for_file(archive_path),
                "archiveSize": archive_path.stat().st_size,
                "installedSize": installed_size_for_archive(archive_path),
                "manifestRelativePath": "translation-manifest.json",
                "minAppVersion": resolved_min_app_version,
            }
        )

    return write_catalog_payload(resolved_output, packages=packages, requested_version=requested_version)


def _supported_languages_for_artifact(artifact: ArtifactSpec) -> list[str]:
    if artifact.family == "gguf_causal_llm":
        return ["zho", "eng", "jpn", "kor", "fra", "deu", "rus", "spa", "ita"]

    supported_codes = [
        artifact.package_source or artifact.source_langs[0],
        artifact.package_target or artifact.target_langs[0],
    ]
    return list(dict.fromkeys(supported_codes))
