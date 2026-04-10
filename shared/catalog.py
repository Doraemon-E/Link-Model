from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .files import ensure_directory, resolve_catalog_version


def write_catalog_payload(
    output_path: Path,
    *,
    packages: list[dict[str, object]],
    requested_version: int | None,
) -> Path:
    payload = {
        "version": resolve_catalog_version(output_path, requested_version),
        "generatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "packages": packages,
    }
    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path
