from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pptt.io.manifests import sha256_file


def guarded_output_directory(output_path: Path, results_root: Path) -> Path:
    """Resolve an output directory only when it is contained by results_root."""

    root = Path(results_root).resolve()
    directory = Path(output_path).resolve().parent
    try:
        directory.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Output directory is outside the guarded results root: {directory}"
        ) from error
    if directory == root:
        raise ValueError("A replay output must use a dedicated result directory")
    return directory


def compare_json_outputs(before: Path, after: Path) -> dict[str, Any]:
    """Compare deterministic JSON artifacts at byte and decoded-payload levels."""

    before_path = Path(before)
    after_path = Path(after)
    before_payload = json.loads(before_path.read_text(encoding="utf-8"))
    after_payload = json.loads(after_path.read_text(encoding="utf-8"))
    before_digest = sha256_file(before_path)
    after_digest = sha256_file(after_path)
    return {
        "before_sha256": before_digest,
        "after_sha256": after_digest,
        "byte_identical": before_digest == after_digest,
        "payload_identical": before_payload == after_payload,
        "numeric_summary": after_payload,
    }
