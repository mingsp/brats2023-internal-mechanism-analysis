from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq
import yaml

from pptt.io.manifests import sha256_file


_CONFIG_SUFFIXES = {".json", ".yaml", ".yml"}
_WEIGHT_SUFFIXES = {".ckpt", ".pt", ".pth"}
_FIGURE_SUFFIXES = {".jpeg", ".jpg", ".pdf", ".png", ".svg", ".tif", ".tiff"}
_TABLE_SUFFIXES = {".csv", ".parquet", ".tsv"}
_NONFORMAL_MARKERS = (
    "debug",
    "dry_run",
    "overfit",
    "preflight",
    "resume_check",
    "smoke",
)


def format_environment_lock(
    *,
    environment: Mapping[str, Any],
    packages: list[str],
) -> str:
    """Serialize runtime metadata and package pins in a diff-friendly form."""

    lines = ["[runtime]"]
    for key in sorted(environment):
        value = environment[key]
        if isinstance(value, (list, tuple)):
            rendered = " | ".join(str(item) for item in value)
        elif isinstance(value, Mapping):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            rendered = str(value)
        lines.append(f"{key}={rendered}")
    lines.extend(["", "[packages]"])
    lines.extend(sorted(str(package).strip() for package in packages if str(package).strip()))
    return "\n".join(lines) + "\n"


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    file_path = Path(path)
    return {
        "relative_path": file_path.relative_to(root).as_posix(),
        "size_bytes": int(file_path.stat().st_size),
        "sha256": sha256_file(file_path),
    }


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _FIGURE_SUFFIXES:
        return "figure"
    if suffix in _TABLE_SUFFIXES:
        return "table"
    if suffix in _WEIGHT_SUFFIXES:
        return "checkpoint"
    if suffix == ".json":
        return "json"
    if suffix == ".npz":
        return "trace"
    if suffix in {".log", ".txt"}:
        return "log"
    return "other"


def _table_row_count(path: Path) -> int | None:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return int(pq.ParquetFile(path).metadata.num_rows)
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = csv.reader(stream, delimiter=delimiter)
            return max(sum(1 for _ in rows) - 1, 0)
    return None


def _is_formal_eligible(relative_path: str) -> bool:
    normalized = relative_path.lower()
    return not any(marker in normalized for marker in _NONFORMAL_MARKERS)


def _collect_config_files(workspace_root: Path) -> list[dict[str, Any]]:
    config_root = workspace_root / "configs"
    if not config_root.is_dir():
        return []
    paths = sorted(
        (
            path
            for path in config_root.rglob("*")
            if path.is_file() and path.suffix.lower() in _CONFIG_SUFFIXES
        ),
        key=lambda path: path.relative_to(workspace_root).as_posix(),
    )
    return [_file_record(path, workspace_root) for path in paths]


def _collect_data_manifest(asset_root: Path) -> dict[str, Any]:
    path = asset_root / "manifests" / "assets.json"
    if not path.is_file():
        return {
            "available": False,
            "root": "asset",
            "relative_path": "manifests/assets.json",
        }
    record = _file_record(path, asset_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "root": "asset",
        **record,
        "schema_version": payload.get("schema_version"),
        "splits": payload.get("splits", {}),
        "full_hash_manifests": payload.get("full_hash_manifests", []),
    }


def _resolve_root(root_name: str, *, workspace_root: Path, asset_root: Path) -> Path:
    roots = {"workspace": workspace_root, "asset": asset_root}
    try:
        return roots[root_name]
    except KeyError as error:
        raise ValueError(f"Unknown inventory root: {root_name}") from error


def _collect_weights(
    workspace_root: Path,
    asset_root: Path,
) -> list[dict[str, Any]]:
    matrix_path = workspace_root / "manifests" / "model_matrix.yaml"
    if not matrix_path.is_file():
        return []
    payload = yaml.safe_load(matrix_path.read_text(encoding="utf-8")) or {}
    records: list[dict[str, Any]] = []
    for job in payload.get("jobs", []):
        checkpoint = job["checkpoint"]
        root_name = str(checkpoint["root"])
        root = _resolve_root(
            root_name,
            workspace_root=workspace_root,
            asset_root=asset_root,
        )
        relative_path = Path(str(checkpoint["path"]))
        path = root / relative_path
        record: dict[str, Any] = {
            "model": str(job["model"]),
            "seed": int(job["seed"]),
            "root": root_name,
            "relative_path": relative_path.as_posix(),
            "available": path.is_file(),
        }
        if path.is_file():
            record.update(
                {
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
        records.append(record)
    return records


def _collect_result_artifacts(results_root: Path) -> list[dict[str, Any]]:
    if not results_root.is_dir():
        return []
    paths = sorted(
        (path for path in results_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(results_root).as_posix(),
    )
    records: list[dict[str, Any]] = []
    for path in paths:
        record = _file_record(path, results_root)
        kind = _artifact_kind(path)
        record.update(
            {
                "root": "results",
                "artifact_kind": kind,
                "formal_eligible": _is_formal_eligible(record["relative_path"]),
            }
        )
        row_count = _table_row_count(path)
        if row_count is not None:
            record["row_count"] = row_count
        records.append(record)
    return records


def build_result_inventory(
    *,
    workspace_root: Path,
    asset_root: Path,
    results_root: Path,
    run_commands: Mapping[str, str],
    git_commit: str,
    git_dirty: bool,
    environment: Mapping[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    """Build a content-addressed inventory without modifying any artifact."""

    workspace = Path(workspace_root).resolve()
    assets = Path(asset_root).resolve()
    results = Path(results_root).resolve()
    artifacts = _collect_result_artifacts(results)
    kind_counts = Counter(record["artifact_kind"] for record in artifacts)
    formal_count = sum(bool(record["formal_eligible"]) for record in artifacts)
    return {
        "schema_version": 1,
        "generated_at": str(generated_at),
        "code": {"commit": str(git_commit), "dirty": bool(git_dirty)},
        "environment": dict(environment),
        "run_commands": dict(sorted(run_commands.items())),
        "config_files": _collect_config_files(workspace),
        "data_manifest": _collect_data_manifest(assets),
        "weights": _collect_weights(workspace, assets),
        "result_artifacts": artifacts,
        "result_summary": {
            "artifact_count": len(artifacts),
            "formal_eligible_count": formal_count,
            "nonformal_count": len(artifacts) - formal_count,
            "by_kind": dict(sorted(kind_counts.items())),
        },
    }


def _verify_record(
    record: Mapping[str, Any],
    root: Path,
    *,
    label: str,
) -> list[str]:
    if record.get("available") is False:
        return []
    relative_path = Path(str(record["relative_path"]))
    path = root / relative_path
    if not path.is_file():
        return [f"Missing {label}: {relative_path.as_posix()}"]
    problems: list[str] = []
    expected_size = record.get("size_bytes")
    if expected_size is not None and int(path.stat().st_size) != int(expected_size):
        problems.append(f"Size mismatch for {label}: {relative_path.as_posix()}")
    expected_digest = record.get("sha256")
    if expected_digest is not None and sha256_file(path) != str(expected_digest):
        problems.append(f"SHA-256 mismatch for {label}: {relative_path.as_posix()}")
    return problems


def verify_result_inventory(
    inventory: Mapping[str, Any],
    *,
    workspace_root: Path,
    asset_root: Path,
    results_root: Path,
) -> list[str]:
    """Return every integrity failure; an empty list is a successful audit."""

    workspace = Path(workspace_root).resolve()
    assets = Path(asset_root).resolve()
    results = Path(results_root).resolve()
    problems: list[str] = []
    for record in inventory.get("config_files", []):
        problems.extend(_verify_record(record, workspace, label="config"))
    data_manifest = inventory.get("data_manifest", {})
    if data_manifest:
        problems.extend(_verify_record(data_manifest, assets, label="data manifest"))
    for record in inventory.get("weights", []):
        root = _resolve_root(
            str(record["root"]),
            workspace_root=workspace,
            asset_root=assets,
        )
        problems.extend(_verify_record(record, root, label="weight"))
    for record in inventory.get("result_artifacts", []):
        problems.extend(_verify_record(record, results, label="result artifact"))
    return problems
