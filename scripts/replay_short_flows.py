from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import yaml

from pptt.io.manifests import write_assets_json
from pptt.reproducibility.replay import (
    compare_json_outputs,
    guarded_output_directory,
)


SHORT_FLOWS = (
    ("V0", Path("configs/experiments/v0_math.yaml"), Path("scripts/run_v0_math.py")),
    (
        "V2",
        Path("configs/experiments/v2_synthetic.yaml"),
        Path("scripts/run_v2_synthetic.py"),
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_output(config_path: Path, workspace: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "output" not in config:
        raise ValueError(f"Replay config does not declare output: {config_path}")
    return (workspace / str(config["output"])).resolve()


def _run_one(
    *,
    label: str,
    config_path: Path,
    script_path: Path,
    workspace: Path,
    results_root: Path,
    backup_root: Path,
) -> dict[str, Any]:
    output_path = _load_output(config_path, workspace)
    output_directory = guarded_output_directory(output_path, results_root)
    if not output_path.is_file():
        raise FileNotFoundError(f"Missing baseline output for {label}: {output_path}")
    relative_output = output_path.relative_to(output_directory)
    backup_directory = backup_root / label.lower()
    shutil.move(os.fspath(output_directory), os.fspath(backup_directory))
    if output_directory.exists():
        raise RuntimeError(f"Replay directory was not emptied: {output_directory}")

    command = [sys.executable, os.fspath(script_path), "--config", os.fspath(config_path)]
    started_at = _utc_now()
    completed = subprocess.run(
        command,
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    ended_at = _utc_now()
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0:
        if output_directory.exists():
            shutil.rmtree(output_directory)
        shutil.move(os.fspath(backup_directory), os.fspath(output_directory))
        raise RuntimeError(f"{label} replay failed with exit code {completed.returncode}")

    before_path = backup_directory / relative_output
    comparison = compare_json_outputs(before_path, output_path)
    record = {
        "label": label,
        "command": " ".join(["python", script_path.as_posix(), "--config", config_path.as_posix()]),
        "started_at": started_at,
        "ended_at": ended_at,
        "result_directory_absent_before_run": True,
        "return_code": completed.returncode,
        **comparison,
    }
    if not comparison["byte_identical"] or not comparison["payload_identical"]:
        failure_root = results_root / "replay_failure_backup" / label.lower()
        if failure_root.exists():
            raise FileExistsError(f"Replay failure backup already exists: {failure_root}")
        failure_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(backup_directory, failure_root)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay deterministic V0 and V2 flows from empty result directories."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("manifests/short_replay_audit.json"),
    )
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    results_root = (workspace / "results").resolve()
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=".pptt_replay_", dir=workspace) as directory:
        backup_root = Path(directory)
        for label, config, script in SHORT_FLOWS:
            records.append(
                _run_one(
                    label=label,
                    config_path=workspace / config,
                    script_path=script,
                    workspace=workspace,
                    results_root=results_root,
                    backup_root=backup_root,
                )
            )

    audit = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "all_byte_identical": all(row["byte_identical"] for row in records),
        "all_payload_identical": all(row["payload_identical"] for row in records),
        "flows": records,
    }
    audit_path = args.audit if args.audit.is_absolute() else workspace / args.audit
    write_assets_json(audit_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    if not audit["all_byte_identical"] or not audit["all_payload_identical"]:
        raise AssertionError("Short-flow replay did not reproduce the registered outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
