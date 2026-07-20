#!/usr/bin/env python3
"""Read-only console monitor for the locked V8 TransUNet experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG = Path(
    "configs/experiments/v8_transunet_mechanism_replication.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    return payload


def _load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"unreadable status {path}: {type(error).__name__}: {error}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"status is not a JSON object: {path}")
        return None
    return payload


def _completed_artifacts(path: Path, pattern: str) -> list[Path]:
    if not path.is_dir():
        return []
    return [item for item in path.glob(pattern) if not item.name.startswith(".")]


def _artifact_count(path: Path, pattern: str) -> tuple[int, str | None]:
    artifacts = _completed_artifacts(path, pattern)
    if not artifacts:
        return 0, None
    latest = max(artifacts, key=lambda item: item.stat().st_mtime)
    return len(artifacts), latest.stem


def _artifact_eta(path: Path, pattern: str, *, expected: int) -> tuple[float, float] | None:
    artifacts = _completed_artifacts(path, pattern)
    if len(artifacts) < 2 or len(artifacts) >= expected:
        return None
    timestamps = sorted(item.stat().st_mtime for item in artifacts)
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        return None
    throughput = (len(artifacts) - 1) / elapsed
    if throughput <= 0:
        return None
    return throughput, (expected - len(artifacts)) / throughput


def _status_failed(value: object) -> bool:
    status = str(value or "").upper()
    return "FAIL" in status or "ERROR" in status


def _validation_seed_progress(
    workspace: Path,
    config: Mapping[str, Any],
    seed: int,
    errors: list[str],
) -> dict[str, Any]:
    model = str(config["model"])
    expected = int(config["validation"]["expected_patient_count_per_seed"])
    trace_root = (
        workspace
        / str(config["validation"]["worker_root"])
        / f"seed_{seed}"
        / model
        / f"seed_{seed}"
        / str(config["discovery_split"])
        / "case_traces"
    )
    completed, last_patient = _artifact_count(trace_root, "*.npz")
    status_path = (
        workspace
        / str(config["validation"]["worker_root"])
        / f"seed_{seed}"
        / "transfer_status.json"
    )
    job_status = _load_json(status_path, errors)
    timing = _artifact_eta(trace_root, "*.npz", expected=expected)
    status = "NOT_STARTED"
    if completed > expected:
        status = "COUNT_MISMATCH"
        errors.append(
            f"validation seed {seed}: {completed} traces exceed expected {expected}"
        )
    elif job_status and _status_failed(job_status.get("status")):
        status = str(job_status.get("status"))
        errors.append(f"validation seed {seed}: {status}")
    elif completed == expected:
        status = "COMPLETE"
    elif completed:
        status = "RUNNING"
    return {
        "seed": seed,
        "status": status,
        "completed": completed,
        "total": expected,
        "last_patient_id": last_patient,
        "patients_per_second": timing[0] if timing else None,
        "eta_seconds": timing[1] if timing else None,
        "trace_root": str(trace_root),
    }


def _formal_seed_progress(
    workspace: Path,
    config: Mapping[str, Any],
    seed: int,
    errors: list[str],
) -> dict[str, Any]:
    model = str(config["model"])
    expected = int(config["formal"]["expected_patient_count_per_seed"])
    job_root = (
        workspace
        / str(config["formal_output_root"])
        / model
        / f"seed_{seed}"
    )
    completed, last_artifact = _artifact_count(job_root / "patient_results", "*.json")
    progress = _load_json(job_root / "progress.json", errors) or {}
    job_status = _load_json(job_root / "job_status.json", errors) or {}
    progress_completed = int(progress.get("completed_patients", completed) or 0)
    if job_status and progress_completed != completed:
        errors.append(
            f"formal seed {seed}: progress={progress_completed}, atomic files={completed}"
        )
    status = str(job_status.get("status") or progress.get("status") or "NOT_STARTED")
    if completed and status == "NOT_STARTED":
        status = "RUNNING"
    if completed == expected and not _status_failed(status):
        status = str(job_status.get("status") or "COMPLETE")
    if completed > expected:
        status = "COUNT_MISMATCH"
        errors.append(
            f"formal seed {seed}: {completed} patients exceed expected {expected}"
        )
    error = progress.get("error")
    if error:
        errors.append(f"formal seed {seed}: {error}")
    if _status_failed(status) and not error:
        errors.append(f"formal seed {seed}: {status}")
    return {
        "seed": seed,
        "status": status,
        "completed": completed,
        "total": expected,
        "last_patient_id": progress.get("last_patient_id") or last_artifact,
        "patients_per_second": progress.get("patients_per_second"),
        "eta_seconds": progress.get("eta_seconds"),
        "error": error,
        "job_root": str(job_root),
    }


def _system_metrics(workspace: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(workspace)
    metrics: dict[str, Any] = {
        "disk_free_gib": usage.free / (1024**3),
        "disk_total_gib": usage.total / (1024**3),
        "gpu": [],
    }
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return metrics
    if completed.returncode == 0:
        metrics["gpu"] = [
            line.strip() for line in completed.stdout.splitlines() if line.strip()
        ]
    return metrics


def collect_v8_progress(
    workspace_root: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    include_system: bool = False,
) -> dict[str, Any]:
    """Collect one immutable snapshot without starting or modifying experiments."""
    workspace = Path(workspace_root).resolve()
    active_config = Path(config_path)
    if not active_config.is_absolute():
        active_config = workspace / active_config
    config = _load_yaml(active_config.resolve())
    seeds = tuple(int(value) for value in config["model_seeds"])
    errors: list[str] = []

    validation_seeds = [
        _validation_seed_progress(workspace, config, seed, errors) for seed in seeds
    ]
    formal_seeds = [
        _formal_seed_progress(workspace, config, seed, errors) for seed in seeds
    ]
    validation = {
        "completed": sum(int(row["completed"]) for row in validation_seeds),
        "total": sum(int(row["total"]) for row in validation_seeds),
    }
    formal = {
        "completed": sum(int(row["completed"]) for row in formal_seeds),
        "total": sum(int(row["total"]) for row in formal_seeds),
    }

    candidate_path = (
        workspace
        / str(config["validation"]["candidate_root"])
        / "candidate_registration.json"
    )
    candidate_payload = _load_json(candidate_path, errors)
    candidate = {
        "status": (
            str(candidate_payload.get("status"))
            if candidate_payload
            else "NOT_AVAILABLE"
        ),
        "path_id": (
            str(candidate_payload["candidate"].get("path_id"))
            if candidate_payload and isinstance(candidate_payload.get("candidate"), dict)
            else None
        ),
    }
    lock = _load_json(workspace / str(config["protocol_lock"]), errors)
    final_status = _load_json(
        workspace / str(config["formal_output_root"]) / "v8_status.json",
        errors,
    )
    experiment_root = (workspace / str(config["formal_output_root"])).resolve().parent
    pipeline_status = _load_json(experiment_root / "pipeline_status.json", errors)

    has_failure = bool(errors) or any(
        _status_failed(row["status"]) for row in validation_seeds + formal_seeds
    )
    if pipeline_status and _status_failed(pipeline_status.get("status")):
        has_failure = True
        message = pipeline_status.get("message") or pipeline_status.get("error")
        errors.append(f"pipeline: {message or pipeline_status.get('status')}")

    scientific_status = None
    if final_status and final_status.get("execution_status") == "PASS":
        phase = "COMPLETE"
        scientific_status = final_status.get("scientific_status")
    elif has_failure:
        phase = "FAILED"
    elif candidate["status"] == "NO_REGISTERED_TRANSUNET_CANDIDATE" or (
        lock and lock.get("status") == "NO_REGISTERED_TRANSUNET_CANDIDATE"
    ):
        phase = "TERMINATED_NO_CANDIDATE"
    elif formal["completed"] > 0 or any(
        row["status"] not in {"NOT_STARTED", ""} for row in formal_seeds
    ):
        phase = "FORMAL_INTERVENTION"
    elif lock and lock.get("test_intervention_authorized") is True:
        phase = "PROTOCOL_LOCKED"
    elif candidate["status"] == "REGISTERED_TRANSUNET_CANDIDATE":
        phase = "CANDIDATE_SELECTED"
    elif validation["completed"] == validation["total"] and validation["total"]:
        phase = "VALIDATION_COMPLETE"
    elif validation["completed"]:
        phase = "VALIDATION_TRACES"
    else:
        phase = "NOT_STARTED"

    formal_eta_values = [
        float(row["eta_seconds"])
        for row in formal_seeds
        if row.get("eta_seconds") is not None
    ]
    validation_eta_values = [
        float(row["eta_seconds"])
        for row in validation_seeds
        if row.get("eta_seconds") is not None
    ]
    if phase == "FORMAL_INTERVENTION" and formal_eta_values:
        eta_seconds = max(formal_eta_values)
    elif phase == "VALIDATION_TRACES" and validation_eta_values:
        eta_seconds = max(validation_eta_values)
    else:
        eta_seconds = None
    snapshot: dict[str, Any] = {
        "read_only": True,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "candidate": candidate,
        "validation": validation,
        "validation_seeds": validation_seeds,
        "formal": formal,
        "formal_seeds": formal_seeds,
        "eta_seconds": eta_seconds,
        "has_failure": has_failure,
        "errors": list(dict.fromkeys(errors)),
        "scientific_status": scientific_status,
    }
    if include_system:
        snapshot["system"] = _system_metrics(workspace)
    return snapshot


def _duration(seconds: object) -> str:
    if seconds is None:
        return "pending throughput"
    value = max(0, int(float(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_progress(snapshot: Mapping[str, Any]) -> str:
    lines = [
        "PPTT V8 TransUNet mechanism replication",
        f"phase: {snapshot['phase']}",
        (
            "candidate: "
            f"{snapshot['candidate']['status']}"
            + (
                f" / {snapshot['candidate']['path_id']}"
                if snapshot["candidate"].get("path_id")
                else ""
            )
        ),
        (
            f"validation: {snapshot['validation']['completed']}/"
            f"{snapshot['validation']['total']}"
        ),
        f"formal test: {snapshot['formal']['completed']}/{snapshot['formal']['total']}",
    ]
    for row in snapshot["formal_seeds"]:
        lines.append(
            f"  seed {row['seed']}: {row['completed']}/{row['total']} "
            f"status={row['status']} last={row['last_patient_id'] or '-'} "
            f"eta={_duration(row.get('eta_seconds'))}"
        )
    lines.append(f"parallel ETA: {_duration(snapshot.get('eta_seconds'))}")
    if snapshot.get("scientific_status"):
        lines.append(f"scientific status: {snapshot['scientific_status']}")
    system = snapshot.get("system")
    if system:
        lines.append(
            f"disk free: {system['disk_free_gib']:.1f}/"
            f"{system['disk_total_gib']:.1f} GiB"
        )
        for gpu in system.get("gpu", []):
            lines.append(f"GPU: {gpu}")
    if snapshot["errors"]:
        lines.append("errors:")
        lines.extend(f"  - {value}" for value in snapshot["errors"])
    else:
        lines.append("errors: none")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only progress monitor for V8 TransUNet replication."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    while True:
        snapshot = collect_v8_progress(
            args.workspace_root,
            config_path=args.config,
            include_system=True,
        )
        rendered = (
            json.dumps(snapshot, ensure_ascii=False, indent=2)
            if args.json
            else format_progress(snapshot)
        )
        if args.watch:
            print("\033[2J\033[H", end="")
        print(rendered, flush=True)
        if not args.watch:
            return 1 if snapshot["has_failure"] else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
