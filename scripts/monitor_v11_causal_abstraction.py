#!/usr/bin/env python3
"""Read-only monitor for the locked V11 causal-abstraction jobs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG = Path("configs/experiments/v11_causal_abstraction.yaml")
DEFAULT_LOG_ROOT = Path("logs/v11_causal_abstraction/formal")
PROGRESS_PATTERN = re.compile(
    r"v11-progress model=(?P<model>\S+) seed=(?P<seed>\d+) "
    r"patient=(?P<patient>\S+) node=(?P<node>\S+) "
    r"condition=(?P<condition>\S+) status=(?P<status>\S+)"
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
        errors.append(f"unreadable JSON {path}: {type(error).__name__}: {error}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"JSON payload is not an object: {path}")
        return None
    return payload


def _completed_patient_artifacts(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.glob("*/patient_result.json")
        if not path.parent.name.startswith(".")
    )


def _recent_throughput_and_eta(
    artifacts: list[Path],
    *,
    expected: int,
    recent_window: int = 20,
) -> tuple[float | None, float | None]:
    if len(artifacts) < 2 or len(artifacts) >= expected:
        return None, 0.0 if len(artifacts) >= expected else None
    timestamps = sorted(path.stat().st_mtime for path in artifacts)[-recent_window:]
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        return None, None
    patients_per_second = (len(timestamps) - 1) / elapsed
    if patients_per_second <= 0:
        return None, None
    return patients_per_second, (expected - len(artifacts)) / patients_per_second


def _pid_is_alive(pid_path: Path) -> tuple[int | None, bool]:
    if not pid_path.is_file():
        return None, False
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return None, False
    return pid, True


def _parse_progress_log(path: Path) -> dict[str, Any]:
    completed = {"task": set(), "null": set()}
    current = {"patient": None, "node": None, "condition": None, "status": None}
    failures: list[str] = []
    if not path.is_file():
        return {
            "condition_counts": {"task": 0, "null": 0},
            "current": current,
            "failures": failures,
        }
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        return {
            "condition_counts": {"task": 0, "null": 0},
            "current": current,
            "failures": [f"log read failed: {type(error).__name__}: {error}"],
        }
    for line in lines:
        match = PROGRESS_PATTERN.search(line)
        if match:
            event = match.groupdict()
            current = {
                "patient": event["patient"],
                "node": None if event["node"] == "-" else event["node"],
                "condition": (
                    None if event["condition"] == "-" else event["condition"]
                ),
                "status": event["status"],
            }
            if event["status"] == "COMPLETE" and event["condition"] in completed:
                completed[event["condition"]].add(
                    (event["patient"], event["node"])
                )
        upper = line.upper()
        if "TRACEBACK (MOST RECENT CALL LAST)" in upper or re.search(
            r"(?:\bFAILED\b|\bFATAL\b|(?:ERROR|EXCEPTION)\s*:)", upper
        ):
            failures.append(line.strip()[:500])
    return {
        "condition_counts": {
            condition: len(values) for condition, values in completed.items()
        },
        "current": current,
        "failures": failures[-3:],
    }


def _job_progress(
    workspace: Path,
    configuration: Mapping[str, Any],
    *,
    model: str,
    model_seed: int,
    log_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    expected = int(configuration["coverage"]["expected_formal_patients"])
    job_root = (
        workspace
        / str(configuration["formal_job_root"])
        / model
        / f"seed_{model_seed}"
    )
    artifacts = _completed_patient_artifacts(job_root / "patient_results")
    completed = len(artifacts)
    if completed > expected:
        errors.append(f"{model}/seed_{model_seed}: patient count exceeds protocol")
    rate, eta = _recent_throughput_and_eta(artifacts, expected=expected)
    status_payload = _load_json(job_root / "job_status.json", errors) or {}
    log_path = log_root / model / f"seed_{model_seed}.log"
    parsed = _parse_progress_log(log_path)
    pid, alive = _pid_is_alive(log_root / model / f"seed_{model_seed}.pid")
    status_text = str(status_payload.get("status") or "").upper()
    if "FAIL" in status_text or "ERROR" in status_text or parsed["failures"]:
        status = "FAILED"
    elif status_text == "COMPLETE" and completed == expected:
        status = "COMPLETE"
    elif alive:
        status = "RUNNING"
    elif completed == expected:
        status = "AWAITING_AUDIT"
    elif completed:
        status = "STOPPED_PARTIAL"
    else:
        status = "PENDING"
    if status == "FAILED":
        errors.extend(
            f"{model}/seed_{model_seed}: {message}"
            for message in parsed["failures"]
        )
    return {
        "model": model,
        "seed": int(model_seed),
        "status": status,
        "completed_patients": completed,
        "expected_patients": expected,
        "current_patient": parsed["current"]["patient"],
        "current_node": parsed["current"]["node"],
        "current_condition": parsed["current"]["condition"],
        "task_conditions_complete": parsed["condition_counts"]["task"],
        "null_conditions_complete": parsed["condition_counts"]["null"],
        "patients_per_hour": rate * 3600.0 if rate is not None else None,
        "eta_seconds": eta,
        "pid": pid,
        "pid_alive": alive,
        "log": str(log_path),
        "job_root": str(job_root),
        "failures": parsed["failures"],
    }


def _system_metrics(workspace: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(workspace)
    output: dict[str, Any] = {
        "disk_free_gib": usage.free / (1024**3),
        "disk_total_gib": usage.total / (1024**3),
        "gpus": [],
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
        return output
    if completed.returncode != 0:
        return output
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        output["gpus"].append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mib": int(parts[2]),
                "memory_total_mib": int(parts[3]),
                "utilization_percent": int(parts[4]),
            }
        )
    return output


def collect_v11_progress(
    workspace_root: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    log_root: str | Path = DEFAULT_LOG_ROOT,
    include_system: bool = True,
) -> dict[str, Any]:
    """Collect one immutable progress snapshot without modifying any job."""
    workspace = Path(workspace_root).resolve()
    active_config = Path(config_path)
    if not active_config.is_absolute():
        active_config = workspace / active_config
    active_log_root = Path(log_root)
    if not active_log_root.is_absolute():
        active_log_root = workspace / active_log_root
    configuration = _load_yaml(active_config.resolve())
    errors: list[str] = []
    models = tuple(configuration["main_models"]) + tuple(
        configuration["control_models"]
    )
    seeds = tuple(int(value) for value in configuration["model_seeds"])
    jobs = [
        _job_progress(
            workspace,
            configuration,
            model=str(model),
            model_seed=seed,
            log_root=active_log_root.resolve(),
            errors=errors,
        )
        for model in models
        for seed in seeds
    ]
    snapshot = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "jobs": jobs,
        "completed_jobs": sum(row["status"] == "COMPLETE" for row in jobs),
        "running_jobs": sum(row["status"] == "RUNNING" for row in jobs),
        "failed_jobs": sum(row["status"] == "FAILED" for row in jobs),
        "completed_patients": sum(row["completed_patients"] for row in jobs),
        "expected_patients": sum(row["expected_patients"] for row in jobs),
        "errors": errors,
    }
    if include_system:
        snapshot["system"] = _system_metrics(workspace)
    return snapshot


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    value = max(0, int(round(seconds)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s"


def format_v11_progress(snapshot: Mapping[str, Any]) -> str:
    headers = (
        "job",
        "status",
        "patients",
        "current",
        "task/null",
        "patients/h",
        "ETA",
    )
    rows = []
    for job in snapshot["jobs"]:
        current = "/".join(
            str(value)
            for value in (
                job.get("current_patient"),
                job.get("current_node"),
                job.get("current_condition"),
            )
            if value
        ) or "-"
        rate = job.get("patients_per_hour")
        rows.append(
            (
                f"{job['model']}/s{job['seed']}",
                str(job["status"]),
                f"{job['completed_patients']}/{job['expected_patients']}",
                current,
                f"{job['task_conditions_complete']}/{job['null_conditions_complete']}",
                "-" if rate is None else f"{rate:.2f}",
                _format_duration(job.get("eta_seconds")),
            )
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    output = [
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    output.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    output.append(
        f"jobs complete/running/failed: {snapshot['completed_jobs']}/"
        f"{snapshot['running_jobs']}/{snapshot['failed_jobs']} | patients: "
        f"{snapshot['completed_patients']}/{snapshot['expected_patients']}"
    )
    system = snapshot.get("system")
    if isinstance(system, Mapping):
        output.append(f"disk free: {float(system['disk_free_gib']):.1f} GiB")
        for gpu in system.get("gpus", []):
            output.append(
                f"GPU {gpu['index']} {gpu['name']}: {gpu['memory_used_mib']}/"
                f"{gpu['memory_total_mib']} MiB, util {gpu['utilization_percent']}%"
            )
    if snapshot.get("errors"):
        output.append("failures:")
        output.extend(f"  {message}" for message in snapshot["errors"])
    return "\n".join(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--watch-seconds", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-clear", action="store_true")
    args = parser.parse_args()
    while True:
        snapshot = collect_v11_progress(
            args.workspace_root,
            config_path=args.config,
            log_root=args.log_root,
            include_system=True,
        )
        rendered = (
            json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)
            if args.json
            else format_v11_progress(snapshot)
        )
        if args.watch_seconds > 0 and not args.no_clear:
            print("\033[2J\033[H", end="")
        print(rendered, flush=True)
        if args.watch_seconds <= 0:
            return 1 if snapshot["failed_jobs"] else 0
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
