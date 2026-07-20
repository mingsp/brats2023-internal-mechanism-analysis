from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.monitor_v11_causal_abstraction import (
    collect_v11_progress,
    format_v11_progress,
)


def _write_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "main_models": ["unet_baseline", "transunet_r50_vit_b16"],
                "control_models": ["unet_noskip"],
                "model_seeds": [42, 123, 3407],
                "formal_job_root": "results/v11/formal_jobs",
                "coverage": {"expected_formal_patients": 2},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_monitor_is_read_only_and_counts_unique_condition_events(tmp_path: Path):
    config = tmp_path / "v11.yaml"
    _write_config(config)
    job = tmp_path / "results/v11/formal_jobs/unet_baseline/seed_42"
    first = job / "patient_results/p1/patient_result.json"
    first.parent.mkdir(parents=True)
    first.write_text("{}", encoding="utf-8")
    (job / "job_status.json").write_text(
        json.dumps({"status": "RUNNING"}),
        encoding="utf-8",
    )
    log = tmp_path / "logs/unet_baseline/seed_42.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "\n".join(
            [
                "v11-progress model=unet_baseline seed=42 patient=p2 node=- condition=- status=STARTED",
                "v11-progress model=unet_baseline seed=42 patient=p2 node=down1 condition=task status=RUNNING",
                "v11-progress model=unet_baseline seed=42 patient=p2 node=down1 condition=task status=COMPLETE",
                "v11-progress model=unet_baseline seed=42 patient=p2 node=down1 condition=task status=COMPLETE",
                "v11-progress model=unet_baseline seed=42 patient=p2 node=down1 condition=null status=RUNNING",
            ]
        ),
        encoding="utf-8",
    )
    before = {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*")}

    snapshot = collect_v11_progress(
        tmp_path,
        config_path=config,
        log_root=tmp_path / "logs",
        include_system=False,
    )

    after = {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*")}
    assert before == after
    active = snapshot["jobs"][0]
    assert active["completed_patients"] == 1
    assert active["task_conditions_complete"] == 1
    assert active["null_conditions_complete"] == 0
    assert active["current_patient"] == "p2"
    assert active["current_node"] == "down1"
    assert active["current_condition"] == "null"
    assert len(snapshot["jobs"]) == 9
    assert "unet_baseline/s42" in format_v11_progress(snapshot)


def test_monitor_surfaces_log_failure_without_mutating_job(tmp_path: Path):
    config = tmp_path / "v11.yaml"
    _write_config(config)
    log = tmp_path / "logs/transunet_r50_vit_b16/seed_123.log"
    log.parent.mkdir(parents=True)
    log.write_text("RuntimeError: CUDA out of memory\n", encoding="utf-8")

    snapshot = collect_v11_progress(
        tmp_path,
        config_path=config,
        log_root=tmp_path / "logs",
        include_system=False,
    )

    failed = [job for job in snapshot["jobs"] if job["status"] == "FAILED"]
    assert [(job["model"], job["seed"]) for job in failed] == [
        ("transunet_r50_vit_b16", 123)
    ]
    assert snapshot["failed_jobs"] == 1
    assert snapshot["errors"]
