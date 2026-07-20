from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.monitor_v8_progress import collect_v8_progress, format_progress


SEEDS = (42, 123, 3407)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _workspace(tmp_path: Path) -> Path:
    config = {
        "model": "transunet_r50_vit_b16",
        "model_seeds": list(SEEDS),
        "discovery_split": "val",
        "validation": {
            "worker_root": "results/v8/validation_workers",
            "candidate_root": "results/v8/validation_candidate",
            "expected_patient_count_per_seed": 2,
        },
        "formal_output_root": "results/v8/formal_test",
        "protocol_lock": "results/v8/protocol_lock.json",
        "formal": {"expected_patient_count_per_seed": 3},
    }
    path = tmp_path / "configs/experiments/v8.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return tmp_path


def _validation_trace(workspace: Path, seed: int, patient: str) -> None:
    path = (
        workspace
        / "results/v8/validation_workers"
        / f"seed_{seed}"
        / "transunet_r50_vit_b16"
        / f"seed_{seed}"
        / "val/case_traces"
        / f"{patient}.npz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"trace")


def _formal_patient(workspace: Path, seed: int, patient: str) -> None:
    path = (
        workspace
        / "results/v8/formal_test/transunet_r50_vit_b16"
        / f"seed_{seed}/patient_results/{patient}.json"
    )
    _write_json(path, {"patient_id": patient, "status": "PASS"})


def test_progress_summary_reports_phase_counts_and_dynamic_eta(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    config_path = workspace / "configs/experiments/v8.yaml"
    initial = collect_v8_progress(workspace, config_path=config_path)
    assert initial["phase"] == "NOT_STARTED"
    assert initial["validation"]["completed"] == 0

    for seed in SEEDS:
        _validation_trace(workspace, seed, "patient_a")
        _validation_trace(workspace, seed, "patient_b")
    temporary = (
        workspace
        / "results/v8/validation_workers/seed_42"
        / "transunet_r50_vit_b16/seed_42/val/case_traces/.patient_c.tmp.npz"
    )
    temporary.write_bytes(b"incomplete")
    _write_json(
        workspace / "results/v8/validation_candidate/candidate_registration.json",
        {
            "status": "REGISTERED_TRANSUNET_CANDIDATE",
            "candidate": {"path_id": "skip_down2_to_up2"},
            "test_intervention_authorized": True,
        },
    )
    _write_json(
        workspace / "results/v8/protocol_lock.json",
        {
            "status": "LOCKED_BEFORE_FIRST_TEST_INTERVENTION",
            "candidate": {"path_id": "skip_down2_to_up2"},
            "test_intervention_authorized": True,
        },
    )
    for seed in SEEDS:
        _formal_patient(workspace, seed, "patient_a")
        _write_json(
            workspace
            / "results/v8/formal_test/transunet_r50_vit_b16"
            / f"seed_{seed}/progress.json",
            {
                "status": "RUNNING",
                "completed_patients": 1,
                "total_patients": 3,
                "last_patient_id": "patient_a",
                "patients_per_second": 0.5,
                "eta_seconds": 4.0 + seed / 1000,
                "error": None,
            },
        )

    running = collect_v8_progress(workspace, config_path=config_path)
    assert running["phase"] == "FORMAL_INTERVENTION"
    assert running["candidate"]["path_id"] == "skip_down2_to_up2"
    assert running["validation"] == {"completed": 6, "total": 6}
    assert running["formal"]["completed"] == 3
    assert running["formal"]["total"] == 9
    assert running["eta_seconds"] > 4.0
    rendered = format_progress(running)
    assert "skip_down2_to_up2" in rendered
    assert "3/9" in rendered


def test_progress_summary_surfaces_failed_seed_without_restarting(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    config_path = workspace / "configs/experiments/v8.yaml"
    _write_json(
        workspace
        / "results/v8/formal_test/transunet_r50_vit_b16/seed_123/progress.json",
        {
            "status": "FAILED",
            "completed_patients": 1,
            "total_patients": 3,
            "last_patient_id": "patient_a",
            "patients_per_second": 0.1,
            "eta_seconds": None,
            "error": "RuntimeError: audit mismatch",
        },
    )
    summary = collect_v8_progress(workspace, config_path=config_path)
    assert summary["phase"] == "FAILED"
    assert summary["has_failure"] is True
    assert any("audit mismatch" in value for value in summary["errors"])
    assert summary["read_only"] is True


def test_progress_summary_handles_no_candidate_and_complete(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    config_path = workspace / "configs/experiments/v8.yaml"
    candidate_path = (
        workspace / "results/v8/validation_candidate/candidate_registration.json"
    )
    _write_json(
        candidate_path,
        {
            "status": "NO_REGISTERED_TRANSUNET_CANDIDATE",
            "candidate": None,
            "test_intervention_authorized": False,
        },
    )
    no_candidate = collect_v8_progress(workspace, config_path=config_path)
    assert no_candidate["phase"] == "TERMINATED_NO_CANDIDATE"
    assert no_candidate["candidate"]["status"] == "NO_REGISTERED_TRANSUNET_CANDIDATE"

    candidate_path.unlink()
    _write_json(
        workspace / "results/v8/formal_test/v8_status.json",
        {
            "execution_status": "PASS",
            "scientific_status": "INTERVENTIONALLY_FAITHFUL_TRANSUNET_REPLICATION",
        },
    )
    complete = collect_v8_progress(workspace, config_path=config_path)
    assert complete["phase"] == "COMPLETE"
    assert complete["scientific_status"] == (
        "INTERVENTIONALLY_FAITHFUL_TRANSUNET_REPLICATION"
    )
