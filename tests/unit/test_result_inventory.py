from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from pptt.reproducibility.inventory import (
    build_result_inventory,
    format_environment_lock,
    verify_result_inventory,
)


def _write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def test_inventory_records_configs_weights_tables_figures_and_commands(tmp_path: Path):
    workspace = tmp_path / "workspace"
    asset_root = tmp_path / "assets"
    results_root = workspace / "results"
    config = workspace / "configs" / "experiments" / "v0.yaml"
    _write_yaml(config, {"seed": 7, "output": "results/v0/report.json"})

    asset_manifest = asset_root / "manifests" / "assets.json"
    asset_manifest.parent.mkdir(parents=True)
    asset_manifest.write_text(
        json.dumps({"schema_version": 1, "splits": {"test": {"pair_count": 2}}}),
        encoding="utf-8",
    )
    weight = asset_root / "validation_checkpoints" / "model.pth"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"fixed-weight")
    _write_yaml(
        workspace / "manifests" / "model_matrix.yaml",
        {
            "jobs": [
                {
                    "model": "unet",
                    "seed": 42,
                    "checkpoint": {
                        "root": "asset",
                        "path": "validation_checkpoints/model.pth",
                    },
                },
                {
                    "model": "transunet",
                    "seed": 123,
                    "checkpoint": {
                        "root": "workspace",
                        "path": "results/training/missing.pth",
                    },
                },
            ]
        },
    )

    results_root.mkdir(parents=True)
    pd.DataFrame({"value": [1, 2, 3]}).to_parquet(results_root / "table.parquet")
    (results_root / "figure.png").write_bytes(b"png-placeholder")
    debug_dir = results_root / "v4_debug_smoke"
    debug_dir.mkdir()
    (debug_dir / "audit.csv").write_text("x\n1\n2\n", encoding="utf-8")

    inventory = build_result_inventory(
        workspace_root=workspace,
        asset_root=asset_root,
        results_root=results_root,
        run_commands={"V0": "python scripts/run_v0_math.py --config v0.yaml"},
        git_commit="0123456789abcdef",
        git_dirty=False,
        environment={"python": "3.10-test", "gpu_names": ["test-gpu"]},
        generated_at="2026-07-15T12:00:00+00:00",
    )

    assert inventory["code"] == {
        "commit": "0123456789abcdef",
        "dirty": False,
    }
    assert inventory["run_commands"]["V0"].startswith("python scripts/")
    assert inventory["data_manifest"]["available"] is True
    assert inventory["config_files"][0]["relative_path"] == "configs/experiments/v0.yaml"

    weights = {(row["model"], row["seed"]): row for row in inventory["weights"]}
    assert weights[("unet", 42)]["available"] is True
    assert len(weights[("unet", 42)]["sha256"]) == 64
    assert weights[("transunet", 123)]["available"] is False

    artifacts = {row["relative_path"]: row for row in inventory["result_artifacts"]}
    assert artifacts["table.parquet"]["row_count"] == 3
    assert artifacts["figure.png"]["artifact_kind"] == "figure"
    assert artifacts["figure.png"]["size_bytes"] == len(b"png-placeholder")
    assert artifacts["table.parquet"]["formal_eligible"] is True
    assert artifacts["v4_debug_smoke/audit.csv"]["formal_eligible"] is False


def test_inventory_verifier_detects_result_tampering(tmp_path: Path):
    workspace = tmp_path / "workspace"
    asset_root = tmp_path / "assets"
    results_root = workspace / "results"
    (workspace / "configs").mkdir(parents=True)
    (workspace / "manifests").mkdir(parents=True)
    _write_yaml(workspace / "manifests" / "model_matrix.yaml", {"jobs": []})
    (asset_root / "manifests").mkdir(parents=True)
    (asset_root / "manifests" / "assets.json").write_text("{}\n", encoding="utf-8")
    results_root.mkdir(parents=True)
    result = results_root / "report.json"
    result.write_text('{"passed": true}\n', encoding="utf-8")

    inventory = build_result_inventory(
        workspace_root=workspace,
        asset_root=asset_root,
        results_root=results_root,
        run_commands={},
        git_commit="abc",
        git_dirty=True,
        environment={},
        generated_at="2026-07-15T12:00:00+00:00",
    )
    assert verify_result_inventory(
        inventory,
        workspace_root=workspace,
        asset_root=asset_root,
        results_root=results_root,
    ) == []

    result.write_text('{"passed": false}\n', encoding="utf-8")
    problems = verify_result_inventory(
        inventory,
        workspace_root=workspace,
        asset_root=asset_root,
        results_root=results_root,
    )
    assert any("SHA-256 mismatch" in problem for problem in problems)


def test_environment_lock_has_stable_sections_and_sorted_packages():
    text = format_environment_lock(
        environment={
            "python": "3.10.16",
            "torch": "2.5.1",
            "cuda_runtime": "12.4",
            "gpu_names": ["GPU B", "GPU A"],
        },
        packages=["zeta==2", "alpha==1"],
    )

    assert "[runtime]" in text
    assert "python=3.10.16" in text
    assert "gpu_names=GPU B | GPU A" in text
    assert "[packages]" in text
    assert text.index("alpha==1") < text.index("zeta==2")
