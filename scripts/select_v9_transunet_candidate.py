from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
import yaml

from pptt.interventions.transunet_paths import declared_transunet_decoder_paths
from pptt.statistics.mechanism_replication import (
    select_feasible_registered_candidate,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
    temporary.replace(path)


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select one TransUNet path after validation-only process and "
            "matching-feasibility gates."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/v9_transunet_specificity.yaml"),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    config_path = args.config if args.config.is_absolute() else workspace / args.config
    config = _load_yaml(config_path.resolve())
    seeds = tuple(int(value) for value in config["model_seeds"])
    process_root = workspace / str(config["validation"]["process_candidate_root"])
    process_registration = _load_json(process_root / "candidate_registration.json")
    if process_registration.get("discovery_split") != "val" or bool(
        process_registration.get("test_data_read_during_selection", True)
    ):
        raise ValueError("V8 process candidate assets are not validation-only")

    patient_frames = []
    summary_frames = []
    for seed in seeds:
        seed_root = (
            workspace
            / str(config["validation"]["feasibility_root"])
            / f"seed_{seed}"
        )
        status = _load_json(seed_root / "feasibility_status.json")
        if status.get("status") != "COMPLETE_VALIDATION_FEASIBILITY":
            raise ValueError(f"validation feasibility is incomplete for seed {seed}")
        if status.get("test_data_read") is not False:
            raise ValueError(f"validation feasibility read test data for seed {seed}")
        expected = int(config["validation"]["expected_patient_count_per_seed"])
        if int(status.get("patient_count", -1)) != expected:
            raise ValueError(f"validation patient count differs for seed {seed}")
        patient_frames.append(pd.read_parquet(seed_root / "patient_feasibility.parquet"))
        summary_frames.append(pd.read_parquet(seed_root / "feasibility_summary.parquet"))
    patients = pd.concat(patient_frames, ignore_index=True)
    feasibility = pd.concat(summary_frames, ignore_index=True)
    registration = select_feasible_registered_candidate(
        process_registration,
        feasibility,
        required_seeds=seeds,
        minimum_feasible_patients_per_seed=int(
            config["eligibility"]["minimum_feasible_patients_per_seed"]
        ),
    )
    paths = {path.path_id: path.to_dict() for path in declared_transunet_decoder_paths()}
    if tuple(paths) != tuple(config["candidate_paths"]):
        raise ValueError("V9 candidate paths differ from the declared graph")
    if registration["candidate"] is not None:
        selected_id = str(registration["candidate"]["path_id"])
        registration["candidate"]["path"] = paths[selected_id]
    registration.update(
        {
            "discovery_split": "val",
            "formal_intervention_split": "test",
            "declared_paths": list(paths.values()),
            "test_data_read_during_selection": False,
            "v8_result_modified": False,
            "statistical_thresholds": dict(config["statistics"]),
        }
    )
    output = workspace / str(config["validation"]["candidate_root"])
    _write_parquet_atomic(output / "validation_feasibility_patient_rows.parquet", patients)
    _write_parquet_atomic(output / "validation_feasibility_summary.parquet", feasibility)
    _write_json_atomic(output / "candidate_registration.json", registration)
    print(json.dumps(registration, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
