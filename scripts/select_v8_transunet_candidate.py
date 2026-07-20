from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pptt.interventions.transunet_paths import (
    declared_transunet_decoder_paths,
)
from pptt.io.artifacts import load_case_trace
from pptt.lineage.direct_paths import (
    class_balanced_direct_net_recovery,
    output_anchored_direct_path_cohorts,
)
from pptt.statistics.mechanism_replication import (
    select_registered_candidate,
    summarize_validation_candidates,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            _json_ready(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _trace_directory(
    workspace: Path,
    config: dict[str, Any],
    *,
    model_seed: int,
) -> Path:
    return (
        workspace
        / str(config["validation"]["worker_root"])
        / f"seed_{int(model_seed)}"
        / str(config["model"])
        / f"seed_{int(model_seed)}"
        / str(config["discovery_split"])
        / "case_traces"
    )


def _patient_rows(workspace: Path, config: dict[str, Any]) -> pd.DataFrame:
    paths = declared_transunet_decoder_paths()
    expected_ids = tuple(str(value) for value in config["candidate_paths"])
    if tuple(path.path_id for path in paths) != expected_ids:
        raise ValueError("configured candidate paths differ from the declared graph")
    expected_patients = int(
        config["validation"]["expected_patient_count_per_seed"]
    )
    minimum_slice_pixels = int(
        config["eligibility"]["minimum_target_pixels_on_selected_slice"]
    )
    truth_classes = tuple(int(value) for value in config["truth_classes"])
    rows: list[dict[str, Any]] = []
    for model_seed in (int(value) for value in config["model_seeds"]):
        directory = _trace_directory(workspace, config, model_seed=model_seed)
        trace_paths = sorted(directory.glob("*.npz"))
        if len(trace_paths) != expected_patients:
            raise ValueError(
                f"seed {model_seed} has {len(trace_paths)} validation traces; "
                f"expected {expected_patients}"
            )
        for trace_path in trace_paths:
            trace = load_case_trace(trace_path)
            if (
                trace.truth is None
                or trace.final_model_state is None
                or trace.slice_ids is None
            ):
                raise ValueError(f"validation trace lacks required fields: {trace_path}")
            for path in paths:
                cohorts = output_anchored_direct_path_cohorts(
                    trace.states,
                    trace.truth,
                    trace.reliable,
                    trace.final_model_state,
                    source_index=path.source_index,
                    receiver_index=path.receiver_index,
                    truth_classes=truth_classes,
                )
                slice_counts = cohorts.correction.sum(axis=(1, 2), dtype=np.int64)
                selected_slice_count = int(slice_counts.max(initial=0))
                rows.append(
                    {
                        "path_id": path.path_id,
                        "topology_order": path.topology_order,
                        "source_node": path.source_node,
                        "receiver_node": path.receiver_node,
                        "source_index": path.source_index,
                        "receiver_index": path.receiver_index,
                        "model_seed": model_seed,
                        "patient_id": trace_path.stem,
                        "net_recovery": class_balanced_direct_net_recovery(
                            cohorts,
                            trace.truth,
                            truth_classes=truth_classes,
                        ),
                        "selected_slice_target_pixel_count": selected_slice_count,
                        "eligible": selected_slice_count >= minimum_slice_pixels,
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select at most one TransUNet mechanism candidate using validation "
            "PPTT traces only."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/v8_transunet_mechanism_replication.yaml"
        ),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    config = _load_yaml(args.config.resolve())
    if str(config["discovery_split"]) != "val":
        raise ValueError("V8 candidate discovery is locked to the validation split")
    if str(config["intervention_split"]) != "test":
        raise ValueError("V8 formal intervention is locked to the test split")
    patient = _patient_rows(workspace, config)
    statistics, coverage = summarize_validation_candidates(
        patient,
        required_seeds=tuple(int(value) for value in config["model_seeds"]),
        bootstrap_iterations=int(config["statistics"]["bootstrap_iterations"]),
        bootstrap_seed=int(config["statistics"]["bootstrap_seed"]),
    )
    registration = select_registered_candidate(
        statistics,
        coverage,
        required_seeds=tuple(int(value) for value in config["model_seeds"]),
        minimum_supported_seeds=int(
            config["statistics"]["minimum_supported_seeds"]
        ),
        minimum_patients_per_seed=int(
            config["eligibility"]["minimum_candidate_patients_per_seed"]
        ),
        minimum_target_pixels_per_seed=int(
            config["eligibility"]["minimum_candidate_target_pixels_per_seed"]
        ),
        alpha=float(config["statistics"]["alpha"]),
        minimum_standardized_effect=float(
            config["statistics"]["minimum_standardized_effect"]
        ),
    )
    path_by_id = {
        path.path_id: path.to_dict()
        for path in declared_transunet_decoder_paths()
    }
    if registration["candidate"] is not None:
        selected_id = str(registration["candidate"]["path_id"])
        registration["candidate"]["path"] = path_by_id[selected_id]
    registration.update(
        {
            "discovery_split": "val",
            "formal_intervention_split": "test",
            "declared_paths": list(path_by_id.values()),
            "validation_patient_row_count": int(len(patient)),
            "test_data_read_during_selection": False,
        }
    )
    output_root = workspace / str(config["validation"]["candidate_root"])
    _write_parquet_atomic(
        output_root / "validation_candidate_patient_scores.parquet",
        patient,
    )
    _write_parquet_atomic(
        output_root / "validation_candidate_statistics.parquet",
        statistics,
    )
    _write_parquet_atomic(
        output_root / "validation_candidate_coverage.parquet",
        coverage,
    )
    _write_json_atomic(output_root / "candidate_registration.json", registration)
    print(json.dumps(_json_ready(registration), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
