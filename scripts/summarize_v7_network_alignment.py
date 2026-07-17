from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from pptt.statistics.network_alignment import (
    compute_global_alignment_statistics,
    evaluate_network_alignment_gate,
    summarize_alignment_cells,
)

try:
    from scripts.lock_v7_network_alignment_protocol import (
        load_and_validate_formal_lock,
        sha256_file,
        source_tree_sha256,
    )
except ModuleNotFoundError:
    from lock_v7_network_alignment_protocol import (
        load_and_validate_formal_lock,
        sha256_file,
        source_tree_sha256,
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


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            _json_ready(dict(payload)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary: Path | None = None
    try:
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
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_operator_audit(patient: Mapping[str, Any]) -> bool:
    if patient.get("status") == "INELIGIBLE_UNION_PIXELS":
        return True
    audit = patient.get("audit", {})
    return bool(
        audit.get("clean_final_state_mismatch_count") == 0
        and audit.get("channel_multiset_preserved") is True
        and audit.get("root_restore_pass") is True
        and audit.get("hook_count_before") == audit.get("hook_count_after")
        and np.isfinite(float(audit.get("root_restore_max_abs_logit_error", np.nan)))
    )


def _collect_formal_rows(
    output_root: Path,
    *,
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    patient_ids: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    operator_failures: list[str] = []
    for model in models:
        for seed in seeds:
            job_root = output_root / model / f"seed_{seed}"
            status_path = job_root / "job_status.json"
            manifest_path = job_root / "job_manifest.json"
            if not status_path.is_file() or not manifest_path.is_file():
                raise FileNotFoundError(f"formal V7 job is incomplete: {job_root}")
            status = _load_json(status_path)
            manifest = _load_json(manifest_path)
            if (
                status.get("status") != "COMPLETE"
                or status.get("execution_mode") != "formal"
                or int(status.get("patient_count", -1)) != 250
            ):
                raise ValueError(f"formal V7 job status is invalid: {job_root}")
            if manifest.get("patient_ids") != list(patient_ids):
                raise ValueError(f"formal V7 patient registry differs: {job_root}")
            patient_paths = sorted((job_root / "patient_results").glob("*.json"))
            if len(patient_paths) != 250 or {
                path.stem for path in patient_paths
            } != set(patient_ids):
                raise ValueError(f"formal V7 patient files are incomplete: {job_root}")
            jobs.append(
                {
                    "model": model,
                    "model_seed": seed,
                    "patient_count": len(patient_paths),
                    "status": status["status"],
                }
            )
            for patient_path in patient_paths:
                patient = _load_json(patient_path)
                patient_id = str(patient["patient_id"])
                cells = patient.get("matrix_cells")
                if not isinstance(cells, list) or len(cells) != 49:
                    raise ValueError(f"patient matrix is not 7x7: {patient_path}")
                identities = {
                    (int(cell["transition_index"]), str(cell["restore_node"]))
                    for cell in cells
                }
                if len(identities) != 49:
                    raise ValueError(f"patient matrix contains duplicate cells: {patient_path}")
                if not _validate_operator_audit(patient):
                    operator_failures.append(f"{model}/seed_{seed}/{patient_id}")
                for cell in cells:
                    if (
                        cell["status"] == "TARGET_ONLY_CONTROL_UNAVAILABLE"
                        and cell.get("specific_effect") is not None
                    ):
                        raise ValueError(
                            f"undefined specificity was encoded as a value: {patient_path}"
                        )
                    rows.append(
                        {
                            "model": model,
                            "model_seed": seed,
                            "patient_id": patient_id,
                            "selected_slice_id": patient.get("selected_slice_id"),
                            "union_pixel_count": int(patient.get("union_pixel_count", 0)),
                            **cell,
                        }
                    )
    frame = pd.DataFrame(rows)
    expected_rows = len(models) * len(seeds) * len(patient_ids) * 49
    if len(frame) != expected_rows:
        raise ValueError(f"formal V7 row count differs: {len(frame)} != {expected_rows}")
    return frame, {
        "jobs": jobs,
        "operator_failures": operator_failures,
        "operator_audit_pass": not operator_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize the locked V7 network-alignment matrix and gate claims."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/v7_network_process_intervention_alignment.yaml"
        ),
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path(
            "results/v7_network_process_intervention_alignment/v7_protocol_lock.json"
        ),
    )
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    config_path = args.config if args.config.is_absolute() else workspace / args.config
    lock_path = (
        args.protocol_lock
        if args.protocol_lock.is_absolute()
        else workspace / args.protocol_lock
    )
    config_path = config_path.resolve()
    lock_path = lock_path.resolve()
    config = _load_yaml(config_path)
    raw_lock = _load_json(lock_path)
    patient_ids = tuple(str(value) for value in raw_lock["registered_patient_ids"])
    lock = load_and_validate_formal_lock(
        lock_path,
        active_configuration_sha256=sha256_file(config_path),
        active_source_tree_sha256=source_tree_sha256(workspace),
        patient_ids=patient_ids,
    )
    output_root = workspace / str(config["output_root"])
    models = tuple(str(value) for value in config["models"])
    seeds = tuple(int(value) for value in config["model_seeds"])
    patient_cells, collection_audit = _collect_formal_rows(
        output_root,
        models=models,
        seeds=seeds,
        patient_ids=patient_ids,
    )
    eligibility = config["eligibility"]
    statistics = config["statistics"]
    cell_summary = summarize_alignment_cells(
        patient_cells,
        bootstrap_iterations=int(statistics["bootstrap_iterations"]),
        bootstrap_seed=int(statistics["bootstrap_seed"]),
        minimum_patients=int(eligibility["minimum_patients_per_transition"]),
        minimum_aggregate_pixels=int(
            eligibility["minimum_aggregate_pixels_per_transition"]
        ),
    )
    nodes = tuple(str(value) for value in config["nodes"])
    receiving = {
        f"{left}->{right}": right for left, right in zip(nodes[:-1], nodes[1:])
    }
    global_statistics = compute_global_alignment_statistics(
        patient_cells,
        receiving_node_by_transition=receiving,
        bootstrap_iterations=int(statistics["bootstrap_iterations"]),
        bootstrap_seed=int(statistics["bootstrap_seed"]),
    )
    patient_global = global_statistics.attrs.get("patient_values", pd.DataFrame())
    v0_path = workspace / "results" / "v0_math" / "v0_math_report.json"
    v0 = _load_json(v0_path)
    audit = {
        "mathematical_properties_pass": v0.get("passed") is True,
        "protocol_identity_pass": lock.get("formal_intervention_authorized") is True,
        "formal_job_matrix_pass": len(collection_audit["jobs"]) == 6,
        "operator_audit_pass": collection_audit["operator_audit_pass"],
    }
    gate = evaluate_network_alignment_gate(
        global_statistics,
        cell_summary,
        audit,
        minimum_supported_seeds=int(statistics["minimum_supported_seeds"]),
        minimum_evaluable_transitions=int(
            statistics["minimum_evaluable_transitions"]
        ),
        minimum_positive_receiving_transitions=int(
            statistics["minimum_positive_receiving_transitions"]
        ),
        alpha=float(statistics["alpha"]),
    )

    patient_cells.to_parquet(output_root / "network_alignment_patient_cells.parquet", index=False)
    cell_summary.to_parquet(output_root / "network_alignment_cell_summary.parquet", index=False)
    patient_global.to_parquet(output_root / "network_alignment_patient_global.parquet", index=False)
    global_statistics.to_parquet(
        output_root / "network_alignment_global_statistics.parquet", index=False
    )
    _write_json_atomic(output_root / "v7_conclusion_gate.json", gate)
    full_audit = {
        **audit,
        "protocol_lock_sha256": sha256_file(lock_path),
        "formal_row_count": int(len(patient_cells)),
        "job_count": len(collection_audit["jobs"]),
        "operator_failure_count": len(collection_audit["operator_failures"]),
        "operator_failures": collection_audit["operator_failures"],
    }
    _write_json_atomic(output_root / "v7_audit.json", full_audit)
    status = {
        "status": gate["status"],
        "formal_job_count": 6,
        "formal_patient_count_per_job": 250,
        "formal_matrix_row_count": int(len(patient_cells)),
        "network_wide_claim_authorized": gate["network_wide_claim_authorized"],
    }
    _write_json_atomic(output_root / "v7_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
