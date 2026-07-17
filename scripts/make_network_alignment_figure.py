from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from pptt.data.brats2d import discover_slice_records
from pptt.experiments.model_matrix import load_model_matrix
from pptt.interventions.network_runtime import run_network_alignment_forwards
from pptt.io.artifacts import load_case_trace
from pptt.lineage.cohorts import (
    output_anchored_persistent_correction_cohorts,
    select_process_slice,
)
from pptt.visualization.network_alignment import render_network_alignment_figure

try:
    from scripts.run_v7_network_alignment import (
        _cell_masks_and_metadata,
        _group_records,
        _job_for,
        _load_runtime,
        _load_slice,
        _registered_checkpoint,
        _state_from_logits,
    )
except ModuleNotFoundError:
    from run_v7_network_alignment import (
        _cell_masks_and_metadata,
        _group_records,
        _job_for,
        _load_runtime,
        _load_slice,
        _registered_checkpoint,
        _state_from_logits,
    )


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    return payload


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload


def _select_representative_condition(cells: pd.DataFrame) -> dict:
    required = {
        "model",
        "model_seed",
        "patient_id",
        "selected_slice_id",
        "transition_index",
        "transition",
        "restore_node",
        "is_receiving_node_cell",
        "status",
        "specific_effect",
    }
    missing = required - set(cells.columns)
    if missing:
        raise ValueError(f"patient cell table lacks columns: {sorted(missing)}")
    selected = cells[
        cells["is_receiving_node_cell"].astype(bool)
        & (cells["status"] == "EVALUABLE")
    ].copy()
    selected["specific_effect"] = pd.to_numeric(
        selected["specific_effect"], errors="coerce"
    )
    selected = selected[np.isfinite(selected["specific_effect"])]
    pivot = selected.pivot_table(
        index=["model", "model_seed", "patient_id"],
        columns="transition_index",
        values="specific_effect",
        aggfunc="first",
    ).reindex(columns=range(7))
    pivot = pivot[pivot.notna().sum(axis=1) >= 4]
    if pivot.empty:
        raise ValueError("no patient has at least four evaluable receiving-node intervals")
    medians = pivot.median(axis=0, skipna=True).to_numpy(dtype=np.float64)
    candidates = []
    for identity, row in pivot.iterrows():
        values = row.to_numpy(dtype=np.float64)
        finite = np.isfinite(values) & np.isfinite(medians)
        distance = float(np.abs(values[finite] - medians[finite]).mean())
        model, model_seed, patient_id = identity
        candidates.append(
            {
                "model": str(model),
                "model_seed": int(model_seed),
                "patient_id": str(patient_id),
                "distance_to_coordinate_median_l1": distance,
                "evaluable_transition_count": int(finite.sum()),
            }
        )
    chosen = sorted(
        candidates,
        key=lambda value: (
            value["distance_to_coordinate_median_l1"],
            value["model"],
            value["model_seed"],
            value["patient_id"],
        ),
    )[0]
    patient = selected[
        (selected["model"] == chosen["model"])
        & (selected["model_seed"].astype(int) == chosen["model_seed"])
        & (selected["patient_id"] == chosen["patient_id"])
    ].copy()
    patient_median = float(patient["specific_effect"].median())
    patient["distance_to_patient_effect_median"] = (
        patient["specific_effect"] - patient_median
    ).abs()
    condition = patient.sort_values(
        ["distance_to_patient_effect_median", "transition_index"],
        kind="mergesort",
    ).iloc[0]
    chosen.update(
        {
            "slice_id": str(condition["selected_slice_id"]),
            "transition_index": int(condition["transition_index"]),
            "transition": str(condition["transition"]),
            "restore_node": str(condition["restore_node"]),
            "specific_effect": float(condition["specific_effect"]),
            "selection_rule": "minimum_l1_distance_to_matrix_median_then_interval_median",
            "candidate_count": len(candidates),
        }
    )
    return chosen


def _reconstruct_example(
    workspace: Path,
    *,
    asset_root: Path,
    config: dict,
    condition: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray, dict]:
    matrix = load_model_matrix(
        workspace / str(config["model_matrix"]),
        workspace_root=workspace,
        asset_root_override=asset_root,
    )
    job = _job_for(
        matrix.jobs,
        model=str(condition["model"]),
        model_seed=int(condition["model_seed"]),
    )
    runtime = _load_runtime(job, device=device)
    lock = _load_json(
        workspace
        / "results"
        / "v7_network_process_intervention_alignment"
        / "v7_protocol_lock.json"
    )
    registered = _registered_checkpoint(
        lock, str(condition["model"]), int(condition["model_seed"])
    )
    if runtime.checkpoint_sha256 != registered:
        raise ValueError("active checkpoint differs from the locked V7 checkpoint")
    data_config = _load_yaml(workspace / "configs" / "data" / "brats2023_2d.yaml")
    split_config = data_config["splits"][str(config["primary_split"])]
    grouped = _group_records(
        discover_slice_records(
            matrix.asset_root / str(split_config["image_dir"]),
            matrix.asset_root / str(split_config["mask_dir"]),
        )
    )
    patient_id = str(condition["patient_id"])
    trace_path = (
        workspace
        / str(config["v3_root"])
        / str(condition["model"])
        / f"seed_{int(condition['model_seed'])}"
        / str(config["primary_split"])
        / "case_traces"
        / f"{patient_id}.npz"
    )
    trace = load_case_trace(trace_path)
    if trace.truth is None or trace.final_model_state is None:
        raise ValueError("formal trace lacks truth or native final output")
    cohorts_all = output_anchored_persistent_correction_cohorts(
        trace.states,
        trace.truth,
        trace.reliable,
        trace.final_model_state,
        truth_classes=tuple(int(value) for value in config["truth_classes"]),
    )
    selection = select_process_slice(cohorts_all)
    records = grouped[patient_id]
    record = records[selection.slice_index]
    if record.slice_id != str(condition["slice_id"]):
        raise ValueError("reconstructed slice differs from the formal patient-cell row")
    image_tensor, truth = _load_slice(record)
    image_tensor = image_tensor.to(device)
    with torch.inference_mode():
        clean_trace = runtime.adapter.trace(image_tensor)
    clean_state = clean_trace.logits.argmax(dim=1)[0].detach().cpu().numpy()
    if not np.array_equal(clean_state, trace.final_model_state[selection.slice_index]):
        raise ValueError("reconstructed clean output differs from the formal V3 output")
    cohorts = cohorts_all[:, selection.slice_index]
    nodes = tuple(str(value) for value in config["nodes"])
    masks, metadata = _cell_masks_and_metadata(
        clean_activations=clean_trace.activations,
        cohorts=cohorts,
        states=trace.states[:, selection.slice_index],
        truth=truth,
        reliable=trace.reliable[:, selection.slice_index],
        final_model_state=trace.final_model_state[selection.slice_index],
        nodes=nodes,
        config=config,
    )
    forward_set = run_network_alignment_forwards(
        runtime.adapter,
        image_tensor,
        root_node=str(config["intervention"]["root_node"]),
        restore_nodes=tuple(str(value) for value in config["intervention"]["restore_nodes"]),
        masks=masks,
        input_equivalent_shift_yx=tuple(
            int(value) for value in config["intervention"]["input_equivalent_shift_yx"]
        ),
        logit_tolerance=float(config["intervention"]["logit_tolerance"]),
        clean_trace=clean_trace,
    )
    transition_index = int(condition["transition_index"])
    restore_node = str(condition["restore_node"])
    key = f"{restore_node}/transition_{transition_index}"
    control_logits = forward_set.control_restored_logits.get(key)
    if key not in forward_set.target_restored_logits or control_logits is None:
        raise ValueError("representative condition lacks target or matched-control output")
    states = {
        "clean": _state_from_logits(forward_set.clean_logits),
        "corrupt": _state_from_logits(forward_set.corrupt_logits),
        "restore_target": _state_from_logits(forward_set.target_restored_logits[key]),
        "restore_control": _state_from_logits(control_logits),
    }
    image = image_tensor[0, 0].detach().cpu().numpy()
    condition["target_pixel_count"] = int(cohorts[transition_index].sum())
    condition["target_feature_count"] = int(
        metadata[(transition_index, restore_node)]["target_feature_count"]
    )
    return image, truth, states, cohorts[transition_index], condition


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the locked network-wide process-intervention alignment figure."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("/root/autodl-tmp/A_scheme_workspace/brats2023_data"),
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    config = _load_yaml(
        workspace
        / "configs"
        / "experiments"
        / "v7_network_process_intervention_alignment.yaml"
    )
    result_root = workspace / str(config["output_root"])
    cell_summary = pd.read_parquet(result_root / "network_alignment_cell_summary.parquet")
    patient_global = pd.read_parquet(result_root / "network_alignment_patient_global.parquet")
    patient_cells = pd.read_parquet(result_root / "network_alignment_patient_cells.parquet")
    condition = _select_representative_condition(patient_cells)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    image, truth, states, target_mask, condition = _reconstruct_example(
        workspace,
        asset_root=args.asset_root,
        config=config,
        condition=condition,
        device=device,
    )
    output_root = (
        args.output_root
        if args.output_root is not None
        else workspace / "results" / "paper_outputs" / args.language
    ).resolve()
    output_base = output_root / "fig3_network_process_intervention_alignment"
    nodes = tuple(str(value) for value in config["nodes"])
    transitions = tuple(
        f"{left}->{right}" for left, right in zip(nodes[:-1], nodes[1:], strict=True)
    )
    manifest = render_network_alignment_figure(
        output_base=output_base,
        language=args.language,
        cell_summary=cell_summary,
        patient_global=patient_global,
        transitions=transitions,
        restore_nodes=tuple(str(value) for value in config["intervention"]["restore_nodes"]),
        example_image=image,
        example_truth=truth,
        example_states=states,
        example_target_mask=target_mask,
        example_identity=condition,
        dpi=args.dpi,
        bootstrap_iterations=int(config["statistics"]["bootstrap_iterations"]),
        bootstrap_seed=int(config["statistics"]["bootstrap_seed"]),
    )
    gate = _load_json(result_root / "v7_conclusion_gate.json")
    manifest["registered_conclusion_status"] = gate["status"]
    manifest["network_wide_claim_authorized"] = bool(
        gate["network_wide_claim_authorized"]
    )
    output_base.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_base), "example": condition}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
