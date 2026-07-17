from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pptt.data.brats2d import ensure_chw, normalize_label
from pptt.io.artifacts import load_case_trace
from pptt.lineage.cohorts import output_anchored_persistent_correction_cohorts
from pptt.visualization.pixel_fate import (
    render_pixel_fate_figure,
    select_median_process_case,
)


NODES = ("down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4")
VECTOR_COLUMNS = tuple(f"persistent_net_t{index}_difference" for index in range(7))


def _load_trace(
    workspace: Path,
    *,
    model: str,
    model_seed: int,
    patient_id: str,
):
    path = (
        workspace
        / "results"
        / "v3_unet_pair"
        / model
        / f"seed_{model_seed}"
        / "test"
        / "case_traces"
        / f"{patient_id}.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_case_trace(path), path


def _representative_patient(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    selected = frame[frame["split"] == "test"].copy()
    if selected.empty:
        raise ValueError("process table has no test observations")
    rows = []
    for patient_id, group in selected.groupby("patient_id", sort=True):
        reference_events = group[
            [f"persistent_net_t{index}_reference" for index in range(7)]
        ].abs()
        comparator_events = group[
            [f"persistent_net_t{index}_comparator" for index in range(7)]
        ].abs()
        event_count = int(
            ((reference_events.to_numpy() + comparator_events.to_numpy()) > 0)
            .any(axis=0)
            .sum()
        )
        rows.append(
            {
                "patient_id": str(patient_id),
                "reference_tumor_pixels": int(group["lesion_area_reference"].min()),
                "comparator_tumor_pixels": int(group["lesion_area_comparator"].min()),
                "process_event_interval_count": event_count,
                **{
                    column: float(group[column].mean()) for column in VECTOR_COLUMNS
                },
            }
        )
    candidates = pd.DataFrame(rows)
    candidates = candidates[
        (candidates["reference_tumor_pixels"] >= 500)
        & (candidates["comparator_tumor_pixels"] >= 500)
        & (candidates["process_event_interval_count"] >= 4)
    ].copy()
    if candidates.empty:
        raise ValueError("no patient satisfies the locked representative-case criteria")
    chosen = select_median_process_case(candidates, vector_columns=VECTOR_COLUMNS)
    patient_rows = selected[selected["patient_id"] == chosen["patient_id"]].copy()
    patient_vector = np.asarray([chosen[column] for column in VECTOR_COLUMNS], dtype=np.float64)
    patient_rows["distance_to_patient_mean_l1"] = np.abs(
        patient_rows.loc[:, VECTOR_COLUMNS].to_numpy(dtype=np.float64)
        - patient_vector[None, :]
    ).sum(axis=1)
    seed_row = patient_rows.sort_values(
        ["distance_to_patient_mean_l1", "model_seed"], kind="mergesort"
    ).iloc[0]
    chosen["model_seed"] = int(seed_row["model_seed"])
    chosen["seed_selection_rule"] = "minimum_l1_distance_to_patient_three_seed_mean"
    chosen["eligible_patient_count"] = int(len(candidates))
    return chosen, seed_row.to_frame().T


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the locked all-node PPTT pixel-fate figure."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("/root/autodl-tmp/A_scheme_workspace/brats2023_data/processed_2d"),
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    process_root = workspace / "results" / "process_effect_gate"
    paired = pd.read_parquet(process_root / "patient_process_contrasts.parquet")
    effects = pd.read_parquet(process_root / "primary_process_statistics.parquet")
    representative, seed_frame = _representative_patient(paired)
    patient_id = str(representative["patient_id"])
    model_seed = int(representative["model_seed"])
    baseline, baseline_path = _load_trace(
        workspace,
        model="unet_baseline",
        model_seed=model_seed,
        patient_id=patient_id,
    )
    noskip, noskip_path = _load_trace(
        workspace,
        model="unet_noskip",
        model_seed=model_seed,
        patient_id=patient_id,
    )
    if (
        baseline.truth is None
        or noskip.truth is None
        or baseline.final_model_state is None
        or noskip.final_model_state is None
    ):
        raise ValueError("formal V3 traces must contain truth and native final output")
    if baseline.slice_ids != noskip.slice_ids or not np.array_equal(baseline.truth, noskip.truth):
        raise ValueError("paired traces do not share one patient grid")
    baseline_cohorts = output_anchored_persistent_correction_cohorts(
        baseline.states,
        baseline.truth,
        baseline.reliable,
        baseline.final_model_state,
        truth_classes=(1, 2, 3),
    )
    noskip_cohorts = output_anchored_persistent_correction_cohorts(
        noskip.states,
        noskip.truth,
        noskip.reliable,
        noskip.final_model_state,
        truth_classes=(1, 2, 3),
    )
    joint = baseline_cohorts | noskip_cohorts
    joint_counts = joint.any(axis=0).sum(axis=(1, 2), dtype=np.int64)
    slice_index = int(np.argmax(joint_counts))
    slice_id = str(baseline.slice_ids[slice_index])
    image_path = args.asset_root / "test_Image" / f"{slice_id}.npy"
    mask_path = args.asset_root / "test_Mask" / f"{slice_id}.npy"
    image = ensure_chw(np.load(image_path, allow_pickle=False))[0]
    truth = normalize_label(np.load(mask_path, allow_pickle=False))
    if not np.array_equal(truth, baseline.truth[slice_index]):
        raise ValueError("source mask and formal trace truth differ")
    output_root = (
        args.output_root
        if args.output_root is not None
        else workspace / "results" / "paper_outputs" / args.language
    ).resolve()
    output_base = output_root / "fig2_pptt_pixel_fate"
    seed_row = seed_frame.iloc[0]
    manifest = render_pixel_fate_figure(
        output_base=output_base,
        language=args.language,
        scatter=paired[paired["split"] == "test"],
        patient_id=patient_id,
        model_seed=model_seed,
        slice_id=slice_id,
        image=image,
        truth=truth,
        baseline_states=baseline.states[:, slice_index],
        noskip_states=noskip.states[:, slice_index],
        nodes=NODES,
        process_vector=[float(seed_row[column]) for column in VECTOR_COLUMNS],
        effect_statistics=effects,
        process_masks=joint[:, slice_index],
        dpi=args.dpi,
    )
    manifest.update(
        {
            "representative_selection": representative,
            "joint_process_slice_rule": "maximum_union_process_pixels_smallest_index_tie",
            "joint_process_pixel_count": int(joint_counts[slice_index]),
            "source_paths": {
                "baseline_trace": str(baseline_path),
                "noskip_trace": str(noskip_path),
                "image": str(image_path),
                "mask": str(mask_path),
            },
        }
    )
    output_base.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_base), "patient_id": patient_id, "slice_id": slice_id}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
