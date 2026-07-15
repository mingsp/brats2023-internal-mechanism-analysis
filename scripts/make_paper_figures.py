from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pptt.io.artifacts import load_case_trace
from pptt.statistics.patient import select_representative_case
from pptt.transitions.fields import TransitionEvent, classify_transition_events
from pptt.visualization.common import configure_figure_style
from pptt.visualization.controls import (
    render_intervention_dose,
    render_observer_selectivity,
    render_synthetic_region_comparison,
    render_transfer_admission,
)
from pptt.visualization.flow_curves import render_flow_and_reconstruction
from pptt.visualization.lineage_maps import render_lineage_depth
from pptt.visualization.transition_maps import render_transition_map


NODES = ("down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4")


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_ready(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _case_selection_frame(
    node_frame: pd.DataFrame,
    transition_frame: pd.DataFrame,
    *,
    model: str,
    model_seed: int,
    split: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    nodes = node_frame[
        (node_frame.model == model)
        & (node_frame.model_seed == model_seed)
        & (node_frame.split == split)
        & (node_frame.node == "up4")
        & node_frame.class_index.isin([1, 2, 3])
    ]
    if nodes.empty:
        raise ValueError("no node rows are available for representative-case selection")
    area = nodes.groupby("patient_id").truth_pixel_count.sum().rename("lesion_area")
    presence = (
        nodes.pivot_table(
            index="patient_id",
            columns="class_index",
            values="class_available",
            aggfunc="max",
        )
        .reindex(columns=[1, 2, 3], fill_value=False)
        .rename(
            columns={
                1: "class_1_present",
                2: "class_2_present",
                3: "class_3_present",
            }
        )
    )
    transitions = transition_frame[
        (transition_frame.model == model)
        & (transition_frame.model_seed == model_seed)
        & (transition_frame.split == split)
    ]
    reliability = transitions.groupby("patient_id").reliable_fraction.mean().rename(
        "reliable_fraction"
    )
    vectors = transitions.pivot_table(
        index="patient_id",
        columns="transition",
        values="persistent_net_rate",
        aggfunc="mean",
    )
    vector_columns = tuple(f"net_{name}" for name in vectors.columns)
    vectors.columns = list(vector_columns)
    frame = pd.concat([area, reliability, presence, vectors], axis=1).dropna()
    return frame.reset_index(), vector_columns


def _record_pair(
    records: list[dict[str, Any]],
    kind: str,
    paths: tuple[Path, Path],
    source: str,
) -> None:
    records.append(
        {
            "kind": kind,
            "png": paths[0],
            "pdf": paths[1],
            "png_bytes": paths[0].stat().st_size,
            "pdf_bytes": paths[1].stat().st_size,
            "source": source,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate English or Chinese PPTT paper figures from result tables."
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--v1-root", type=Path)
    parser.add_argument("--v2-root", type=Path)
    parser.add_argument("--v3-root", type=Path)
    parser.add_argument("--v4-root", type=Path)
    parser.add_argument("--v5-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    parser.add_argument("--model", default="unet_baseline")
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--split", default="test")
    parser.add_argument("--area-quantiles", type=float, nargs=2, default=(0.25, 0.75))
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    v1_root = (args.v1_root or results_root / "v1_observers").resolve()
    v2_root = (args.v2_root or results_root / "v2_synthetic").resolve()
    v3_root = (args.v3_root or results_root / "v3_unet_pair").resolve()
    v4_root = (args.v4_root or results_root / "v4_up4_intervention").resolve()
    v5_root = (args.v5_root or results_root / "v5_transunet").resolve()
    output_root = (
        args.output_root or results_root / "paper_outputs" / args.language
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    font = configure_figure_style(args.language)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    node_path = v3_root / "node_state_by_patient.parquet"
    transition_path = v3_root / "transition_event_by_patient.parquet"
    lineage_path = v3_root / "lineage_depth_summary.parquet"
    for required in (node_path, transition_path, lineage_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    node_frame = pd.read_parquet(node_path)
    transition_frame = pd.read_parquet(transition_path)
    lineage_frame = pd.read_parquet(lineage_path)
    case_frame, vector_columns = _case_selection_frame(
        node_frame,
        transition_frame,
        model=args.model,
        model_seed=args.model_seed,
        split=args.split,
    )
    selection = select_representative_case(
        case_frame,
        vector_columns=vector_columns,
        area_quantiles=tuple(args.area_quantiles),
    )
    _write_json(
        output_root / "case_selection.json",
        {
            **asdict(selection),
            "model": args.model,
            "model_seed": args.model_seed,
            "split": args.split,
            "area_quantiles": args.area_quantiles,
            "vector_columns": vector_columns,
        },
    )
    trace_path = (
        v3_root
        / args.model
        / f"seed_{args.model_seed}"
        / args.split
        / "case_traces"
        / f"{selection.representative_patient_id}.npz"
    )
    trace = load_case_trace(trace_path)
    if trace.truth is None:
        raise ValueError("selected V3 trace has no truth labels")
    for transition_index, (left, right) in enumerate(
        zip(NODES[:-1], NODES[1:], strict=True)
    ):
        events = classify_transition_events(
            trace.truth,
            trace.states[transition_index],
            trace.states[transition_index + 1],
            num_classes=4,
            reliable=trace.reliable[transition_index],
        )
        changing = np.isin(
            events,
            [
                int(TransitionEvent.CORRECTION),
                int(TransitionEvent.DESTRUCTION),
                int(TransitionEvent.WRONG_REENCODING),
            ],
        ).sum(axis=(1, 2))
        slice_index = int(np.argmax(changing))
        base = output_root / f"transition_{transition_index + 1}_{left}_to_{right}"
        paths = render_transition_map(
            events[slice_index],
            base,
            transition_label=f"{left} -> {right}",
            language=args.language,
            dpi=args.dpi,
        )
        _record_pair(
            records,
            "pixel_transition_map",
            paths,
            f"{trace_path}#slice={trace.slice_ids[slice_index]}",
        )

    selected_nodes = node_frame[
        (node_frame.split == args.split) & (node_frame.model_seed == args.model_seed)
    ]
    selected_transitions = transition_frame[
        (transition_frame.split == args.split)
        & (transition_frame.model_seed == args.model_seed)
    ]
    paths = render_flow_and_reconstruction(
        selected_nodes,
        selected_transitions,
        output_root / "persistent_flow_and_reconstructed_dice",
        nodes=NODES,
        language=args.language,
        dpi=args.dpi,
    )
    _record_pair(records, "flow_and_metric", paths, str(v3_root))

    selected_lineage = lineage_frame[
        (lineage_frame.model == args.model)
        & (lineage_frame.model_seed == args.model_seed)
        & (lineage_frame.split == args.split)
    ]
    for kind, short_name in (
        ("final_state_depth", "dF"),
        ("correct_state_depth", "dY"),
        ("terminal_error_origin_depth", "o"),
    ):
        paths = render_lineage_depth(
            selected_lineage,
            output_root / f"lineage_{short_name}",
            kind=kind,
            nodes=NODES,
            language=args.language,
            dpi=args.dpi,
        )
        _record_pair(records, f"lineage_{short_name}", paths, str(lineage_path))

    synthetic_path = v2_root / "v2_construct_validation.json"
    if synthetic_path.is_file():
        paths = render_synthetic_region_comparison(
            json.loads(synthetic_path.read_text(encoding="utf-8")),
            output_root / "synthetic_region_localization",
            language=args.language,
            dpi=args.dpi,
        )
        _record_pair(records, "synthetic_control", paths, str(synthetic_path))
    else:
        skipped.append({"kind": "synthetic_control", "reason": str(synthetic_path)})

    observer_path = (
        v1_root
        / args.model
        / f"seed_{args.model_seed}"
        / "observer_control_selectivity.parquet"
    )
    if observer_path.is_file():
        paths = render_observer_selectivity(
            pd.read_parquet(observer_path),
            output_root / "observer_selectivity_controls",
            language=args.language,
            dpi=args.dpi,
        )
        _record_pair(records, "observer_control", paths, str(observer_path))
    else:
        skipped.append({"kind": "observer_control", "reason": str(observer_path)})

    intervention_path = v4_root / "intervention_effects_by_patient.parquet"
    if intervention_path.is_file():
        paths = render_intervention_dose(
            pd.read_parquet(intervention_path),
            output_root / "intervention_dose_response",
            language=args.language,
            dpi=args.dpi,
        )
        _record_pair(records, "intervention", paths, str(intervention_path))
    else:
        skipped.append({"kind": "intervention", "reason": str(intervention_path)})

    transfer_path = v5_root / "observer_transfer_table.parquet"
    if transfer_path.is_file():
        transfer_frame = pd.read_parquet(transfer_path)
        if not transfer_frame.empty:
            paths = render_transfer_admission(
                transfer_frame,
                output_root / "cross_architecture_transfer",
                language=args.language,
                dpi=args.dpi,
            )
            _record_pair(records, "architecture_transfer", paths, str(transfer_path))
        else:
            skipped.append({"kind": "architecture_transfer", "reason": "empty table"})
    else:
        skipped.append({"kind": "architecture_transfer", "reason": str(transfer_path)})

    _write_json(
        output_root / "figure_manifest.json",
        {
            "language": args.language,
            "dpi": args.dpi,
            "font": font,
            "representative_case": selection.representative_patient_id,
            "figure_count": len(records),
            "figures": records,
            "skipped_optional_figures": skipped,
        },
    )
    print(f"Generated {len(records)} figure pairs under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
