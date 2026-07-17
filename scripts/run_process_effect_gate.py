from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yaml

from pptt.experiments.pilot import stable_seed
from pptt.io.artifacts import load_case_trace
from pptt.statistics.hypotheses import paired_patient_statistics
from pptt.statistics.process_effects import (
    class_balanced_damage_rate,
    class_balanced_path_tv,
    class_balanced_persistent_net_by_transition,
    evaluate_causal_entry_gate,
    holm_adjust,
    select_robust_transition_candidate,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration is not a mapping: {path}")
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _transition_names(nodes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        f"{left}->{right}" for left, right in zip(nodes[:-1], nodes[1:], strict=True)
    )


def _configured_indices(
    configured: list[str],
    transitions: tuple[str, ...],
) -> tuple[int, ...]:
    names = tuple(str(value) for value in configured)
    missing = set(names) - set(transitions)
    if missing:
        raise ValueError(f"Unknown configured transitions: {sorted(missing)}")
    return tuple(transitions.index(name) for name in names)


def _load_transition_paths(
    frame: pd.DataFrame,
    *,
    models: tuple[str, str],
    model_seeds: tuple[int, ...],
    transition_count: int,
    class_count: int,
) -> dict[tuple[str, int, str, str], np.ndarray]:
    selected = frame[
        (frame.scope == "full")
        & frame.model.isin(models)
        & frame.model_seed.isin(model_seeds)
    ].copy()
    key_columns = ["model", "model_seed", "split", "patient_id"]
    index_columns = ["transition_index", "truth_class", "state_from", "state_to"]
    expected_rows = transition_count * class_count**3
    expected_indices = np.stack(
        np.meshgrid(
            np.arange(transition_count),
            np.arange(class_count),
            np.arange(class_count),
            np.arange(class_count),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 4)
    paths: dict[tuple[str, int, str, str], np.ndarray] = {}
    for raw_key, group in selected.groupby(key_columns, sort=True, observed=True):
        key = (str(raw_key[0]), int(raw_key[1]), str(raw_key[2]), str(raw_key[3]))
        ordered = group.sort_values(index_columns, kind="mergesort")
        if len(ordered) != expected_rows:
            raise ValueError(f"Incomplete transition path for {key}: {len(ordered)}")
        actual_indices = ordered[index_columns].to_numpy(dtype=np.int64)
        if not np.array_equal(actual_indices, expected_indices):
            raise ValueError(f"Transition tensor grid is not complete for {key}")
        paths[key] = ordered["count"].to_numpy(dtype=np.int64).reshape(
            transition_count,
            class_count,
            class_count,
            class_count,
        )
    if not paths:
        raise ValueError("No formal transition paths were loaded")
    return paths


def _endpoint_lookup(
    frame: pd.DataFrame,
    *,
    final_node: str,
    models: tuple[str, str],
    model_seeds: tuple[int, ...],
    truth_classes: tuple[int, ...],
) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    selected = frame[
        (frame.node == final_node)
        & frame.model.isin(models)
        & frame.model_seed.isin(model_seeds)
        & frame.class_index.isin((-1, *truth_classes))
    ].copy()
    lookup: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    key_columns = ["model", "model_seed", "split", "patient_id"]
    for raw_key, group in selected.groupby(key_columns, sort=True, observed=True):
        key = (str(raw_key[0]), int(raw_key[1]), str(raw_key[2]), str(raw_key[3]))
        macro = group[group.class_index == -1]
        if len(macro) != 1:
            raise ValueError(f"Missing unique endpoint macro row for {key}")
        row = macro.iloc[0]
        availability = {
            int(item.class_index): bool(item.class_available)
            for item in group.itertuples(index=False)
            if int(item.class_index) in truth_classes
        }
        lookup[key] = {
            "terminal_dice": float(row.dice),
            "lesion_area": int(row.truth_pixel_count),
            **{
                f"class_{class_index}_present": bool(
                    availability.get(class_index, False)
                )
                for class_index in truth_classes
            },
        }
    return lookup


def _model_process_frame(
    *,
    paths: dict[tuple[str, int, str, str], np.ndarray],
    endpoint_lookup: dict[tuple[str, int, str, str], dict[str, Any]],
    v3_root: Path,
    truth_classes: tuple[int, ...],
    middle_indices: tuple[int, ...],
    late_indices: tuple[int, ...],
    transition_count: int,
) -> pd.DataFrame:
    rows = []
    total = len(paths)
    for position, (key, transition_path) in enumerate(sorted(paths.items()), start=1):
        model, model_seed, split, patient_id = key
        endpoint = endpoint_lookup.get(key)
        if endpoint is None:
            raise ValueError(f"Missing endpoint state row for {key}")
        trace_path = (
            v3_root
            / model
            / f"seed_{model_seed}"
            / split
            / "case_traces"
            / f"{patient_id}.npz"
        )
        if not trace_path.is_file():
            raise FileNotFoundError(f"Missing formal case trace: {trace_path}")
        trace = load_case_trace(trace_path)
        if trace.truth is None:
            raise ValueError(f"Trace has no truth labels: {trace_path}")
        persistent = class_balanced_persistent_net_by_transition(
            trace.states,
            trace.truth,
            truth_classes=truth_classes,
        )
        if persistent.shape != (transition_count,):
            raise ValueError(f"Unexpected persistent path length for {key}")
        rows.append(
            {
                "model": model,
                "model_seed": model_seed,
                "split": split,
                "patient_id": patient_id,
                "middle_damage": class_balanced_damage_rate(
                    transition_path[list(middle_indices)],
                    truth_classes=truth_classes,
                ),
                "late_persistent_net_recovery": float(
                    persistent[list(late_indices)].sum()
                ),
                **{
                    f"persistent_net_t{index}": float(value)
                    for index, value in enumerate(persistent)
                },
                **endpoint,
            }
        )
        if position % 100 == 0 or position == total:
            print(f"process traces {position}/{total}", flush=True)
    return pd.DataFrame(rows)


def _pair_models(
    model_frame: pd.DataFrame,
    *,
    paths: dict[tuple[str, int, str, str], np.ndarray],
    reference_model: str,
    comparator_model: str,
    truth_classes: tuple[int, ...],
    transition_count: int,
) -> pd.DataFrame:
    identifiers = ["model_seed", "split", "patient_id"]
    reference = model_frame[model_frame.model == reference_model].drop(
        columns="model"
    )
    comparator = model_frame[model_frame.model == comparator_model].drop(
        columns="model"
    )
    paired = reference.merge(
        comparator,
        on=identifiers,
        suffixes=("_reference", "_comparator"),
        validate="one_to_one",
    )
    metric_names = (
        "middle_damage",
        "late_persistent_net_recovery",
        "terminal_dice",
    )
    for metric in metric_names:
        paired[f"{metric}_difference"] = (
            paired[f"{metric}_reference"] - paired[f"{metric}_comparator"]
        )
    for index in range(transition_count):
        paired[f"persistent_net_t{index}_difference"] = (
            paired[f"persistent_net_t{index}_reference"]
            - paired[f"persistent_net_t{index}_comparator"]
        )
    process_distances = []
    for row in paired.itertuples(index=False):
        reference_key = (
            reference_model,
            int(row.model_seed),
            str(row.split),
            str(row.patient_id),
        )
        comparator_key = (
            comparator_model,
            int(row.model_seed),
            str(row.split),
            str(row.patient_id),
        )
        process_distances.append(
            class_balanced_path_tv(
                paths[reference_key],
                paths[comparator_key],
                truth_classes=truth_classes,
            )
        )
    paired["full_path_transition_tv"] = process_distances
    paired["absolute_terminal_difference"] = paired[
        "terminal_dice_difference"
    ].abs()
    for class_index in truth_classes:
        left = f"class_{class_index}_present_reference"
        right = f"class_{class_index}_present_comparator"
        if not (paired[left] == paired[right]).all():
            raise ValueError("Paired models disagree on ground-truth class presence")
        paired[f"class_{class_index}_present"] = paired[left].astype(bool)
    if not (
        paired.lesion_area_reference == paired.lesion_area_comparator
    ).all():
        raise ValueError("Paired models disagree on lesion area")
    paired["lesion_area"] = paired.lesion_area_reference.astype(np.int64)
    return paired


def _primary_statistics(
    paired: pd.DataFrame,
    *,
    primary_split: str,
    model_seeds: tuple[int, ...],
    iterations: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    selected = paired[paired.split == primary_split].copy()
    metrics = (
        "middle_damage",
        "late_persistent_net_recovery",
        "terminal_dice",
    )
    rows = []
    for model_seed in model_seeds:
        seed_frame = selected[selected.model_seed == model_seed]
        for metric in metrics:
            result = paired_patient_statistics(
                seed_frame[f"{metric}_reference"].to_numpy(dtype=np.float64),
                seed_frame[f"{metric}_comparator"].to_numpy(dtype=np.float64),
                iterations=iterations,
                seed=stable_seed(bootstrap_seed, "process_gate", metric, model_seed),
            )
            rows.append(
                {
                    "scope": "model_seed",
                    "model_seed": int(model_seed),
                    "metric": metric,
                    **result.to_dict(),
                }
            )
    patient_average = selected.groupby("patient_id", as_index=False)[
        [
            f"{metric}_{side}"
            for metric in metrics
            for side in ("reference", "comparator")
        ]
    ].mean()
    for metric in metrics:
        result = paired_patient_statistics(
            patient_average[f"{metric}_reference"].to_numpy(dtype=np.float64),
            patient_average[f"{metric}_comparator"].to_numpy(dtype=np.float64),
            iterations=iterations,
            seed=stable_seed(bootstrap_seed, "process_gate", metric, "fixed_seeds"),
        )
        rows.append(
            {
                "scope": "fixed_seed_patient_average",
                "model_seed": -1,
                "metric": metric,
                **result.to_dict(),
            }
        )
    statistics = pd.DataFrame(rows)
    statistics["holm_p"] = holm_adjust(
        statistics.wilcoxon_p.to_numpy(dtype=np.float64)
    )
    statistics["multiplicity_family_size"] = len(statistics)
    return statistics


def _candidate_statistics(
    paired: pd.DataFrame,
    *,
    discovery_split: str,
    candidate_indices: tuple[int, ...],
    transitions: tuple[str, ...],
    model_seeds: tuple[int, ...],
    iterations: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    discovery = paired[paired.split == discovery_split]
    rows = []
    for transition_index in candidate_indices:
        metric = f"persistent_net_t{transition_index}"
        for model_seed in model_seeds:
            selected = discovery[discovery.model_seed == model_seed]
            result = paired_patient_statistics(
                selected[f"{metric}_reference"].to_numpy(dtype=np.float64),
                selected[f"{metric}_comparator"].to_numpy(dtype=np.float64),
                iterations=iterations,
                seed=stable_seed(
                    bootstrap_seed,
                    "candidate",
                    transition_index,
                    model_seed,
                ),
            )
            rows.append(
                {
                    "transition_index": transition_index,
                    "transition": transitions[transition_index],
                    "model_seed": model_seed,
                    **result.to_dict(),
                }
            )
    frame = pd.DataFrame(rows)
    frame["holm_p"] = holm_adjust(frame.wilcoxon_p.to_numpy(dtype=np.float64))
    return frame


def _process_relation(
    paired: pd.DataFrame,
    *,
    primary_split: str,
    model_seeds: tuple[int, ...],
) -> pd.DataFrame:
    selected = paired[paired.split == primary_split]
    rows = []
    for model_seed in model_seeds:
        current = selected[selected.model_seed == model_seed]
        result = spearmanr(
            current.absolute_terminal_difference,
            current.full_path_transition_tv,
        )
        rows.append(
            {
                "scope": "model_seed",
                "model_seed": model_seed,
                "patient_count": len(current),
                "spearman_rho": float(result.statistic),
                "spearman_p": float(result.pvalue),
            }
        )
    averaged = selected.groupby("patient_id", as_index=False)[
        ["absolute_terminal_difference", "full_path_transition_tv"]
    ].mean()
    result = spearmanr(
        averaged.absolute_terminal_difference,
        averaged.full_path_transition_tv,
    )
    rows.append(
        {
            "scope": "fixed_seed_patient_average",
            "model_seed": -1,
            "patient_count": len(averaged),
            "spearman_rho": float(result.statistic),
            "spearman_p": float(result.pvalue),
        }
    )
    return pd.DataFrame(rows)


def _representative_case(
    paired: pd.DataFrame,
    *,
    primary_split: str,
    truth_classes: tuple[int, ...],
    maximum_absolute_terminal_difference: float,
    area_quantiles: tuple[float, float],
) -> dict[str, Any]:
    selected = paired[paired.split == primary_split].copy()
    aggregations: dict[str, Any] = {
        "full_path_transition_tv": "mean",
        "terminal_dice_difference": "mean",
        "lesion_area": "first",
    }
    for class_index in truth_classes:
        aggregations[f"class_{class_index}_present"] = "all"
    averaged = selected.groupby("patient_id", as_index=False).agg(aggregations)
    low, high = np.quantile(averaged.lesion_area, area_quantiles)
    all_classes = averaged[
        [f"class_{class_index}_present" for class_index in truth_classes]
    ].all(axis=1)
    eligible = averaged[
        all_classes
        & averaged.lesion_area.between(low, high, inclusive="both")
        & (
            averaged.terminal_dice_difference.abs()
            <= maximum_absolute_terminal_difference
        )
    ].copy()
    fallback_used = False
    if eligible.empty:
        fallback_used = True
        eligible = averaged[
            all_classes
            & averaged.lesion_area.between(low, high, inclusive="both")
        ].copy()
    if eligible.empty:
        raise ValueError("No representative patient satisfies class and area rules")
    chosen = eligible.sort_values(
        ["full_path_transition_tv", "patient_id"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    patient_rows = selected[selected.patient_id == chosen.patient_id].copy()
    patient_rows["distance_to_patient_mean"] = (
        patient_rows.full_path_transition_tv - chosen.full_path_transition_tv
    ).abs()
    chosen_seed = patient_rows.sort_values(
        ["distance_to_patient_mean", "model_seed"],
        kind="mergesort",
    ).iloc[0]
    return {
        "patient_id": str(chosen.patient_id),
        "model_seed": int(chosen_seed.model_seed),
        "full_path_transition_tv": float(chosen.full_path_transition_tv),
        "terminal_dice_difference": float(chosen.terminal_dice_difference),
        "lesion_area": int(chosen.lesion_area),
        "maximum_absolute_terminal_difference": float(
            maximum_absolute_terminal_difference
        ),
        "lesion_area_interval": [float(low), float(high)],
        "fallback_used": fallback_used,
        "selection_role": "descriptive_visualization_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute the formal PPTT process-effect gate and causal candidate."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/process_effect_gate.yaml"),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--v3-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    config = _load_yaml(args.config)
    v3_root = (
        args.v3_root
        if args.v3_root is not None
        else workspace / str(config["v3_root"])
    ).resolve()
    output_root = (
        args.output_root
        if args.output_root is not None
        else workspace / str(config["output_root"])
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    reference_model = str(config["models"]["reference"])
    comparator_model = str(config["models"]["comparator"])
    models = (reference_model, comparator_model)
    model_seeds = tuple(int(value) for value in config["model_seeds"])
    nodes = tuple(str(value) for value in config["nodes"])
    transitions = _transition_names(nodes)
    truth_classes = tuple(int(value) for value in config["truth_classes"])
    class_count = max(truth_classes) + 1
    middle_indices = _configured_indices(
        config["middle_damage_transitions"], transitions
    )
    late_indices = _configured_indices(
        config["late_persistent_transitions"], transitions
    )
    candidate_indices = _configured_indices(
        config["candidate_decoder_transitions"], transitions
    )
    statistics_config = config["statistics"]
    iterations = int(statistics_config["bootstrap_iterations"])
    bootstrap_seed = int(statistics_config["bootstrap_seed"])

    tensor_frame = pd.read_parquet(v3_root / "transition_tensor_by_patient.parquet")
    node_frame = pd.read_parquet(v3_root / "node_state_by_patient.parquet")
    paths = _load_transition_paths(
        tensor_frame,
        models=models,
        model_seeds=model_seeds,
        transition_count=len(transitions),
        class_count=class_count,
    )
    endpoints = _endpoint_lookup(
        node_frame,
        final_node=nodes[-1],
        models=models,
        model_seeds=model_seeds,
        truth_classes=truth_classes,
    )
    model_process = _model_process_frame(
        paths=paths,
        endpoint_lookup=endpoints,
        v3_root=v3_root,
        truth_classes=truth_classes,
        middle_indices=middle_indices,
        late_indices=late_indices,
        transition_count=len(transitions),
    )
    paired = _pair_models(
        model_process,
        paths=paths,
        reference_model=reference_model,
        comparator_model=comparator_model,
        truth_classes=truth_classes,
        transition_count=len(transitions),
    )
    primary_statistics = _primary_statistics(
        paired,
        primary_split=str(config["primary_split"]),
        model_seeds=model_seeds,
        iterations=iterations,
        bootstrap_seed=bootstrap_seed,
    )
    candidate_statistics = _candidate_statistics(
        paired,
        discovery_split=str(config["discovery_split"]),
        candidate_indices=candidate_indices,
        transitions=transitions,
        model_seeds=model_seeds,
        iterations=iterations,
        bootstrap_seed=bootstrap_seed,
    )
    candidate = select_robust_transition_candidate(
        candidate_statistics.to_dict("records"),
        allowed_transition_indices=candidate_indices,
        required_model_seeds=model_seeds,
    )
    path_map = config["candidate_path_map"][candidate["transition"]]
    candidate["path"] = {
        "module": str(path_map["module"]),
        "argument_index": int(path_map["argument_index"]),
        "source": str(path_map["source"]),
    }

    seed_primary = primary_statistics[
        primary_statistics.scope == "model_seed"
    ].copy()
    gate_config = config["gate"]
    gate = evaluate_causal_entry_gate(
        seed_primary.to_dict("records"),
        minimum_supported_seeds=int(gate_config["minimum_supported_seeds"]),
        alpha=float(gate_config["alpha"]),
        minimum_standardized_effect=float(
            gate_config["minimum_standardized_effect"]
        ),
    )
    gate["candidate_selected_on_discovery_split"] = True
    gate["candidate"] = candidate
    gate["causal_experiment_authorized"] = bool(gate["passed"])
    relation = _process_relation(
        paired,
        primary_split=str(config["primary_split"]),
        model_seeds=model_seeds,
    )
    representative_config = config["representative_case"]
    representative = _representative_case(
        paired,
        primary_split=str(config["primary_split"]),
        truth_classes=truth_classes,
        maximum_absolute_terminal_difference=float(
            representative_config["maximum_absolute_terminal_difference"]
        ),
        area_quantiles=tuple(
            float(value) for value in representative_config["lesion_area_quantiles"]
        ),
    )

    model_process.to_parquet(output_root / "model_process_metrics.parquet", index=False)
    paired.to_parquet(output_root / "patient_process_contrasts.parquet", index=False)
    primary_statistics.to_parquet(
        output_root / "primary_process_statistics.parquet", index=False
    )
    candidate_statistics.to_parquet(
        output_root / "candidate_transition_statistics.parquet", index=False
    )
    relation.to_parquet(output_root / "endpoint_process_relation.parquet", index=False)
    _write_json(output_root / "causal_entry_gate.json", gate)
    _write_json(output_root / "representative_case.json", representative)
    _write_json(
        output_root / "process_effect_status.json",
        {
            "execution_status": "PASS",
            "config": config,
            "input_root": v3_root,
            "model_process_row_count": len(model_process),
            "paired_patient_row_count": len(paired),
            "primary_statistics_row_count": len(primary_statistics),
            "candidate_statistics_row_count": len(candidate_statistics),
            "causal_experiment_authorized": gate[
                "causal_experiment_authorized"
            ],
            "candidate": candidate,
            "representative_case": representative,
        },
    )
    print(json.dumps(_json_ready(gate), ensure_ascii=False, indent=2), flush=True)
    print(f"Process effect gate complete: {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
