from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from itertools import product
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import yaml

from pptt.causal_abstraction.gates import evaluate_causal_abstraction_gate
from pptt.causal_abstraction.kernels import TransitionProcess
from pptt.causal_abstraction.metrics import (
    STATE_DISTRIBUTION_NAMES,
    cross_architecture_transfer,
    paired_fidelity_difference,
    structural_process_contrast,
    summarize_dose_direction,
    summarize_intervention_specificity,
    summarize_path_fidelity,
)
from scripts.lock_v11_causal_abstraction_protocol import (
    EXPECTED_CONTROL_MODELS,
    EXPECTED_MAIN_MODELS,
    EXPECTED_MODEL_SEEDS,
    sha256_file,
    source_tree_sha256,
    validate_v11_configuration,
)
from scripts.run_v11_causal_abstraction import (
    CONDITION_NAMES,
    FORMAL_DOSES,
    NODE_NAMES,
    _active_validation_hashes,
    _load_json,
    _validate_patient_payload,
    validate_locked_plan_manifest,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
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
    encoded = json.dumps(
        _json_ready(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.parquet")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _state_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{name}" for name in STATE_DISTRIBUTION_NAMES]


def _load_process(path: Path) -> TransitionProcess:
    return TransitionProcess.from_payload(_load_json(path))


def _validated_state_count_row(row: Mapping[str, Any]) -> np.ndarray:
    counts = np.asarray(row.get("state_counts"), dtype=np.int64)
    reliable = int(row.get("reliable_pixel_count", -1))
    selected = int(row.get("selected_pixel_count", -1))
    if (
        counts.shape != (5,)
        or np.any(counts < 0)
        or reliable < 0
        or selected < 0
        or reliable > selected
        or int(counts.sum()) != reliable
    ):
        raise ValueError("formal state-count row is internally inconsistent")
    return counts


def _validate_condition_grid(
    condition: Mapping[str, Any],
    *,
    patient_id: str,
    node: str,
    node_status: str,
    pair_count: int,
) -> dict[tuple[int, int], int]:
    node_index = NODE_NAMES.index(node)
    path = [*NODE_NAMES[node_index:], "Y"]
    rows = list(condition.get("state_counts", []))
    if node_status == "NO_MATCH":
        if pair_count != 0 or rows:
            raise ValueError(f"NO_MATCH node contains intervention rows: {patient_id}/{node}")
        return {}
    if node_status not in {"PASS", "FAILED_OPERATOR_OR_OOD_AUDIT"}:
        raise ValueError(f"unknown formal node status: {patient_id}/{node}/{node_status}")
    if pair_count <= 0 or not rows:
        raise ValueError(f"matched formal node lacks state rows: {patient_id}/{node}")

    seen: set[tuple[float, str, int, int]] = set()
    group_sizes: dict[tuple[int, int], set[int]] = {}
    for row in rows:
        _validated_state_count_row(row)
        dose = float(row.get("dose", float("nan")))
        depth = str(row.get("depth"))
        downstream_depth = int(row.get("downstream_depth", -1))
        source_state = int(row.get("source_state", -1))
        truth_class = int(row.get("truth_class", -1))
        selected = int(row.get("selected_pixel_count", -1))
        if (
            dose not in FORMAL_DOSES
            or depth not in path
            or downstream_depth != path.index(depth)
            or not 0 <= source_state < 5
            or not 0 <= truth_class < 4
            or selected <= 0
        ):
            raise ValueError(f"invalid formal condition row: {patient_id}/{node}")
        key = (dose, depth, source_state, truth_class)
        if key in seen:
            raise ValueError(f"duplicate formal state-count condition: {patient_id}/{node}/{key}")
        seen.add(key)
        group_sizes.setdefault((source_state, truth_class), set()).add(selected)

    if any(len(values) != 1 for values in group_sizes.values()):
        raise ValueError(f"selected pixel count changes across the path: {patient_id}/{node}")
    expected = {
        (dose, depth, source_state, truth_class)
        for dose, depth, (source_state, truth_class) in product(
            FORMAL_DOSES,
            path,
            sorted(group_sizes),
        )
    }
    if seen != expected:
        raise ValueError(f"formal condition grid is incomplete: {patient_id}/{node}")
    fixed_sizes = {group: next(iter(values)) for group, values in group_sizes.items()}
    if sum(fixed_sizes.values()) != pair_count:
        raise ValueError(f"formal condition pair count differs: {patient_id}/{node}")
    return fixed_sizes


def collect_formal_job(
    job_root: Path,
    *,
    model: str,
    model_seed: int,
    patient_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load one complete formal job and reject partial or smoke contamination."""

    manifest_path = job_root / "job_manifest.json"
    status_path = job_root / "job_status.json"
    if not manifest_path.is_file() or not status_path.is_file():
        raise FileNotFoundError(f"formal job manifest or status is missing: {job_root}")
    manifest = _load_json(manifest_path)
    status = _load_json(status_path)
    expected_patients = list(patient_ids)
    if (
        manifest.get("execution_mode") != "formal"
        or manifest.get("patient_ids") != expected_patients
        or int(manifest.get("patient_count", -1)) != len(expected_patients)
        or manifest.get("full_activations_persisted") is not False
        or status.get("status") != "COMPLETE"
        or status.get("execution_mode") != "formal"
        or status.get("formal_summary_eligible") is not True
        or int(status.get("completed_patients", -1)) != len(expected_patients)
    ):
        raise ValueError(f"formal job is incomplete or contaminated: {job_root}")
    identity = manifest.get("job_identity")
    if (
        not isinstance(identity, dict)
        or identity.get("model") != model
        or int(identity.get("model_seed", -1)) != int(model_seed)
        or status.get("job_identity") != identity
    ):
        raise ValueError(f"formal job identity differs: {job_root}")
    patient_root = job_root / "patient_results"
    actual_directories = sorted(
        path.name for path in patient_root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    if actual_directories != expected_patients:
        raise ValueError(f"formal patient output registry differs: {job_root}")

    state_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    for patient_id in expected_patients:
        path = patient_root / patient_id / "patient_result.json"
        payload = _load_json(path)
        _validate_patient_payload(
            payload,
            patient_id=patient_id,
            job_identity=identity,
        )
        if (
            payload.get("execution_mode") != "formal"
            or payload.get("formal_summary_eligible") is not True
            or payload.get("formal_claim_eligible") is not False
            or payload.get("storage_audit", {}).get("full_feature_tensors_persisted") is not False
            or payload.get("storage_audit", {}).get("full_output_logits_persisted") is not False
        ):
            raise ValueError(f"patient result is not an admissible formal artifact: {patient_id}")
        for node_result in payload["node_results"]:
            node = str(node_result["node"])
            node_status = str(node_result.get("status"))
            effect = dict(node_result.get("effect_summary", {}))
            pair_count = int(effect.get("pair_count", -1))
            effect_rows.append(
                {
                    "architecture": model,
                    "model_seed": int(model_seed),
                    "patient_id": patient_id,
                    "intervention_node": node,
                    "node_status": node_status,
                    **effect,
                }
            )
            condition_signatures = {}
            for condition_name in CONDITION_NAMES:
                condition = node_result["conditions"][condition_name]
                if (
                    condition.get("condition") != condition_name
                    or condition.get("doses") != list(FORMAL_DOSES)
                    or condition.get("downstream_path")
                    != [*NODE_NAMES[NODE_NAMES.index(node) :], "Y"]
                ):
                    raise ValueError(f"patient condition identity differs: {patient_id}/{node}")
                condition_signatures[condition_name] = _validate_condition_grid(
                    condition,
                    patient_id=patient_id,
                    node=node,
                    node_status=node_status,
                    pair_count=pair_count,
                )
                for count_row in condition.get("state_counts", []):
                    counts = _validated_state_count_row(count_row)
                    state_rows.append(
                        {
                            "architecture": model,
                            "model_seed": int(model_seed),
                            "patient_id": patient_id,
                            "intervention_node": node,
                            "source_state": int(count_row["source_state"]),
                            "truth_class": int(count_row["truth_class"]),
                            "condition": condition_name,
                            "dose": float(count_row["dose"]),
                            "downstream_node": str(count_row["depth"]),
                            "downstream_depth": int(count_row["downstream_depth"]),
                            "selected_pixel_count": int(count_row["selected_pixel_count"]),
                            "reliable_pixel_count": int(count_row["reliable_pixel_count"]),
                            "node_status": node_status,
                            **{
                                f"count_{name}": int(counts[index])
                                for index, name in enumerate(STATE_DISTRIBUTION_NAMES)
                            },
                        }
                    )
            if condition_signatures["task"] != condition_signatures["null"]:
                raise ValueError(
                    f"task and null conditions use different pixels: {patient_id}/{node}"
                )
    states = pd.DataFrame(state_rows)
    effects = pd.DataFrame(effect_rows)
    if states.empty or effects.empty:
        raise ValueError(f"formal job contains no state or effect rows: {job_root}")
    return states, effects, {
        "job_manifest_sha256": sha256_file(manifest_path),
        "job_status_sha256": sha256_file(status_path),
        "patient_count": len(expected_patients),
        "state_row_count": len(states),
        "effect_row_count": len(effects),
    }


def _network_distributions(state_rows: pd.DataFrame) -> pd.DataFrame:
    count_columns = [f"count_{name}" for name in STATE_DISTRIBUTION_NAMES]
    frame = state_rows[state_rows["reliable_pixel_count"] > 0].copy()
    if frame.empty:
        raise ValueError("no reliable formal state distributions remain")
    denominator = frame["reliable_pixel_count"].to_numpy(dtype=np.float64)
    counts = frame[count_columns].to_numpy(dtype=np.float64)
    probabilities = counts / denominator[:, None]
    for index, column in enumerate(_state_columns("network")):
        frame[column] = probabilities[:, index]
    return frame


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("paired-control values must be a nonempty finite vector")
    if array.size == 1:
        return float(array[0]), float(array[0])
    generator = np.random.default_rng(int(seed))
    positions = generator.integers(0, array.size, size=(int(iterations), array.size))
    means = array[positions].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _patient_error_advantage(
    correct: pd.DataFrame,
    control: pd.DataFrame,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, float, float, float, float]:
    excluded = {"tv_error", "process_source"}
    keys = [
        column
        for column in correct.columns
        if column in control.columns
        and column not in excluded
        and not column.startswith("network_")
        and not column.startswith("high_level_")
    ]
    if "patient_id" not in keys:
        raise ValueError("paired control comparison lacks patient identity")
    left = correct[[*keys, "tv_error"]].rename(columns={"tv_error": "correct_tv"})
    right = control[[*keys, "tv_error"]].rename(columns={"tv_error": "control_tv"})
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if paired.empty:
        raise ValueError("paired control and correct process share no conditions")
    paired["advantage"] = paired["control_tv"] - paired["correct_tv"]
    patient = paired.groupby("patient_id", sort=True)["advantage"].mean().reset_index()
    values = patient["advantage"].to_numpy(dtype=np.float64)
    low, high = _bootstrap_mean_interval(
        values,
        iterations=int(bootstrap_iterations),
        seed=int(bootstrap_seed),
    )
    if np.allclose(values, 0.0):
        p_value = 1.0
    else:
        try:
            p_value = float(
                wilcoxon(
                    values,
                    alternative="greater",
                    zero_method="pratt",
                ).pvalue
            )
        except ValueError:
            p_value = 1.0
    return patient, float(values.mean()), low, high, p_value


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Holm correction requires finite p-values")
    order = np.argsort(values, kind="mergesort")
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def _permuted_depth_process(
    process: TransitionProcess,
    permutation: Sequence[int],
) -> TransitionProcess:
    order = np.asarray(tuple(int(value) for value in permutation), dtype=np.int64)
    if sorted(order.tolist()) != list(range(len(process.node_names))):
        raise ValueError("depth permutation must contain every registered transition once")
    return TransitionProcess(
        node_names=process.node_names,
        kernels=process.kernels[order],
        coverage=process.coverage[order],
        alpha=process.alpha,
        source=f"depth_permuted:{process.source}",
    )


def _evaluate_process_by_node(
    network_rows: pd.DataFrame,
    process: TransitionProcess,
    *,
    evaluation: str,
    direction: str | None,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], dict[str, Any]]:
    evaluated_frames = []
    summary_rows = []
    summaries = {}
    architecture = str(network_rows["architecture"].iloc[0])
    model_seed = int(network_rows["model_seed"].iloc[0])
    for node_index, node in enumerate(NODE_NAMES):
        selected = network_rows[network_rows["intervention_node"] == node]
        if selected.empty:
            continue
        transfer = cross_architecture_transfer(
            process,
            selected,
            bootstrap_iterations=int(bootstrap_iterations),
            bootstrap_seed=int(bootstrap_seed) + node_index,
        )
        evaluated = transfer.evaluated_rows.copy()
        evaluated["evaluation"] = evaluation
        evaluated["direction"] = direction
        evaluated_frames.append(evaluated)
        summary = transfer.summary
        summaries[node] = summary
        summary_rows.append(
            {
                "kind": "fidelity",
                "evaluation": evaluation,
                "direction": direction,
                "architecture": architecture,
                "model_seed": model_seed,
                "node": node,
                "process_source": process.source,
                "macro_tv": summary.macro_tv,
                "worst_tv": summary.worst_tv,
                "worst_ci_high": summary.worst_ci_high,
                "patient_count": summary.patient_count,
                "worst_condition_json": json.dumps(
                    _json_ready(summary.worst_condition),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    return evaluated_frames, summary_rows, summaries


def _coverage_and_operator_rows(
    state_rows: pd.DataFrame,
    effect_rows: pd.DataFrame,
    configuration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    coverage_limits = configuration["coverage"]
    intervention_rows = state_rows[
        (state_rows["condition"] == "task")
        & np.isclose(state_rows["dose"], 1.0)
        & (state_rows["downstream_node"] == state_rows["intervention_node"])
    ]
    node_patient_counts = (
        intervention_rows.groupby(
            ["architecture", "model_seed", "intervention_node"],
            sort=True,
        )["patient_id"]
        .nunique()
        .to_dict()
    )
    eligible_states: dict[tuple[str, int, str], set[int]] = {}
    for identity, group in intervention_rows.groupby(
        ["architecture", "model_seed", "intervention_node", "source_state"],
        sort=True,
    ):
        architecture, model_seed, node, source_state = identity
        patient_count = int(group["patient_id"].nunique())
        pixel_count = int(group["selected_pixel_count"].sum())
        evaluable = bool(
            patient_count >= int(coverage_limits["minimum_state_patients"])
            and pixel_count >= int(coverage_limits["minimum_state_pixels"])
        )
        if evaluable:
            eligible_states.setdefault(
                (str(architecture), int(model_seed), str(node)), set()
            ).add(int(source_state))
        output.append(
            {
                "kind": "coverage",
                "architecture": architecture,
                "model_seed": int(model_seed),
                "node": node,
                "source_state": int(source_state),
                "patient_count": patient_count,
                "node_patient_count": int(
                    node_patient_counts[(architecture, model_seed, node)]
                ),
                "pixel_count": pixel_count,
                "evaluable": evaluable,
            }
        )
    reliability_keys = [
        "condition",
        "dose",
        "downstream_node",
        "source_state",
        "truth_class",
    ]
    for identity, states in sorted(eligible_states.items()):
        architecture, model_seed, node = identity
        selected = state_rows[
            (state_rows["architecture"] == architecture)
            & (state_rows["model_seed"] == model_seed)
            & (state_rows["intervention_node"] == node)
            & state_rows["source_state"].isin(states)
        ]
        aggregated = (
            selected.groupby(reliability_keys, sort=True, dropna=False)[
                ["selected_pixel_count", "reliable_pixel_count"]
            ]
            .sum()
            .reset_index()
        )
        aggregated = aggregated[aggregated["selected_pixel_count"] > 0].copy()
        if aggregated.empty:
            continue
        aggregated["reliable_fraction"] = (
            aggregated["reliable_pixel_count"]
            / aggregated["selected_pixel_count"]
        )
        worst = aggregated.loc[aggregated["reliable_fraction"].idxmin()]
        output.append(
            {
                "kind": "reliability",
                "architecture": architecture,
                "model_seed": int(model_seed),
                "node": node,
                "minimum_reliable_fraction": float(worst["reliable_fraction"]),
                "macro_reliable_fraction": float(
                    aggregated["reliable_fraction"].mean()
                ),
                "worst_condition_json": json.dumps(
                    {
                        key: _json_ready(worst[key]) for key in reliability_keys
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    matched = effect_rows[pd.to_numeric(effect_rows["pair_count"], errors="coerce") > 0]
    for identity, group in matched.groupby(
        ["architecture", "model_seed", "intervention_node"],
        sort=True,
    ):
        architecture, model_seed, node = identity
        output.append(
            {
                "kind": "operator",
                "architecture": architecture,
                "model_seed": int(model_seed),
                "node": node,
                "reconstruction_error": float(
                    group["operator_reconstruction_max_abs_error"].max()
                ),
                "state_realization": float(
                    group["target_state_realization_rate"].min()
                ),
                "restore_error": float(group["restore_max_abs_logit_error"].max()),
                "ood_pass": bool(
                    group["norm_pass"].astype(bool).all()
                    and group["leakage_pass"].astype(bool).all()
                    and group["node_status"].eq("PASS").all()
                ),
                "patient_count": int(group["patient_id"].nunique()),
                "maximum_null_target_logit_change": float(
                    group["null_target_max_abs_logit_change"].max()
                ),
            }
        )
    return output


def _rename_distribution(frame: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    return frame.rename(
        columns={
            f"{source}_{name}": f"{target}_{name}"
            for name in STATE_DISTRIBUTION_NAMES
        }
    )


def _specificity_and_dose_for_cell(
    cell_rows: pd.DataFrame,
    process: TransitionProcess,
    *,
    node: str,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    keys = [
        "architecture",
        "model_seed",
        "patient_id",
        "intervention_node",
        "source_state",
        "downstream_node",
        "downstream_depth",
        "truth_class",
    ]
    clean = cell_rows[
        (cell_rows["condition"] == "task") & np.isclose(cell_rows["dose"], 0.0)
    ][[*keys, *_state_columns("network")]]
    task = cell_rows[
        (cell_rows["condition"] == "task") & np.isclose(cell_rows["dose"], 1.0)
    ].copy()
    null = cell_rows[
        (cell_rows["condition"] == "null") & np.isclose(cell_rows["dose"], 1.0)
    ][[*keys, *_state_columns("network")]]
    target_transfer = cross_architecture_transfer(
        process,
        task,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed),
    )
    target = target_transfer.evaluated_rows[
        [*keys, *_state_columns("high_level")]
    ]
    combined = _rename_distribution(clean, "network", "clean").merge(
        _rename_distribution(task[[*keys, *_state_columns("network")]], "network", "task"),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    combined = combined.merge(
        _rename_distribution(null, "network", "null"),
        on=keys,
        how="inner",
        validate="one_to_one",
    ).merge(
        _rename_distribution(target, "high_level", "target"),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    specificity = summarize_intervention_specificity(
        combined,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed),
    )

    dose_input = cell_rows[cell_rows["condition"] == "task"].copy()
    dose_transfer = cross_architecture_transfer(
        process,
        dose_input,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed) + 1,
    )
    dose_rows = dose_transfer.evaluated_rows
    complete_keys = [column for column in keys if column != "architecture"]
    complete = dose_rows.groupby(complete_keys, sort=True)["dose"].transform(
        lambda values: tuple(sorted(float(value) for value in values)) == FORMAL_DOSES
    )
    dose_rows = dose_rows[complete].copy()
    if dose_rows.empty:
        raise ValueError(f"no complete dose trajectories remain for {node}")
    dose = summarize_dose_direction(
        dose_rows,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed) + 2,
    )
    architecture = str(cell_rows["architecture"].iloc[0])
    model_seed = int(cell_rows["model_seed"].iloc[0])
    specificity_row = {
        "kind": "specificity",
        "architecture": architecture,
        "model_seed": model_seed,
        "node": node,
        "task_gain": specificity.task_gain,
        "null_gain": specificity.null_gain,
        "delta_gain": specificity.delta_gain,
        "delta_ci_low": specificity.delta_ci_low,
        "delta_ci_high": specificity.delta_ci_high,
        "null_clean_tv": specificity.null_clean_tv,
        "patient_count": specificity.patient_count,
    }
    dose_row = {
        "kind": "dose",
        "architecture": architecture,
        "model_seed": model_seed,
        "node": node,
        "aligned_fraction": dose.aligned_fraction,
        "reverse_condition_count": dose.reverse_condition_count,
        "patient_count": int(dose.patient_rows["patient_id"].nunique()),
    }
    return specificity_row, dose_row, dose.condition_rows


def _control_rows_for_cell(
    cell_rows: pd.DataFrame,
    process: TransitionProcess,
    correct_patient_rows: pd.DataFrame,
    *,
    node: str,
    configuration: Mapping[str, Any],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    controls = configuration["controls"]
    depth_process = _permuted_depth_process(
        process,
        controls["depth_permutation"],
    )
    depth_transfer = cross_architecture_transfer(
        depth_process,
        cell_rows,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed),
    )
    _, order_advantage, order_low, order_high, order_p = _patient_error_advantage(
        correct_patient_rows,
        depth_transfer.summary.patient_rows,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed) + 1,
    )

    permutation = np.asarray(controls["state_permutation"], dtype=np.int64)
    state_input = cell_rows.copy()
    registered_states = state_input["source_state"].to_numpy(dtype=np.int64)
    state_input["registered_source_state"] = registered_states
    state_input["source_state"] = permutation[registered_states]
    state_transfer = cross_architecture_transfer(
        process,
        state_input,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed) + 2,
    )
    state_evaluated = state_transfer.evaluated_rows.copy()
    state_evaluated["source_state"] = state_evaluated.pop(
        "registered_source_state"
    ).astype(np.int64)
    state_summary = summarize_path_fidelity(
        state_evaluated,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed) + 3,
    )
    _, state_advantage, state_low, state_high, state_p = _patient_error_advantage(
        correct_patient_rows,
        state_summary.patient_rows,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=int(bootstrap_seed) + 4,
    )
    architecture = str(cell_rows["architecture"].iloc[0])
    model_seed = int(cell_rows["model_seed"].iloc[0])
    return (
        {
            "kind": "order",
            "architecture": architecture,
            "model_seed": model_seed,
            "node": node,
            "advantage": order_advantage,
            "ci_low": order_low,
            "ci_high": order_high,
            "raw_p": order_p,
        },
        {
            "kind": "randomized",
            "control_name": "state_permutation",
            "architecture": architecture,
            "model_seed": model_seed,
            "node": node,
            "advantage": state_advantage,
            "ci_low": state_low,
            "ci_high": state_high,
            "raw_p": state_p,
            "separation_pass": bool(
                state_advantage
                >= float(controls["minimum_state_permutation_advantage"])
                and state_low
                > float(controls["minimum_state_permutation_ci_low"])
            ),
        },
    )


def _observer_admission_audit(
    workspace: Path,
    configuration: Mapping[str, Any],
) -> tuple[bool, bool, dict[str, Any]]:
    rows = {}
    admission_pass = True
    randomization_pass = True
    for model in (*EXPECTED_MAIN_MODELS, *EXPECTED_CONTROL_MODELS):
        for model_seed in EXPECTED_MODEL_SEEDS:
            identity = f"{model}/seed_{model_seed}"
            path = (
                workspace
                / str(configuration["observer_root"])
                / model
                / f"seed_{model_seed}"
                / "v1_status.json"
            )
            status = _load_json(path)
            primary = str(status.get("status"))
            randomization = str(status.get("parameter_randomization_status"))
            admission_ok = primary == "PASS"
            randomization_ok = (
                randomization == "PASS"
                if model_seed == 42
                else randomization in {"PASS", "NOT_APPLICABLE"}
            )
            admission_pass &= admission_ok
            randomization_pass &= randomization_ok
            rows[identity] = {
                "status": primary,
                "parameter_randomization_status": randomization,
                "sha256": sha256_file(path),
            }
    return admission_pass, randomization_pass, rows


def _gate_thresholds(
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "main_architectures": list(configuration["main_models"]),
        "model_seeds": list(configuration["model_seeds"]),
        "nodes": list(configuration["nodes"]),
        "transfer_directions": list(configuration["transfer_directions"]),
        **dict(configuration["coverage"]),
        "maximum_reconstruction_error": configuration["intervention"][
            "maximum_reconstruction_error"
        ],
        "minimum_state_realization": configuration["intervention"][
            "minimum_state_realization"
        ],
        "maximum_restore_error": configuration["intervention"][
            "maximum_restore_error"
        ],
        "holm_alpha": configuration["statistics"]["holm_alpha"],
        **dict(configuration["gates"]),
    }


def summarize_v11(
    *,
    workspace: Path,
    config_path: Path,
    protocol_lock_path: Path,
) -> dict[str, Any]:
    configuration = _load_yaml(config_path)
    validated = validate_v11_configuration(configuration)
    lock = _load_json(protocol_lock_path)
    if (
        lock.get("status") != "LOCKED_BEFORE_FORMAL_INTERVENTION"
        or lock.get("formal_intervention_authorized") is not True
        or lock.get("configuration_sha256") != sha256_file(config_path)
        or lock.get("source_identity", {}).get("source_tree_sha256")
        != source_tree_sha256(workspace)
    ):
        raise ValueError("active source or configuration differs from the formal V11 lock")
    process_hashes, calibration_hashes = _active_validation_hashes(
        workspace,
        configuration,
    )
    if (
        lock.get("process_hashes") != process_hashes
        or lock.get("calibration_hashes") != calibration_hashes
    ):
        raise ValueError("active validation processes or calibrations differ from the lock")
    patient_ids = list(lock.get("test_patient_ids", []))
    if len(patient_ids) != validated.formal_patient_count:
        raise ValueError("formal lock has the wrong test patient count")

    all_state_frames = []
    all_effect_frames = []
    job_audits = {}
    formal_root = workspace / str(configuration["formal_job_root"])
    plan_root = workspace / str(configuration["intervention_plan_root"])
    for model in validated.main_models + validated.control_models:
        for model_seed in validated.model_seeds:
            identity = f"{model}/seed_{model_seed}"
            plan_manifest_path = (
                plan_root / model / f"seed_{model_seed}" / "plan_manifest.json"
            )
            plan_manifest, _ = validate_locked_plan_manifest(
                plan_manifest_path,
                lock=lock,
                model=model,
                model_seed=model_seed,
            )
            state_frame, effect_frame, job_audit = collect_formal_job(
                formal_root / model / f"seed_{model_seed}",
                model=model,
                model_seed=model_seed,
                patient_ids=patient_ids,
            )
            job_manifest = _load_json(
                formal_root
                / model
                / f"seed_{model_seed}"
                / "job_manifest.json"
            )
            job_identity = job_manifest["job_identity"]
            if (
                job_identity.get("protocol_lock_sha256")
                != sha256_file(protocol_lock_path)
                or job_identity.get("plan_manifest_sha256")
                != sha256_file(plan_manifest_path)
                or job_identity.get("checkpoint_sha256")
                != plan_manifest.get("checkpoint_sha256")
            ):
                raise ValueError(f"formal job was not executed from its locked plan: {identity}")
            all_state_frames.append(state_frame)
            all_effect_frames.append(effect_frame)
            job_audits[identity] = job_audit
    state_rows = pd.concat(all_state_frames, ignore_index=True)
    effect_rows = pd.concat(all_effect_frames, ignore_index=True)
    network_rows = _network_distributions(state_rows)

    fit_root = workspace / str(configuration["validation_fit_root"])
    processes = {
        "unet_baseline": _load_process(fit_root / "H_U.json"),
        "transunet_r50_vit_b16": _load_process(fit_root / "H_T.json"),
        "shared": _load_process(fit_root / "H_shared.json"),
    }
    bootstrap_iterations = int(configuration["statistics"]["bootstrap_iterations"])
    bootstrap_seed = int(configuration["statistics"]["bootstrap_seed"])
    admissible = network_rows[
        network_rows["node_status"].eq("PASS")
        & network_rows["architecture"].isin(validated.main_models)
    ].copy()
    full_task = admissible[
        (admissible["condition"] == "task") & np.isclose(admissible["dose"], 1.0)
    ]

    patient_counterfactual_frames: list[pd.DataFrame] = []
    condition_frames: list[pd.DataFrame] = []
    fidelity_rows: list[dict[str, Any]] = []
    summary_cache: dict[tuple[str, int, str, str], Any] = {}
    evaluated_cache: dict[tuple[str, int, str], pd.DataFrame] = {}
    for architecture in validated.main_models:
        process = processes[architecture]
        cross_source = (
            processes["transunet_r50_vit_b16"]
            if architecture == "unet_baseline"
            else processes["unet_baseline"]
        )
        cross_direction = (
            "transunet_r50_vit_b16->unet_baseline"
            if architecture == "unet_baseline"
            else "unet_baseline->transunet_r50_vit_b16"
        )
        for model_seed in validated.model_seeds:
            cell = full_task[
                (full_task["architecture"] == architecture)
                & (full_task["model_seed"] == model_seed)
            ]
            if cell.empty:
                continue
            for evaluation, active_process, direction in (
                ("within", process, None),
                ("shared", processes["shared"], None),
                ("cross", cross_source, cross_direction),
            ):
                evaluated, rows, summaries = _evaluate_process_by_node(
                    cell,
                    active_process,
                    evaluation=evaluation,
                    direction=direction,
                    bootstrap_iterations=bootstrap_iterations,
                    bootstrap_seed=bootstrap_seed + 1000 * int(model_seed),
                )
                patient_counterfactual_frames.extend(evaluated)
                fidelity_rows.extend(rows)
                for node, summary in summaries.items():
                    summary_cache[(architecture, int(model_seed), node, evaluation)] = summary
                    condition = summary.condition_rows.copy()
                    condition["evaluation"] = evaluation
                    condition["direction"] = direction
                    condition_frames.append(condition)
                    if evaluation == "within":
                        evaluated_cache[(architecture, int(model_seed), node)] = summary.patient_rows

    shared_rows = []
    for architecture in validated.main_models:
        for model_seed in validated.model_seeds:
            for node in NODE_NAMES:
                specific = summary_cache.get((architecture, model_seed, node, "within"))
                shared = summary_cache.get((architecture, model_seed, node, "shared"))
                if specific is None or shared is None:
                    continue
                difference = paired_fidelity_difference(
                    shared,
                    specific,
                    bootstrap_iterations=bootstrap_iterations,
                    bootstrap_seed=bootstrap_seed + model_seed + NODE_NAMES.index(node),
                )
                shared_rows.append(
                    {
                        "kind": "shared_noninferiority",
                        "architecture": architecture,
                        "model_seed": model_seed,
                        "node": node,
                        "tv_increment": difference.estimate,
                        "ci_low": difference.ci_low,
                        "ci_high": difference.ci_high,
                        "patient_count": difference.patient_count,
                    }
                )

    specificity_rows = []
    dose_rows = []
    dose_condition_frames = []
    order_rows = []
    randomized_rows = []
    for architecture in validated.main_models:
        process = processes[architecture]
        for model_seed in validated.model_seeds:
            for node_index, node in enumerate(NODE_NAMES):
                cell = admissible[
                    (admissible["architecture"] == architecture)
                    & (admissible["model_seed"] == model_seed)
                    & (admissible["intervention_node"] == node)
                ]
                correct = evaluated_cache.get((architecture, model_seed, node))
                if cell.empty or correct is None:
                    continue
                specificity, dose, dose_conditions = _specificity_and_dose_for_cell(
                    cell,
                    process,
                    node=node,
                    bootstrap_iterations=bootstrap_iterations,
                    bootstrap_seed=bootstrap_seed + 10_000 * model_seed + node_index,
                )
                specificity_rows.append(specificity)
                dose_rows.append(dose)
                dose_conditions["architecture"] = architecture
                dose_conditions["model_seed"] = model_seed
                dose_conditions["node"] = node
                dose_condition_frames.append(dose_conditions)
                order, randomized = _control_rows_for_cell(
                    cell[
                        (cell["condition"] == "task")
                        & np.isclose(cell["dose"], 1.0)
                    ],
                    process,
                    correct,
                    node=node,
                    configuration=configuration,
                    bootstrap_iterations=bootstrap_iterations,
                    bootstrap_seed=bootstrap_seed + 20_000 * model_seed + node_index,
                )
                order_rows.append(order)
                randomized_rows.append(randomized)
    if order_rows:
        adjusted = _holm_adjust([row["raw_p"] for row in order_rows])
        for row, holm_p in zip(order_rows, adjusted, strict=True):
            row["holm_p"] = float(holm_p)

    structural_input = network_rows[
        network_rows["node_status"].eq("PASS")
        & (network_rows["condition"] == "task")
        & np.isclose(network_rows["dose"], 1.0)
        & (network_rows["downstream_depth"] == 1)
        & network_rows["architecture"].isin(
            ("unet_baseline", "unet_noskip")
        )
    ].copy()
    structural_input["node"] = structural_input["intervention_node"]
    for name in STATE_DISTRIBUTION_NAMES:
        structural_input[f"process_{name}"] = structural_input[f"network_{name}"]
    structural = structural_process_contrast(
        structural_input[structural_input["architecture"] == "unet_baseline"],
        structural_input[structural_input["architecture"] == "unet_noskip"],
        structural_input[structural_input["architecture"] == "unet_baseline"],
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    structural_gate_row = {
        "kind": "structure",
        "delta_distance": structural.delta_distance,
        "ci_low": structural.ci_low,
        "ci_high": structural.ci_high,
        "between_distance": structural.between,
        "within_distance": structural.within,
        "patient_count": structural.patient_count,
        "reported_node_count": int(structural.node_rows["node"].nunique()),
    }

    metric_rows = [
        *_coverage_and_operator_rows(state_rows, effect_rows, configuration),
        *fidelity_rows,
        *shared_rows,
        *dose_rows,
        structural_gate_row,
    ]
    control_rows = [*specificity_rows, *order_rows, *randomized_rows]
    metrics = pd.DataFrame(metric_rows)
    controls = pd.DataFrame(control_rows)
    observer_pass, randomization_pass, observer_audit = _observer_admission_audit(
        workspace,
        configuration,
    )
    history = _load_json(fit_root / "history_admission.json")
    audit = {
        "asset_identity_pass": True,
        "protocol_lock_pass": True,
        "patient_registry_pass": True,
        "observer_admission_pass": observer_pass,
        "observer_randomization_pass": randomization_pass,
        "source_clean_before_formal_pass": lock.get("source_identity", {}).get("code_dirty") is False,
        "history_status": history.get("status"),
        "job_count": len(job_audits),
        "expected_job_count": 9,
        "patient_count_per_job": len(patient_ids),
        "state_row_count": len(state_rows),
        "effect_row_count": len(effect_rows),
        "job_audits": job_audits,
        "observer_audit": observer_audit,
        "protocol_lock_sha256": sha256_file(protocol_lock_path),
        "configuration_sha256": sha256_file(config_path),
        "source_tree_sha256": source_tree_sha256(workspace),
    }
    gate = evaluate_causal_abstraction_gate(
        metrics,
        controls,
        audit,
        _gate_thresholds(configuration),
    )

    output_root = workspace / str(configuration["output_root"])
    patient_counterfactuals = pd.concat(
        patient_counterfactual_frames,
        ignore_index=True,
    )
    condition_fidelity = pd.concat(condition_frames, ignore_index=True)
    cross_transfer = condition_fidelity[
        condition_fidelity["evaluation"] == "cross"
    ].copy()
    specificity_and_dose = pd.concat(
        [
            pd.DataFrame(specificity_rows),
            pd.DataFrame(dose_rows),
            pd.DataFrame(order_rows),
            pd.DataFrame(randomized_rows),
        ],
        ignore_index=True,
        sort=False,
    )
    structural_output = pd.concat(
        [
            structural.patient_rows.assign(kind="patient"),
            structural.node_rows.assign(kind="node"),
            pd.DataFrame([structural_gate_row]).assign(kind="overall"),
        ],
        ignore_index=True,
        sort=False,
    )
    outputs = {
        "patient_counterfactuals.parquet": patient_counterfactuals,
        "condition_fidelity.parquet": condition_fidelity,
        "cross_architecture_transfer.parquet": cross_transfer,
        "shared_noninferiority.parquet": pd.DataFrame(shared_rows),
        "specificity_and_dose.parquet": specificity_and_dose,
        "noskip_structural_contrast.parquet": structural_output,
        "formal_gate_metrics.parquet": metrics,
        "formal_gate_controls.parquet": controls,
    }
    for name, frame in outputs.items():
        _write_parquet_atomic(output_root / name, frame)
    _write_json_atomic(output_root / "v11_gate.json", gate.to_dict())
    _write_json_atomic(output_root / "v11_audit.json", audit)
    status = {
        "status": gate.status.value,
        "formal_summary_complete": True,
        "shared_full_network_claim_authorized": gate.full_network_claim_authorized,
        "causal_claim_authorized": gate.causal_claim_authorized,
        "cross_architecture_claim_authorized": gate.cross_architecture_claim_authorized,
        "structural_sensitivity_supported": gate.structural_sensitivity_supported,
        "allowed_claim": gate.allowed_claim,
        "forbidden_claims": list(gate.forbidden_claims),
        "failed_conditions": list(gate.failed_conditions),
        "only_this_file_authorizes_formal_claims": True,
        "formal_output_hashes": {
            name: sha256_file(output_root / name) for name in outputs
        },
    }
    _write_json_atomic(output_root / "v11_status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize locked V11 patient-level causal-abstraction results."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/v11_causal_abstraction.yaml"),
    )
    parser.add_argument(
        "--protocol-lock",
        type=Path,
        default=Path("results/v11_causal_abstraction/v11_protocol_lock.json"),
    )
    args = parser.parse_args()
    workspace = args.workspace_root.resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (workspace / args.config).resolve()
    )
    lock_path = (
        args.protocol_lock.resolve()
        if args.protocol_lock.is_absolute()
        else (workspace / args.protocol_lock).resolve()
    )
    try:
        status = summarize_v11(
            workspace=workspace,
            config_path=config_path,
            protocol_lock_path=lock_path,
        )
    except Exception as error:
        configuration = _load_yaml(config_path)
        output_root = workspace / str(configuration["output_root"])
        failed = {
            "status": "FAILED_AUDIT",
            "formal_summary_complete": False,
            "shared_full_network_claim_authorized": False,
            "causal_claim_authorized": False,
            "cross_architecture_claim_authorized": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "only_this_file_authorizes_formal_claims": True,
        }
        _write_json_atomic(output_root / "v11_status.json", failed)
        raise
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
