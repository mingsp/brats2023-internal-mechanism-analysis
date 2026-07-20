"""Publication figure for endpoint-similar but process-different pixel trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from pptt.visualization.common import configure_figure_style, save_publication_figure
from pptt.visualization.flagship import BRATS_CLASS_COLORS, segmentation_overlay


_EFFECT_ORDER = (
    "middle_damage",
    "late_persistent_net_recovery",
    "terminal_dice",
)
_EFFECT_COLORS = {
    "middle_damage": "#D97706",
    "late_persistent_net_recovery": "#087F8C",
    "terminal_dice": "#315E9A",
}
_SEED_COLOR = "#315E9A"
_SEED_MARKERS = {42: "o", 123: "s", 3407: "^"}
_CORRECTION = "#0F8B8D"
_DAMAGE = "#E07A2D"
_REENCODING = "#7551A6"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def select_median_process_case(
    frame: pd.DataFrame,
    *,
    vector_columns: Sequence[str],
) -> dict[str, Any]:
    """Select the deterministic observation nearest the coordinate-wise median."""
    columns = tuple(str(value) for value in vector_columns)
    required = {"patient_id", *columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"representative-case table lacks columns: {sorted(missing)}")
    if frame.empty or not columns:
        raise ValueError("representative-case table and process vector must be nonempty")
    selected = frame.copy()
    vectors = selected.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(vectors.to_numpy(dtype=np.float64)).all(axis=1)
    selected = selected.loc[finite].copy()
    vectors = vectors.loc[finite]
    if selected.empty:
        raise ValueError("no finite process vector is available")
    median = np.median(vectors.to_numpy(dtype=np.float64), axis=0)
    selected["distance_to_process_median_l1"] = np.abs(
        vectors.to_numpy(dtype=np.float64) - median[None, :]
    ).sum(axis=1)
    sort_columns = ["distance_to_process_median_l1", "patient_id"]
    if "model_seed" in selected.columns:
        sort_columns.append("model_seed")
    chosen = selected.sort_values(sort_columns, kind="mergesort").iloc[0]
    record = {key: _json_ready(value) for key, value in chosen.to_dict().items()}
    record["process_median"] = median.tolist()
    record["selection_rule"] = "minimum_l1_distance_to_coordinatewise_median"
    return record


def _effect_labels(language: str) -> dict[str, str]:
    if language == "zh":
        return {
            "middle_damage": "中部破坏与重编码",
            "late_persistent_net_recovery": "后期持续净恢复",
            "terminal_dice": "终点肿瘤 Dice",
        }
    return {
        "middle_damage": "Mid-path damage and re-encoding",
        "late_persistent_net_recovery": "Late persistent net recovery",
        "terminal_dice": "Terminal tumor Dice",
    }


def _draw_scatter(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    patient_id: str,
    model_seed: int,
    language: str,
) -> dict[str, Any]:
    required = {
        "patient_id",
        "model_seed",
        "terminal_dice_difference",
        "full_path_transition_tv",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"scatter table lacks columns: {sorted(missing)}")
    selected = frame.copy()
    selected["terminal_abs"] = pd.to_numeric(
        selected["terminal_dice_difference"], errors="coerce"
    ).abs()
    selected["process_distance"] = pd.to_numeric(
        selected["full_path_transition_tv"], errors="coerce"
    )
    selected = selected[
        np.isfinite(selected["terminal_abs"])
        & np.isfinite(selected["process_distance"])
    ]
    if selected.empty:
        raise ValueError("scatter table has no finite observations")
    endpoint_close = selected["terminal_abs"] <= 0.02
    endpoint_close_count = int(endpoint_close.sum())
    close_process_median = float(
        selected.loc[endpoint_close, "process_distance"].median()
    )
    axis.axvspan(0.0, 2.0, color="#E5E7EB", alpha=0.70, zorder=0)
    points: list[dict[str, Any]] = []
    for seed, group in selected.groupby("model_seed", sort=True):
        marker = _SEED_MARKERS.get(int(seed), "o")
        axis.scatter(
            group["terminal_abs"] * 100.0,
            group["process_distance"],
            s=17,
            alpha=0.42,
            color=_SEED_COLOR,
            marker=marker,
            edgecolors="white",
            linewidths=0.25,
            rasterized=True,
        )
        points.extend(
            {
                "patient_id": str(row.patient_id),
                "model_seed": int(row.model_seed),
                "terminal_absolute_difference_pp": float(row.terminal_abs * 100.0),
                "process_distance": float(row.process_distance),
            }
            for row in group.itertuples()
        )
    chosen = selected[
        (selected["patient_id"].astype(str) == str(patient_id))
        & (selected["model_seed"].astype(int) == int(model_seed))
    ]
    if len(chosen) != 1:
        raise ValueError("highlighted patient-seed observation must be unique")
    row = chosen.iloc[0]
    x_value = float(row["terminal_abs"] * 100.0)
    y_value = float(row["process_distance"])
    axis.scatter(
        [x_value],
        [y_value],
        s=88,
        marker="*",
        color="#B42318",
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )
    axis.annotate(
        str(patient_id).replace("BraTS-GLI-", ""),
        (x_value, y_value),
        xytext=(5, 6),
        textcoords="offset points",
        fontsize=7.3,
        color="#7A271A",
    )
    axis.set_xlabel(
        "终点 Dice 绝对差（百分点）"
        if language == "zh"
        else "Absolute terminal Dice difference (pp)"
    )
    axis.set_ylabel("全路径转移距离" if language == "zh" else "Full-path transition distance")
    annotation = (
        f"终点差 ≤ 2 pp：{endpoint_close_count}/{len(selected)}\n"
        f"过程距离中位数 = {close_process_median:.3f}"
        if language == "zh"
        else f"Endpoint gap ≤ 2 pp: {endpoint_close_count}/{len(selected)}\n"
        f"Median process distance = {close_process_median:.3f}"
    )
    axis.text(
        0.025,
        0.97,
        annotation,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        color="#334E68",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.0},
    )
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=_SEED_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=4.6,
            label=f"seed {int(seed)}",
        )
        for seed in sorted(selected["model_seed"].unique())
    ]
    for handle, seed in zip(handles, sorted(selected["model_seed"].unique()), strict=True):
        handle.set_marker(_SEED_MARKERS.get(int(seed), "o"))
    axis.legend(handles=handles, frameon=False, loc="lower right", ncol=min(3, len(handles)))
    return {
        "point_count": len(points),
        "endpoint_close_threshold_pp": 2.0,
        "endpoint_close_count": endpoint_close_count,
        "endpoint_close_fraction": endpoint_close_count / len(selected),
        "endpoint_close_process_distance_median": close_process_median,
        "seed_encoding": {
            str(int(seed)): {"color": _SEED_COLOR, "marker": _SEED_MARKERS.get(int(seed), "o")}
            for seed in sorted(selected["model_seed"].unique())
        },
        "points": points,
        "highlighted": {
            "patient_id": str(patient_id),
            "model_seed": int(model_seed),
            "terminal_absolute_difference_pp": x_value,
            "process_distance": y_value,
        },
    }


def _draw_effects(
    axis: plt.Axes,
    statistics: pd.DataFrame,
    *,
    language: str,
) -> list[dict[str, Any]]:
    required = {"scope", "metric", "mean_difference", "ci_low", "ci_high"}
    missing = required - set(statistics.columns)
    if missing:
        raise ValueError(f"effect table lacks columns: {sorted(missing)}")
    summary = statistics[statistics["scope"] == "fixed_seed_patient_average"]
    labels = _effect_labels(language)
    records: list[dict[str, Any]] = []
    positions = np.arange(len(_EFFECT_ORDER))[::-1]
    for position, metric in zip(positions, _EFFECT_ORDER, strict=True):
        rows = summary[summary["metric"] == metric]
        if len(rows) != 1:
            raise ValueError(f"effect table must contain one fixed estimate for {metric}")
        row = rows.iloc[0]
        mean = float(row["mean_difference"] * 100.0)
        low = float(row["ci_low"] * 100.0)
        high = float(row["ci_high"] * 100.0)
        color = _EFFECT_COLORS[metric]
        axis.plot([low, high], [position, position], color=color, linewidth=2.2)
        axis.scatter([mean], [position], s=50, color=color, edgecolors="white", linewidths=0.6)
        axis.text(high + 0.45, position, f"{mean:+.1f} [{low:.1f}, {high:.1f}]", va="center", fontsize=7.4)
        records.append(
            {
                "metric": metric,
                "mean_difference_pp": mean,
                "ci_low_pp": low,
                "ci_high_pp": high,
                "paired_cohens_d": _json_ready(row.get("paired_cohens_d", np.nan)),
                "holm_p": _json_ready(row.get("holm_p", np.nan)),
            }
        )
    axis.axvline(0.0, color="#4B5563", linewidth=0.8, linestyle="--")
    axis.set_yticks(positions, [labels[value] for value in _EFFECT_ORDER])
    axis.set_xlabel("有跳接 − 无跳接（百分点）" if language == "zh" else "Skip − no-skip (pp)")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.55)
    axis.tick_params(axis="y", length=0)
    axis.spines[["top", "right", "left"]].set_visible(False)
    by_metric = {row["metric"]: row for row in records}
    ratio = (
        by_metric["late_persistent_net_recovery"]["mean_difference_pp"]
        / by_metric["terminal_dice"]["mean_difference_pp"]
    )
    axis.text(
        0.98,
        0.04,
        (
            f"后期持续净恢复 / 终点 Dice = {ratio:.2f}×"
            if language == "zh"
            else f"Late persistent recovery / terminal Dice = {ratio:.2f}×"
        ),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.8,
        color="#087F8C",
        fontweight="bold",
    )
    records.append({"metric": "late_to_terminal_ratio", "ratio": ratio})
    return records


def _trajectory_events(
    states: np.ndarray,
    truth: np.ndarray,
    reliable: np.ndarray,
    final_state: np.ndarray,
) -> np.ndarray:
    events = np.zeros((states.shape[0] - 1, *truth.shape), dtype=np.uint8)
    tumor = truth > 0
    reliable_suffix = np.logical_and.accumulate(reliable[::-1], axis=0)[::-1]
    for transition in range(states.shape[0] - 1):
        current = states[transition]
        following = states[transition + 1]
        persistent_correct = np.all(
            states[transition + 1 :] == truth[np.newaxis, ...],
            axis=0,
        )
        correction = (
            tumor
            & reliable_suffix[transition]
            & (current != truth)
            & (following == truth)
            & persistent_correct
            & (final_state == truth)
        )
        damage = (
            tumor
            & reliable[transition]
            & (current == truth)
            & (following != truth)
        )
        reencoding = (
            tumor
            & reliable[transition]
            & (current != truth)
            & (following != truth)
            & (current != following)
        )
        events[transition, correction] = 1
        events[transition, damage] = 2
        events[transition, reencoding] = 3
    return events


def _crop_for_trajectory(truth: np.ndarray, events: np.ndarray) -> tuple[slice, slice]:
    selected = (truth > 0) | np.any(events > 0, axis=(0, 1))
    rows, columns = np.where(selected)
    if not rows.size:
        return slice(0, truth.shape[0]), slice(0, truth.shape[1])
    margin = 8
    return (
        slice(max(0, int(rows.min()) - margin), min(truth.shape[0], int(rows.max()) + margin + 1)),
        slice(max(0, int(columns.min()) - margin), min(truth.shape[1], int(columns.max()) + margin + 1)),
    )


def _draw_trajectory(
    figure: plt.Figure,
    container: Any,
    *,
    image: np.ndarray,
    truth: np.ndarray,
    baseline_states: np.ndarray,
    noskip_states: np.ndarray,
    baseline_reliable: np.ndarray,
    noskip_reliable: np.ndarray,
    baseline_final_state: np.ndarray,
    noskip_final_state: np.ndarray,
    nodes: Sequence[str],
    language: str,
) -> dict[str, Any]:
    node_names = tuple(str(value) for value in nodes)
    grid = container.subgridspec(
        2,
        len(node_names) + 2,
        width_ratios=[1.05, 0.68, *([1.0] * len(node_names))],
        wspace=0.045,
        hspace=0.08,
    )
    expected_reliable = (len(node_names) - 1, *truth.shape)
    if (
        baseline_reliable.shape != expected_reliable
        or noskip_reliable.shape != expected_reliable
        or baseline_reliable.dtype != np.bool_
        or noskip_reliable.dtype != np.bool_
        or baseline_final_state.shape != truth.shape
        or noskip_final_state.shape != truth.shape
    ):
        raise ValueError("trajectory reliability and final states do not align")
    baseline_events = _trajectory_events(
        baseline_states,
        truth,
        baseline_reliable,
        baseline_final_state,
    )
    noskip_events = _trajectory_events(
        noskip_states,
        truth,
        noskip_reliable,
        noskip_final_state,
    )
    crop = _crop_for_trajectory(truth, np.stack([baseline_events, noskip_events]))
    reference_axis = figure.add_subplot(grid[:, 0])
    reference_axis.imshow(image[crop], cmap="gray", interpolation="nearest")
    reference_axis.contour(
        (truth[crop] > 0).astype(float),
        levels=[0.5],
        colors=["white"],
        linewidths=0.7,
    )
    reference_axis.set_title("同一 MRI 与真值边界" if language == "zh" else "Same MRI and GT contour", fontsize=7.8, pad=3)
    reference_axis.axis("off")
    event_cmap = ListedColormap([_CORRECTION, _DAMAGE, _REENCODING])
    event_totals: dict[str, dict[str, int]] = {}
    for row_index, (states, events, label, key) in enumerate(
        (
            (baseline_states, baseline_events, "有跳接\nU-Net" if language == "zh" else "Skip\nU-Net", "skip"),
            (noskip_states, noskip_events, "无跳接\nU-Net" if language == "zh" else "No-skip\nU-Net", "no_skip"),
        )
    ):
        label_axis = figure.add_subplot(grid[row_index, 1])
        label_axis.text(0.5, 0.5, label, ha="center", va="center", fontsize=8.3)
        label_axis.axis("off")
        event_totals[key] = {
            "persistent_correction": int(np.count_nonzero(events == 1)),
            "damage": int(np.count_nonzero(events == 2)),
            "wrong_reencoding": int(np.count_nonzero(events == 3)),
        }
        for node_index, node in enumerate(node_names):
            axis = figure.add_subplot(grid[row_index, node_index + 2])
            axis.imshow(image[crop], cmap="gray", interpolation="nearest")
            predicted_tumor = states[node_index][crop] > 0
            if np.any(predicted_tumor) and np.any(~predicted_tumor):
                axis.contour(
                    predicted_tumor.astype(float),
                    levels=[0.5],
                    colors=["#AAB4BE"],
                    linewidths=0.42,
                )
            if node_index == 0:
                initial_correct = (truth > 0) & (states[0] == truth)
                axis.imshow(
                    np.ma.masked_where(~initial_correct[crop], initial_correct[crop]),
                    cmap=ListedColormap(["#6FA7A3"]),
                    alpha=0.48,
                    interpolation="nearest",
                )
            else:
                current_events = events[node_index - 1][crop]
                axis.imshow(
                    np.ma.masked_where(current_events == 0, current_events),
                    cmap=event_cmap,
                    vmin=1,
                    vmax=3,
                    alpha=0.88,
                    interpolation="nearest",
                )
            axis.contour(
                (truth[crop] > 0).astype(float),
                levels=[0.5],
                colors=["white"],
                linewidths=0.55,
            )
            if row_index == 0:
                axis.set_title(node, fontsize=7.8, pad=3)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#D1D5DB")
                spine.set_linewidth(0.45)
    figure.legend(
        handles=[
            Patch(facecolor=_CORRECTION, label="持续纠正" if language == "zh" else "Persistent correction"),
            Patch(facecolor=_DAMAGE, label="破坏" if language == "zh" else "Damage"),
            Patch(facecolor=_REENCODING, label="错误重编码" if language == "zh" else "Wrong re-encoding"),
        ],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.58, 0.01),
        ncol=3,
        fontsize=7.2,
    )
    return {
        "truth_class_pixel_counts": {
            str(class_id): int(np.count_nonzero(truth == class_id))
            for class_id in sorted(BRATS_CLASS_COLORS)
        },
        "event_totals": event_totals,
        "first_node_encoding": "correct_tumor_state",
        "later_node_encoding": "events_relative_to_previous_node",
        "ground_truth_encoding": "white_contour",
    }


def render_pixel_fate_figure(
    *,
    output_base: str | Path,
    language: str,
    scatter: pd.DataFrame,
    patient_id: str,
    model_seed: int,
    slice_id: str,
    image: np.ndarray,
    truth: np.ndarray,
    baseline_states: np.ndarray,
    noskip_states: np.ndarray,
    baseline_reliable: np.ndarray | None = None,
    noskip_reliable: np.ndarray | None = None,
    baseline_final_state: np.ndarray | None = None,
    noskip_final_state: np.ndarray | None = None,
    nodes: Sequence[str],
    process_vector: Sequence[float],
    effect_statistics: pd.DataFrame,
    dpi: int = 600,
) -> dict[str, Any]:
    """Render the endpoint/process contrast and one fixed all-node pixel trajectory."""
    configure_figure_style(language)
    node_names = tuple(str(value) for value in nodes)
    image_array = np.asarray(image, dtype=np.float32)
    truth_array = np.asarray(truth)
    baseline = np.asarray(baseline_states)
    noskip = np.asarray(noskip_states)
    if len(node_names) != 8:
        raise ValueError("pixel-fate figure requires the locked eight-node path")
    expected = (8, *truth_array.shape)
    if image_array.shape != truth_array.shape or baseline.shape != expected or noskip.shape != expected:
        raise ValueError("image, truth, and both eight-node paths must share one spatial grid")
    baseline_reliability = (
        np.ones((7, *truth_array.shape), dtype=bool)
        if baseline_reliable is None
        else np.asarray(baseline_reliable)
    )
    noskip_reliability = (
        np.ones((7, *truth_array.shape), dtype=bool)
        if noskip_reliable is None
        else np.asarray(noskip_reliable)
    )
    baseline_final = (
        baseline[-1] if baseline_final_state is None else np.asarray(baseline_final_state)
    )
    noskip_final = (
        noskip[-1] if noskip_final_state is None else np.asarray(noskip_final_state)
    )
    vector = np.asarray(tuple(process_vector), dtype=np.float64)
    if vector.shape != (7,) or not np.isfinite(vector).all():
        raise ValueError("process_vector must contain seven finite transition values")

    figure = plt.figure(figsize=(14.6, 7.4), constrained_layout=False)
    outer = figure.add_gridspec(
        2,
        2,
        height_ratios=[0.90, 1.28],
        width_ratios=[1.02, 0.98],
        left=0.055,
        right=0.99,
        top=0.965,
        bottom=0.065,
        hspace=0.28,
        wspace=0.25,
    )
    scatter_record = _draw_scatter(
        figure.add_subplot(outer[0, 0]),
        scatter,
        patient_id=str(patient_id),
        model_seed=int(model_seed),
        language=language,
    )
    effect_record = _draw_effects(
        figure.add_subplot(outer[0, 1]), effect_statistics, language=language
    )
    trajectory_record = _draw_trajectory(
        figure,
        outer[1, :],
        image=image_array,
        truth=truth_array,
        baseline_states=baseline,
        noskip_states=noskip,
        baseline_reliable=baseline_reliability,
        noskip_reliable=noskip_reliability,
        baseline_final_state=baseline_final,
        noskip_final_state=noskip_final,
        nodes=node_names,
        language=language,
    )
    png_path, pdf_path = save_publication_figure(figure, output_base, dpi=dpi)
    manifest = {
        "figure": "endpoint_similar_process_different",
        "language": language,
        "patient_id": str(patient_id),
        "model_seed": int(model_seed),
        "slice_id": str(slice_id),
        "nodes": list(node_names),
        "node_count": len(node_names),
        "same_case_and_slice_across_panels": True,
        "process_vector": vector.tolist(),
        "scatter": scatter_record,
        "effects": effect_record,
        "trajectory": trajectory_record,
        "outputs": {"png": str(png_path), "pdf": str(pdf_path)},
        "dpi": int(dpi),
    }
    manifest_path = Path(output_base).with_suffix(".json")
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = ["render_pixel_fate_figure", "select_median_process_case"]
