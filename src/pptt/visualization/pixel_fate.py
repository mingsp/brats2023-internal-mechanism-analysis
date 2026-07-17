"""Publication figure for endpoint-similar but process-different pixel trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
_SEED_COLORS = {42: "#315E9A", 123: "#D97706", 3407: "#168A76"}


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
    points: list[dict[str, Any]] = []
    for seed, group in selected.groupby("model_seed", sort=True):
        color = _SEED_COLORS.get(int(seed), "#6B7280")
        axis.scatter(
            group["terminal_abs"] * 100.0,
            group["process_distance"],
            s=13,
            alpha=0.40,
            color=color,
            edgecolors="none",
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
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=_SEED_COLORS.get(int(seed), "#6B7280"),
            markeredgecolor="none",
            markersize=4.6,
            label=f"seed {int(seed)}",
        )
        for seed in sorted(selected["model_seed"].unique())
    ]
    axis.legend(handles=handles, frameon=False, loc="lower right", ncol=min(3, len(handles)))
    return {
        "point_count": len(points),
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
    return records


def _draw_trajectory(
    figure: plt.Figure,
    container: Any,
    *,
    image: np.ndarray,
    truth: np.ndarray,
    baseline_states: np.ndarray,
    noskip_states: np.ndarray,
    nodes: Sequence[str],
    process_masks: np.ndarray | None,
    language: str,
) -> dict[str, Any]:
    node_names = tuple(str(value) for value in nodes)
    grid = container.subgridspec(
        2,
        len(node_names) + 3,
        width_ratios=[1.05, 1.05, 0.55, *([1.0] * len(node_names))],
        wspace=0.045,
        hspace=0.08,
    )
    blank = np.zeros_like(truth)
    reference_images = (
        (segmentation_overlay(image, blank, colors=BRATS_CLASS_COLORS, alpha=0.0), "T2-FLAIR"),
        (
            segmentation_overlay(blank.astype(np.float32), truth, colors=BRATS_CLASS_COLORS, alpha=1.0),
            "金标准" if language == "zh" else "Ground truth",
        ),
    )
    for column, (panel, title) in enumerate(reference_images):
        axis = figure.add_subplot(grid[:, column])
        axis.imshow(panel, interpolation="nearest")
        axis.set_title(title, fontsize=8.2, pad=3)
        axis.axis("off")
    union = None
    if process_masks is not None:
        masks = np.asarray(process_masks)
        if masks.shape != (len(node_names) - 1, *truth.shape) or masks.dtype != np.bool_:
            raise ValueError("process_masks must be boolean with shape 7xHxW")
        union = masks.any(axis=0)
    for row_index, (states, label) in enumerate(
        (
            (baseline_states, "有跳接\nU-Net" if language == "zh" else "Skip\nU-Net"),
            (noskip_states, "无跳接\nU-Net" if language == "zh" else "No-skip\nU-Net"),
        )
    ):
        label_axis = figure.add_subplot(grid[row_index, 2])
        label_axis.text(0.5, 0.5, label, ha="center", va="center", fontsize=8.3)
        label_axis.axis("off")
        for node_index, node in enumerate(node_names):
            axis = figure.add_subplot(grid[row_index, node_index + 3])
            axis.imshow(
                segmentation_overlay(image, states[node_index], colors=BRATS_CLASS_COLORS, alpha=0.53),
                interpolation="nearest",
            )
            if union is not None and np.any(union) and np.any(~union):
                axis.contour(union.astype(float), levels=[0.5], colors=["white"], linewidths=0.42)
            if row_index == 0:
                axis.set_title(node, fontsize=7.8, pad=3)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#D1D5DB")
                spine.set_linewidth(0.45)
    return {
        "truth_class_pixel_counts": {
            str(class_id): int(np.count_nonzero(truth == class_id))
            for class_id in sorted(BRATS_CLASS_COLORS)
        },
        "process_union_pixel_count": None if union is None else int(union.sum()),
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
    nodes: Sequence[str],
    process_vector: Sequence[float],
    effect_statistics: pd.DataFrame,
    process_masks: np.ndarray | None = None,
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
        nodes=node_names,
        process_masks=process_masks,
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
