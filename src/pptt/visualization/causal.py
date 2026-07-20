"""Deterministic helpers for PPTT causal-faithfulness figures."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from pptt.visualization.common import configure_figure_style, save_publication_figure


SEED_COLORS = {42: "#2563A6", 123: "#D97706", 3407: "#16897C"}
SEED_MARKERS = {42: "o", 123: "s", 3407: "^"}
SEED_LINESTYLES = {42: "-", 123: "--", 3407: "-."}
RETAINED_COLOR = "#78B7B2"
LOST_COLOR = "#E07A2D"
RECOVERED_COLOR = "#007C83"


def display_channel(image: np.ndarray, *, channel_index: int) -> np.ndarray:
    """Extract one two-dimensional modality from CHW or 1xCHW input."""
    values = np.asarray(image)
    if values.ndim == 4:
        if values.shape[0] != 1:
            raise ValueError("batched display input must contain exactly one image")
        values = values[0]
    if values.ndim == 2:
        if int(channel_index) != 0:
            raise ValueError("a two-dimensional image only has channel zero")
        selected = values
    elif values.ndim == 3:
        index = int(channel_index)
        if not 0 <= index < values.shape[0]:
            raise ValueError("channel_index is outside the image")
        selected = values[index]
    else:
        raise ValueError("display input must have shape HxW, CxHxW, or 1xCxHxW")
    if not np.isfinite(selected).all():
        raise ValueError("display channel contains non-finite values")
    return np.asarray(selected)


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap interval for a sample mean."""
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size < 2:
        raise ValueError("bootstrap interval requires a vector with at least two values")
    if not np.isfinite(sample).all():
        raise ValueError("bootstrap values must be finite")
    if iterations < 100:
        raise ValueError("bootstrap iterations must be at least 100")
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, sample.size, size=(iterations, sample.size))
    means = sample[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def causal_event_map(
    states: np.ndarray,
    truth: np.ndarray,
    target_mask: np.ndarray,
    *,
    transition_index: int,
) -> np.ndarray:
    """Encode persistent retention (1) and loss (2) on a fixed clean target set."""
    state_path = np.asarray(states)
    labels = np.asarray(truth)
    target = np.asarray(target_mask)
    if state_path.ndim != 3 or state_path.shape[1:] != labels.shape:
        raise ValueError("states must have shape KxHxW and align with truth")
    if target.shape != labels.shape or target.dtype != np.bool_:
        raise ValueError("target_mask must be boolean and align with truth")
    if not 0 <= int(transition_index) < state_path.shape[0] - 1:
        raise ValueError("transition_index is outside the state path")
    retained = np.all(
        state_path[int(transition_index) + 1 :] == labels[np.newaxis, ...],
        axis=0,
    )
    events = np.zeros(labels.shape, dtype=np.uint8)
    events[target & retained] = 1
    events[target & ~retained] = 2
    return events


def comparative_causal_event_map(
    states: np.ndarray,
    corrupt_states: np.ndarray,
    truth: np.ndarray,
    target_mask: np.ndarray,
    *,
    transition_index: int,
    show_recovery: bool,
) -> np.ndarray:
    """Encode retained, lost and corruption-reversed target pixels."""
    current = causal_event_map(
        states,
        truth,
        target_mask,
        transition_index=transition_index,
    )
    corrupt = causal_event_map(
        corrupt_states,
        truth,
        target_mask,
        transition_index=transition_index,
    )
    retained = current == 1
    lost = current == 2
    recovered = retained & (corrupt == 2) & bool(show_recovery)
    events = np.zeros_like(current)
    events[retained] = 1
    events[lost] = 2
    events[recovered] = 3
    return events


def _roi_bounds(mask: np.ndarray, *, margin: int = 8) -> tuple[slice, slice]:
    selected = np.asarray(mask)
    if selected.ndim != 2 or selected.dtype != np.bool_ or not selected.any():
        raise ValueError("ROI mask must be a nonempty two-dimensional boolean array")
    rows, columns = np.where(selected)
    height, width = selected.shape
    row_start = max(0, int(rows.min()) - margin)
    row_stop = min(height, int(rows.max()) + margin + 1)
    column_start = max(0, int(columns.min()) - margin)
    column_stop = min(width, int(columns.max()) + margin + 1)
    return slice(row_start, row_stop), slice(column_start, column_stop)


def _normalized_grayscale(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("image must be a finite two-dimensional array")
    lower, upper = np.quantile(values, [0.01, 0.99])
    if upper <= lower:
        return np.zeros_like(values)
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0)


def _draw_reference(
    axis: plt.Axes,
    *,
    image: np.ndarray,
    truth: np.ndarray,
    target_mask: np.ndarray,
    roi: tuple[slice, slice],
    language: str,
) -> None:
    axis.imshow(
        _normalized_grayscale(image)[roi],
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    tumor = truth[roi] > 0
    if tumor.any() and np.any(~tumor):
        axis.contour(
            tumor.astype(np.uint8),
            levels=[0.5],
            colors=["white"],
            linewidths=0.8,
        )
    target = target_mask[roi]
    if target.any() and np.any(~target):
        axis.contour(
            target.astype(np.uint8),
            levels=[0.5],
            colors=[RECOVERED_COLOR],
            linewidths=1.25,
        )
    axis.set_title(
        "固定持续纠正像素" if language == "zh" else "Fixed persistent-correction set",
        fontsize=8.3,
        pad=3,
    )
    axis.axis("off")


def _draw_condition(
    axis: plt.Axes,
    *,
    image: np.ndarray,
    truth: np.ndarray,
    target_mask: np.ndarray,
    states: np.ndarray,
    corrupt_states: np.ndarray,
    transition_index: int,
    roi: tuple[slice, slice],
    title: str,
    show_recovery: bool,
) -> float:
    grayscale = _normalized_grayscale(image)[roi]
    axis.imshow(grayscale, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    events = comparative_causal_event_map(
        states,
        corrupt_states,
        truth,
        target_mask,
        transition_index=transition_index,
        show_recovery=show_recovery,
    )[roi]
    masked = np.ma.masked_where(events == 0, events)
    axis.imshow(
        masked,
        cmap=ListedColormap([RETAINED_COLOR, LOST_COLOR, RECOVERED_COLOR]),
        vmin=1,
        vmax=3,
        alpha=0.86,
        interpolation="nearest",
    )
    tumor = truth[roi] > 0
    if tumor.any() and np.any(~tumor):
        axis.contour(
            tumor.astype(np.uint8),
            levels=[0.5],
            colors=["white"],
            linewidths=0.65,
    )
    target_count = int(np.count_nonzero(events))
    q_value = float(np.count_nonzero(np.isin(events, (1, 3))) / target_count)
    axis.set_title(f"{title}\nQ = {q_value:.2f}", fontsize=8.3, pad=3)
    axis.axis("off")
    return q_value


def _draw_dose(axis: plt.Axes, dose: pd.DataFrame, *, language: str) -> None:
    required = {
        "model_seed",
        "alpha",
        "mean_persistent_retention",
        "ci_low",
        "ci_high",
    }
    if not required.issubset(dose.columns):
        raise ValueError("dose table lacks required columns")
    for seed, group in dose.groupby("model_seed", sort=True):
        ordered = group.sort_values("alpha")
        color = SEED_COLORS.get(int(seed), "#4B5563")
        x = ordered.alpha.to_numpy(dtype=float)
        y = ordered.mean_persistent_retention.to_numpy(dtype=float)
        low = ordered.ci_low.to_numpy(dtype=float)
        high = ordered.ci_high.to_numpy(dtype=float)
        axis.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
        axis.plot(
            x,
            y,
            color=color,
            marker=SEED_MARKERS.get(int(seed), "o"),
            linestyle=SEED_LINESTYLES.get(int(seed), "-"),
            markersize=4.3,
            label=f"seed {seed}",
        )
        axis.text(
            x[-1] + 0.025,
            y[-1],
            str(int(seed)),
            color=color,
            fontsize=7.2,
            va="center",
        )
    axis.set_xlim(-0.03, 1.14)
    axis.set_ylim(0.0, 1.02)
    axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axis.set_xlabel("目标路径恢复比例" if language == "zh" else "Target-path restoration fraction")
    axis.set_ylabel("持续纠正保留率 Q" if language == "zh" else "Persistent-correction retention Q")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.65)


def _draw_effects(axis: plt.Axes, statistics: pd.DataFrame, *, language: str) -> None:
    required = {
        "endpoint",
        "model_seed",
        "mean_difference",
        "ci_low",
        "ci_high",
        "paired_cohens_d",
        "patient_count",
    }
    if not required.issubset(statistics.columns):
        raise ValueError("causal statistics table lacks required columns")
    order = ("necessity", "restoration", "specificity")
    labels = (
        {
            "necessity": "必要性",
            "restoration": "恢复性",
            "specificity": "区域特异性",
        }
        if language == "zh"
        else {
            "necessity": "Necessity",
            "restoration": "Restoration",
            "specificity": "Regional specificity",
        }
    )
    y_base = np.arange(len(order), dtype=float)[::-1]
    offsets = {42: 0.16, 123: 0.0, 3407: -0.16}
    for base, endpoint in zip(y_base, order, strict=True):
        rows = statistics[statistics.endpoint == endpoint].sort_values("model_seed")
        if rows.empty:
            raise ValueError(f"missing causal endpoint: {endpoint}")
        for row in rows.itertuples():
            seed = int(row.model_seed)
            y = base + offsets.get(seed, 0.0)
            color = SEED_COLORS.get(seed, "#4B5563")
            axis.plot(
                [float(row.ci_low) * 100.0, float(row.ci_high) * 100.0],
                [y, y],
                color=color,
                linewidth=1.7,
            )
            axis.scatter(
                [float(row.mean_difference) * 100.0],
                [y],
                color=color,
                edgecolors="white",
                linewidths=0.6,
                s=34,
                marker=SEED_MARKERS.get(seed, "o"),
                zorder=4,
            )
            axis.text(
                float(row.ci_high) * 100.0 + 0.55,
                y,
                f"n={int(row.patient_count)}",
                fontsize=6.7,
                va="center",
                color="#52606D",
            )
    axis.axvline(0.0, color="#4B5563", linewidth=0.8, linestyle="--")
    axis.set_yticks(y_base, [labels[value] for value in order])
    axis.set_xlabel("因果效应（百分点）" if language == "zh" else "Causal effect (percentage points)")
    axis.set_ylim(-0.55, 2.55)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.65)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=SEED_COLORS[seed],
            label=f"seed {seed}",
        )
        for seed in (42, 123, 3407)
    ]
    for handle, seed in zip(handles, (42, 123, 3407), strict=True):
        handle.set_marker(SEED_MARKERS[seed])
    axis.legend(handles=handles, frameon=False, loc="lower right", fontsize=7.2)


def render_causal_result_figure(
    *,
    output_base: str | Path,
    language: str,
    image: np.ndarray,
    truth: np.ndarray,
    target_mask: np.ndarray,
    condition_states: Mapping[str, np.ndarray],
    transition_index: int,
    dose: pd.DataFrame,
    statistics: pd.DataFrame,
    dpi: int = 400,
) -> tuple[Path, Path]:
    """Render the single formal PPTT causal-faithfulness result figure."""
    required_conditions = ("clean", "corrupt", "restore_target", "restore_control")
    if tuple(condition_states) != required_conditions:
        raise ValueError(f"condition_states must follow {required_conditions}")
    configure_figure_style(language)
    roi = _roi_bounds(np.asarray(target_mask, dtype=bool), margin=10)
    figure = plt.figure(figsize=(15.0, 5.4), constrained_layout=False)
    outer = figure.add_gridspec(
        1,
        3,
        width_ratios=[2.15, 1.0, 1.18],
        left=0.025,
        right=0.99,
        top=0.93,
        bottom=0.14,
        wspace=0.27,
    )
    case_grid = outer[0].subgridspec(
        2,
        3,
        width_ratios=[1.05, 1.0, 1.0],
        wspace=0.06,
        hspace=0.22,
    )
    reference_axis = figure.add_subplot(case_grid[:, 0])
    _draw_reference(
        reference_axis,
        image=image,
        truth=truth,
        target_mask=target_mask,
        roi=roi,
        language=language,
    )
    condition_labels = (
        {
            "clean": "原始前向",
            "corrupt": "空间对齐破坏",
            "restore_target": "恢复等量目标区域",
            "restore_control": "恢复匹配对照",
        }
        if language == "zh"
        else {
            "clean": "Clean",
            "corrupt": "Spatial corruption",
            "restore_target": "Matched-target restore",
            "restore_control": "Matched-control restore",
        }
    )
    condition_positions = {
        "clean": (0, 1),
        "corrupt": (0, 2),
        "restore_target": (1, 1),
        "restore_control": (1, 2),
    }
    for condition in required_conditions:
        row, column = condition_positions[condition]
        axis = figure.add_subplot(case_grid[row, column])
        _draw_condition(
            axis,
            image=image,
            truth=truth,
            target_mask=target_mask,
            states=np.asarray(condition_states[condition]),
            corrupt_states=np.asarray(condition_states["corrupt"]),
            transition_index=transition_index,
            roi=roi,
            title=condition_labels[condition],
            show_recovery=condition in {"restore_target", "restore_control"},
        )
    dose_axis = figure.add_subplot(outer[1])
    _draw_dose(dose_axis, dose, language=language)
    effect_axis = figure.add_subplot(outer[2])
    _draw_effects(effect_axis, statistics, language=language)
    figure.text(0.008, 0.955, "a", fontsize=11, fontweight="bold")
    figure.text(0.548, 0.955, "b", fontsize=11, fontweight="bold")
    figure.text(0.765, 0.955, "c", fontsize=11, fontweight="bold")
    event_labels = (
        ("持续保留", "干预后丢失", "相对破坏条件恢复")
        if language == "zh"
        else (
            "Persistently retained",
            "Lost after intervention",
            "Recovered from corruption",
        )
    )
    figure.legend(
        handles=[
            Patch(facecolor=RETAINED_COLOR, label=event_labels[0]),
            Patch(facecolor=LOST_COLOR, label=event_labels[1]),
            Patch(facecolor=RECOVERED_COLOR, label=event_labels[2]),
        ],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.185, 0.015),
        ncol=3,
        fontsize=7.5,
    )
    return save_publication_figure(figure, output_base, dpi=dpi)


__all__ = [
    "bootstrap_mean_interval",
    "causal_event_map",
    "comparative_causal_event_map",
    "display_channel",
    "render_causal_result_figure",
]
