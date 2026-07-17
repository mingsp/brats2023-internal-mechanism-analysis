"""Publication figure for the network-wide process-intervention alignment matrix."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from pptt.visualization.common import configure_figure_style, save_publication_figure
from pptt.visualization.flagship import BRATS_CLASS_COLORS, segmentation_overlay


_MODEL_ORDER = ("unet_baseline", "unet_noskip")
_MODEL_COLORS = {"unet_baseline": "#315E9A", "unet_noskip": "#D97706"}


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


def _alignment_matrices(
    rows: pd.DataFrame,
    *,
    transitions: Sequence[str],
    restore_nodes: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    required = {
        "model",
        "model_seed",
        "transition_index",
        "transition",
        "restore_node",
        "evaluable",
        "mean_specific_effect",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"alignment summary lacks columns: {sorted(missing)}")
    transition_names = tuple(str(value) for value in transitions)
    restore_names = tuple(str(value) for value in restore_nodes)
    if len(transition_names) != 7 or len(restore_names) != 7:
        raise ValueError("network-alignment figure requires a 7x7 registered matrix")
    matrices: dict[str, np.ndarray] = {}
    support: dict[str, np.ndarray] = {}
    for model in _MODEL_ORDER:
        model_rows = rows[rows["model"] == model]
        if model_rows.empty:
            raise ValueError(f"alignment summary lacks model: {model}")
        matrix = np.full((7, 7), np.nan, dtype=np.float64)
        counts = np.zeros((7, 7), dtype=np.int64)
        for transition_index, transition in enumerate(transition_names):
            for restore_index, restore_node in enumerate(restore_names):
                cell = model_rows[
                    (model_rows["transition_index"].astype(int) == transition_index)
                    & (model_rows["transition"].astype(str) == transition)
                    & (model_rows["restore_node"].astype(str) == restore_node)
                ]
                values = pd.to_numeric(cell["mean_specific_effect"], errors="coerce")
                valid = cell["evaluable"].astype(bool) & np.isfinite(values)
                finite = values.loc[valid].to_numpy(dtype=np.float64)
                if finite.size:
                    matrix[transition_index, restore_index] = float(finite.mean())
                    counts[transition_index, restore_index] = int(finite.size)
        matrices[model] = matrix
        support[model] = counts
    return matrices, support


def _patient_bootstrap_summary(
    rows: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    required = {"model", "model_seed", "patient_id", "endpoint", "value"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"patient alignment table lacks columns: {sorted(missing)}")
    if iterations < 100:
        raise ValueError("bootstrap_iterations must be at least 100")
    frame = rows.copy()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[np.isfinite(frame["value"])]
    averaged = (
        frame.groupby(["model", "patient_id", "endpoint"], as_index=False, sort=True)["value"]
        .mean()
    )
    records: list[dict[str, Any]] = []
    generator = np.random.default_rng(int(seed))
    for model in _MODEL_ORDER:
        for endpoint in ("macro", "micro"):
            values = averaged[
                (averaged["model"] == model) & (averaged["endpoint"] == endpoint)
            ]["value"].to_numpy(dtype=np.float64)
            if not values.size:
                records.append(
                    {
                        "model": model,
                        "endpoint": endpoint,
                        "patient_count": 0,
                        "mean": None,
                        "ci_low": None,
                        "ci_high": None,
                    }
                )
                continue
            indices = generator.integers(0, values.size, size=(iterations, values.size))
            boot = values[indices].mean(axis=1)
            low, high = np.quantile(boot, [0.025, 0.975])
            records.append(
                {
                    "model": model,
                    "endpoint": endpoint,
                    "patient_count": int(values.size),
                    "mean": float(values.mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            )
    return records


def _draw_matrix(
    axis: plt.Axes,
    matrix: np.ndarray,
    *,
    model: str,
    transitions: Sequence[str],
    restore_nodes: Sequence[str],
    norm: TwoSlopeNorm,
    cmap: Any,
    language: str,
) -> Any:
    masked = np.ma.masked_invalid(matrix)
    image = axis.imshow(masked, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
    for row in range(7):
        for column in range(7):
            value = matrix[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value * 100:+.1f}", ha="center", va="center", fontsize=6.5)
    for index in range(7):
        axis.add_patch(Rectangle((index - 0.48, index - 0.48), 0.96, 0.96, fill=False, edgecolor="#111827", linewidth=1.0))
    axis.set_xticks(range(7), restore_nodes, rotation=40, ha="right", rotation_mode="anchor")
    axis.set_yticks(range(7), transitions)
    axis.set_xlabel("恢复节点" if language == "zh" else "Restored node")
    axis.set_ylabel("过程区间" if language == "zh" else "Process interval")
    title = (
        "有跳接 U-Net" if model == "unet_baseline" and language == "zh" else
        "无跳接 U-Net" if model == "unet_noskip" and language == "zh" else
        "Skip U-Net" if model == "unet_baseline" else "No-skip U-Net"
    )
    axis.set_title(title, fontsize=9.5, pad=6)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def _draw_example(
    figure: plt.Figure,
    container: Any,
    *,
    image: np.ndarray,
    truth: np.ndarray,
    states: Mapping[str, np.ndarray],
    target_mask: np.ndarray,
    language: str,
) -> dict[str, Any]:
    required = ("clean", "corrupt", "restore_target", "restore_control")
    missing = set(required) - set(states)
    if missing:
        raise ValueError(f"example states lack conditions: {sorted(missing)}")
    image_array = np.asarray(image, dtype=np.float32)
    truth_array = np.asarray(truth)
    target = np.asarray(target_mask)
    if image_array.shape != truth_array.shape or target.shape != truth_array.shape or target.dtype != np.bool_:
        raise ValueError("example image, truth, and target mask must share one spatial grid")
    grid = container.subgridspec(2, 2, wspace=0.06, hspace=0.15)
    titles = {
        "clean": "原始" if language == "zh" else "Clean",
        "corrupt": "破坏" if language == "zh" else "Corrupt",
        "restore_target": "恢复目标" if language == "zh" else "Restore target",
        "restore_control": "恢复对照" if language == "zh" else "Restore control",
    }
    condition_counts: dict[str, int] = {}
    for position, condition in enumerate(required):
        state = np.asarray(states[condition])
        if state.shape != truth_array.shape:
            raise ValueError(f"example state {condition} does not align with truth")
        axis = figure.add_subplot(grid[position // 2, position % 2])
        axis.imshow(segmentation_overlay(image_array, state, colors=BRATS_CLASS_COLORS, alpha=0.53), interpolation="nearest")
        if target.any() and (~target).any():
            axis.contour(target.astype(float), levels=[0.5], colors=["white"], linewidths=0.65)
        axis.set_title(titles[condition], fontsize=8.0, pad=2)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.45)
            spine.set_color("#D1D5DB")
        condition_counts[condition] = int((state[target] == truth_array[target]).sum()) if target.any() else 0
    return {
        "target_pixel_count": int(target.sum()),
        "target_correct_pixel_counts": condition_counts,
    }


def _draw_global_effects(
    axis: plt.Axes,
    records: Sequence[Mapping[str, Any]],
    *,
    language: str,
) -> None:
    order = [
        ("unet_baseline", "macro"),
        ("unet_baseline", "micro"),
        ("unet_noskip", "macro"),
        ("unet_noskip", "micro"),
    ]
    by_key = {(str(row["model"]), str(row["endpoint"])): row for row in records}
    labels = []
    positions = np.arange(len(order))[::-1]
    for position, identity in zip(positions, order, strict=True):
        row = by_key[identity]
        model, endpoint = identity
        model_label = (
            "有跳接" if model == "unet_baseline" and language == "zh" else
            "无跳接" if model == "unet_noskip" and language == "zh" else
            "Skip" if model == "unet_baseline" else "No-skip"
        )
        endpoint_label = "宏" if endpoint == "macro" and language == "zh" else "微" if endpoint == "micro" and language == "zh" else endpoint.capitalize()
        labels.append(f"{model_label} {endpoint_label}")
        if row["mean"] is None:
            continue
        mean = float(row["mean"] * 100.0)
        low = float(row["ci_low"] * 100.0)
        high = float(row["ci_high"] * 100.0)
        color = _MODEL_COLORS[model]
        axis.plot([low, high], [position, position], color=color, linewidth=2.0)
        axis.scatter([mean], [position], color=color, s=43, edgecolors="white", linewidths=0.6)
        axis.text(high + 0.25, position, f"{mean:+.1f} [{low:.1f}, {high:.1f}]", va="center", fontsize=7.1)
    axis.axvline(0.0, color="#4B5563", linestyle="--", linewidth=0.8)
    axis.set_yticks(positions, labels)
    axis.set_xlabel("目标恢复 − 匹配对照（百分点）" if language == "zh" else "Target restore − matched control (pp)")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.55)
    axis.tick_params(axis="y", length=0)
    axis.spines[["top", "right", "left"]].set_visible(False)


def render_network_alignment_figure(
    *,
    output_base: str | Path,
    language: str,
    cell_summary: pd.DataFrame,
    patient_global: pd.DataFrame,
    transitions: Sequence[str],
    restore_nodes: Sequence[str],
    example_image: np.ndarray,
    example_truth: np.ndarray,
    example_states: Mapping[str, np.ndarray],
    example_target_mask: np.ndarray,
    example_identity: Mapping[str, Any],
    dpi: int = 600,
    bootstrap_iterations: int = 10000,
    bootstrap_seed: int = 20260717,
) -> dict[str, Any]:
    """Render complete 7x7 matrices, one fixed counterfactual case, and global effects."""
    configure_figure_style(language)
    transition_names = tuple(str(value) for value in transitions)
    restore_names = tuple(str(value) for value in restore_nodes)
    matrices, support = _alignment_matrices(
        cell_summary, transitions=transition_names, restore_nodes=restore_names
    )
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices.values()])
    limit = max(0.01, float(np.max(np.abs(finite))) if finite.size else 0.01)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#D9DDE3")
    global_records = _patient_bootstrap_summary(
        patient_global, iterations=bootstrap_iterations, seed=bootstrap_seed
    )

    figure = plt.figure(figsize=(14.8, 7.5), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.92],
        height_ratios=[1.12, 0.72],
        left=0.055,
        right=0.985,
        top=0.96,
        bottom=0.075,
        wspace=0.32,
        hspace=0.34,
    )
    images = []
    for column, model in enumerate(_MODEL_ORDER):
        images.append(
            _draw_matrix(
                figure.add_subplot(grid[0, column]),
                matrices[model],
                model=model,
                transitions=transition_names,
                restore_nodes=restore_names,
                norm=norm,
                cmap=cmap,
                language=language,
            )
        )
    colorbar = figure.colorbar(images[0], ax=[figure.axes[0], figure.axes[1]], fraction=0.025, pad=0.02)
    colorbar.set_label("特异恢复率" if language == "zh" else "Specific recovery rate")
    example_record = _draw_example(
        figure,
        grid[0, 2],
        image=example_image,
        truth=example_truth,
        states=example_states,
        target_mask=example_target_mask,
        language=language,
    )
    _draw_global_effects(figure.add_subplot(grid[1, :]), global_records, language=language)
    png_path, pdf_path = save_publication_figure(figure, output_base, dpi=dpi)
    identity = {str(key): _json_ready(value) for key, value in example_identity.items()}
    identity.update(example_record)
    identity["same_case_and_slice_across_conditions"] = True
    manifest = {
        "figure": "network_process_intervention_alignment",
        "language": language,
        "matrix_shape": [7, 7],
        "dual_axis": False,
        "transitions": list(transition_names),
        "restore_nodes": list(restore_names),
        "matrices": {model: _json_ready(matrix) for model, matrix in matrices.items()},
        "evaluable_seed_counts": {model: support[model].tolist() for model in _MODEL_ORDER},
        "color_limit": limit,
        "global_effects": global_records,
        "example": identity,
        "outputs": {"png": str(png_path), "pdf": str(pdf_path)},
        "dpi": int(dpi),
    }
    Path(output_base).with_suffix(".json").write_text(
        json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = ["render_network_alignment_figure"]
