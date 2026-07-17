"""Publication overview for Pixel-wise Prediction Transition Tracing (PPTT)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

from pptt.visualization.common import configure_figure_style, save_publication_figure
from pptt.visualization.flagship import BRATS_CLASS_COLORS, segmentation_overlay


NODES = ("down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4")
EVENT_COLORS = {
    "correction": "#0F766E",
    "destruction": "#D97706",
    "wrong_reencoding": "#7C3AED",
    "stable_correct": "#4F7CAC",
}


def _normalize(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("image must be a finite two-dimensional array")
    low, high = np.quantile(values, [0.01, 0.99])
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _roi(mask: np.ndarray, *, margin: int = 8) -> tuple[slice, slice]:
    rows, columns = np.where(mask)
    if rows.size == 0:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    return (
        slice(max(0, int(rows.min()) - margin), min(mask.shape[0], int(rows.max()) + margin + 1)),
        slice(max(0, int(columns.min()) - margin), min(mask.shape[1], int(columns.max()) + margin + 1)),
    )


def _select_process_pixel(
    states: np.ndarray,
    truth: np.ndarray,
    reliable: np.ndarray,
) -> tuple[int, int, int]:
    tumor = truth > 0
    center = np.asarray(np.where(tumor)).mean(axis=1) if tumor.any() else np.asarray(truth.shape) / 2
    best: tuple[int, np.ndarray] | None = None
    for transition in range(states.shape[0] - 1):
        persistent = np.all(
            states[transition + 1 :] == truth[np.newaxis, ...],
            axis=0,
        )
        candidates = (
            tumor
            & reliable[transition]
            & (states[transition] != truth)
            & (states[transition + 1] == truth)
            & persistent
        )
        if candidates.any() and (best is None or candidates.sum() > best[1].sum()):
            best = (transition, candidates)
    if best is None:
        candidates = tumor if tumor.any() else np.ones_like(truth, dtype=bool)
        transition = states.shape[0] - 2
    else:
        transition, candidates = best
    coordinates = np.column_stack(np.where(candidates))
    distance = np.square(coordinates - center[np.newaxis, :]).sum(axis=1)
    row, column = coordinates[int(np.argmin(distance))]
    return int(transition), int(row), int(column)


def _event_fields(
    states: np.ndarray,
    truth: np.ndarray,
    reliable: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    event_maps = np.zeros((states.shape[0] - 1, *truth.shape), dtype=np.uint8)
    rates = np.zeros((states.shape[0] - 1, 3), dtype=np.float64)
    tumor = truth > 0
    for transition in range(states.shape[0] - 1):
        current = states[transition]
        following = states[transition + 1]
        eligible = tumor & reliable[transition]
        correction = eligible & (current != truth) & (following == truth)
        destruction = eligible & (current == truth) & (following != truth)
        reencoding = (
            eligible
            & (current != truth)
            & (following != truth)
            & (current != following)
        )
        stable = eligible & (current == truth) & (following == truth)
        event_maps[transition, correction] = 1
        event_maps[transition, destruction] = 2
        event_maps[transition, reencoding] = 3
        event_maps[transition, stable] = 4
        denominator = max(1, int(eligible.sum()))
        rates[transition] = (
            correction.sum() / denominator,
            destruction.sum() / denominator,
            reencoding.sum() / denominator,
        )
    return event_maps, rates


def _transition_matrix(
    states: np.ndarray,
    truth: np.ndarray,
    reliable: np.ndarray,
    *,
    transition: int,
    truth_class: int,
) -> np.ndarray:
    selected = (truth == truth_class) & reliable[transition]
    matrix = np.zeros((4, 4), dtype=np.float64)
    for source, target in zip(
        states[transition][selected],
        states[transition + 1][selected],
        strict=True,
    ):
        matrix[int(source), int(target)] += 1.0
    if matrix.sum() > 0:
        matrix /= matrix.sum()
    return matrix


def _persistent_depth(states: np.ndarray, truth: np.ndarray) -> np.ndarray:
    tumor = truth > 0
    depth = np.full(truth.shape, states.shape[0], dtype=np.int16)
    for node in range(states.shape[0]):
        persistent = np.all(states[node:] == truth[np.newaxis, ...], axis=0)
        depth[(depth == states.shape[0]) & tumor & persistent] = node
    return depth[tumor]


def _figure_arrow(
    figure: plt.Figure,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#52606D",
    linewidth: float = 1.2,
    style: str = "-|>",
    linestyle: str = "-",
) -> None:
    figure.add_artist(
        FancyArrowPatch(
            start,
            end,
            transform=figure.transFigure,
            arrowstyle=style,
            mutation_scale=10,
            linewidth=linewidth,
            color=color,
            linestyle=linestyle,
        )
    )


def _draw_network(
    axis: plt.Axes,
    figure: plt.Figure,
    state_centers: list[float],
    *,
    language: str,
) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    x_values = np.linspace(0.06, 0.94, len(NODES))
    y_values = np.asarray([0.82, 0.61, 0.40, 0.19, 0.19, 0.40, 0.61, 0.82])
    widths = np.asarray([0.055, 0.050, 0.046, 0.042, 0.042, 0.046, 0.050, 0.055])
    colors = ["#7EA6D8"] * 4 + ["#77A86D"] * 4
    for index, (x, y, width, color) in enumerate(
        zip(x_values, y_values, widths, colors, strict=True)
    ):
        if index:
            axis.annotate(
                "",
                xy=(x - width / 2, y),
                xytext=(x_values[index - 1] + widths[index - 1] / 2, y_values[index - 1]),
                arrowprops={"arrowstyle": "->", "color": "#52606D", "lw": 1.1},
            )
        for offset in (0.014, 0.007, 0.0):
            axis.add_patch(
                Rectangle(
                    (x - width / 2 + offset, y - 0.10 + offset),
                    width,
                    0.20,
                    facecolor=color,
                    edgecolor="#30445A",
                    linewidth=0.65,
                    alpha=0.35 if offset else 0.92,
                )
            )
        axis.text(x, y - 0.145, NODES[index], ha="center", va="top", fontsize=7.2)
    for encoder, decoder in ((0, 7), (1, 6), (2, 5)):
        axis.annotate(
            "",
            xy=(x_values[decoder], y_values[decoder] + 0.10),
            xytext=(x_values[encoder], y_values[encoder] + 0.10),
            arrowprops={
                "arrowstyle": "->",
                "color": "#9AA5B1",
                "lw": 0.8,
                "linestyle": "--",
                "connectionstyle": "arc3,rad=-0.08",
            },
        )
    axis.text(
        0.5,
        1.03,
        "冻结的分割网络" if language == "zh" else "Frozen segmentation network",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#243B53",
    )
    bbox = axis.get_position()
    for x, target_x in zip(x_values, state_centers, strict=True):
        source = (bbox.x0 + bbox.width * x, bbox.y0 + bbox.height * 0.02)
        _figure_arrow(
            figure,
            source,
            (target_x, 0.615),
            color="#7B8794",
            linewidth=0.75,
            style="-|>",
        )


def render_pptt_method_figure(
    *,
    output_base: str | Path,
    language: str,
    image: np.ndarray,
    truth: np.ndarray,
    states: np.ndarray,
    reliable: np.ndarray,
    final_state: np.ndarray,
    dpi: int = 400,
) -> tuple[Path, Path]:
    """Render a continuous, non-numbered graphical explanation of PPTT."""
    image_array = _normalize(image)
    truth_array = np.asarray(truth)
    state_path = np.asarray(states)
    reliability = np.asarray(reliable)
    final = np.asarray(final_state)
    if (
        truth_array.ndim != 2
        or state_path.shape != (len(NODES), *truth_array.shape)
        or reliability.shape != (len(NODES) - 1, *truth_array.shape)
        or reliability.dtype != np.bool_
        or final.shape != truth_array.shape
    ):
        raise ValueError("method figure inputs do not match the locked eight-node path")
    configure_figure_style(language)
    tumor = truth_array > 0
    crop = _roi(tumor, margin=10)
    transition, pixel_row, pixel_column = _select_process_pixel(
        state_path,
        truth_array,
        reliability,
    )
    event_maps, event_rates = _event_fields(state_path, truth_array, reliability)
    truth_class = int(truth_array[pixel_row, pixel_column])
    tensor = _transition_matrix(
        state_path,
        truth_array,
        reliability,
        transition=transition,
        truth_class=truth_class,
    )
    depths = _persistent_depth(state_path, truth_array)

    figure = plt.figure(figsize=(16.0, 8.4), constrained_layout=False)
    input_axis = figure.add_axes([0.025, 0.715, 0.105, 0.215])
    input_axis.imshow(image_array[crop], cmap="gray", vmin=0, vmax=1)
    input_axis.set_title("MRI 输入" if language == "zh" else "MRI input", fontsize=8.5)
    input_axis.axis("off")
    output_axis = figure.add_axes([0.875, 0.715, 0.105, 0.215])
    output_axis.imshow(
        segmentation_overlay(
            image_array,
            final,
            colors=BRATS_CLASS_COLORS,
            alpha=0.55,
        )[crop]
    )
    output_axis.set_title("最终分割" if language == "zh" else "Final segmentation", fontsize=8.5)
    output_axis.axis("off")
    state_centers = np.linspace(0.18, 0.82, len(NODES)).tolist()
    network_axis = figure.add_axes([0.155, 0.715, 0.68, 0.215])
    _draw_network(network_axis, figure, state_centers, language=language)
    _figure_arrow(figure, (0.132, 0.82), (0.153, 0.82), color="#365F91", linewidth=1.8)
    _figure_arrow(figure, (0.84, 0.82), (0.872, 0.82), color="#365F91", linewidth=1.8)

    figure.add_artist(
        plt.Line2D([0.02, 0.98], [0.675, 0.675], transform=figure.transFigure, color="#2F6FAB", lw=1.4)
    )
    figure.text(
        0.03,
        0.655,
        "PPTT  像素预测转移追踪" if language == "zh" else "PPTT  Pixel-wise Prediction Transition Tracing",
        color="#1F4E79",
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    figure.text(
        0.79,
        0.655,
        r"$\pi_k(h_k)\rightarrow s_k(p),\quad k=1,\ldots,K$",
        color="#52606D",
        fontsize=8.0,
        va="top",
    )

    map_width = 0.072
    map_height = 0.135
    class_colors = {0: "#A8B2BD", 1: "#D1495B", 2: "#2CA25F", 3: "#E9C46A"}
    crop_row = pixel_row - crop[0].start
    crop_column = pixel_column - crop[1].start
    for node_index, center_x in enumerate(state_centers):
        axis = figure.add_axes([center_x - map_width / 2, 0.455, map_width, map_height])
        axis.imshow(
            segmentation_overlay(
                image_array,
                state_path[node_index],
                colors=BRATS_CLASS_COLORS,
                alpha=0.54,
            )[crop]
        )
        axis.scatter(
            [crop_column],
            [crop_row],
            s=19,
            facecolors="none",
            edgecolors="white",
            linewidths=0.9,
        )
        axis.axis("off")
        axis.text(0.5, -0.07, NODES[node_index], transform=axis.transAxes, ha="center", va="top", fontsize=7.1)
        predicted = int(state_path[node_index, pixel_row, pixel_column])
        tile_axis = figure.add_axes([center_x - 0.010, 0.405, 0.020, 0.025])
        tile_axis.add_patch(
            Rectangle((0, 0), 1, 1, facecolor=class_colors[predicted], edgecolor="#243B53", linewidth=0.7)
        )
        tile_axis.axis("off")
        if node_index < len(NODES) - 1:
            next_center = state_centers[node_index + 1]
            arrow_color = EVENT_COLORS["correction"] if node_index == transition else "#7B8794"
            _figure_arrow(
                figure,
                (center_x + 0.012, 0.417),
                (next_center - 0.012, 0.417),
                color=arrow_color,
                linewidth=2.2 if node_index == transition else 0.9,
            )
    truth_axis = figure.add_axes([0.115, 0.405, 0.024, 0.030])
    truth_axis.add_patch(
        Rectangle((0, 0), 1, 1, facecolor=class_colors[truth_class], edgecolor="#243B53", linewidth=0.7)
    )
    truth_axis.axis("off")
    figure.text(0.108, 0.441, r"$y(p)$", fontsize=8.0)
    _figure_arrow(figure, (0.142, 0.418), (state_centers[0] - 0.014, 0.418), color="#7B8794", linewidth=0.9)
    figure.text(
        (state_centers[transition] + state_centers[transition + 1]) / 2,
        0.385,
        "纠正并持续" if language == "zh" else "correction → persistence",
        ha="center",
        va="top",
        color=EVENT_COLORS["correction"],
        fontsize=7.5,
        fontweight="bold",
    )

    tensor_axis = figure.add_axes([0.045, 0.075, 0.19, 0.23])
    tensor_axis.imshow(tensor, cmap="Blues", vmin=0, vmax=max(0.01, float(tensor.max())))
    tensor_axis.set_xticks(range(4), range(4), fontsize=7)
    tensor_axis.set_yticks(range(4), range(4), fontsize=7)
    tensor_axis.set_xlabel(r"$s_{t+1}(p)$", fontsize=8)
    tensor_axis.set_ylabel(r"$s_t(p)$", fontsize=8)
    tensor_axis.set_title(rf"$T_{{t={transition + 1}}}^{{(y={truth_class})}}$", fontsize=9)
    for row in range(4):
        for column in range(4):
            if tensor[row, column] > 0:
                tensor_axis.text(column, row, f"{tensor[row, column]:.2f}", ha="center", va="center", fontsize=6.4)

    flow_axis = figure.add_axes([0.285, 0.075, 0.27, 0.23])
    x = np.arange(1, len(NODES))
    transition_labels = [f"{NODES[index]}→{NODES[index + 1]}" for index in range(len(NODES) - 1)]
    for column, (label, color) in enumerate(
        (
            (("纠正" if language == "zh" else "Correction"), EVENT_COLORS["correction"]),
            (("破坏" if language == "zh" else "Destruction"), EVENT_COLORS["destruction"]),
            (("错误重编码" if language == "zh" else "Wrong re-encoding"), EVENT_COLORS["wrong_reencoding"]),
        )
    ):
        flow_axis.plot(x, event_rates[:, column], marker="o", markersize=3.5, color=color, label=label)
    flow_axis.axvline(transition + 1, color="#0F766E", linestyle="--", linewidth=0.9)
    flow_axis.set_xticks(x, transition_labels, rotation=32, ha="right", fontsize=6.5)
    flow_axis.set_ylabel("像素比例" if language == "zh" else "Pixel fraction")
    flow_axis.set_ylim(bottom=0)
    flow_axis.spines[["top", "right"]].set_visible(False)
    flow_axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    flow_axis.legend(frameon=False, fontsize=7.0, loc="upper left")

    spatial_axis = figure.add_axes([0.605, 0.075, 0.14, 0.23])
    spatial_axis.imshow(image_array[crop], cmap="gray", vmin=0, vmax=1)
    selected_events = event_maps[transition][crop]
    event_cmap = ListedColormap(
        [
            EVENT_COLORS["correction"],
            EVENT_COLORS["destruction"],
            EVENT_COLORS["wrong_reencoding"],
            EVENT_COLORS["stable_correct"],
        ]
    )
    spatial_axis.imshow(
        np.ma.masked_where(selected_events == 0, selected_events),
        cmap=event_cmap,
        vmin=1,
        vmax=4,
        alpha=0.78,
        interpolation="nearest",
    )
    spatial_axis.set_title("空间转移图" if language == "zh" else "Spatial transition map", fontsize=8.7)
    spatial_axis.axis("off")

    depth_axis = figure.add_axes([0.79, 0.075, 0.185, 0.23])
    valid_depths = depths[depths < len(NODES)]
    counts = np.bincount(valid_depths, minlength=len(NODES)).astype(float)
    denominator = max(1.0, float(depths.size))
    depth_axis.bar(np.arange(len(NODES)), counts / denominator, color="#4F7CAC", edgecolor="#243B53", linewidth=0.45)
    depth_axis.set_xticks(np.arange(len(NODES)), NODES, rotation=32, ha="right", fontsize=6.5)
    depth_axis.set_ylabel("像素比例" if language == "zh" else "Pixel fraction")
    depth_axis.set_title("持续正确起点" if language == "zh" else "Persistent-correct onset", fontsize=8.7)
    depth_axis.spines[["top", "right"]].set_visible(False)
    depth_axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)

    _figure_arrow(
        figure,
        ((state_centers[transition] + state_centers[transition + 1]) / 2, 0.375),
        (0.14, 0.315),
        color="#2F6FAB",
        linewidth=1.2,
    )
    _figure_arrow(figure, (0.245, 0.19), (0.278, 0.19), color="#2F6FAB", linewidth=1.3)
    _figure_arrow(figure, (0.565, 0.19), (0.598, 0.19), color="#2F6FAB", linewidth=1.3)
    _figure_arrow(figure, (0.752, 0.19), (0.782, 0.19), color="#2F6FAB", linewidth=1.3)
    return save_publication_figure(figure, output_base, dpi=dpi)


__all__ = ["render_pptt_method_figure"]
