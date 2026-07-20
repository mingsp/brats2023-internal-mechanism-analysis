"""Focused publication overview for Pixel Prediction Transition Tracing."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

from pptt.visualization.common import configure_figure_style, save_publication_figure
from pptt.visualization.flagship import BRATS_CLASS_COLORS, segmentation_overlay


NODES = ("down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4")
CORRECTION = "#0F8B8D"
DAMAGE = "#E07A2D"
REENCODING = "#7551A6"
NEUTRAL = "#60758A"


def _normalize(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("image must be a finite two-dimensional array")
    low, high = np.quantile(values, [0.01, 0.99])
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _roi(mask: np.ndarray, *, margin: int = 10) -> tuple[slice, slice]:
    rows, columns = np.where(mask)
    if rows.size == 0:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    return (
        slice(
            max(0, int(rows.min()) - margin),
            min(mask.shape[0], int(rows.max()) + margin + 1),
        ),
        slice(
            max(0, int(columns.min()) - margin),
            min(mask.shape[1], int(columns.max()) + margin + 1),
        ),
    )


def _persistent_events(
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
        persistent_wrong = np.all(
            states[transition + 1 :] != truth[np.newaxis, ...],
            axis=0,
        )
        eligible = tumor & reliable_suffix[transition]
        correction = (
            eligible
            & (current != truth)
            & (following == truth)
            & persistent_correct
            & (final_state == truth)
        )
        damage = (
            eligible
            & (current == truth)
            & (following != truth)
            & persistent_wrong
            & (final_state != truth)
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


def _select_process_pixel(
    events: np.ndarray,
    truth: np.ndarray,
) -> tuple[int, int, int]:
    correction_counts = (events == 1).sum(axis=(1, 2))
    if correction_counts.max(initial=0) > 0:
        transition = int(np.argmax(correction_counts))
        candidates = events[transition] == 1
    else:
        transition = int(np.argmax((events > 0).sum(axis=(1, 2))))
        candidates = events[transition] > 0
    if not candidates.any():
        candidates = truth > 0
    if not candidates.any():
        candidates = np.ones_like(truth, dtype=bool)
    center = (
        np.asarray(np.where(truth > 0)).mean(axis=1)
        if np.any(truth > 0)
        else np.asarray(truth.shape) / 2
    )
    coordinates = np.column_stack(np.where(candidates))
    distances = np.square(coordinates - center[np.newaxis, :]).sum(axis=1)
    row, column = coordinates[int(np.argmin(distances))]
    return transition, int(row), int(column)


def _transition_tensor(
    states: np.ndarray,
    truth: np.ndarray,
    reliable: np.ndarray,
    *,
    transition: int,
    truth_class: int,
) -> np.ndarray:
    selected = (truth == truth_class) & reliable[transition]
    matrix = np.zeros((4, 4), dtype=np.float64)
    np.add.at(
        matrix,
        (states[transition][selected], states[transition + 1][selected]),
        1.0,
    )
    if matrix.sum() > 0:
        matrix /= matrix.sum()
    return matrix


def _formation_depth(states: np.ndarray, truth: np.ndarray) -> np.ndarray:
    depth = np.full(truth.shape, -1, dtype=np.int16)
    for node in range(states.shape[0]):
        persistent = np.all(states[node:] == truth[np.newaxis, ...], axis=0)
        depth[(depth < 0) & (truth > 0) & persistent] = node
    return depth


def _arrow(
    figure: plt.Figure,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#486581",
    linewidth: float = 1.3,
) -> None:
    figure.add_artist(
        FancyArrowPatch(
            start,
            end,
            transform=figure.transFigure,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=linewidth,
            color=color,
        )
    )


def _draw_network(axis: plt.Axes, *, language: str) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    x = np.linspace(0.05, 0.95, 8)
    y = np.asarray([0.80, 0.60, 0.40, 0.20, 0.20, 0.40, 0.60, 0.80])
    widths = np.asarray([0.050, 0.047, 0.044, 0.041, 0.041, 0.044, 0.047, 0.050])
    colors = ["#78A6D3"] * 4 + ["#79A96B"] * 4
    for index, (center_x, center_y, width, color) in enumerate(
        zip(x, y, widths, colors, strict=True)
    ):
        if index:
            axis.annotate(
                "",
                xy=(center_x - width / 2, center_y),
                xytext=(x[index - 1] + widths[index - 1] / 2, y[index - 1]),
                arrowprops={"arrowstyle": "->", "color": "#52606D", "lw": 1.0},
            )
        for offset, alpha in ((0.012, 0.28), (0.006, 0.48), (0.0, 0.92)):
            axis.add_patch(
                Rectangle(
                    (center_x - width / 2 + offset, center_y - 0.095 + offset),
                    width,
                    0.19,
                    facecolor=color,
                    edgecolor="#334E68",
                    linewidth=0.6,
                    alpha=alpha,
                )
            )
        axis.text(center_x, center_y - 0.13, NODES[index], ha="center", va="top", fontsize=7.0)
    for encoder, decoder in ((0, 7), (1, 6), (2, 5)):
        axis.annotate(
            "",
            xy=(x[decoder], y[decoder] + 0.10),
            xytext=(x[encoder], y[encoder] + 0.10),
            arrowprops={
                "arrowstyle": "->",
                "color": "#A0AEC0",
                "lw": 0.75,
                "linestyle": "--",
                "connectionstyle": "arc3,rad=-0.08",
            },
        )
    axis.text(
        0.5,
        1.01,
        (
            "冻结网络与有序检查点（U 形网络为适配示例）"
            if language == "zh"
            else "Frozen network and ordered checkpoints (U-shaped adapter shown)"
        ),
        ha="center",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color="#243B53",
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
    dpi: int = 600,
) -> tuple[Path, Path]:
    """Render one continuous data flow from frozen network to process outputs."""
    if language not in {"en", "zh"}:
        raise ValueError("language must be 'en' or 'zh'")
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
        raise ValueError("method figure inputs do not match the eight-node path")
    configure_figure_style(language)
    crop = _roi(truth_array > 0)
    events = _persistent_events(state_path, truth_array, reliability, final)
    transition, pixel_row, pixel_column = _select_process_pixel(events, truth_array)
    truth_class = int(truth_array[pixel_row, pixel_column])
    tensor = _transition_tensor(
        state_path,
        truth_array,
        reliability,
        transition=transition,
        truth_class=truth_class,
    )
    depths = _formation_depth(state_path, truth_array)

    figure = plt.figure(figsize=(15.8, 7.0), constrained_layout=False)
    input_axis = figure.add_axes([0.025, 0.705, 0.105, 0.225])
    input_axis.imshow(image_array[crop], cmap="gray", vmin=0, vmax=1)
    input_axis.set_title("MRI 输入" if language == "zh" else "MRI input", fontsize=8.2)
    input_axis.axis("off")
    network_axis = figure.add_axes([0.155, 0.705, 0.68, 0.225])
    _draw_network(network_axis, language=language)
    output_axis = figure.add_axes([0.87, 0.705, 0.105, 0.225])
    output_axis.imshow(
        segmentation_overlay(image_array, final, colors=BRATS_CLASS_COLORS, alpha=0.55)[crop]
    )
    output_axis.set_title("最终输出" if language == "zh" else "Final output", fontsize=8.2)
    output_axis.axis("off")
    _arrow(figure, (0.132, 0.815), (0.153, 0.815), linewidth=1.8)
    _arrow(figure, (0.84, 0.815), (0.868, 0.815), linewidth=1.8)

    figure.text(
        0.025,
        0.655,
        (
            "PPTT：同一像素的预测状态轨迹"
            if language == "zh"
            else "PPTT: prediction-state trajectory of the same pixel"
        ),
        fontsize=11.0,
        fontweight="bold",
        color="#1F4E79",
        va="center",
    )
    figure.text(
        0.745,
        0.655,
        r"$s_k(p)=\arg\max_c\,\pi_k(h_k(p))$",
        fontsize=9.0,
        color="#52606D",
        va="center",
    )
    map_centers = np.linspace(0.135, 0.91, len(NODES))
    map_width = 0.084
    map_height = 0.155
    crop_row = pixel_row - int(crop[0].start)
    crop_column = pixel_column - int(crop[1].start)
    class_colors = {0: "#AEB8C2", 1: "#D1495B", 2: "#2CA25F", 3: "#E9C46A"}
    for node_index, center_x in enumerate(map_centers):
        axis = figure.add_axes([center_x - map_width / 2, 0.425, map_width, map_height])
        axis.imshow(
            segmentation_overlay(
                image_array,
                state_path[node_index],
                colors=BRATS_CLASS_COLORS,
                alpha=0.52,
            )[crop],
            interpolation="nearest",
        )
        axis.scatter(
            [crop_column],
            [crop_row],
            s=22,
            facecolors="none",
            edgecolors="white",
            linewidths=1.0,
        )
        axis.set_title(NODES[node_index], fontsize=7.6, pad=2)
        axis.axis("off")
        predicted = int(state_path[node_index, pixel_row, pixel_column])
        state_axis = figure.add_axes([center_x - 0.011, 0.385, 0.022, 0.026])
        state_axis.add_patch(
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=class_colors[predicted],
                edgecolor="#243B53",
                linewidth=0.7,
            )
        )
        state_axis.axis("off")
        if node_index < len(NODES) - 1:
            event = int(events[node_index, pixel_row, pixel_column])
            color = {1: CORRECTION, 2: DAMAGE, 3: REENCODING}.get(event, "#9AA5B1")
            _arrow(
                figure,
                (center_x + 0.013, 0.398),
                (map_centers[node_index + 1] - 0.013, 0.398),
                color=color,
                linewidth=2.1 if event else 0.8,
            )
    figure.text(
        0.045,
        0.398,
        (r"$y(p)$:" if language == "en" else r"$y(p)$："),
        fontsize=8.0,
        color="#52606D",
        ha="right",
        va="center",
    )
    truth_state_axis = figure.add_axes([0.048, 0.385, 0.022, 0.026])
    truth_state_axis.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=class_colors[truth_class],
            edgecolor="#243B53",
            linewidth=0.7,
        )
    )
    truth_state_axis.axis("off")

    output_y = 0.045
    output_h = 0.205
    tensor_axis = figure.add_axes([0.06, output_y, 0.20, output_h])
    tensor_axis.imshow(tensor, cmap="Blues", vmin=0, vmax=max(0.01, float(tensor.max())))
    tensor_axis.set_xticks(range(4), range(4), fontsize=7)
    tensor_axis.set_yticks(range(4), range(4), fontsize=7)
    tensor_axis.set_xlabel(r"$s_{t+1}(p)$", fontsize=8)
    tensor_axis.set_ylabel(r"$s_t(p)$", fontsize=8)
    tensor_axis.set_title(
        "条件转移张量" if language == "zh" else "Conditional transition tensor",
        fontsize=8.8,
        fontweight="bold",
    )
    for row in range(4):
        for column in range(4):
            if tensor[row, column] > 0:
                tensor_axis.text(
                    column,
                    row,
                    f"{tensor[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                )

    event_axis = figure.add_axes([0.36, output_y, 0.25, output_h])
    selected_events = events[transition][crop]
    event_axis.imshow(image_array[crop], cmap="gray", vmin=0, vmax=1)
    event_axis.imshow(
        np.ma.masked_where(selected_events == 0, selected_events),
        cmap=ListedColormap([CORRECTION, DAMAGE, REENCODING]),
        vmin=1,
        vmax=3,
        alpha=0.82,
        interpolation="nearest",
    )
    event_axis.contour(
        (truth_array[crop] > 0).astype(np.uint8),
        levels=[0.5],
        colors=["white"],
        linewidths=0.55,
    )
    event_axis.set_title(
        (
            f"像素事件：{NODES[transition]}→{NODES[transition + 1]}"
            if language == "zh"
            else f"Pixel events: {NODES[transition]}→{NODES[transition + 1]}"
        ),
        fontsize=8.8,
        fontweight="bold",
    )
    event_axis.axis("off")
    event_labels = (
        ("持续纠正", "持续破坏", "错误重编码")
        if language == "zh"
        else ("Persistent correction", "Persistent damage", "Wrong re-encoding")
    )
    event_axis.legend(
        handles=[
            Rectangle((0, 0), 1, 1, facecolor=color, label=label)
            for color, label in zip(
                (CORRECTION, DAMAGE, REENCODING),
                event_labels,
                strict=True,
            )
        ],
        frameon=False,
        fontsize=6.8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
    )

    depth_axis = figure.add_axes([0.70, output_y, 0.25, output_h])
    tumor_depths = depths[(truth_array > 0) & (depths >= 0)]
    counts = np.bincount(tumor_depths, minlength=len(NODES)).astype(np.float64)
    fractions = counts / max(1.0, float((truth_array > 0).sum()))
    depth_axis.bar(
        np.arange(len(NODES)),
        fractions,
        color="#4F7CAC",
        edgecolor="#334E68",
        linewidth=0.45,
    )
    depth_axis.set_xticks(np.arange(len(NODES)), NODES, rotation=30, ha="right", fontsize=6.5)
    depth_axis.set_ylabel("像素比例" if language == "zh" else "Pixel fraction")
    depth_axis.set_title(
        "持续正确形成深度" if language == "zh" else "Persistent-correct formation depth",
        fontsize=8.8,
        fontweight="bold",
    )
    depth_axis.spines[["top", "right"]].set_visible(False)
    depth_axis.grid(axis="y", color="#E5E7EB", linewidth=0.55)

    operator = FancyBboxPatch(
        (0.415, 0.306),
        0.17,
        0.037,
        transform=figure.transFigure,
        boxstyle="round,pad=0.004,rounding_size=0.005",
        facecolor="#F7FAFC",
        edgecolor="#486581",
        linewidth=0.9,
    )
    figure.add_artist(operator)
    figure.text(
        0.50,
        0.324,
        r"$\mathcal{P}:\{s_k(p)\}_{k=1}^{K}\mapsto(\mathbf{T},\mathbf{e},d_F)$",
        fontsize=8.0,
        color="#243B53",
        ha="center",
        va="center",
    )
    _arrow(figure, (0.50, 0.372), (0.50, 0.346), color=NEUTRAL, linewidth=1.15)
    figure.add_artist(
        plt.Line2D(
            [0.16, 0.825],
            [0.279, 0.279],
            transform=figure.transFigure,
            color=NEUTRAL,
            linewidth=0.9,
        )
    )
    _arrow(figure, (0.50, 0.306), (0.50, 0.282), color=NEUTRAL, linewidth=1.0)
    for output_x in (0.16, 0.485, 0.825):
        _arrow(figure, (output_x, 0.279), (output_x, 0.260), color=NEUTRAL, linewidth=0.9)
    return save_publication_figure(figure, output_base, dpi=dpi)


__all__ = ["render_pptt_method_figure"]
