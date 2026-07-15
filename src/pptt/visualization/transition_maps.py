from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import patches, pyplot as plt

from pptt.transitions.fields import TransitionEvent
from pptt.visualization.common import save_publication_figure


EVENT_COLORS = {
    TransitionEvent.STABLE_CORRECT: np.array([0.86, 0.88, 0.90]),
    TransitionEvent.CORRECTION: np.array([0.12, 0.62, 0.33]),
    TransitionEvent.DESTRUCTION: np.array([0.84, 0.18, 0.20]),
    TransitionEvent.STABLE_ERROR: np.array([0.43, 0.46, 0.50]),
    TransitionEvent.WRONG_REENCODING: np.array([0.95, 0.70, 0.12]),
    TransitionEvent.UNCERTAIN: np.array([0.95, 0.95, 0.95]),
}


def event_rgb_image(events: np.ndarray) -> np.ndarray:
    values = np.asarray(events)
    if values.ndim != 2:
        raise ValueError("events must be a two-dimensional map")
    allowed = {int(event) for event in TransitionEvent}
    if not set(np.unique(values)).issubset(allowed):
        raise ValueError("events contain an unsupported transition code")
    rgb = np.zeros((*values.shape, 3), dtype=np.float32)
    for event, color in EVENT_COLORS.items():
        rgb[values == int(event)] = color
    uncertain = values == int(TransitionEvent.UNCERTAIN)
    rows, columns = np.indices(values.shape)
    texture = uncertain & ((rows + columns) % 6 < 2)
    rgb[texture] = np.array([0.20, 0.22, 0.25])
    return rgb


def render_transition_map(
    events: np.ndarray,
    output_base: str | Path,
    *,
    transition_label: str,
    language: str,
    dpi: int = 400,
) -> tuple[Path, Path]:
    labels = {
        "en": {
            "title": f"Pixel transitions: {transition_label}",
            "stable_correct": "Stable correct",
            "correction": "Correction",
            "destruction": "Destruction",
            "stable_error": "Stable error",
            "wrong": "Wrong re-encoding",
            "uncertain": "Uncertain",
        },
        "zh": {
            "title": f"像素状态转移：{transition_label}",
            "stable_correct": "稳定正确",
            "correction": "修正",
            "destruction": "破坏",
            "stable_error": "稳定错误",
            "wrong": "错误重编码",
            "uncertain": "不确定",
        },
    }[language]
    figure, axis = plt.subplots(figsize=(5.0, 4.5), constrained_layout=True)
    axis.imshow(event_rgb_image(events), interpolation="nearest")
    axis.set_title(labels["title"], pad=8)
    axis.set_axis_off()
    legend = [
        patches.Patch(color=EVENT_COLORS[event], label=labels[name])
        for event, name in (
            (TransitionEvent.STABLE_CORRECT, "stable_correct"),
            (TransitionEvent.CORRECTION, "correction"),
            (TransitionEvent.DESTRUCTION, "destruction"),
            (TransitionEvent.STABLE_ERROR, "stable_error"),
            (TransitionEvent.WRONG_REENCODING, "wrong"),
        )
    ]
    legend.append(
        patches.Patch(
            facecolor=EVENT_COLORS[TransitionEvent.UNCERTAIN],
            edgecolor="#34383f",
            hatch="///",
            label=labels["uncertain"],
        )
    )
    axis.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
    )
    return save_publication_figure(figure, output_base, dpi=dpi)
