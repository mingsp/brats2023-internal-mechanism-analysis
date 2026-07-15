from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pptt.visualization.common import save_publication_figure


MODEL_COLORS = {
    "unet_baseline": "#2166ac",
    "unet_noskip": "#d6604d",
    "transunet_r50_vit_b16": "#1b7837",
}


def render_flow_and_reconstruction(
    node_frame: pd.DataFrame,
    transition_frame: pd.DataFrame,
    output_base: str | Path,
    *,
    nodes: tuple[str, ...],
    language: str,
    dpi: int = 400,
) -> tuple[Path, Path]:
    required_node = {"model", "node", "class_index", "dice"}
    required_transition = {"model", "transition", "persistent_net_rate"}
    if not required_node.issubset(node_frame) or not required_transition.issubset(
        transition_frame
    ):
        raise ValueError("flow figure input tables are incomplete")
    labels = {
        "en": {
            "title": "Persistent state flow and reconstructed Dice",
            "flow": "Persistent net flow",
            "dice": "Macro tumor Dice",
            "node": "Declared checkpoint",
            "baseline": "U-Net",
            "noskip": "No-skip U-Net",
            "transunet": "TransUNet",
        },
        "zh": {
            "title": "持续状态流与重构 Dice",
            "flow": "持续净流",
            "dice": "肿瘤宏平均 Dice",
            "node": "预声明检查点",
            "baseline": "有跳接 U-Net",
            "noskip": "无跳接 U-Net",
            "transunet": "TransUNet",
        },
    }[language]
    model_names = {
        "unet_baseline": labels["baseline"],
        "unet_noskip": labels["noskip"],
        "transunet_r50_vit_b16": labels["transunet"],
    }
    figure, left_axis = plt.subplots(figsize=(7.4, 4.2))
    right_axis = left_axis.twinx()
    x = np.arange(len(nodes))
    handles = []
    for model in sorted(set(node_frame.model) & set(transition_frame.model)):
        color = MODEL_COLORS.get(model, "#4d4d4d")
        node_selected = node_frame[
            (node_frame.model == model) & (node_frame.class_index == -1)
        ]
        node_values = node_selected.groupby("node").dice.mean().reindex(nodes)
        transitions = tuple(
            f"{left}->{right}" for left, right in zip(nodes[:-1], nodes[1:], strict=True)
        )
        transition_selected = transition_frame[transition_frame.model == model]
        flow_values = (
            transition_selected.groupby("transition")
            .persistent_net_rate.mean()
            .reindex(transitions)
        )
        flow_line = left_axis.plot(
            x[1:],
            flow_values.to_numpy(dtype=np.float64),
            color=color,
            marker="o",
            linestyle="-",
            label=f"{model_names.get(model, model)}: {labels['flow']}",
        )[0]
        dice_line = right_axis.plot(
            x,
            node_values.to_numpy(dtype=np.float64),
            color=color,
            marker="s",
            linestyle="--",
            alpha=0.82,
            label=f"{model_names.get(model, model)}: {labels['dice']}",
        )[0]
        handles.extend([flow_line, dice_line])
    left_axis.axhline(0.0, color="#777777", linewidth=0.8, zorder=0)
    left_axis.set_xticks(x, nodes, rotation=25, ha="right")
    left_axis.set_xlabel(labels["node"])
    left_axis.set_ylabel(labels["flow"])
    right_axis.set_ylabel(labels["dice"])
    right_axis.set_ylim(0.0, 1.0)
    left_axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    left_axis.set_title(labels["title"])
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        frameon=False,
    )
    figure.subplots_adjust(bottom=0.25, top=0.90, left=0.10, right=0.90)
    return save_publication_figure(figure, output_base, dpi=dpi)
