from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pptt.visualization.common import save_publication_figure


def render_synthetic_region_comparison(
    payload: dict[str, Any],
    output_base: str | Path,
    *,
    language: str,
    dpi: int = 400,
) -> tuple[Path, Path]:
    values = [
        float(payload["pptt"]["region_iou"]),
        float(payload["comparators"]["gradcam"]["region_iou"]),
        float(payload["comparators"]["layercam"]["region_iou"]),
    ]
    title = (
        "Known-region recovery on the controlled benchmark"
        if language == "en"
        else "受控基准中的已知区域恢复"
    )
    ylabel = "Region IoU" if language == "en" else "区域 IoU"
    figure, axis = plt.subplots(figsize=(4.8, 3.5), constrained_layout=True)
    bars = axis.bar(
        ["PPTT", "Grad-CAM", "LayerCAM"],
        values,
        color=["#2166ac", "#d6604d", "#f4a582"],
        width=0.62,
    )
    axis.bar_label(bars, fmt="%.2f", padding=3)
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    return save_publication_figure(figure, output_base, dpi=dpi)


def render_observer_selectivity(
    frame: pd.DataFrame,
    output_base: str | Path,
    *,
    language: str,
    dpi: int = 400,
) -> tuple[Path, Path]:
    required = {"node", "control", "mean_difference", "ci_low", "ci_high"}
    if not required.issubset(frame) or frame.empty:
        raise ValueError("observer selectivity table is incomplete")
    summary = frame.groupby(["node", "control"], as_index=False).agg(
        mean_difference=("mean_difference", "mean"),
        ci_low=("ci_low", "mean"),
        ci_high=("ci_high", "mean"),
    )
    nodes = tuple(dict.fromkeys(summary.node))
    figure, axis = plt.subplots(figsize=(7.0, 3.8))
    styles = {
        "patient_permutation": ("#762a83", "o"),
        "spatial_shift": ("#1b7837", "s"),
    }
    for control, selected in summary.groupby("control"):
        selected = selected.set_index("node").reindex(nodes)
        color, marker = styles.get(control, ("#4d4d4d", "o"))
        center = selected.mean_difference.to_numpy(dtype=np.float64)
        low = selected.ci_low.to_numpy(dtype=np.float64)
        high = selected.ci_high.to_numpy(dtype=np.float64)
        axis.errorbar(
            np.arange(len(nodes)),
            center,
            yerr=np.vstack([center - low, high - center]),
            color=color,
            marker=marker,
            capsize=3,
            label=control.replace("_", " "),
        )
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set_xticks(np.arange(len(nodes)), nodes, rotation=25, ha="right")
    axis.set_ylabel(
        "Control minus real JSD" if language == "en" else "控制减真实观察器 JSD"
    )
    axis.set_title(
        "Observer selectivity controls" if language == "en" else "观察器选择性控制"
    )
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    figure.subplots_adjust(bottom=0.22)
    return save_publication_figure(figure, output_base, dpi=dpi)


def render_intervention_dose(
    frame: pd.DataFrame,
    output_base: str | Path,
    *,
    language: str,
    dpi: int = 400,
) -> tuple[Path, Path]:
    selected = frame[frame.condition.str.startswith("replacement_alpha_")]
    if selected.empty or not {"alpha", "selective_path_effect"}.issubset(selected):
        raise ValueError("intervention dose table is incomplete")
    patient_curves = selected.pivot_table(
        index="patient_id",
        columns="alpha",
        values="selective_path_effect",
    ).sort_index(axis=1)
    alpha = patient_curves.columns.to_numpy(dtype=np.float64)
    values = patient_curves.to_numpy(dtype=np.float64)
    mean = values.mean(axis=0)
    ci_low, ci_high = np.quantile(values, [0.025, 0.975], axis=0)
    figure, axis = plt.subplots(figsize=(4.8, 3.5), constrained_layout=True)
    axis.fill_between(alpha, ci_low, ci_high, color="#92c5de", alpha=0.45)
    axis.plot(alpha, mean, color="#2166ac", marker="o")
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set_xlabel("Replacement dose" if language == "en" else "替换剂量")
    axis.set_ylabel("Selective path effect" if language == "en" else "选择性路径效应")
    axis.set_title("Natural-activation dose response" if language == "en" else "自然激活替换剂量响应")
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    return save_publication_figure(figure, output_base, dpi=dpi)


def render_transfer_admission(
    frame: pd.DataFrame,
    output_base: str | Path,
    *,
    language: str,
    dpi: int = 400,
) -> tuple[Path, Path]:
    if frame.empty or not {"node", "node_admitted"}.issubset(frame):
        raise ValueError("transfer admission table is incomplete")
    summary = frame.groupby("node", sort=False).node_admitted.mean()
    figure, axis = plt.subplots(figsize=(6.2, 3.3), constrained_layout=True)
    colors = np.where(summary.to_numpy() == 1.0, "#1b7837", "#d6604d")
    axis.bar(summary.index, summary.to_numpy(), color=colors)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Admitted model fraction" if language == "en" else "通过模型比例")
    axis.set_title("Cross-architecture node admission" if language == "en" else "跨架构节点准入")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    return save_publication_figure(figure, output_base, dpi=dpi)
