from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pptt.visualization.common import save_publication_figure


def render_lineage_depth(
    lineage_frame: pd.DataFrame,
    output_base: str | Path,
    *,
    kind: str,
    nodes: tuple[str, ...],
    language: str,
    dpi: int = 400,
) -> tuple[Path, Path]:
    required = {"kind", "class_index", "depth", "count"}
    if not required.issubset(lineage_frame):
        raise ValueError("lineage figure input table is incomplete")
    selected = lineage_frame[lineage_frame.kind == kind]
    if selected.empty:
        raise ValueError(f"lineage kind is unavailable: {kind}")
    title_map = {
        "en": {
            "final_state_depth": "Final-state determination depth (dF)",
            "correct_state_depth": "Correct-state determination depth (dY)",
            "terminal_error_origin_depth": "Terminal-error origin depth (o)",
        },
        "zh": {
            "final_state_depth": "最终状态确定深度（dF）",
            "correct_state_depth": "正确状态确定深度（dY）",
            "terminal_error_origin_depth": "终末错误来源深度（o）",
        },
    }[language]
    class_labels = (
        ("Background", "NCR/NET", "ED", "ET")
        if language == "en"
        else ("背景", "坏死/非增强核心", "水肿", "增强肿瘤")
    )
    depths = tuple(range(-1, len(nodes) + 1))
    counts = (
        selected.groupby(["class_index", "depth"])["count"]
        .sum()
        .unstack(fill_value=0)
    )
    matrix = np.zeros((4, len(depths)), dtype=np.float64)
    for class_index in range(4):
        for column, depth in enumerate(depths):
            if class_index in counts.index and depth in counts.columns:
                matrix[class_index, column] = counts.loc[class_index, depth]
        total = matrix[class_index].sum()
        if total:
            matrix[class_index] /= total
    depth_labels = ["N/A" if depth == -1 else (nodes[depth] if depth < len(nodes) else "final") for depth in depths]
    figure, axis = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=max(0.01, matrix.max()))
    axis.set_xticks(np.arange(len(depths)), depth_labels, rotation=30, ha="right")
    axis.set_yticks(np.arange(4), class_labels)
    axis.set_xlabel("Depth" if language == "en" else "深度")
    axis.set_ylabel("Truth class" if language == "en" else "真实类别")
    axis.set_title(title_map[kind])
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Patient-aggregated rate" if language == "en" else "患者汇总比例")
    return save_publication_figure(figure, output_base, dpi=dpi)
