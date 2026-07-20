"""Supplementary effect-and-coverage audit for the locked V7 matrix."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from pptt.visualization.common import configure_figure_style, save_publication_figure


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


def aggregate_v7_matrices(
    cell_summary: pd.DataFrame,
    *,
    transitions: Sequence[str],
    restore_nodes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "transition",
        "restore_node",
        "mean_specific_effect",
        "specific_patient_count",
        "evaluable",
    }
    missing = required - set(cell_summary.columns)
    if missing:
        raise ValueError(f"V7 cell summary lacks columns: {sorted(missing)}")
    transition_names = tuple(str(value) for value in transitions)
    node_names = tuple(str(value) for value in restore_nodes)
    if len(transition_names) != 7 or len(node_names) != 7:
        raise ValueError("V7 coverage audit requires a complete 7x7 registry")
    effect = np.full((7, 7), np.nan, dtype=np.float64)
    coverage = np.zeros((7, 7), dtype=np.int64)
    available = np.zeros((7, 7), dtype=bool)
    for row_index, transition in enumerate(transition_names):
        for column_index, node in enumerate(node_names):
            selected = cell_summary[
                (cell_summary["transition"].astype(str) == transition)
                & (cell_summary["restore_node"].astype(str) == node)
            ].copy()
            if selected.empty:
                raise ValueError(f"V7 matrix cell is missing: {transition}/{node}")
            patient_counts = pd.to_numeric(
                selected["specific_patient_count"],
                errors="raise",
            ).to_numpy(dtype=np.int64)
            if np.any(patient_counts < 0):
                raise ValueError("V7 patient coverage must be nonnegative")
            coverage[row_index, column_index] = int(patient_counts.sum())
            values = pd.to_numeric(
                selected["mean_specific_effect"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            valid = np.isfinite(values) & (patient_counts > 0)
            cell_available = bool(selected["evaluable"].astype(bool).any() and valid.any())
            available[row_index, column_index] = cell_available
            if cell_available:
                effect[row_index, column_index] = float(
                    np.average(values[valid], weights=patient_counts[valid])
                )
    return effect, coverage, available


def render_v7_coverage_figure(
    *,
    output_base: str | Path,
    language: str,
    cell_summary: pd.DataFrame,
    transitions: Sequence[str],
    restore_nodes: Sequence[str],
    status: str,
    dpi: int = 600,
) -> dict[str, Any]:
    if language not in {"en", "zh"}:
        raise ValueError("language must be 'en' or 'zh'")
    if status != "INSUFFICIENT_NETWORK_COVERAGE":
        raise ValueError("V7 supplementary figure must preserve its locked status")
    configure_figure_style(language)
    transition_names = tuple(str(value) for value in transitions)
    node_names = tuple(str(value) for value in restore_nodes)
    effect, coverage, available = aggregate_v7_matrices(
        cell_summary,
        transitions=transition_names,
        restore_nodes=node_names,
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), constrained_layout=True)
    finite_effect = effect[np.isfinite(effect)]
    maximum = max(0.01, float(np.max(np.abs(finite_effect))) if finite_effect.size else 0.01)
    effect_cmap = plt.get_cmap("RdBu_r").copy()
    effect_cmap.set_bad("#D1D5DB")
    effect_image = axes[0].imshow(
        np.ma.masked_invalid(effect),
        cmap=effect_cmap,
        norm=TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
        interpolation="nearest",
    )
    coverage_cmap = plt.get_cmap("Blues").copy()
    coverage_cmap.set_bad("#D1D5DB")
    coverage_values = coverage.astype(np.float64)
    coverage_values[~available] = np.nan
    coverage_image = axes[1].imshow(
        np.ma.masked_invalid(coverage_values),
        cmap=coverage_cmap,
        vmin=0,
        vmax=max(1.0, float(np.nanmax(coverage_values)) if np.isfinite(coverage_values).any() else 1.0),
        interpolation="nearest",
    )
    titles = (
        ("特异恢复效应", "可评估患者覆盖")
        if language == "zh"
        else ("Specific restoration effect", "Evaluable-patient coverage")
    )
    for axis, title in zip(axes, titles, strict=True):
        axis.set_title(title, fontsize=10.0, fontweight="bold", pad=8)
        axis.set_xticks(range(7), node_names, rotation=32, ha="right", fontsize=7.2)
        axis.set_yticks(range(7), transition_names, fontsize=7.2)
        axis.set_xlabel("恢复节点" if language == "zh" else "Restored node")
        axis.set_ylabel("像素转移" if language == "zh" else "Pixel transition")
        axis.set_xticks(np.arange(-0.5, 7, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 7, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.65)
        axis.tick_params(which="minor", bottom=False, left=False)
    for row in range(7):
        for column in range(7):
            if available[row, column]:
                axes[0].text(
                    column,
                    row,
                    f"{effect[row, column]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="#1F2933",
                )
                axes[1].text(
                    column,
                    row,
                    str(int(coverage[row, column])),
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="#1F2933",
                )
            else:
                for axis in axes:
                    axis.text(
                        column,
                        row,
                        "NA",
                        ha="center",
                        va="center",
                        fontsize=6.0,
                        color="#52606D",
                    )
    figure.colorbar(
        effect_image,
        ax=axes[0],
        shrink=0.78,
        label="效应" if language == "zh" else "Effect",
    )
    figure.colorbar(
        coverage_image,
        ax=axes[1],
        shrink=0.78,
        label="患者数" if language == "zh" else "Patients",
    )
    figure.suptitle(
        (
            "全网络干预覆盖审计：覆盖不足"
            if language == "zh"
            else "Network-wide intervention coverage audit: insufficient coverage"
        ),
        fontsize=11.2,
        fontweight="bold",
        color="#7A271A",
    )
    png_path, pdf_path = save_publication_figure(figure, output_base, dpi=dpi)
    manifest = {
        "figure": "v7_network_coverage_audit",
        "language": language,
        "status": status,
        "network_wide_claim_authorized": False,
        "matrix_shape": [7, 7],
        "transitions": list(transition_names),
        "restore_nodes": list(node_names),
        "effect_matrix": _json_ready(effect),
        "coverage_matrix": coverage.tolist(),
        "available_matrix": available.tolist(),
        "unavailable_encoding": "gray_NA_not_zero",
        "dpi": int(dpi),
        "outputs": {"png": str(png_path), "pdf": str(pdf_path)},
    }
    manifest_path = Path(output_base).with_suffix(".json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = ["aggregate_v7_matrices", "render_v7_coverage_figure"]
