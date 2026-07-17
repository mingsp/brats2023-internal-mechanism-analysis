from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from pptt.data.brats2d import ensure_chw, normalize_label
from pptt.io.artifacts import CaseTrace, load_case_trace
from pptt.visualization.common import configure_figure_style, save_publication_figure
from pptt.visualization.flagship import (
    BRATS_CLASS_COLORS,
    segmentation_overlay,
    select_representative_slice,
)


NODES = ("down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4")
MODEL_COLORS = {42: "#2563A6", 123: "#D97706", 3407: "#16897C"}
EFFECT_COLORS = {
    "late_persistent_net_recovery": "#007F73",
    "middle_damage": "#D97706",
    "terminal_dice": "#365F91",
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_case_pair(
    v3_root: Path,
    *,
    patient_id: str,
    model_seed: int,
    split: str,
) -> tuple[CaseTrace, CaseTrace, Path, Path]:
    paths = tuple(
        v3_root
        / model
        / f"seed_{model_seed}"
        / split
        / "case_traces"
        / f"{patient_id}.npz"
        for model in ("unet_baseline", "unet_noskip")
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline, noskip = (load_case_trace(path) for path in paths)
    if baseline.truth is None or noskip.truth is None:
        raise ValueError("both formal traces must contain ground truth")
    if baseline.slice_ids != noskip.slice_ids:
        raise ValueError("paired model traces do not contain identical slice identifiers")
    if not np.array_equal(baseline.truth, noskip.truth):
        raise ValueError("paired model traces do not contain identical ground truth")
    if baseline.states.shape != noskip.states.shape:
        raise ValueError("paired model traces do not have identical node grids")
    if baseline.states.shape[0] != len(NODES):
        raise ValueError("formal trace does not contain the locked eight-node path")
    return baseline, noskip, paths[0], paths[1]


def _load_verified_flair(
    image_root: Path,
    mask_root: Path,
    *,
    slice_id: str,
    expected_truth: np.ndarray,
) -> tuple[np.ndarray, Path, Path]:
    image_path = image_root / f"{slice_id}.npy"
    mask_path = mask_root / f"{slice_id}.npy"
    if not image_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(f"missing image/mask pair for {slice_id}")
    image = ensure_chw(np.load(image_path, allow_pickle=False))
    truth = normalize_label(np.load(mask_path, allow_pickle=False))
    if not np.array_equal(truth, expected_truth):
        raise ValueError("raw mask and formal trace truth disagree")
    return image[0], image_path, mask_path


def _effect_labels(language: str) -> dict[str, str]:
    if language == "zh":
        return {
            "late_persistent_net_recovery": "后期持续净恢复",
            "middle_damage": "中部破坏与错误重编码",
            "terminal_dice": "终点肿瘤 Dice",
        }
    return {
        "late_persistent_net_recovery": "Late persistent net recovery",
        "middle_damage": "Mid-path damage and re-encoding",
        "terminal_dice": "Terminal tumor Dice",
    }


def _draw_scatter(
    axis: plt.Axes,
    paired: pd.DataFrame,
    representative: dict[str, Any],
    relation: pd.DataFrame,
    *,
    language: str,
) -> dict[str, Any]:
    selected = paired[paired.split == "test"].copy()
    if selected.empty:
        raise ValueError("no test contrasts are available for the flagship scatter")
    for model_seed, group in selected.groupby("model_seed", sort=True):
        axis.scatter(
            group.terminal_dice_difference * 100.0,
            group.full_path_transition_tv,
            s=13,
            alpha=0.42,
            color=MODEL_COLORS[int(model_seed)],
            edgecolors="none",
            rasterized=True,
        )
    axis.axvspan(-5.0, 5.0, color="#D9DDE3", alpha=0.32, linewidth=0)
    axis.axvline(0.0, color="#4B5563", linewidth=0.8, linestyle="--")
    chosen = selected[
        (selected.patient_id == str(representative["patient_id"]))
        & (selected.model_seed == int(representative["model_seed"]))
    ]
    if len(chosen) != 1:
        raise ValueError("representative patient-seed row is not unique")
    row = chosen.iloc[0]
    x_value = float(row.terminal_dice_difference * 100.0)
    y_value = float(row.full_path_transition_tv)
    axis.scatter(
        [x_value],
        [y_value],
        s=76,
        marker="*",
        color="#B42318",
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )
    axis.annotate(
        str(representative["patient_id"]).replace("BraTS-GLI-", ""),
        (x_value, y_value),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=7.5,
        color="#7A271A",
    )
    averaged = relation[relation.scope == "fixed_seed_patient_average"]
    if len(averaged) != 1:
        raise ValueError("missing fixed-seed endpoint-process correlation")
    rho = float(averaged.iloc[0].spearman_rho)
    correlation_label = (
        f"患者均值 Spearman ρ = {rho:.2f}"
        if language == "zh"
        else f"Patient-average Spearman ρ = {rho:.2f}"
    )
    axis.text(
        0.02,
        0.98,
        correlation_label,
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        color="#374151",
    )
    axis.set_xlabel(
        "终点 Dice 差异（有跳接 − 无跳接，百分点）"
        if language == "zh"
        else "Terminal Dice difference (skip − no-skip, pp)"
    )
    axis.set_ylabel(
        "全路径转移距离"
        if language == "zh"
        else "Full-path transition distance"
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=MODEL_COLORS[seed],
            markeredgecolor="none",
            markersize=5,
            label=f"seed {seed}",
        )
        for seed in sorted(MODEL_COLORS)
    ]
    axis.legend(handles=handles, loc="lower right", frameon=False, ncol=3)
    return {
        "point_count": int(len(selected)),
        "representative_terminal_difference_pp": x_value,
        "representative_process_distance": y_value,
        "patient_average_spearman_rho": rho,
    }


def _draw_forest(
    axis: plt.Axes,
    statistics: pd.DataFrame,
    *,
    language: str,
) -> list[dict[str, Any]]:
    order = (
        "late_persistent_net_recovery",
        "middle_damage",
        "terminal_dice",
    )
    labels = _effect_labels(language)
    summary = statistics[statistics.scope == "fixed_seed_patient_average"]
    seed_rows = statistics[statistics.scope == "model_seed"]
    y_positions = np.arange(len(order), dtype=np.float64)[::-1]
    records = []
    for y_value, metric in zip(y_positions, order, strict=True):
        fixed = summary[summary.metric == metric]
        if len(fixed) != 1:
            raise ValueError(f"missing unique fixed-seed result for {metric}")
        row = fixed.iloc[0]
        mean = float(row.mean_difference * 100.0)
        low = float(row.ci_low * 100.0)
        high = float(row.ci_high * 100.0)
        color = EFFECT_COLORS[metric]
        axis.plot([low, high], [y_value, y_value], color=color, linewidth=2.2)
        axis.scatter(
            [mean],
            [y_value],
            s=52,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            zorder=4,
        )
        current_seeds = seed_rows[seed_rows.metric == metric].sort_values("model_seed")
        offsets = np.linspace(-0.13, 0.13, len(current_seeds))
        for offset, seed_row in zip(offsets, current_seeds.itertuples(), strict=True):
            axis.scatter(
                [float(seed_row.mean_difference * 100.0)],
                [y_value + offset],
                s=19,
                facecolors="white",
                edgecolors=MODEL_COLORS[int(seed_row.model_seed)],
                linewidths=1.0,
                zorder=5,
            )
        axis.text(
            high + 0.8,
            y_value,
            f"{mean:+.1f} [{low:.1f}, {high:.1f}]  d={float(row.paired_cohens_d):.2f}",
            va="center",
            ha="left",
            fontsize=7.4,
            color="#111827",
        )
        records.append(
            {
                "metric": metric,
                "mean_difference_pp": mean,
                "ci_low_pp": low,
                "ci_high_pp": high,
                "paired_cohens_d": float(row.paired_cohens_d),
                "holm_p": float(row.holm_p),
            }
        )
    axis.axvline(0.0, color="#4B5563", linewidth=0.8, linestyle="--")
    axis.set_yticks(y_positions, [labels[metric] for metric in order])
    axis.set_xlabel(
        "有跳接 − 无跳接（百分点）"
        if language == "zh"
        else "Skip − no-skip difference (pp)"
    )
    axis.set_xlim(left=min(-2.0, axis.get_xlim()[0]), right=24.0)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    return records


def _draw_case_trajectory(
    figure: plt.Figure,
    container: Any,
    *,
    flair: np.ndarray,
    truth: np.ndarray,
    baseline_states: np.ndarray,
    noskip_states: np.ndarray,
    language: str,
) -> None:
    subgrid = container.subgridspec(
        2,
        11,
        width_ratios=[1.08, 1.08, 0.48, 1, 1, 1, 1, 1, 1, 1, 1],
        wspace=0.055,
        hspace=0.08,
    )
    blank = np.zeros_like(truth, dtype=np.float32)
    mri_rgb = segmentation_overlay(
        flair,
        np.zeros_like(truth),
        colors=BRATS_CLASS_COLORS,
        alpha=0.0,
    )
    truth_rgb = segmentation_overlay(
        blank,
        truth,
        colors=BRATS_CLASS_COLORS,
        alpha=1.0,
    )
    for column, (image, title) in enumerate(
        (
            (mri_rgb, "T2-FLAIR"),
            (truth_rgb, "金标准" if language == "zh" else "Ground truth"),
        )
    ):
        axis = figure.add_subplot(subgrid[:, column])
        axis.imshow(image, interpolation="nearest")
        axis.set_title(title, pad=4, fontsize=8.5)
        axis.axis("off")
    for row_index, (row_states, row_label) in enumerate(
        (
            (
                baseline_states,
                "有跳接 U-Net" if language == "zh" else "Skip U-Net",
            ),
            (
                noskip_states,
                "无跳接 U-Net" if language == "zh" else "No-skip U-Net",
            ),
        )
    ):
        label_axis = figure.add_subplot(subgrid[row_index, 2])
        label_axis.text(
            0.5,
            0.5,
            row_label.replace(" U-Net", "\nU-Net"),
            ha="center",
            va="center",
            fontsize=8.5,
            linespacing=1.15,
        )
        label_axis.axis("off")
        for node_index, node in enumerate(NODES):
            axis = figure.add_subplot(subgrid[row_index, node_index + 3])
            axis.imshow(
                segmentation_overlay(
                    flair,
                    row_states[node_index],
                    colors=BRATS_CLASS_COLORS,
                    alpha=0.53,
                ),
                interpolation="nearest",
            )
            if row_index == 0:
                axis.set_title(node, pad=4, fontsize=8.0)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.45)
                spine.set_edgecolor("#D1D5DB")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the formal PPTT flagship figure from frozen results."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    effect_root = workspace / "results" / "process_effect_gate"
    v3_root = workspace / "results" / "v3_unet_pair"
    asset_root = Path(
        "/root/autodl-tmp/A_scheme_workspace/brats2023_data/processed_2d"
    )
    output_root = (
        args.output_root
        if args.output_root is not None
        else workspace / "results" / "paper_outputs" / args.language
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    configure_figure_style(args.language)

    representative = json.loads(
        (effect_root / "representative_case.json").read_text(encoding="utf-8")
    )
    paired = pd.read_parquet(effect_root / "patient_process_contrasts.parquet")
    statistics = pd.read_parquet(effect_root / "primary_process_statistics.parquet")
    relation = pd.read_parquet(effect_root / "endpoint_process_relation.parquet")
    baseline, noskip, baseline_path, noskip_path = _load_case_pair(
        v3_root,
        patient_id=str(representative["patient_id"]),
        model_seed=int(representative["model_seed"]),
        split="test",
    )
    slice_index = select_representative_slice(
        baseline.truth,
        truth_classes=(1, 2, 3),
    )
    slice_id = baseline.slice_ids[slice_index]
    flair, image_path, mask_path = _load_verified_flair(
        asset_root / "test_Image",
        asset_root / "test_Mask",
        slice_id=slice_id,
        expected_truth=baseline.truth[slice_index],
    )

    figure = plt.figure(figsize=(14.8, 7.8), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.35],
        width_ratios=[1.05, 1.0],
        left=0.055,
        right=0.985,
        top=0.965,
        bottom=0.075,
        hspace=0.30,
        wspace=0.27,
    )
    scatter_axis = figure.add_subplot(grid[0, 0])
    forest_axis = figure.add_subplot(grid[0, 1])
    scatter_record = _draw_scatter(
        scatter_axis,
        paired,
        representative,
        relation,
        language=args.language,
    )
    effect_records = _draw_forest(forest_axis, statistics, language=args.language)
    _draw_case_trajectory(
        figure,
        grid[1, :],
        flair=flair,
        truth=baseline.truth[slice_index],
        baseline_states=baseline.states[:, slice_index],
        noskip_states=noskip.states[:, slice_index],
        language=args.language,
    )

    figure.text(0.012, 0.965, "a", fontsize=11, fontweight="bold")
    figure.text(0.53, 0.965, "b", fontsize=11, fontweight="bold")
    figure.text(0.012, 0.49, "c", fontsize=11, fontweight="bold")
    legend_labels = (
        {1: "坏死/非增强核心", 2: "水肿", 3: "增强区"}
        if args.language == "zh"
        else {1: "NCR/NET", 2: "Edema", 3: "Enhancing tumor"}
    )
    class_handles = [
        Patch(facecolor=BRATS_CLASS_COLORS[class_id], label=legend_labels[class_id])
        for class_id in (1, 2, 3)
    ]
    figure.legend(
        handles=class_handles,
        loc="lower center",
        bbox_to_anchor=(0.64, 0.012),
        frameon=False,
        ncol=3,
        columnspacing=1.4,
        handlelength=1.1,
    )

    output_base = output_root / "fig2_similar_endpoints_different_processes"
    png_path, pdf_path = save_publication_figure(
        figure,
        output_base,
        dpi=args.dpi,
    )
    manifest = {
        "status": "PASS",
        "scientific_question": "similar_endpoints_different_internal_processes",
        "language": args.language,
        "source_results": effect_root,
        "representative_case": representative,
        "representative_slice": {
            "slice_index_in_trace": slice_index,
            "slice_id": slice_id,
            "selection_rule": "all_three_classes_then_maximum_tumor_area",
            "same_slice_for_both_models_and_all_nodes": True,
            "mri_channel": 0,
            "mri_modality": "T2-FLAIR",
            "channel_order_source": "preprocess_brats2023_gli.py: t2f,t1n,t1c,t2w",
            "image_path": image_path,
            "mask_path": mask_path,
            "baseline_trace": baseline_path,
            "noskip_trace": noskip_path,
        },
        "scatter": scatter_record,
        "forest": effect_records,
        "outputs": {"png": png_path, "pdf": pdf_path},
    }
    manifest_path = output_root / "fig2_similar_endpoints_different_processes.json"
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
