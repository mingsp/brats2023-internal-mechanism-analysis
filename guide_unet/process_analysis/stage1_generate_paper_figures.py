"""Generate paper-ready figures for Stage 1 explanation closure.

The figures generated here are intentionally single-purpose:
visual filmstrips contain only visual evidence, while quantitative readouts are
saved as separate plots/tables. No new inference is performed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont


DEFAULT_STAGE1_TABLE = (
    "results/skip_case_baseline_vs_noskip_mechanism_comparison/"
    "table_10_stage1_node_state_comparison.csv"
)
DEFAULT_E4_TABLE = (
    "stage1_explanation_suite_20260517/results/e7_claim_evidence_matrix/"
    "table_e4_key_adjusted_drop_formal_n512.csv"
)
DEFAULT_E5_TABLE = (
    "stage1_explanation_suite_20260517/results/e7_claim_evidence_matrix/"
    "table_e5_late_transition_aggregation_readout_formal_n512.csv"
)
DEFAULT_E6_TABLE = (
    "stage1_explanation_suite_20260517/results/e7_claim_evidence_matrix/"
    "table_e6_cam_variant_trend_readout_formal_n512.csv"
)
DEFAULT_CASE_SUMMARY_TABLE = (
    "stage1_explanation_suite_20260517/results/e8_visual_data_readout/"
    "table_top10_visual_case_interpretation_summary.csv"
)
DEFAULT_OUTPUT_ROOT = "stage1_explanation_suite_20260517/results/e8_paper_figures"

NODES = ["down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4"]
DECODER_NODES = ["up1", "up2", "up3", "up4"]
CORE_NODES = ["down4", "up1", "up3", "up4"]
ALL_VISUAL_LABELS = ["MRI+GT", "MRI+Pred", "down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4"]
MAIN_VISUAL_INDICES = [0, 1, 5, 6, 8, 9]
SUPPORT_CASE = "BraTS-GLI-00568-000_78"
SUPPORT_CASES = [
    "BraTS-GLI-00568-000_78",
    "BraTS-GLI-00429-000_66",
    "BraTS-GLI-01240-000_43",
    "BraTS-GLI-01430-000_73",
]
MIXED_CASES = [
    "BraTS-GLI-00611-001_58",
    "BraTS-GLI-01144-000_73",
    "BraTS-GLI-01455-000_67",
]
BOUNDARY_CASES = [
    "BraTS-GLI-01364-000_83",
    "BraTS-GLI-01433-000_75",
    "BraTS-GLI-01478-000_91",
]

MODEL_LABEL = {"baseline": "baseline U-Net", "noskip_unet": "no-skip U-Net"}
MODEL_COLOR = {"baseline": "#1f77b4", "noskip_unet": "#d55e00"}
RESPONSE_COLOR = {"cam": "#0072b2", "feature_l2": "#009e73"}
NODE_LABEL = {
    "down1": "down1",
    "down2": "down2",
    "down3": "down3",
    "down4": "down4",
    "up1": "up1",
    "up2": "up2",
    "up3": "up3",
    "up4": "up4",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, out_stem: Path) -> None:
    ensure_dir(out_stem.parent)
    fig.savefig(str(out_stem.with_suffix(".png")), bbox_inches="tight")
    fig.savefig(str(out_stem.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)


def slug_figure_id(figure_id: str) -> str:
    slug = figure_id.lower().replace("fig.", "fig").strip()
    slug = slug.replace("fig s", "fig_s")
    slug = slug.replace("fig ", "fig")
    return slug.replace(" ", "_")


def short_case_name(case_name: str) -> str:
    core = case_name.replace("BraTS-GLI-", "")
    if "-000_" in core:
        prefix, slice_id = core.split("-000_", 1)
        try:
            prefix = str(int(prefix))
        except ValueError:
            pass
        return f"{prefix}_{slice_id}"
    return core


def find_pil_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/msyhbd.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def text_wh(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]
    return draw.textsize(text, font=font)


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return []
    runs = []
    start = int(indices[0])
    prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx - prev > 1:
            runs.append((start, prev))
            start = idx
        prev = idx
    runs.append((start, prev))
    return runs


def compact_visual_image(src: Path, dst: Path) -> None:
    """Trim outer and large internal whitespace without adding annotations."""

    image = Image.open(src).convert("RGB")
    arr = np.asarray(image)
    nonwhite = np.any(arr < 245, axis=2)
    if not nonwhite.any():
        shutil.copy2(src, dst)
        return

    rows = np.where(nonwhite.any(axis=1))[0]
    cols = np.where(nonwhite.any(axis=0))[0]
    pad = 18
    x0 = max(0, int(cols[0]) - pad)
    x1 = min(image.width, int(cols[-1]) + pad + 1)

    row_density = nonwhite[:, x0:x1].sum(axis=1)
    row_has_content = row_density > max(8, int((x1 - x0) * 0.002))
    row_indices = np.where(row_has_content)[0]
    if len(row_indices) == 0:
        shutil.copy2(src, dst)
        return

    raw_runs = []
    start = int(row_indices[0])
    prev = int(row_indices[0])
    for idx in row_indices[1:]:
        idx = int(idx)
        if idx - prev > 1:
            raw_runs.append((start, prev))
            start = idx
        prev = idx
    raw_runs.append((start, prev))

    merged = []
    merge_gap = 70
    for start, end in raw_runs:
        if not merged or start - merged[-1][1] > merge_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = end

    sections = []
    for start, end in merged:
        y0 = max(0, start - pad)
        y1 = min(image.height, end + pad + 1)
        if y1 - y0 >= 6:
            sections.append(image.crop((x0, y0, x1, y1)))

    if not sections:
        shutil.copy2(src, dst)
        return

    gap = 26
    new_width = max(section.width for section in sections)
    new_height = sum(section.height for section in sections) + gap * (len(sections) - 1)
    compact = Image.new("RGB", (new_width, new_height), "white")
    y = 0
    for section in sections:
        x = (new_width - section.width) // 2
        compact.paste(section, (x, y))
        y += section.height + gap
    compact.save(dst)


def build_clean_filmstrip(
    src: Path,
    dst: Path,
    case_name: str,
    kind: str,
    selected_indices: list[int] | None = None,
) -> None:
    """Rebuild a visual-only filmstrip from existing panel images."""

    image = Image.open(src).convert("RGB")
    arr = np.asarray(image)
    nonwhite = np.any(arr < 245, axis=2)
    row_density = nonwhite.sum(axis=1)
    image_row_mask = row_density > max(900, int(image.width * 0.42))
    row_runs = find_runs(image_row_mask)
    row_runs = [(a, b) for a, b in row_runs if b - a > 100]
    if len(row_runs) < 2:
        compact_visual_image(src, dst)
        return
    row_runs = row_runs[:2]

    panel_grid = []
    col_runs_ref = None
    for y0, y1 in row_runs:
        band = nonwhite[y0 : y1 + 1, :]
        col_density = band.sum(axis=0)
        col_mask = col_density > max(40, int((y1 - y0 + 1) * 0.20))
        col_runs = [(a, b) for a, b in find_runs(col_mask) if b - a > 100]
        if len(col_runs) < 10:
            compact_visual_image(src, dst)
            return
        col_runs = col_runs[:10]
        if col_runs_ref is None:
            col_runs_ref = col_runs
        row_panels = [image.crop((x0, y0, x1 + 1, y1 + 1)) for x0, x1 in col_runs]
        panel_grid.append(row_panels)

    selected_indices = selected_indices or list(range(len(ALL_VISUAL_LABELS)))
    labels = [ALL_VISUAL_LABELS[i] for i in selected_indices]
    panel_grid = [[row[i] for i in selected_indices] for row in panel_grid]
    panel_w = max(panel.width for row in panel_grid for panel in row)
    panel_h = max(panel.height for row in panel_grid for panel in row)
    gap_x = 24
    gap_y = 74
    left = 145
    right = 20
    top = 62
    label_h = 44
    bottom = 24
    width = left + len(labels) * panel_w + (len(labels) - 1) * gap_x + right
    height = top + label_h + panel_h * 2 + gap_y + bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = find_pil_font(28, bold=False)
    label_font = find_pil_font(22, bold=False)
    row_font = find_pil_font(22, bold=False)

    scope = "key-node" if selected_indices == MAIN_VISUAL_INDICES else "full-node"
    title = f"{case_name} | {'CAM' if kind == 'cam' else 'raw feature response'} {scope} trajectory"
    tw, th = text_wh(draw, title, title_font)
    draw.text(((width - tw) // 2, 14), title, fill=(20, 20, 20), font=title_font)

    x_positions = [left + i * (panel_w + gap_x) for i in range(len(labels))]
    for x, label in zip(x_positions, labels):
        lw, lh = text_wh(draw, label, label_font)
        draw.text((x + (panel_w - lw) // 2, top), label, fill=(20, 20, 20), font=label_font)

    row_names = ["baseline", "no-skip"]
    y_positions = [top + label_h, top + label_h + panel_h + gap_y]
    for row_idx, (row_name, y) in enumerate(zip(row_names, y_positions)):
        rw, rh = text_wh(draw, row_name, row_font)
        draw.text((left - rw - 22, y + (panel_h - rh) // 2), row_name, fill=(20, 20, 20), font=row_font)
        for col_idx, panel in enumerate(panel_grid[row_idx]):
            x = x_positions[col_idx] + (panel_w - panel.width) // 2
            canvas.paste(panel, (x, y))

    canvas.save(dst)


def read_stage1_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values("node_order")
    return df


def plot_stage1_curve(stage1: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    x = np.arange(len(stage1))
    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    ax.plot(
        x,
        stage1["task_dice_gt_baseline"],
        marker="o",
        linewidth=2.1,
        color=MODEL_COLOR["baseline"],
        label=MODEL_LABEL["baseline"],
    )
    ax.plot(
        x,
        stage1["task_dice_gt_noskip"],
        marker="s",
        linewidth=2.1,
        color=MODEL_COLOR["noskip_unet"],
        label=MODEL_LABEL["noskip_unet"],
    )
    ax.axvspan(3.5, 7.2, color="#eeeeee", alpha=0.65, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([NODE_LABEL[n] for n in stage1["node"]])
    ax.set_ylabel("Semantic readability (Dice to GT)")
    ax.set_xlabel("Stage 1 graph node")
    ax.set_ylim(0.40, 0.90)
    ax.set_title("Stage 1 semantic readability curve")
    ax.legend(loc="lower right", frameon=False)
    out = out_dir / "main" / "fig1_stage1_semantic_readability_curve"
    save_figure(fig, out)
    return {
        "figure_id": "Fig. 1",
        "path_png": str(out.with_suffix(".png")),
        "content": "Stage 1 node-level semantic readability curve",
        "paper_role": "Main phenomenon definition",
    }


def plot_decoder_increments(stage1: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    baseline = stage1.set_index("node")["task_dice_gt_baseline"]
    noskip = stage1.set_index("node")["task_dice_gt_noskip"]
    transitions = [("up1", "up2"), ("up2", "up3"), ("up3", "up4")]
    labels = [f"{a}->{b}" for a, b in transitions]
    base_vals = [baseline[b] - baseline[a] for a, b in transitions]
    noskip_vals = [noskip[b] - noskip[a] for a, b in transitions]

    x = np.arange(len(transitions))
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.3, 2.8))
    ax.bar(x - width / 2, base_vals, width, color=MODEL_COLOR["baseline"], label=MODEL_LABEL["baseline"])
    ax.bar(x + width / 2, noskip_vals, width, color=MODEL_COLOR["noskip_unet"], label=MODEL_LABEL["noskip_unet"])
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Increment of semantic readability")
    ax.set_title("Late decoder increments")
    ax.legend(loc="upper left", frameon=False)
    out = out_dir / "supplement" / "fig_s1_decoder_semantic_increments"
    save_figure(fig, out)
    return {
        "figure_id": "Fig. S1",
        "path_png": str(out.with_suffix(".png")),
        "content": "Decoder increment pattern behind acceleration/deceleration",
        "paper_role": "Supplementary phenomenon quantification",
    }


def plot_framework(out_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(7.1, 2.8))
    ax.axis("off")
    boxes = [
        (0.02, 0.57, 0.16, 0.25, "Node graph\nreadout"),
        (0.23, 0.57, 0.17, 0.25, "CAM / raw\ntrajectory"),
        (0.45, 0.57, 0.17, 0.25, "Case-level\ndelta alignment"),
        (0.67, 0.57, 0.16, 0.25, "Response\nguided\nperturbation"),
        (0.84, 0.57, 0.14, 0.25, "Claim\nlevels"),
        (0.23, 0.16, 0.17, 0.20, "CAM variant\nsensitivity"),
        (0.45, 0.16, 0.17, 0.20, "Aggregation\nrobustness"),
    ]
    for x, y, w, h, label in boxes:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.018",
            linewidth=1.0,
            edgecolor="#555555",
            facecolor="#f7f7f7",
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.5, transform=ax.transAxes)

    arrow_pairs = [
        ((0.18, 0.695), (0.23, 0.695)),
        ((0.40, 0.695), (0.45, 0.695)),
        ((0.62, 0.695), (0.67, 0.695)),
        ((0.83, 0.695), (0.84, 0.695)),
        ((0.315, 0.57), (0.315, 0.36)),
        ((0.535, 0.57), (0.535, 0.36)),
        ((0.40, 0.26), (0.45, 0.26)),
        ((0.62, 0.26), (0.75, 0.57)),
    ]
    for start, end in arrow_pairs:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color="#555555",
            transform=ax.transAxes,
        )
        ax.add_patch(arrow)
    ax.set_title("Evidence pipeline for Stage 1 phenomenon explanation", pad=4)
    out = out_dir / "main" / "fig2_evidence_pipeline"
    save_figure(fig, out)
    return {
        "figure_id": "Fig. 2",
        "path_png": str(out.with_suffix(".png")),
        "content": "Evidence pipeline and claim-control framework",
        "paper_role": "Method overview",
    }


def copy_visual_case_figures(case_summary: pd.DataFrame, out_dir: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    main_dir = out_dir / "main"
    supp_dir = out_dir / "supplement"
    ensure_dir(main_dir)
    ensure_dir(supp_dir)

    by_case = case_summary.set_index("case_name")
    selected = [(SUPPORT_CASE, "cam", "Fig. 3"), (SUPPORT_CASE, "feature", "Fig. 4")]
    for case_name, kind, figure_id in selected:
        src_col = "cam_figure" if kind == "cam" else "feature_figure"
        src = Path(str(by_case.loc[case_name, src_col]))
        dst = main_dir / f"{slug_figure_id(figure_id)}_{kind}_trajectory_{case_name}.png"
        build_clean_filmstrip(src, dst, case_name, kind, selected_indices=MAIN_VISUAL_INDICES)
        records.append(
            {
                "figure_id": figure_id,
                "path_png": str(dst),
                "content": f"{kind.upper()} trajectory filmstrip for representative support case {case_name}",
                "paper_role": "Main visual evidence",
            }
        )
        if kind == "cam":
            full_dst = main_dir / f"{slug_figure_id(figure_id)}_{kind}_fullnode_trajectory_{case_name}.png"
            build_clean_filmstrip(src, full_dst, case_name, kind, selected_indices=None)
            records.append(
                {
                    "figure_id": f"{figure_id}-full",
                    "path_png": str(full_dst),
                    "content": f"{kind.upper()} full-node trajectory filmstrip for representative support case {case_name}",
                    "paper_role": "Main visual evidence for full Stage 1 node sequence",
                }
            )

    supp_index = 2
    for case_name in SUPPORT_CASES + MIXED_CASES + BOUNDARY_CASES:
        if case_name == SUPPORT_CASE:
            continue
        if case_name not in by_case.index:
            continue
        for kind in ["cam", "feature"]:
            src_col = "cam_figure" if kind == "cam" else "feature_figure"
            src = Path(str(by_case.loc[case_name, src_col]))
            dst = supp_dir / f"fig_s{supp_index}_{kind}_trajectory_{case_name}.png"
            build_clean_filmstrip(src, dst, case_name, kind)
            records.append(
                {
                    "figure_id": f"Fig. S{supp_index}",
                    "path_png": str(dst),
                    "content": f"{kind.upper()} trajectory filmstrip for {case_name}",
                    "paper_role": "Supplementary visual evidence",
                }
            )
            supp_index += 1
    return records


def plot_case_numeric_readout(case_summary: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    support = case_summary[case_summary["case_name"].isin(SUPPORT_CASES)].copy()
    support["case_short"] = support["case_name"].map(short_case_name)
    support = support.set_index("case_name").loc[SUPPORT_CASES].reset_index()

    metrics = [
        ("Task delta", "baseline_task_up3_to_up4", "noskip_task_up3_to_up4"),
        ("CAM delta", "baseline_cam_finalprob_up3_to_up4", "noskip_cam_finalprob_up3_to_up4"),
        ("Raw-response delta", "baseline_feature_finalprob_up3_to_up4", "noskip_feature_finalprob_up3_to_up4"),
        ("Up4 perturbation drop", "baseline_e4_up4_cam_adj_final_fg_drop", "noskip_e4_up4_cam_adj_final_fg_drop"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6), sharex=True)
    axes = axes.ravel()
    x = np.arange(len(support))
    width = 0.36
    for ax, (title, base_col, noskip_col) in zip(axes, metrics):
        ax.bar(x - width / 2, support[base_col], width, color=MODEL_COLOR["baseline"], label=MODEL_LABEL["baseline"])
        ax.bar(x + width / 2, support[noskip_col], width, color=MODEL_COLOR["noskip_unet"], label=MODEL_LABEL["noskip_unet"])
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(support["case_short"], rotation=18, ha="right")
    axes[0].set_ylabel("Value")
    axes[2].set_ylabel("Value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.995), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.2, w_pad=1.2)
    out = out_dir / "main" / "fig5_support_case_numeric_readout"
    save_figure(fig, out)
    return {
        "figure_id": "Fig. 5",
        "path_png": str(out.with_suffix(".png")),
        "content": "Numeric readout paired with support-case visualizations",
        "paper_role": "Main quantitative readout for visual evidence",
    }


def plot_e4_perturbation(e4: pd.DataFrame, out_dir: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for metric, suffix, ylabel, figure_id in [
        ("random_adjusted_final_fg_prob_drop", "finalfg", "Adjusted final foreground probability drop", "Fig. 6"),
        ("random_adjusted_gt_fg_prob_drop", "gtfg", "Adjusted GT foreground probability drop", "Fig. S20"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
        for ax, response_source in zip(axes, ["cam", "feature_l2"]):
            sub = e4[(e4["response_source"] == response_source) & (e4["node"].isin(CORE_NODES))].copy()
            sub["node"] = pd.Categorical(sub["node"], categories=CORE_NODES, ordered=True)
            sub = sub.sort_values(["node", "model"])
            x = np.arange(len(CORE_NODES))
            width = 0.34
            for offset, model in [(-width / 2, "baseline"), (width / 2, "noskip_unet")]:
                m = sub[sub["model"] == model].set_index("node").loc[CORE_NODES]
                y = m[metric].values
                low = m[f"{metric}_ci_low"].values
                high = m[f"{metric}_ci_high"].values
                yerr = np.vstack([y - low, high - y])
                ax.bar(
                    x + offset,
                    y,
                    width,
                    yerr=yerr,
                    capsize=2,
                    color=MODEL_COLOR[model],
                    label=MODEL_LABEL[model],
                )
            ax.axhline(0, color="#444444", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(CORE_NODES)
            ax.set_title("CAM" if response_source == "cam" else "Raw feature response")
        axes[0].set_ylabel(ylabel)
        axes[0].legend(loc="upper left", frameon=False)
        fig.suptitle("Response-guided perturbation", y=0.995, fontsize=10)
        fig.tight_layout()
        subdir = "main" if figure_id == "Fig. 6" else "supplement"
        out = out_dir / subdir / f"{slug_figure_id(figure_id)}_response_guided_perturbation_{suffix}"
        save_figure(fig, out)
        records.append(
            {
                "figure_id": figure_id,
                "path_png": str(out.with_suffix(".png")),
                "content": f"Response-guided perturbation ({suffix})",
                "paper_role": "Main faithfulness evidence" if figure_id == "Fig. 6" else "Supplementary faithfulness evidence",
            }
        )
    return records


def plot_e6_cam_sensitivity(e6: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    metric = "final_prob_pearson"
    target_mode = "gt_fg"
    sub = e6[(e6["metric"] == metric) & (e6["target_mode"] == target_mode)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    x = np.arange(len(CORE_NODES))
    markers = {"baseline": "o", "noskip_unet": "s"}
    titles = {"gradcam": "Grad-CAM", "hirescam": "HiResCAM"}
    for ax, variant in zip(axes, ["gradcam", "hirescam"]):
        for model in ["baseline", "noskip_unet"]:
            row = sub[(sub["model"] == model) & (sub["cam_variant"] == variant)]
            if row.empty:
                continue
            y = row[CORE_NODES].iloc[0].values.astype(float)
            ax.plot(
                x,
                y,
                linestyle="-",
                marker=markers[model],
                linewidth=2.2,
                markersize=5,
                color=MODEL_COLOR[model],
                label=MODEL_LABEL[model],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(CORE_NODES)
        ax.set_title(titles[variant])
        ax.set_ylim(0.0, 0.95)
        ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#dddddd", ncol=1)
    axes[0].set_ylabel("CAM-final probability Pearson")
    fig.suptitle("CAM variant sensitivity", y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.2)
    out = out_dir / "supplement" / "fig_s21_cam_variant_sensitivity"
    save_figure(fig, out)
    return {
        "figure_id": "Fig. S21",
        "path_png": str(out.with_suffix(".png")),
        "content": "CAM variant sensitivity for the main alignment trend",
        "paper_role": "Supplementary robustness evidence",
    }


def plot_e5_aggregation_robustness(e5: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    metric = "final_prob_pearson"
    sub = e5[(e5["metric"] == metric)].copy()
    aggs = ["l2", "mean_abs", "positive_only", "max_abs", "topk_channel"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    x = np.arange(len(DECODER_NODES))
    colors = {
        "l2": "#000000",
        "mean_abs": "#0072b2",
        "positive_only": "#009e73",
        "max_abs": "#d55e00",
        "topk_channel": "#cc79a7",
    }
    for ax, model in zip(axes, ["baseline", "noskip_unet"]):
        msub = sub[sub["model"] == model]
        for agg in aggs:
            row = msub[msub["aggregation"] == agg]
            if row.empty:
                continue
            y = row[DECODER_NODES].iloc[0].values.astype(float)
            ax.plot(x, y, marker="o", linewidth=1.6, color=colors[agg], label=agg)
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(DECODER_NODES)
        ax.set_title(MODEL_LABEL[model])
    axes[0].set_ylabel("Raw response-final probability Pearson")
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle("Raw-response aggregation robustness", y=0.995, fontsize=10)
    fig.tight_layout()
    out = out_dir / "supplement" / "fig_s22_raw_aggregation_robustness"
    save_figure(fig, out)
    return {
        "figure_id": "Fig. S22",
        "path_png": str(out.with_suffix(".png")),
        "content": "Raw feature response aggregation robustness",
        "paper_role": "Supplementary robustness evidence",
    }


def write_visual_numeric_readout(case_summary: pd.DataFrame, out_dir: Path) -> Path:
    ensure_dir(out_dir / "tables")
    columns = [
        "case_name",
        "support_class",
        "baseline_task_up3_to_up4",
        "noskip_task_up3_to_up4",
        "baseline_cam_finalprob_up3_to_up4",
        "noskip_cam_finalprob_up3_to_up4",
        "baseline_feature_finalprob_up3_to_up4",
        "noskip_feature_finalprob_up3_to_up4",
        "baseline_e4_up4_cam_adj_final_fg_drop",
        "noskip_e4_up4_cam_adj_final_fg_drop",
        "baseline_e4_up4_feature_adj_final_fg_drop",
        "noskip_e4_up4_feature_adj_final_fg_drop",
    ]
    out = out_dir / "tables" / "table_visual_case_numeric_readout.csv"
    case_summary[columns].to_csv(out, index=False, encoding="utf-8-sig")
    return out


def write_key_result_tables(stage1: pd.DataFrame, e4: pd.DataFrame, e5: pd.DataFrame, e6: pd.DataFrame, out_dir: Path) -> list[Path]:
    ensure_dir(out_dir / "tables")
    paths: list[Path] = []

    stage = stage1[["node", "task_dice_gt_baseline", "task_dice_gt_noskip"]].copy()
    stage["noskip_minus_baseline"] = stage["task_dice_gt_noskip"] - stage["task_dice_gt_baseline"]
    path = out_dir / "tables" / "table_stage1_curve_readout.csv"
    stage.to_csv(path, index=False, encoding="utf-8-sig")
    paths.append(path)

    e4_key = e4[
        (e4["node"].isin(CORE_NODES))
        & (e4["response_source"].isin(["cam", "feature_l2"]))
        & (e4["top_fraction"] == 0.1)
        & (e4["gamma"] == 1.0)
    ].copy()
    path = out_dir / "tables" / "table_response_guided_perturbation_key_readout.csv"
    e4_key.to_csv(path, index=False, encoding="utf-8-sig")
    paths.append(path)

    e5_key = e5[(e5["metric"] == "final_prob_pearson")].copy()
    path = out_dir / "tables" / "table_raw_aggregation_robustness_key_readout.csv"
    e5_key.to_csv(path, index=False, encoding="utf-8-sig")
    paths.append(path)

    e6_key = e6[(e6["metric"] == "final_prob_pearson") & (e6["target_mode"] == "gt_fg")].copy()
    path = out_dir / "tables" / "table_cam_variant_sensitivity_key_readout.csv"
    e6_key.to_csv(path, index=False, encoding="utf-8-sig")
    paths.append(path)
    return paths


def write_claim_table(out_dir: Path) -> Path:
    ensure_dir(out_dir / "tables")
    rows = [
        {
            "claim_level": "strong",
            "claim_id": "S1",
            "paper_safe_claim": "In this Stage 1 comparison, the no-skip U-Net shows higher early semantic readability than the baseline.",
            "main_evidence": "Stage 1 node readout; CAM/raw alignment; E4 perturbation",
            "must_avoid": "No-skip is universally better in early layers.",
        },
        {
            "claim_level": "strong",
            "claim_id": "S2",
            "paper_safe_claim": "The baseline shows a mid-network semantic valley around down4/up1.",
            "main_evidence": "Stage 1 curve; weak down4/up1 response alignment; low perturbation effect",
            "must_avoid": "The valley is caused only by skip information injection.",
        },
        {
            "claim_level": "strong_with_boundary",
            "claim_id": "S3",
            "paper_safe_claim": "The baseline late increase is consistent with late recovery of task-aligned and task-relevant responses, especially at up4.",
            "main_evidence": "CAM trajectory, raw response trajectory, E4 up4 perturbation, E6 CAM sensitivity",
            "must_avoid": "Skip connections are proven as the sole causal mechanism.",
        },
        {
            "claim_level": "strong",
            "claim_id": "S4",
            "paper_safe_claim": "The no-skip late deceleration is consistent with an earlier high semantic state and limited additional late response gain.",
            "main_evidence": "Stage 1 increments; E5 aggregation; E6 CAM trend; case support table",
            "must_avoid": "Every no-skip case strictly decelerates.",
        },
        {
            "claim_level": "moderate",
            "claim_id": "M1",
            "paper_safe_claim": "Raw feature response provides comparatively stable case-level coupling evidence, strongest for baseline transitions.",
            "main_evidence": "Case-level delta correlation and E5 aggregation robustness",
            "must_avoid": "Raw response perfectly explains every case-level transition.",
        },
        {
            "claim_level": "boundary",
            "claim_id": "B1",
            "paper_safe_claim": "The observations support a focused explanation of this U-Net comparison on BraTS Stage 1 nodes.",
            "main_evidence": "n=512 validation across matched cases",
            "must_avoid": "This is a universal law for all segmentation networks or a complete causal proof.",
        },
    ]
    out = out_dir / "tables" / "table_claim_levels_for_paper.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def write_figure_manifest(records: list[dict[str, str]], out_dir: Path) -> Path:
    ensure_dir(out_dir / "tables")
    out = out_dir / "tables" / "table_paper_figure_selection.csv"
    pd.DataFrame(records).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready Stage 1 explanation figures.")
    parser.add_argument("--stage1-table", default=DEFAULT_STAGE1_TABLE)
    parser.add_argument("--e4-table", default=DEFAULT_E4_TABLE)
    parser.add_argument("--e5-table", default=DEFAULT_E5_TABLE)
    parser.add_argument("--e6-table", default=DEFAULT_E6_TABLE)
    parser.add_argument("--case-summary-table", default=DEFAULT_CASE_SUMMARY_TABLE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    out_dir = Path(args.output_root)
    ensure_dir(out_dir / "main")
    ensure_dir(out_dir / "supplement")
    ensure_dir(out_dir / "tables")

    stage1 = read_stage1_table(Path(args.stage1_table))
    e4 = pd.read_csv(args.e4_table)
    e5 = pd.read_csv(args.e5_table)
    e6 = pd.read_csv(args.e6_table)
    case_summary = pd.read_csv(args.case_summary_table)

    records: list[dict[str, str]] = []
    records.append(plot_stage1_curve(stage1, out_dir))
    records.append(plot_framework(out_dir))
    records.extend(copy_visual_case_figures(case_summary, out_dir))
    records.append(plot_case_numeric_readout(case_summary, out_dir))
    records.extend(plot_e4_perturbation(e4, out_dir))
    records.append(plot_decoder_increments(stage1, out_dir))
    records.append(plot_e6_cam_sensitivity(e6, out_dir))
    records.append(plot_e5_aggregation_robustness(e5, out_dir))

    table_paths = [
        write_visual_numeric_readout(case_summary, out_dir),
        write_claim_table(out_dir),
        write_figure_manifest(records, out_dir),
    ]
    table_paths.extend(write_key_result_tables(stage1, e4, e5, e6, out_dir))

    print(f"Generated {len(records)} figure records under {out_dir}")
    for record in records:
        print(f"{record['figure_id']}: {record['path_png']}")
    print("Generated tables:")
    for path in table_paths:
        print(path)


if __name__ == "__main__":
    main()
