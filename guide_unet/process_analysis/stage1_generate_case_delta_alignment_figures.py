"""Generate paper-ready case-delta alignment figures for Stage 1 explanation.

This script turns the existing n=512 case-transition tables into figures for
the manuscript. It does not run model inference and does not introduce a new
experiment. Its purpose is to expose the already-computed E3 evidence:
same-case task-readout changes versus CAM/raw-response changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ALIGNMENT_TABLE = (
    "results/stage1_case_readout_alignment_n512/"
    "table_case_transition_delta_alignment.csv"
)
DEFAULT_CORR_TABLE = (
    "results/stage1_case_readout_alignment_n512/"
    "table_case_delta_correlation_by_transition.csv"
)
DEFAULT_OUT_ROOT = "stage1_explanation_suite_20260517/results/e8_paper_figures"

DECODER_TRANSITIONS = ["down4->up1", "up1->up2", "up2->up3", "up3->up4"]
ALL_TRANSITIONS = [
    "down1->down2",
    "down2->down3",
    "down3->down4",
    "down4->up1",
    "up1->up2",
    "up2->up3",
    "up3->up4",
]
MODELS = ["baseline", "noskip_unet"]
MODEL_LABEL = {"baseline": "Baseline U-Net", "noskip_unet": "No-skip U-Net"}
MODEL_COLOR = {"baseline": "#1f77b4", "noskip_unet": "#d55e00"}
TRANSITION_COLOR = {
    "down1->down2": "#56b4e9",
    "down2->down3": "#999999",
    "down3->down4": "#8b5a2b",
    "down4->up1": "#0072b2",
    "up1->up2": "#009e73",
    "up2->up3": "#e69f00",
    "up3->up4": "#cc79a7",
}
METRIC_INFO = {
    "cam_final_prob_pearson": {
        "delta_col": "cam_final_prob_pearson_delta",
        "label": "CAM-final probability",
        "short": "CAM-final prob.",
    },
    "feature_final_prob_pearson": {
        "delta_col": "feature_final_prob_pearson_delta",
        "label": "Raw-final probability",
        "short": "Raw-final prob.",
    },
    "cam_gt_mass_fraction": {
        "delta_col": "cam_gt_mass_fraction_delta",
        "label": "CAM GT mass",
        "short": "CAM GT mass",
    },
    "cam_gt_fg_bg_contrast": {
        "delta_col": "cam_gt_fg_bg_contrast_delta",
        "label": "CAM FG-BG contrast",
        "short": "CAM FG-BG",
    },
    "cam_entropy_norm": {
        "delta_col": "cam_entropy_norm_delta",
        "label": "CAM entropy",
        "short": "CAM entropy",
    },
    "feature_gt_mass_fraction": {
        "delta_col": "feature_gt_mass_fraction_delta",
        "label": "Raw GT mass",
        "short": "Raw GT mass",
    },
    "feature_gt_fg_bg_contrast": {
        "delta_col": "feature_gt_fg_bg_contrast_delta",
        "label": "Raw FG-BG contrast",
        "short": "Raw FG-BG",
    },
    "feature_entropy_norm": {
        "delta_col": "feature_entropy_norm_delta",
        "label": "Raw entropy",
        "short": "Raw entropy",
    },
}
HEATMAP_METRICS = [
    "cam_final_prob_pearson",
    "cam_gt_mass_fraction",
    "cam_gt_fg_bg_contrast",
    "cam_entropy_norm",
    "feature_final_prob_pearson",
    "feature_gt_mass_fraction",
    "feature_gt_fg_bg_contrast",
    "feature_entropy_norm",
]
MAIN_METRICS = ["cam_final_prob_pearson", "feature_final_prob_pearson"]


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
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, out_stem: Path) -> None:
    ensure_dir(out_stem.parent)
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fisher_ci(r: float, n: int, z_value: float = 1.96) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r):
        return np.nan, np.nan
    r = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    lo = np.tanh(z - z_value * se)
    hi = np.tanh(z + z_value * se)
    return float(lo), float(hi)


def corr_summary(df: pd.DataFrame, metric: str, scope: str) -> dict[str, object]:
    x = pd.to_numeric(df[METRIC_INFO[metric]["delta_col"]], errors="coerce")
    y = pd.to_numeric(df["task_dice_gt_delta"], errors="coerce")
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]
    n = int(valid.sum())
    pearson = float(x.corr(y, method="pearson")) if n > 2 else np.nan
    spearman = float(x.corr(y, method="spearman")) if n > 2 else np.nan
    lo, hi = fisher_ci(pearson, n)
    return {
        "scope": scope,
        "metric": metric,
        "metric_label": METRIC_INFO[metric]["short"],
        "n": n,
        "pearson": pearson,
        "pearson_ci95_low": lo,
        "pearson_ci95_high": hi,
        "spearman": spearman,
        "task_delta_mean": float(y.mean()),
        "metric_delta_mean": float(x.mean()),
    }


def draw_regression(ax: plt.Axes, x: pd.Series, y: pd.Series, color: str) -> None:
    valid = x.notna() & y.notna()
    xv = x[valid].to_numpy(dtype=float)
    yv = y[valid].to_numpy(dtype=float)
    if len(xv) < 3 or np.nanstd(xv) == 0:
        return
    lo, hi = np.nanpercentile(xv, [2, 98])
    xx = np.linspace(lo, hi, 120)
    slope, intercept = np.polyfit(xv, yv, 1)
    ax.plot(xx, slope * xx + intercept, color=color, linewidth=2.0, zorder=4)


def plot_decoder_scatter(df: pd.DataFrame, out_stem: Path) -> pd.DataFrame:
    decoder_df = df[df["transition"].isin(DECODER_TRANSITIONS)].copy()
    rows = []

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.2), sharex=True, sharey=True)
    for row_idx, model in enumerate(MODELS):
        model_df = decoder_df[decoder_df["model"] == model]
        for col_idx, metric in enumerate(MAIN_METRICS):
            ax = axes[row_idx, col_idx]
            x_col = METRIC_INFO[metric]["delta_col"]
            for transition in DECODER_TRANSITIONS:
                part = model_df[model_df["transition"] == transition]
                ax.scatter(
                    part[x_col],
                    part["task_dice_gt_delta"],
                    s=11,
                    alpha=0.22,
                    color=TRANSITION_COLOR[transition],
                    edgecolors="none",
                    label=transition if row_idx == 0 and col_idx == 0 else None,
                )
            summary = corr_summary(model_df, metric, "decoder pooled")
            rows.append({"model": model, **summary})
            draw_regression(
                ax,
                pd.to_numeric(model_df[x_col], errors="coerce"),
                pd.to_numeric(model_df["task_dice_gt_delta"], errors="coerce"),
                MODEL_COLOR[model],
            )
            ax.axhline(0, color="#555555", linewidth=0.8)
            ax.axvline(0, color="#555555", linewidth=0.8)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-0.9, 0.9)
            ax.set_title(f"{MODEL_LABEL[model]} / {METRIC_INFO[metric]['label']}")
            ax.text(
                0.03,
                0.95,
                f"r={summary['pearson']:.2f}, rho={summary['spearman']:.2f}\n"
                f"n={summary['n']}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2.0},
            )
            if row_idx == 1:
                ax.set_xlabel("Response delta")
            if col_idx == 0:
                ax.set_ylabel("Task Dice-to-GT delta")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle("Case-level decoder delta alignment", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    save_figure(fig, out_stem)
    return pd.DataFrame(rows)


def plot_stage1_scatter(df: pd.DataFrame, out_stem: Path) -> pd.DataFrame:
    stage1_df = df[df["transition"].isin(ALL_TRANSITIONS)].copy()
    rows = []

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.8), sharex=True, sharey=True)
    for row_idx, model in enumerate(MODELS):
        model_df = stage1_df[stage1_df["model"] == model]
        for col_idx, metric in enumerate(MAIN_METRICS):
            ax = axes[row_idx, col_idx]
            x_col = METRIC_INFO[metric]["delta_col"]
            for transition in ALL_TRANSITIONS:
                part = model_df[model_df["transition"] == transition]
                ax.scatter(
                    part[x_col],
                    part["task_dice_gt_delta"],
                    s=9,
                    alpha=0.18,
                    color=TRANSITION_COLOR[transition],
                    edgecolors="none",
                    label=transition if row_idx == 0 and col_idx == 0 else None,
                )
            summary = corr_summary(model_df, metric, "all Stage 1 adjacent transitions")
            rows.append({"model": model, **summary})
            draw_regression(
                ax,
                pd.to_numeric(model_df[x_col], errors="coerce"),
                pd.to_numeric(model_df["task_dice_gt_delta"], errors="coerce"),
                MODEL_COLOR[model],
            )
            ax.axhline(0, color="#555555", linewidth=0.8)
            ax.axvline(0, color="#555555", linewidth=0.8)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-0.9, 0.9)
            ax.set_title(f"{MODEL_LABEL[model]} / {METRIC_INFO[metric]['label']}")
            ax.text(
                0.03,
                0.95,
                f"r={summary['pearson']:.2f}, rho={summary['spearman']:.2f}\n"
                f"n={summary['n']}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2.0},
            )
            if row_idx == 1:
                ax.set_xlabel("Response delta")
            if col_idx == 0:
                ax.set_ylabel("Task Dice-to-GT delta")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.suptitle("Case-level Stage 1 delta alignment", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0.14, 1, 0.97])
    save_figure(fig, out_stem)
    return pd.DataFrame(rows)


def plot_late_transition_scatter(df: pd.DataFrame, out_stem: Path) -> pd.DataFrame:
    late_df = df[df["transition"] == "up3->up4"].copy()
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.8), sharex=True, sharey=True)
    for row_idx, model in enumerate(MODELS):
        model_df = late_df[late_df["model"] == model]
        for col_idx, metric in enumerate(MAIN_METRICS):
            ax = axes[row_idx, col_idx]
            x_col = METRIC_INFO[metric]["delta_col"]
            ax.scatter(
                model_df[x_col],
                model_df["task_dice_gt_delta"],
                s=13,
                alpha=0.30,
                color=MODEL_COLOR[model],
                edgecolors="none",
            )
            summary = corr_summary(model_df, metric, "up3->up4")
            rows.append({"model": model, **summary})
            draw_regression(
                ax,
                pd.to_numeric(model_df[x_col], errors="coerce"),
                pd.to_numeric(model_df["task_dice_gt_delta"], errors="coerce"),
                MODEL_COLOR[model],
            )
            ax.axhline(0, color="#555555", linewidth=0.8)
            ax.axvline(0, color="#555555", linewidth=0.8)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-0.9, 0.9)
            ax.set_title(f"{MODEL_LABEL[model]} / {METRIC_INFO[metric]['label']}")
            ax.text(
                0.03,
                0.95,
                f"r={summary['pearson']:.2f}, rho={summary['spearman']:.2f}\n"
                f"n={summary['n']}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2.0},
            )
            if row_idx == 1:
                ax.set_xlabel("Response delta")
            if col_idx == 0:
                ax.set_ylabel("Task Dice-to-GT delta")

    fig.suptitle("Case-level late-transition delta alignment (up3 to up4)", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    save_figure(fig, out_stem)
    return pd.DataFrame(rows)


def plot_decoder_heatmap(corr_df: pd.DataFrame, out_stem: Path) -> None:
    decoder_corr = corr_df[corr_df["transition"].isin(DECODER_TRANSITIONS)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), constrained_layout=True)
    vmin, vmax = -0.35, 0.35
    image = None
    for ax, model in zip(axes, MODELS):
        matrix = np.full((len(HEATMAP_METRICS), len(DECODER_TRANSITIONS)), np.nan)
        for r_idx, metric in enumerate(HEATMAP_METRICS):
            for c_idx, transition in enumerate(DECODER_TRANSITIONS):
                row = decoder_corr[
                    (decoder_corr["model"] == model)
                    & (decoder_corr["metric"] == metric)
                    & (decoder_corr["transition"] == transition)
                ]
                if not row.empty:
                    matrix[r_idx, c_idx] = float(row.iloc[0]["pearson_task_delta_vs_metric_delta"])
        image = ax.imshow(matrix, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(MODEL_LABEL[model])
        ax.set_xticks(range(len(DECODER_TRANSITIONS)))
        ax.set_xticklabels(DECODER_TRANSITIONS, rotation=35, ha="right")
        ax.set_yticks(range(len(HEATMAP_METRICS)))
        ax.set_yticklabels([METRIC_INFO[m]["short"] for m in HEATMAP_METRICS])
        ax.grid(False)
        for r_idx in range(matrix.shape[0]):
            for c_idx in range(matrix.shape[1]):
                value = matrix[r_idx, c_idx]
                if np.isfinite(value):
                    ax.text(
                        c_idx,
                        r_idx,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="black",
                    )
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.02)
        cbar.set_label("Pearson r")
    fig.suptitle("Transition-wise case-delta correlation", fontsize=12)
    save_figure(fig, out_stem)


def write_tables(
    out_root: Path,
    stage1_summary: pd.DataFrame,
    decoder_summary: pd.DataFrame,
    late_summary: pd.DataFrame,
) -> None:
    table_dir = out_root / "tables"
    ensure_dir(table_dir)
    for frame in (stage1_summary, decoder_summary, late_summary):
        for col in ["pearson", "pearson_ci95_low", "pearson_ci95_high", "spearman", "task_delta_mean", "metric_delta_mean"]:
            frame[col] = frame[col].astype(float).round(4)
    stage1_summary.to_csv(table_dir / "table_case_delta_stage1_alignment_readout.csv", index=False)
    decoder_summary.to_csv(table_dir / "table_case_delta_decoder_alignment_readout.csv", index=False)
    late_summary.to_csv(table_dir / "table_case_delta_late_transition_readout.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-table", default=DEFAULT_ALIGNMENT_TABLE)
    parser.add_argument("--corr-table", default=DEFAULT_CORR_TABLE)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()

    alignment = pd.read_csv(args.alignment_table)
    corr = pd.read_csv(args.corr_table)
    out_root = Path(args.out_root)
    main_dir = out_root / "main"
    supp_dir = out_root / "supplement"

    stage1_summary = plot_stage1_scatter(
        alignment,
        main_dir / "fig7_case_delta_alignment_stage1_scatter",
    )
    decoder_summary = plot_decoder_scatter(
        alignment,
        supp_dir / "fig_s25_decoder_delta_scatter",
    )
    late_summary = plot_late_transition_scatter(
        alignment,
        supp_dir / "fig_s23_late_transition_delta_scatter",
    )
    plot_decoder_heatmap(
        corr,
        supp_dir / "fig_s24_case_delta_correlation_heatmap",
    )
    write_tables(out_root, stage1_summary, decoder_summary, late_summary)

    print("Generated:")
    for path in [
        main_dir / "fig7_case_delta_alignment_stage1_scatter.png",
        main_dir / "fig7_case_delta_alignment_stage1_scatter.pdf",
        supp_dir / "fig_s23_late_transition_delta_scatter.png",
        supp_dir / "fig_s23_late_transition_delta_scatter.pdf",
        supp_dir / "fig_s24_case_delta_correlation_heatmap.png",
        supp_dir / "fig_s24_case_delta_correlation_heatmap.pdf",
        supp_dir / "fig_s25_decoder_delta_scatter.png",
        supp_dir / "fig_s25_decoder_delta_scatter.pdf",
        out_root / "tables" / "table_case_delta_stage1_alignment_readout.csv",
        out_root / "tables" / "table_case_delta_decoder_alignment_readout.csv",
        out_root / "tables" / "table_case_delta_late_transition_readout.csv",
    ]:
        print(path)


if __name__ == "__main__":
    main()
