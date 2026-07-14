"""Generate English submission figures from completed Stage 1 result tables.

This script performs no inference. It only reads existing CSV summaries and
redraws selected evidence figures with English labels for manuscript use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUTPUT_ROOT = "stage1_explanation_suite_20260517/results/english_submission_figures"
DEFAULT_E9_ROOT = "stage1_explanation_suite_20260517/results/e9_class_region_response_formal_n512"
DEFAULT_E10_ROOT = "stage1_explanation_suite_20260517/results/e10_statistical_validation_formal_n512"
DEFAULT_E11_ROOT = "stage1_explanation_suite_20260517/results/e11_up_source_decomposition_formal_n512"
DEFAULT_E12_ROOT = "stage1_explanation_suite_20260517/results/e12_structural_cause_validation_formal_n512"
DEFAULT_E13_ROOT = "stage1_explanation_suite_20260517/results/e13_up4_skip_counterfactual_mediation_formal_n512"

NODES = ["down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4"]
UP_NODES = ["up1", "up2", "up3", "up4"]

MODEL_LABEL = {"baseline": "baseline U-Net", "noskip_unet": "no-skip U-Net"}
MODEL_COLOR = {"baseline": "#1f77b4", "noskip_unet": "#d55e00"}
SOURCE_LABEL = {"cam": "Grad-CAM", "feature_l2": "raw feature response"}
REGION_LABEL = {
    "background": "background",
    "edema": "edema",
    "necrosis": "necrotic/core",
    "enhancing": "enhancing tumor",
    "tumor_total": "tumor total",
}
REGION_COLOR = {
    "background": "#d8d8d8",
    "edema": "#2ca25f",
    "necrosis": "#d73027",
    "enhancing": "#f2c230",
    "tumor_total": "#4c78a8",
}
ROLE_LABEL = {
    "main_path": "main path",
    "skip_input": "skip input",
    "concat_input": "concat input",
    "conv_output": "fusion output",
}
ROLE_COLOR = {
    "main_path": "#8c8c8c",
    "skip_input": "#1b9e77",
    "concat_input": "#7570b3",
    "conv_output": "#d95f02",
}
PERTURB_LABEL = {
    "main_zero": "main zero",
    "skip_zero": "skip zero",
    "skip_misalign": "skip misalign",
}
COUNTERFACTUAL_LABEL = {
    "boundary_cross": "boundary lesion replacement",
    "boundary_shift": "boundary shift",
    "boundary_background_control": "boundary background control",
    "boundary_random_control": "boundary random control",
    "interior_cross": "interior lesion replacement",
}
CONTRAST_LABEL = {
    "boundary_cross_vs_background": "boundary lesion vs background",
    "boundary_cross_vs_random": "boundary lesion vs random",
    "boundary_shift_vs_background": "boundary shift vs background",
    "interior_cross_vs_background": "interior lesion vs background",
    "interior_cross_vs_random": "interior lesion vs random",
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


def save_figure(fig: plt.Figure, out_stem: Path) -> list[dict[str, str]]:
    ensure_dir(out_stem.parent)
    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [{"name": out_stem.name, "png": str(png), "pdf": str(pdf)}]


def as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def ci_errors(values: pd.Series, low: pd.Series, high: pd.Series) -> np.ndarray:
    vals = as_float(values).to_numpy(dtype=float)
    lows = as_float(low).to_numpy(dtype=float)
    highs = as_float(high).to_numpy(dtype=float)
    return np.vstack([vals - lows, highs - vals])


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def plot_region_mass(summary: pd.DataFrame, source: str, out_dir: Path, name: str) -> list[dict[str, str]]:
    regions = ["background", "edema", "necrosis", "enhancing"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    x = np.arange(len(NODES))
    for ax, model in zip(axes, ["baseline", "noskip_unet"]):
        sub = summary[(summary["model"] == model) & (summary["response_source"] == source)].copy()
        bottom = np.zeros(len(NODES))
        for region in regions:
            vals = []
            for node in NODES:
                row = sub[(sub["node"] == node) & (sub["region"] == region)]
                vals.append(float(row["response_mass_fraction"].iloc[0]) if len(row) else np.nan)
            ax.bar(
                x,
                vals,
                bottom=bottom,
                width=0.72,
                color=REGION_COLOR[region],
                edgecolor="white",
                linewidth=0.3,
                label=REGION_LABEL[region],
            )
            bottom += np.nan_to_num(vals)
        ax.set_title(f"{MODEL_LABEL[model]} | {SOURCE_LABEL[source]}")
        ax.set_xticks(x)
        ax.set_xticklabels(NODES, rotation=30, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Stage 1 node")
    axes[0].set_ylabel("response mass fraction")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Node-wise response allocation across tumor regions", y=1.02)
    fig.subplots_adjust(bottom=0.24, wspace=0.12)
    return save_figure(fig, out_dir / name)


def plot_enrichment(summary: pd.DataFrame, source: str, out_dir: Path, name: str) -> list[dict[str, str]]:
    regions = ["edema", "necrosis", "enhancing"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    x = np.arange(len(NODES))
    for ax, model in zip(axes, ["baseline", "noskip_unet"]):
        sub = summary[(summary["model"] == model) & (summary["response_source"] == source)].copy()
        for region in regions:
            vals = []
            for node in NODES:
                row = sub[(sub["node"] == node) & (sub["region"] == region)]
                vals.append(float(row["top10_enrichment"].iloc[0]) if len(row) else np.nan)
            ax.plot(
                x,
                vals,
                marker="o",
                linewidth=1.8,
                color=REGION_COLOR[region],
                label=REGION_LABEL[region],
            )
        ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.9)
        ax.set_title(f"{MODEL_LABEL[model]} | {SOURCE_LABEL[source]}")
        ax.set_xticks(x)
        ax.set_xticklabels(NODES, rotation=30, ha="right")
        ax.set_xlabel("Stage 1 node")
    axes[0].set_ylabel("top-10% response enrichment")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Tumor-category enrichment in high-response regions", y=1.02)
    fig.subplots_adjust(bottom=0.24, wspace=0.12)
    return save_figure(fig, out_dir / name)


def plot_late_delta(delta: pd.DataFrame, out_dir: Path) -> list[dict[str, str]]:
    regions = ["background", "edema", "necrosis", "enhancing"]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    x = np.arange(len(regions))
    width = 0.34
    for offset, model in [(-width / 2, "baseline"), (width / 2, "noskip_unet")]:
        sub = delta[
            (delta["model"] == model)
            & (delta["response_source"] == "cam")
            & (delta["transition"] == "up3->up4")
            & (delta["region"].isin(regions))
        ].copy()
        sub["region"] = pd.Categorical(sub["region"], categories=regions, ordered=True)
        sub = sub.sort_values("region")
        vals = as_float(sub["response_mass_fraction_delta"])
        err = ci_errors(vals, sub["response_mass_fraction_delta_ci_low"], sub["response_mass_fraction_delta_ci_high"])
        ax.bar(
            x + offset,
            vals,
            width,
            yerr=err,
            capsize=3,
            color=MODEL_COLOR[model],
            alpha=0.88,
            label=MODEL_LABEL[model],
        )
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([REGION_LABEL[r] for r in regions], rotation=20, ha="right")
    ax.set_ylabel("change in response mass fraction")
    ax.set_title("Late decoder transfer in Grad-CAM response allocation (up3 to up4)")
    ax.legend(frameon=False)
    return save_figure(fig, out_dir / "fig9_late_transition_cam_region_delta_en")


def plot_key_effects(tests: pd.DataFrame, out_dir: Path) -> list[dict[str, str]]:
    labels = {
        "C3_late_acceleration_gap": "late semantic increment gap",
        "C4_up4_tumor_mass_crossing": "up4 tumor-response advantage",
        "C5_baseline_late_transfer_background": "baseline background decrease",
        "C5_baseline_late_transfer_tumor_total": "baseline tumor-response increase",
        "C8_up4_perturbation_baseline_cam": "baseline up4 CAM perturbation drop",
        "C9_baseline_up4_drop_exceeds_up3": "baseline perturbation gain at up4",
        "C10_noskip_up2_drop_exceeds_up4": "no-skip early perturbation dominance",
        "C1_endpoint_crossing": "endpoint difference",
    }
    ordered = [claim for claim in labels if claim in set(tests["claim_id"])]
    sub = tests[tests["claim_id"].isin(ordered)].copy()
    sub["claim_id"] = pd.Categorical(sub["claim_id"], categories=ordered, ordered=True)
    sub = sub.sort_values("claim_id")
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    y = np.arange(len(sub))
    vals = as_float(sub["mean_diff"])
    err = ci_errors(vals, sub["ci95_low"], sub["ci95_high"])
    colors = ["#4c78a8" if v >= 0 else "#d65f5f" for v in vals]
    ax.barh(y, vals, xerr=err, capsize=3, color=colors, alpha=0.9)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[item] for item in sub["claim_id"]])
    ax.invert_yaxis()
    ax.set_xlabel("mean effect with bootstrap 95% CI")
    ax.set_title("Key statistical effects in the Stage 1 evidence chain")
    return save_figure(fig, out_dir / "fig10_key_effects_with_ci_en")


def plot_up4_source_composition(source_summary: pd.DataFrame, out_dir: Path) -> list[dict[str, str]]:
    roles = ["main_path", "skip_input", "concat_input", "conv_output"]
    regions = ["background", "edema", "necrosis", "enhancing"]
    sub = source_summary[
        (source_summary["model"] == "baseline")
        & (source_summary["node"] == "up4")
        & (source_summary["source_role"].isin(roles))
        & (source_summary["region"].isin(regions))
    ].copy()
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    x = np.arange(len(roles))
    bottom = np.zeros(len(roles))
    for region in regions:
        vals = []
        for role in roles:
            row = sub[(sub["source_role"] == role) & (sub["region"] == region)]
            vals.append(float(row["response_mass_fraction"].iloc[0]) if len(row) else np.nan)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=REGION_COLOR[region],
            edgecolor="white",
            linewidth=0.3,
            label=REGION_LABEL[region],
        )
        bottom += np.nan_to_num(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([ROLE_LABEL[r] for r in roles], rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("response mass fraction")
    ax.set_title("Up4 source-wise response allocation in baseline U-Net")
    ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0.0, -0.36))
    fig.subplots_adjust(bottom=0.28)
    return save_figure(fig, out_dir / "fig11_up4_source_response_composition_en")


def plot_up_source_tumor_mass(key_readout: pd.DataFrame, out_dir: Path) -> list[dict[str, str]]:
    roles = ["main_path", "skip_input", "concat_input", "conv_output"]
    sub = key_readout[(key_readout["model"] == "baseline") & (key_readout["source_role"].isin(roles))].copy()
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = np.arange(len(UP_NODES))
    for role in roles:
        vals = []
        for node in UP_NODES:
            row = sub[(sub["source_role"] == role) & (sub["node"] == node)]
            vals.append(float(row["tumor_total_response_mass_fraction"].iloc[0]) if len(row) else np.nan)
        ax.plot(x, vals, marker="o", linewidth=1.8, color=ROLE_COLOR[role], label=ROLE_LABEL[role])
    ax.set_xticks(x)
    ax.set_xticklabels(UP_NODES)
    ax.set_xlabel("up node")
    ax.set_ylabel("tumor response mass fraction")
    ax.set_title("Tumor response carried by internal up-node sources")
    ax.legend(frameon=False, ncol=2)
    return save_figure(fig, out_dir / "fig12_up_source_tumor_mass_by_role_en")


def plot_branch_perturbation(branch: pd.DataFrame, out_dir: Path) -> list[dict[str, str]]:
    perturbations = ["main_zero", "skip_zero", "skip_misalign"]
    offsets = [-0.24, 0.0, 0.24]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(UP_NODES))
    for perturb, offset in zip(perturbations, offsets):
        vals = []
        lows = []
        highs = []
        for node in UP_NODES:
            row = branch[(branch["node"] == node) & (branch["perturbation"] == perturb)]
            if len(row):
                vals.append(float(row["final_fg_prob_drop"].iloc[0]))
                lows.append(float(row["final_fg_prob_drop_ci_low"].iloc[0]))
                highs.append(float(row["final_fg_prob_drop_ci_high"].iloc[0]))
            else:
                vals.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
        vals_s = pd.Series(vals)
        err = ci_errors(vals_s, pd.Series(lows), pd.Series(highs))
        ax.bar(
            x + offset,
            vals,
            0.22,
            yerr=err,
            capsize=2.5,
            label=PERTURB_LABEL[perturb],
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(UP_NODES)
    ax.set_xlabel("up node")
    ax.set_ylabel("final foreground probability drop")
    ax.set_title("Output impact of branch-level perturbations in baseline U-Net")
    ax.legend(frameon=False, ncol=3)
    return save_figure(fig, out_dir / "fig13_branch_perturbation_drop_en")


def plot_source_frequency_boundary(freq: pd.DataFrame, out_dir: Path) -> list[dict[str, str]]:
    roles = ["main_path", "skip_input", "concat_input", "conv_output"]
    sub = freq[(freq["model"] == "baseline") & (freq["node"] == "up4") & (freq["source_role"].isin(roles))].copy()
    sub["source_role"] = pd.Categorical(sub["source_role"], categories=roles, ordered=True)
    sub = sub.sort_values("source_role")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    metrics = [
        ("tensor_spatial_gradient_ratio", "spatial-gradient ratio"),
        ("boundary_to_nonboundary_edge_ratio", "boundary/non-boundary edge ratio"),
    ]
    x = np.arange(len(sub))
    colors = [ROLE_COLOR[r] for r in sub["source_role"]]
    for ax, (metric, ylabel) in zip(axes, metrics):
        vals = as_float(sub[metric])
        err = ci_errors(vals, sub[f"{metric}_ci_low"], sub[f"{metric}_ci_high"])
        ax.bar(x, vals, yerr=err, capsize=3, color=colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([ROLE_LABEL[r] for r in sub["source_role"]], rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
    fig.suptitle("Up4 source properties in baseline U-Net", y=1.03)
    fig.subplots_adjust(wspace=0.34, bottom=0.24)
    return save_figure(fig, out_dir / "fig14_up4_source_frequency_boundary_en")


def plot_fusion_prepost(fusion: pd.DataFrame, out_dir: Path) -> list[dict[str, str]]:
    sub = fusion[
        (fusion["model"] == "baseline")
        & (fusion["comparison"] == "output_minus_concat")
        & (fusion["node"].isin(UP_NODES))
    ].copy()
    sub["node"] = pd.Categorical(sub["node"], categories=UP_NODES, ordered=True)
    sub = sub.sort_values("node")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    x = np.arange(len(sub))
    metrics = [
        ("response_mass_fraction_background_delta", "background response change", "#999999"),
        ("response_mass_fraction_tumor_total_delta", "tumor response change", "#4c78a8"),
    ]
    width = 0.34
    for idx, (metric, label, color) in enumerate(metrics):
        vals = as_float(sub[metric])
        err = ci_errors(vals, sub[f"{metric}_ci_low"], sub[f"{metric}_ci_high"])
        axes[0].bar(x + (idx - 0.5) * width, vals, width, yerr=err, capsize=2.5, color=color, label=label)
    axes[0].axhline(0, color="#333333", linewidth=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(list(sub["node"]))
    axes[0].set_ylabel("conv output - concat input")
    axes[0].set_title("response allocation after fusion convolution")
    axes[0].legend(frameon=False)

    metric = "final_prob_pearson_delta"
    vals = as_float(sub[metric])
    err = ci_errors(vals, sub[f"{metric}_ci_low"], sub[f"{metric}_ci_high"])
    axes[1].bar(x, vals, yerr=err, capsize=2.5, color="#1f77b4", alpha=0.9)
    axes[1].axhline(0, color="#333333", linewidth=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(list(sub["node"]))
    axes[1].set_ylabel("change in final-probability Pearson")
    axes[1].set_title("task alignment after fusion convolution")
    fig.suptitle("Baseline U-Net fusion convolution turns concatenated input into task-aligned response", y=1.03)
    fig.subplots_adjust(wspace=0.34)
    return save_figure(fig, out_dir / "fig15_fusion_prepost_delta_en")


def plot_counterfactual(summary: pd.DataFrame, out_dir: Path) -> list[dict[str, str]]:
    variants = [
        "boundary_cross",
        "boundary_background_control",
        "boundary_random_control",
        "boundary_shift",
        "interior_cross",
    ]
    sub = summary[
        (summary["item"].isin(variants))
        & (summary["metric"] == "final_fg_prob_drop")
    ].copy()
    sub["item"] = pd.Categorical(sub["item"], categories=variants, ordered=True)
    sub = sub.sort_values("item")
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    y = np.arange(len(sub))
    vals = as_float(sub["value"])
    err = ci_errors(vals, sub["ci95_low"], sub["ci95_high"])
    colors = ["#1b9e77", "#bdbdbd", "#8c8c8c", "#7570b3", "#d95f02"]
    ax.barh(y, vals, xerr=err, capsize=3, color=colors[: len(sub)], alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([COUNTERFACTUAL_LABEL[i] for i in sub["item"]])
    ax.invert_yaxis()
    ax.set_xlabel("final foreground probability drop")
    ax.set_title("Up4 skip counterfactual region replacement")
    return save_figure(fig, out_dir / "fig16_up4_skip_counterfactual_output_drop_en")


def plot_counterfactual_contrasts(contrasts: pd.DataFrame, out_dir: Path) -> list[dict[str, str]]:
    ordered = [
        "boundary_cross_vs_background",
        "boundary_cross_vs_random",
        "boundary_shift_vs_background",
        "interior_cross_vs_background",
        "interior_cross_vs_random",
    ]
    sub = contrasts[contrasts["comparison"].isin(ordered)].copy()
    sub["comparison"] = pd.Categorical(sub["comparison"], categories=ordered, ordered=True)
    sub = sub.sort_values("comparison")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=True)
    y = np.arange(len(sub))
    metrics = [
        ("final_fg_prob_drop_diff", "output-drop contrast"),
        ("final_prob_pearson_drop_diff", "alignment-drop contrast"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        vals = as_float(sub[metric])
        err = ci_errors(vals, sub[f"{metric}_ci_low"], sub[f"{metric}_ci_high"])
        ax.barh(y, vals, xerr=err, capsize=2.5, color="#4c78a8", alpha=0.9)
        ax.axvline(0, color="#333333", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([CONTRAST_LABEL[c] for c in sub["comparison"]])
        ax.set_xlabel(ylabel)
    axes[0].invert_yaxis()
    axes[1].tick_params(labelleft=False)
    fig.suptitle("Paired counterfactual contrasts for up4 skip content", y=1.03)
    fig.subplots_adjust(wspace=0.18)
    return save_figure(fig, out_dir / "fig17_up4_skip_counterfactual_contrasts_en")


def run(args: argparse.Namespace) -> None:
    configure_style()
    out_root = Path(args.output_root)
    main_dir = out_root / "main"
    supplement_dir = out_root / "supplement"
    ensure_dir(main_dir)
    ensure_dir(supplement_dir)

    e9_root = Path(args.e9_root)
    e10_root = Path(args.e10_root)
    e11_root = Path(args.e11_root)
    e12_root = Path(args.e12_root)
    e13_root = Path(args.e13_root)

    records: list[dict[str, str]] = []

    e9_summary = load_csv(e9_root / "table_class_region_response_summary.csv")
    e9_delta = load_csv(e9_root / "table_class_region_late_transition_delta_summary.csv")
    records += plot_region_mass(e9_summary, "cam", main_dir, "fig7_cam_region_response_mass_by_node_en")
    records += plot_region_mass(e9_summary, "feature_l2", supplement_dir, "fig_s23_feature_region_response_mass_by_node_en")
    records += plot_enrichment(e9_summary, "cam", main_dir, "fig8_cam_top10_region_enrichment_by_node_en")
    records += plot_enrichment(e9_summary, "feature_l2", supplement_dir, "fig_s24_feature_top10_region_enrichment_by_node_en")
    records += plot_late_delta(e9_delta, main_dir)

    e10_tests = load_csv(e10_root / "table_e10_key_statistical_tests.csv")
    records += plot_key_effects(e10_tests, main_dir)

    e11_summary = load_csv(e11_root / "table_up_source_summary.csv")
    e11_key = load_csv(e11_root / "table_up_source_key_readout.csv")
    records += plot_up4_source_composition(e11_summary, main_dir)
    records += plot_up_source_tumor_mass(e11_key, main_dir)

    e12_branch = load_csv(e12_root / "table_e12_branch_perturbation_summary.csv")
    e12_freq = load_csv(e12_root / "table_e12_source_frequency_edge_summary.csv")
    e12_fusion = load_csv(e12_root / "table_e12_fusion_prepost_delta_summary.csv")
    records += plot_branch_perturbation(e12_branch, main_dir)
    records += plot_source_frequency_boundary(e12_freq, main_dir)
    records += plot_fusion_prepost(e12_fusion, main_dir)

    e13_summary = load_csv(e13_root / "table_e13_key_readout.csv")
    e13_contrasts = load_csv(e13_root / "table_e13_paired_contrasts.csv")
    records += plot_counterfactual(e13_summary, main_dir)
    records += plot_counterfactual_contrasts(e13_contrasts, main_dir)

    manifest = pd.DataFrame(records)
    manifest.to_csv(out_root / "run_manifest_english_submission_figures.csv", index=False)
    print(f"Generated {len(manifest)} English figure stems under {out_root}")
    for _, row in manifest.iterrows():
        print(row["png"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--e9-root", default=DEFAULT_E9_ROOT)
    parser.add_argument("--e10-root", default=DEFAULT_E10_ROOT)
    parser.add_argument("--e11-root", default=DEFAULT_E11_ROOT)
    parser.add_argument("--e12-root", default=DEFAULT_E12_ROOT)
    parser.add_argument("--e13-root", default=DEFAULT_E13_ROOT)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
