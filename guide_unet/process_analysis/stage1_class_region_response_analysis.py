from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


GUIDE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GUIDE_ROOT not in sys.path:
    sys.path.insert(0, GUIDE_ROOT)

from process_analysis.stage1_explanation_utils import (  # noqa: E402
    GradCAMStore,
    bootstrap_mean_ci,
    build_eval_loader,
    compute_cam_variant_native,
    ensure_dir,
    feature_response_from_activation,
    load_model,
    parse_csv_list,
    parse_model_specs,
    resolve_device,
    resolve_node_names,
    resize_map,
    segmentation_target_score,
    set_seed,
    top_fraction_mask,
    unpack_logits,
    write_json,
    write_manifest,
    write_run_config,
)


DEFAULT_BASELINE_CKPT = (
    "checkpoints/baseline_brats2023_scratch100_seed42/"
    "baseline/seed_42/baseline_seed42_best_val_loss.pth"
)
DEFAULT_NOSKIP_CKPT = (
    "skip_case_checkpoints/skip_case_brats2023_full100_seed42/"
    "noskip_unet/seed_42/noskip_unet_seed42_best_val_loss.pth"
)
DEFAULT_CASE_SET = "stage1_explanation_suite_20260517/results/common_case_sets/formal_n512_cases.csv"
DEFAULT_E4_ADJUSTED = (
    "stage1_explanation_suite_20260517/results/e4_faithfulness_perturbation_formal_n512_allnodes_main/"
    "table_perturbation_random_adjusted.csv"
)
DEFAULT_OUTPUT_ROOT = "stage1_explanation_suite_20260517/results/e9_class_region_response_formal_n512"

RESPONSE_SOURCES = ("cam", "feature_l2")
NODE_ORDER = ("down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4")
REGIONS = (
    ("background", 0, "#bdbdbd"),
    ("edema", 2, "#1fb141"),
    ("necrosis", 1, "#df1b23"),
    ("enhancing", 3, "#ffc31a"),
)
REGION_ORDER = [item[0] for item in REGIONS]
REGION_COLORS = {name: color for name, _, color in REGIONS}
MODEL_LABEL = {"baseline": "有跳接 U-Net", "noskip_unet": "无跳接 U-Net"}
SOURCE_LABEL = {"cam": "CAM", "feature_l2": "原始特征响应"}
REGION_LABEL = {
    "background": "背景",
    "edema": "水肿",
    "necrosis": "坏死/核心",
    "enhancing": "增强肿瘤",
}


def configure_chinese_font() -> None:
    font_candidates = [
        r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for font_path in font_candidates:
        if os.path.isfile(font_path):
            font_manager.fontManager.addfont(font_path)
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=font_path).get_name()]
            break
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E9 class-wise response-location analysis for Stage-1 explanation."
    )
    parser.add_argument(
        "--model-spec",
        nargs="+",
        default=[
            "baseline={}".format(DEFAULT_BASELINE_CKPT),
            "noskip_unet={}".format(DEFAULT_NOSKIP_CKPT),
        ],
        help="Format: model_name=/path/to/checkpoint",
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--eval-split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--case-set-csv", type=str, default=DEFAULT_CASE_SET)
    parser.add_argument("--case-names", type=str, default="")
    parser.add_argument("--num-cases", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--nodes", type=str, default="all")
    parser.add_argument("--response-sources", type=str, default="cam,feature_l2")
    parser.add_argument("--target-mode", type=str, default="pred_fg", choices=["pred_fg", "gt_fg"])
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--e4-adjusted-csv", type=str, default=DEFAULT_E4_ADJUSTED)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--foreground-only", action="store_true")
    parser.add_argument("--patient-balanced", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def response_maps(store: GradCAMStore, node_name: str, sources: Sequence[str], output_size: tuple[int, int]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    if "cam" in sources:
        cam = compute_cam_variant_native(store=store, node_name=node_name, variant="gradcam")
        out["cam"] = resize_map(cam, output_size=output_size, mode="bilinear")
    if "feature_l2" in sources:
        feat = feature_response_from_activation(store.activations[node_name], aggregation="l2")
        out["feature_l2"] = resize_map(feat, output_size=output_size, mode="bilinear")
    return out


def region_masks(label: np.ndarray) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    for name, class_id, _ in REGIONS:
        if name == "background":
            masks[name] = np.asarray(label) == 0
        else:
            masks[name] = np.asarray(label) == int(class_id)
    return masks


def safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = np.asarray(mask).astype(bool)
    if not np.any(mask_bool):
        return np.nan
    return float(np.asarray(values, dtype=np.float64)[mask_bool].mean())


def safe_sum(values: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = np.asarray(mask).astype(bool)
    if not np.any(mask_bool):
        return 0.0
    return float(np.maximum(np.asarray(values, dtype=np.float64), 0.0)[mask_bool].sum())


def case_region_rows(
    model_name: str,
    batch,
    logits: torch.Tensor,
    store: GradCAMStore,
    node_names: Sequence[str],
    sources: Sequence[str],
    top_fraction: float,
) -> List[Dict]:
    labels_np = batch["label"].detach().cpu().numpy()
    case_names = list(batch["case_name"])
    output_size = tuple(int(v) for v in labels_np.shape[-2:])
    rows: List[Dict] = []

    for node_name in node_names:
        maps = response_maps(store=store, node_name=node_name, sources=sources, output_size=output_size)
        maps_np = {source: tensor.detach().cpu().numpy() for source, tensor in maps.items()}
        for local_idx, case_name in enumerate(case_names):
            label = labels_np[local_idx]
            masks = region_masks(label)
            total_pixels = float(label.size)
            for source in sources:
                response = np.maximum(maps_np[source][local_idx].astype(np.float64), 0.0)
                response_total = float(response.sum())
                top_mask = top_fraction_mask(response, fraction=float(top_fraction), largest=True)
                top_total = max(1, int(top_mask.sum()))
                background_mean = safe_mean(response, masks["background"])
                for region_name in REGION_ORDER:
                    mask = masks[region_name]
                    pixels = int(mask.sum())
                    area_fraction = float(pixels / total_pixels)
                    response_mean = safe_mean(response, mask)
                    response_sum = safe_sum(response, mask)
                    mass_fraction = float(response_sum / response_total) if response_total > 1e-12 else np.nan
                    top_pixels = int(np.logical_and(top_mask, mask).sum())
                    top_fraction_in_region = float(top_pixels / top_total)
                    enrichment = float(top_fraction_in_region / area_fraction) if area_fraction > 1e-12 else np.nan
                    bg_ratio = (
                        float(response_mean / (background_mean + 1e-8))
                        if np.isfinite(response_mean) and np.isfinite(background_mean)
                        else np.nan
                    )
                    rows.append(
                        {
                            "model": model_name,
                            "case_name": str(case_name),
                            "patient_id": str(case_name).rsplit("_", 1)[0],
                            "node": node_name,
                            "node_order": int(node_names.index(node_name)),
                            "response_source": source,
                            "region": region_name,
                            "region_order": int(REGION_ORDER.index(region_name)),
                            "region_pixels": pixels,
                            "region_area_fraction": area_fraction,
                            "response_mean": response_mean,
                            "response_mass_fraction": mass_fraction,
                            "top10_area_fraction": top_fraction_in_region,
                            "top10_enrichment": enrichment,
                            "region_to_background_mean_ratio": bg_ratio,
                            "top_fraction": float(top_fraction),
                        }
                    )
    return rows


def analyze_model(spec, loader, args: argparse.Namespace, device: torch.device, node_names: Sequence[str], sources: Sequence[str]) -> pd.DataFrame:
    model = load_model(model_name=spec.name, ckpt_path=spec.ckpt_path, device=device)
    store = GradCAMStore(model=model, node_names=node_names)
    rows: List[Dict] = []
    for batch in tqdm(loader, desc="E9 class-region {}".format(spec.name)):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        model.zero_grad(set_to_none=True)
        store.clear()
        logits = unpack_logits(model(images))
        score = segmentation_target_score(logits=logits, labels=labels, mode=args.target_mode)
        score.backward()
        rows.extend(
            case_region_rows(
                model_name=spec.name,
                batch=batch,
                logits=logits,
                store=store,
                node_names=node_names,
                sources=sources,
                top_fraction=float(args.top_fraction),
            )
        )
    store.remove()
    return pd.DataFrame(rows)


def summarize_region(case_df: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    metrics = [
        "region_area_fraction",
        "response_mean",
        "response_mass_fraction",
        "top10_area_fraction",
        "top10_enrichment",
        "region_to_background_mean_ratio",
    ]
    group_cols = ["model", "node", "node_order", "response_source", "region", "region_order"]
    rows: List[Dict] = []
    for key, group in case_df.groupby(group_cols, sort=False):
        payload = dict(zip(group_cols, key))
        payload["n_cases"] = int(group["case_name"].nunique())
        for metric in metrics:
            vals = group[metric].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            payload[metric] = float(finite.mean()) if finite.size else np.nan
            lo, hi = bootstrap_mean_ci(finite, n_boot=int(bootstrap), seed=int(seed)) if finite.size else (np.nan, np.nan)
            payload["{}_ci_low".format(metric)] = lo
            payload["{}_ci_high".format(metric)] = hi
        rows.append(payload)
    return pd.DataFrame(rows).sort_values(["model", "response_source", "node_order", "region_order"])


def late_transition_table(case_df: pd.DataFrame, node_names: Sequence[str], bootstrap: int, seed: int) -> pd.DataFrame:
    metrics = ["response_mean", "response_mass_fraction", "top10_area_fraction", "top10_enrichment"]
    rows: List[Dict] = []
    key_cols = ["model", "case_name", "patient_id", "response_source", "region"]
    for key, group in case_df.groupby(key_cols, sort=False):
        pivot = group.set_index("node")
        for src, dst in zip(node_names[:-1], node_names[1:]):
            if src not in pivot.index or dst not in pivot.index:
                continue
            row = dict(zip(key_cols, key))
            row["transition"] = "{}->{}".format(src, dst)
            row["source_node"] = src
            row["target_node"] = dst
            row["source_order"] = int(node_names.index(src))
            for metric in metrics:
                row["{}_source".format(metric)] = float(pivot.loc[src, metric])
                row["{}_target".format(metric)] = float(pivot.loc[dst, metric])
                row["{}_delta".format(metric)] = float(pivot.loc[dst, metric] - pivot.loc[src, metric])
            rows.append(row)
    delta_df = pd.DataFrame(rows)
    if delta_df.empty:
        return delta_df

    summary_rows: List[Dict] = []
    group_cols = ["model", "response_source", "region", "transition", "source_order"]
    for key, group in delta_df.groupby(group_cols, sort=False):
        payload = dict(zip(group_cols, key))
        payload["n_cases"] = int(group["case_name"].nunique())
        for metric in metrics:
            col = "{}_delta".format(metric)
            vals = group[col].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            payload[col] = float(finite.mean()) if finite.size else np.nan
            lo, hi = bootstrap_mean_ci(finite, n_boot=int(bootstrap), seed=int(seed)) if finite.size else (np.nan, np.nan)
            payload["{}_ci_low".format(col)] = lo
            payload["{}_ci_high".format(col)] = hi
        summary_rows.append(payload)
    return pd.DataFrame(summary_rows).sort_values(["model", "response_source", "source_order", "region"])


def wide_region_table(case_df: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["model", "case_name", "patient_id", "node", "node_order", "response_source"]
    value_cols = ["response_mean", "response_mass_fraction", "top10_area_fraction", "top10_enrichment"]
    wide_parts = []
    for value_col in value_cols:
        pivot = case_df.pivot_table(index=index_cols, columns="region", values=value_col, aggfunc="mean")
        pivot = pivot.rename(columns={region: "{}_{}".format(value_col, region) for region in pivot.columns})
        wide_parts.append(pivot)
    return pd.concat(wide_parts, axis=1).reset_index()


def corr_pair(x: Sequence[float], y: Sequence[float], method: str) -> float:
    a = pd.Series(x, dtype="float64")
    b = pd.Series(y, dtype="float64")
    valid = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 6:
        return np.nan
    if float(a[valid].std()) < 1e-10 or float(b[valid].std()) < 1e-10:
        return np.nan
    return float(a[valid].corr(b[valid], method=method))


def perturbation_region_correlation(wide_df: pd.DataFrame, e4_adjusted_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not e4_adjusted_csv or not os.path.isfile(e4_adjusted_csv):
        return pd.DataFrame(), pd.DataFrame()
    e4_cols = [
        "model",
        "case_name",
        "node",
        "response_source",
        "top_fraction",
        "gamma",
        "random_adjusted_final_fg_prob_drop",
        "random_adjusted_final_dice_drop",
        "random_adjusted_gt_fg_prob_drop",
    ]
    e4 = pd.read_csv(e4_adjusted_csv, usecols=lambda col: col in set(e4_cols))
    e4 = e4[(e4["top_fraction"].round(4) == 0.1000) & (e4["gamma"].round(4) == 1.0000)].copy()
    merged = wide_df.merge(e4, on=["model", "case_name", "node", "response_source"], how="inner")
    rows: List[Dict] = []
    predictor_prefixes = ["top10_area_fraction", "response_mass_fraction", "top10_enrichment"]
    outcomes = [
        "random_adjusted_final_fg_prob_drop",
        "random_adjusted_final_dice_drop",
        "random_adjusted_gt_fg_prob_drop",
    ]
    for (model, source, node), group in merged.groupby(["model", "response_source", "node"], sort=False):
        for region in REGION_ORDER:
            for prefix in predictor_prefixes:
                predictor = "{}_{}".format(prefix, region)
                if predictor not in group.columns:
                    continue
                for outcome in outcomes:
                    rows.append(
                        {
                            "model": model,
                            "response_source": source,
                            "node": node,
                            "region": region,
                            "predictor": predictor,
                            "outcome": outcome,
                            "n": int(group[["case_name", predictor, outcome]].dropna().shape[0]),
                            "pearson_r": corr_pair(group[predictor], group[outcome], method="pearson"),
                            "spearman_rho": corr_pair(group[predictor], group[outcome], method="spearman"),
                        }
                    )
    return merged, pd.DataFrame(rows)


def plot_top10_composition(summary: pd.DataFrame, out_path: str) -> None:
    sub = summary[summary["region"].isin(REGION_ORDER)].copy()
    models = ["baseline", "noskip_unet"]
    sources = ["cam", "feature_l2"]
    all_nodes = list(sub.sort_values("node_order")["node"].drop_duplicates())
    fig_width = max(10.2, 0.78 * len(all_nodes) + 4.2)
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 5.9), dpi=180, sharey=True)
    for row_idx, model in enumerate(models):
        for col_idx, source in enumerate(sources):
            ax = axes[row_idx, col_idx]
            panel = sub[(sub["model"] == model) & (sub["response_source"] == source)]
            nodes = list(panel.sort_values("node_order")["node"].drop_duplicates())
            bottom = np.zeros(len(nodes), dtype=float)
            for region in REGION_ORDER:
                vals = []
                for node in nodes:
                    row = panel[(panel["node"] == node) & (panel["region"] == region)]
                    vals.append(float(row["top10_area_fraction"].iloc[0]) if not row.empty else 0.0)
                ax.bar(nodes, vals, bottom=bottom, color=REGION_COLORS[region], label=REGION_LABEL[region], width=0.72)
                bottom += np.asarray(vals)
            ax.set_ylim(0, 1.0)
            ax.set_title("{} | {}".format(MODEL_LABEL.get(model, model), SOURCE_LABEL[source]), fontsize=9)
            ax.grid(axis="y", alpha=0.2)
            ax.tick_params(axis="x", rotation=25)
            if col_idx == 0:
                ax.set_ylabel("Top10 高响应区域占比")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("高响应区域在不同肿瘤类别中的落点", fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def plot_top10_enrichment(summary: pd.DataFrame, out_path: str) -> None:
    models = ["baseline", "noskip_unet"]
    sources = ["cam", "feature_l2"]
    all_nodes = list(summary.sort_values("node_order")["node"].drop_duplicates())
    fig_width = max(10.2, 0.78 * len(all_nodes) + 4.2)
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 5.9), dpi=180, sharey=True)
    for row_idx, model in enumerate(models):
        for col_idx, source in enumerate(sources):
            ax = axes[row_idx, col_idx]
            panel = summary[(summary["model"] == model) & (summary["response_source"] == source)]
            for region in ["edema", "necrosis", "enhancing"]:
                rsub = panel[panel["region"] == region].sort_values("node_order")
                ax.plot(
                    rsub["node"],
                    rsub["top10_enrichment"],
                    marker="o",
                    linewidth=1.8,
                    color=REGION_COLORS[region],
                    label=REGION_LABEL[region],
                )
            ax.axhline(1.0, color="#666666", linewidth=0.8, linestyle="--")
            ax.set_title("{} | {}".format(MODEL_LABEL.get(model, model), SOURCE_LABEL[source]), fontsize=9)
            ax.grid(alpha=0.25)
            ax.tick_params(axis="x", rotation=25)
            if col_idx == 0:
                ax.set_ylabel("Top10 富集倍数")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("高响应区域中的肿瘤类别富集", fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def plot_mass_fraction(summary: pd.DataFrame, out_path: str) -> None:
    models = ["baseline", "noskip_unet"]
    sources = ["cam", "feature_l2"]
    all_nodes = list(summary.sort_values("node_order")["node"].drop_duplicates())
    fig_width = max(10.2, 0.78 * len(all_nodes) + 4.2)
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 5.9), dpi=180, sharey=True)
    for row_idx, model in enumerate(models):
        for col_idx, source in enumerate(sources):
            ax = axes[row_idx, col_idx]
            panel = summary[(summary["model"] == model) & (summary["response_source"] == source)]
            for region in ["edema", "necrosis", "enhancing"]:
                rsub = panel[panel["region"] == region].sort_values("node_order")
                ax.plot(
                    rsub["node"],
                    rsub["response_mass_fraction"],
                    marker="o",
                    linewidth=1.8,
                    color=REGION_COLORS[region],
                    label=REGION_LABEL[region],
                )
            ax.set_title("{} | {}".format(MODEL_LABEL.get(model, model), SOURCE_LABEL[source]), fontsize=9)
            ax.grid(alpha=0.25)
            ax.tick_params(axis="x", rotation=25)
            if col_idx == 0:
                ax.set_ylabel("响应质量占比")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Stage 1 全节点肿瘤类别响应质量", fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def plot_late_delta(delta_summary: pd.DataFrame, out_path: str) -> None:
    sub = delta_summary[delta_summary["transition"] == "up3->up4"].copy()
    if sub.empty:
        return
    regions = ["background", "edema", "necrosis", "enhancing"]
    row_labels = []
    values = []
    for model in ["baseline", "noskip_unet"]:
        for source in ["cam", "feature_l2"]:
            row_labels.append("{} | {}".format(MODEL_LABEL.get(model, model), SOURCE_LABEL[source]))
            row_vals = []
            for region in regions:
                row = sub[(sub["model"] == model) & (sub["response_source"] == source) & (sub["region"] == region)]
                row_vals.append(float(row["response_mass_fraction_delta"].iloc[0]) if not row.empty else np.nan)
            values.append(row_vals)
    arr = np.asarray(values, dtype=float)
    vmax = np.nanmax(np.abs(arr)) if np.any(np.isfinite(arr)) else 1.0
    vmax = max(float(vmax), 1e-4)
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=180)
    im = ax.imshow(arr, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels([REGION_LABEL[item] for item in regions], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, "{:+.3f}".format(arr[i, j]), ha="center", va="center", fontsize=7)
    ax.set_title("后期转移的类别响应质量变化（up3->up4）")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("变化量")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_drop_correlation(corr_df: pd.DataFrame, out_path: str) -> None:
    if corr_df.empty:
        return
    sub = corr_df[
        (corr_df["predictor"].str.startswith("top10_area_fraction_"))
        & (corr_df["outcome"] == "random_adjusted_final_fg_prob_drop")
    ].copy()
    available_nodes = [str(item) for item in sub["node"].drop_duplicates().tolist()]
    nodes = [node for node in NODE_ORDER if node in set(available_nodes)]
    regions = ["background", "edema", "necrosis", "enhancing"]
    fig_width = max(10.2, 0.72 * len(nodes) + 4.8)
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 6.1), dpi=180, sharex=True, sharey=True)
    images = []
    for row_idx, model in enumerate(["baseline", "noskip_unet"]):
        for col_idx, source in enumerate(["cam", "feature_l2"]):
            ax = axes[row_idx, col_idx]
            arr_rows = []
            for region in regions:
                vals = []
                for node in nodes:
                    row = sub[
                        (sub["model"] == model)
                        & (sub["response_source"] == source)
                        & (sub["region"] == region)
                        & (sub["node"] == node)
                    ]
                    vals.append(float(row["pearson_r"].iloc[0]) if not row.empty else np.nan)
                arr_rows.append(vals)
            arr = np.asarray(arr_rows, dtype=float)
            im = ax.imshow(arr, cmap="coolwarm", vmin=-0.7, vmax=0.7)
            images.append(im)
            ax.set_title("{} | {}".format(MODEL_LABEL.get(model, model), SOURCE_LABEL[source]), fontsize=9)
            ax.set_xticks(np.arange(len(nodes)))
            ax.set_xticklabels(nodes, rotation=20, ha="right")
            ax.set_yticks(np.arange(len(regions)))
            ax.set_yticklabels([REGION_LABEL[item] for item in regions])
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    if np.isfinite(arr[i, j]):
                        ax.text(j, i, "{:+.2f}".format(arr[i, j]), ha="center", va="center", fontsize=6.5)
    fig.suptitle("高响应落点与遮挡后输出下降的相关性", fontsize=11, y=0.99)
    fig.subplots_adjust(left=0.16, right=0.86, top=0.88, bottom=0.12, hspace=0.30, wspace=0.26)
    cax = fig.add_axes([0.89, 0.22, 0.025, 0.58])
    cbar = fig.colorbar(images[0], cax=cax)
    cbar.set_label("Pearson 相关")
    fig.savefig(out_path)
    plt.close(fig)


def write_key_readout(summary: pd.DataFrame, delta_summary: pd.DataFrame, corr_df: pd.DataFrame, out_path: str) -> None:
    rows: List[Dict] = []
    for model in ["baseline", "noskip_unet"]:
        for source in ["cam", "feature_l2"]:
            for region in REGION_ORDER:
                late = delta_summary[
                    (delta_summary["model"] == model)
                    & (delta_summary["response_source"] == source)
                    & (delta_summary["region"] == region)
                    & (delta_summary["transition"] == "up3->up4")
                ]
                up4 = summary[
                    (summary["model"] == model)
                    & (summary["response_source"] == source)
                    & (summary["region"] == region)
                    & (summary["node"] == "up4")
                ]
                payload = {"model": model, "response_source": source, "region": region}
                if not up4.empty:
                    payload["up4_response_mass_fraction"] = float(up4["response_mass_fraction"].iloc[0])
                    payload["up4_top10_area_fraction"] = float(up4["top10_area_fraction"].iloc[0])
                    payload["up4_top10_enrichment"] = float(up4["top10_enrichment"].iloc[0])
                if not late.empty:
                    payload["up3_to_up4_mass_delta"] = float(late["response_mass_fraction_delta"].iloc[0])
                    payload["up3_to_up4_top10_delta"] = float(late["top10_area_fraction_delta"].iloc[0])
                if not corr_df.empty:
                    corr = corr_df[
                        (corr_df["model"] == model)
                        & (corr_df["response_source"] == source)
                        & (corr_df["region"] == region)
                        & (corr_df["node"] == "up4")
                        & (corr_df["predictor"] == "top10_area_fraction_{}".format(region))
                        & (corr_df["outcome"] == "random_adjusted_final_fg_prob_drop")
                    ]
                    if not corr.empty:
                        payload["up4_top10_location_vs_drop_pearson"] = float(corr["pearson_r"].iloc[0])
                        payload["up4_top10_location_vs_drop_spearman"] = float(corr["spearman_rho"].iloc[0])
                rows.append(payload)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    configure_chinese_font()
    if args.smoke:
        args.num_cases = min(int(args.num_cases), 4)
        args.batch_size = 1
        args.bootstrap = min(int(args.bootstrap), 50)
        args.output_root = args.output_root + "_smoke"

    ensure_dir(args.output_root)
    set_seed(int(args.seed))
    specs = parse_model_specs(args.model_spec)
    node_names = resolve_node_names(args.nodes)
    sources = parse_csv_list(args.response_sources)
    invalid_sources = sorted(set(sources) - set(RESPONSE_SOURCES))
    if invalid_sources:
        raise ValueError("Unsupported response sources: {}".format(invalid_sources))
    device = resolve_device(args.device)

    _, indices, case_names, loader = build_eval_loader(
        data_dir=args.data_dir,
        split=args.eval_split,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        num_cases=int(args.num_cases),
        seed=int(args.seed),
        foreground_only=bool(args.foreground_only),
        case_set_csv=args.case_set_csv,
        case_names=parse_csv_list(args.case_names),
        patient_balanced=bool(args.patient_balanced),
    )

    frames = [
        analyze_model(spec=spec, loader=loader, args=args, device=device, node_names=node_names, sources=sources)
        for spec in specs
    ]
    case_df = pd.concat(frames, ignore_index=True)
    summary = summarize_region(case_df, bootstrap=int(args.bootstrap), seed=int(args.seed))
    delta_summary = late_transition_table(case_df, node_names=node_names, bootstrap=int(args.bootstrap), seed=int(args.seed))
    wide = wide_region_table(case_df)
    merged_e4, corr_df = perturbation_region_correlation(wide, args.e4_adjusted_csv)

    case_path = os.path.join(args.output_root, "table_class_region_response_case_level.csv")
    summary_path = os.path.join(args.output_root, "table_class_region_response_summary.csv")
    delta_path = os.path.join(args.output_root, "table_class_region_late_transition_delta_summary.csv")
    wide_path = os.path.join(args.output_root, "table_class_region_response_wide_case_level.csv")
    merged_path = os.path.join(args.output_root, "table_class_region_perturbation_joined_case_level.csv")
    corr_path = os.path.join(args.output_root, "table_class_region_location_drop_correlation.csv")
    key_path = os.path.join(args.output_root, "table_class_region_key_readout.csv")

    case_df.to_csv(case_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    delta_summary.to_csv(delta_path, index=False, encoding="utf-8-sig")
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")
    if not merged_e4.empty:
        merged_e4.to_csv(merged_path, index=False, encoding="utf-8-sig")
    if not corr_df.empty:
        corr_df.to_csv(corr_path, index=False, encoding="utf-8-sig")
    write_key_readout(summary, delta_summary, corr_df, key_path)

    plot_top10_composition(summary, os.path.join(args.output_root, "plot_top10_region_composition_by_node.png"))
    plot_top10_enrichment(summary, os.path.join(args.output_root, "plot_top10_region_enrichment_by_node.png"))
    plot_mass_fraction(summary, os.path.join(args.output_root, "plot_region_response_mass_by_node.png"))
    plot_late_delta(delta_summary, os.path.join(args.output_root, "plot_late_transition_region_mass_delta.png"))
    if not corr_df.empty:
        plot_drop_correlation(corr_df, os.path.join(args.output_root, "plot_region_location_drop_correlation.png"))

    write_run_config(args, args.output_root, extra={"device": str(device), "selected_indices": [int(v) for v in indices]})
    write_manifest(
        args.output_root,
        [
            {"output": os.path.basename(path), "path": path}
            for path in [
                case_path,
                summary_path,
                delta_path,
                wide_path,
                merged_path,
                corr_path,
                key_path,
                os.path.join(args.output_root, "plot_top10_region_composition_by_node.png"),
                os.path.join(args.output_root, "plot_top10_region_enrichment_by_node.png"),
                os.path.join(args.output_root, "plot_region_response_mass_by_node.png"),
                os.path.join(args.output_root, "plot_late_transition_region_mass_delta.png"),
                os.path.join(args.output_root, "plot_region_location_drop_correlation.png"),
            ]
        ],
    )
    write_json(
        {
            "purpose": "E9 class-wise response-location analysis",
            "num_cases": int(len(case_names)),
            "models": [spec.name for spec in specs],
            "nodes": list(node_names),
            "response_sources": list(sources),
            "regions": REGION_ORDER,
            "top_fraction": float(args.top_fraction),
            "main_question": "量化响应在背景、水肿、坏死/非增强核心、增强肿瘤中的落点，并检查高响应落点是否与遮挡后输出下降相关。",
        },
        os.path.join(args.output_root, "summary_class_region_response.json"),
    )
    print("Saved E9 class-region response outputs to {}".format(args.output_root))
    print("Device:", device)
    print("Cases:", len(case_names))


if __name__ == "__main__":
    main()
