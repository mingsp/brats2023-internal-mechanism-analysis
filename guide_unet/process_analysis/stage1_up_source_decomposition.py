from __future__ import annotations

import argparse
import json
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
import torch.nn.functional as F
from tqdm import tqdm


GUIDE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GUIDE_ROOT not in sys.path:
    sys.path.insert(0, GUIDE_ROOT)

from analysis_common import safe_corr, safe_cos  # noqa: E402
from process_analysis.stage1_explanation_utils import (  # noqa: E402
    build_eval_loader,
    bootstrap_mean_ci,
    ensure_dir,
    feature_response_from_activation,
    load_model,
    parse_model_specs,
    resize_map,
    resolve_device,
    top_fraction_mask,
    tumor_prob_map,
    unpack_logits,
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
DEFAULT_OUTPUT_ROOT = "stage1_explanation_suite_20260517/results/e11_up_source_decomposition_formal_n512"

UP_NODES = ("up1", "up2", "up3", "up4")
REGIONS = (
    ("background", 0, "#bdbdbd"),
    ("edema", 2, "#1fb141"),
    ("necrosis", 1, "#df1b23"),
    ("enhancing", 3, "#ffc31a"),
)
REGION_ORDER = [item[0] for item in REGIONS]
MODEL_LABEL = {"baseline": "有跳接 U-Net", "noskip_unet": "无跳接 U-Net"}
ROLE_LABEL = {
    "main_path": "主路径上采样",
    "skip_input": "跳接输入",
    "concat_input": "拼接输入",
    "conv_output": "卷积输出",
}
REGION_LABEL = {
    "background": "背景",
    "edema": "水肿",
    "necrosis": "坏死/核心",
    "enhancing": "增强肿瘤",
    "tumor_total": "肿瘤总响应",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E11 up-node source decomposition for Stage-1 explanation.")
    parser.add_argument(
        "--model-spec",
        nargs="+",
        default=[
            "baseline={}".format(DEFAULT_BASELINE_CKPT),
            "noskip_unet={}".format(DEFAULT_NOSKIP_CKPT),
        ],
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--eval-split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--case-set-csv", type=str, default=DEFAULT_CASE_SET)
    parser.add_argument("--num-cases", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


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


def pad_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    diff_y = ref.size()[2] - x.size()[2]
    diff_x = ref.size()[3] - x.size()[3]
    return F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])


class UpSourceStore:
    def __init__(self, model: torch.nn.Module, model_name: str, node_names: Sequence[str]):
        self.model_name = str(model_name)
        self.node_names = tuple(node_names)
        self.sources: Dict[str, Dict[str, torch.Tensor]] = {}
        self._handles = []
        for node_name in self.node_names:
            module = getattr(model, node_name)
            self._handles.append(module.register_forward_pre_hook(self._make_pre_hook(node_name)))
            self._handles.append(module.register_forward_hook(self._make_forward_hook(node_name)))

    def _make_pre_hook(self, node_name: str):
        def hook_fn(module, inputs):
            x_deep = inputs[0]
            x_skip = inputs[1] if len(inputs) > 1 else None
            main = module.up(x_deep)
            if x_skip is not None:
                main = pad_like(main, x_skip)
            payload: Dict[str, torch.Tensor] = {
                "main_path": main,
            }
            if self.model_name == "baseline" and x_skip is not None:
                payload["skip_input"] = x_skip
                payload["concat_input"] = torch.cat([x_skip, main], dim=1)
            self.sources[node_name] = payload

        return hook_fn

    def _make_forward_hook(self, node_name: str):
        def hook_fn(module, inputs, output):
            if node_name not in self.sources:
                self.sources[node_name] = {}
            self.sources[node_name]["conv_output"] = output

        return hook_fn

    def clear(self) -> None:
        self.sources = {}

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


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


def response_from_source(tensor: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    response = feature_response_from_activation(tensor, aggregation="l2")
    return resize_map(response, output_size=output_size, mode="bilinear")


def source_case_rows(
    model_name: str,
    batch: Dict,
    logits: torch.Tensor,
    store: UpSourceStore,
    top_fraction: float,
) -> List[Dict]:
    labels_np = batch["label"].detach().cpu().numpy()
    case_names = list(batch["case_name"])
    output_size = tuple(int(v) for v in labels_np.shape[-2:])
    final_prob_np = tumor_prob_map(logits).detach().cpu().numpy()
    rows: List[Dict] = []

    for node_name in UP_NODES:
        node_payload = store.sources.get(node_name, {})
        for role_name, source_tensor in node_payload.items():
            response_tensor = response_from_source(source_tensor, output_size=output_size)
            response_np = response_tensor.detach().cpu().numpy()
            for local_idx, case_name in enumerate(case_names):
                label = labels_np[local_idx]
                response = np.maximum(response_np[local_idx].astype(np.float64), 0.0)
                final_prob = final_prob_np[local_idx].astype(np.float64)
                response_total = float(response.sum())
                masks = region_masks(label)
                gt_mask = label > 0
                top_mask = top_fraction_mask(response, fraction=float(top_fraction), largest=True)
                top_total = max(1, int(top_mask.sum()))
                background_mean = safe_mean(response, masks["background"])
                common = {
                    "model": model_name,
                    "case_name": str(case_name),
                    "patient_id": str(case_name).rsplit("_", 1)[0],
                    "node": node_name,
                    "node_order": int(UP_NODES.index(node_name)),
                    "source_role": role_name,
                    "role_order": int(["main_path", "skip_input", "concat_input", "conv_output"].index(role_name))
                    if role_name in ("main_path", "skip_input", "concat_input", "conv_output")
                    else 99,
                    "used_in_computation": bool(role_name != "skip_reference_ignored"),
                    "final_prob_pearson": float(safe_corr(response, final_prob)),
                    "final_prob_cosine": float(safe_cos(response, final_prob)),
                    "gt_mass_fraction": float(safe_sum(response, gt_mask) / response_total)
                    if response_total > 1e-12
                    else np.nan,
                    "top_fraction": float(top_fraction),
                }
                for region_name in REGION_ORDER:
                    mask = masks[region_name]
                    pixels = int(mask.sum())
                    area_fraction = float(pixels / float(label.size))
                    response_mean = safe_mean(response, mask)
                    response_sum = safe_sum(response, mask)
                    mass_fraction = float(response_sum / response_total) if response_total > 1e-12 else np.nan
                    top_pixels = int(np.logical_and(top_mask, mask).sum())
                    top_area_fraction = float(top_pixels / top_total)
                    enrichment = float(top_area_fraction / area_fraction) if area_fraction > 1e-12 else np.nan
                    bg_ratio = (
                        float(response_mean / (background_mean + 1e-8))
                        if np.isfinite(response_mean) and np.isfinite(background_mean)
                        else np.nan
                    )
                    row = dict(common)
                    row.update(
                        {
                            "region": region_name,
                            "region_order": int(REGION_ORDER.index(region_name)),
                            "region_pixels": pixels,
                            "region_area_fraction": area_fraction,
                            "response_mean": response_mean,
                            "response_mass_fraction": mass_fraction,
                            "top10_area_fraction": top_area_fraction,
                            "top10_enrichment": enrichment,
                            "region_to_background_mean_ratio": bg_ratio,
                        }
                    )
                    rows.append(row)
    return rows


def analyze_model(spec, loader, args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    model = load_model(model_name=spec.name, ckpt_path=spec.ckpt_path, device=device)
    store = UpSourceStore(model=model, model_name=spec.name, node_names=UP_NODES)
    rows: List[Dict] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="E11 up-source {}".format(spec.name)):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            store.clear()
            logits = unpack_logits(model(images))
            batch = dict(batch)
            batch["label"] = labels
            rows.extend(source_case_rows(spec.name, batch, logits, store, top_fraction=float(args.top_fraction)))
    store.remove()
    return pd.DataFrame(rows)


def summarize(case_df: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    metrics = [
        "region_area_fraction",
        "response_mean",
        "response_mass_fraction",
        "top10_area_fraction",
        "top10_enrichment",
        "region_to_background_mean_ratio",
        "final_prob_pearson",
        "final_prob_cosine",
        "gt_mass_fraction",
    ]
    rows: List[Dict] = []
    group_cols = ["model", "node", "node_order", "source_role", "role_order", "region", "region_order"]
    for keys, group in case_df.groupby(group_cols, sort=False):
        payload = dict(zip(group_cols, keys))
        payload["n_cases"] = int(group["case_name"].nunique())
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            payload[metric] = float(np.nanmean(values))
            ci_low, ci_high = bootstrap_mean_ci(values, n_boot=int(bootstrap), seed=int(seed) + len(rows))
            payload["{}_ci_low".format(metric)] = ci_low
            payload["{}_ci_high".format(metric)] = ci_high
        rows.append(payload)
    return pd.DataFrame(rows).sort_values(["model", "node_order", "role_order", "region_order"])


def wide_case_table(case_df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "region_area_fraction",
        "response_mass_fraction",
        "top10_area_fraction",
        "top10_enrichment",
        "region_to_background_mean_ratio",
        "final_prob_pearson",
        "final_prob_cosine",
        "gt_mass_fraction",
    ]
    wide = case_df.pivot_table(
        index=["model", "case_name", "patient_id", "node", "node_order", "source_role", "role_order"],
        columns="region",
        values=value_cols,
        aggfunc="mean",
    )
    wide.columns = ["{}_{}".format(metric, region) for metric, region in wide.columns]
    wide = wide.reset_index()
    for metric in ("response_mass_fraction", "top10_area_fraction", "region_area_fraction"):
        cols = ["{}_{}".format(metric, r) for r in ("edema", "necrosis", "enhancing")]
        if all(col in wide.columns for col in cols):
            wide["{}_tumor_total".format(metric)] = wide[cols].sum(axis=1)
    return wide


def source_delta_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    metrics = [
        "response_mass_fraction_background",
        "response_mass_fraction_edema",
        "response_mass_fraction_enhancing",
        "response_mass_fraction_tumor_total",
        "final_prob_pearson_background",
    ]
    # final_prob_pearson is duplicated by region after pivot; use background copy as a role-level value.
    if "final_prob_pearson_background" not in wide.columns and "final_prob_pearson_edema" in wide.columns:
        wide["final_prob_pearson_background"] = wide["final_prob_pearson_edema"]
    for (model_name, case_name, node), group in wide.groupby(["model", "case_name", "node"], sort=False):
        pivot = group.set_index("source_role")
        comparisons = []
        if model_name == "baseline":
            comparisons.extend(
                [
                    ("concat_minus_main", "concat_input", "main_path"),
                    ("concat_minus_skip", "concat_input", "skip_input"),
                    ("output_minus_concat", "conv_output", "concat_input"),
                    ("output_minus_main", "conv_output", "main_path"),
                    ("output_minus_skip", "conv_output", "skip_input"),
                ]
            )
        else:
            comparisons.append(("output_minus_main", "conv_output", "main_path"))
        for name, dst, src in comparisons:
            if dst not in pivot.index or src not in pivot.index:
                continue
            row = {
                "model": model_name,
                "case_name": case_name,
                "patient_id": str(case_name).rsplit("_", 1)[0],
                "node": node,
                "node_order": int(UP_NODES.index(node)),
                "comparison": name,
            }
            for metric in metrics:
                if metric in pivot.columns:
                    row["{}_delta".format(metric)] = float(pivot.loc[dst, metric] - pivot.loc[src, metric])
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_delta(delta_df: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    rows: List[Dict] = []
    metric_cols = [col for col in delta_df.columns if col.endswith("_delta")]
    group_cols = ["model", "node", "node_order", "comparison"]
    for keys, group in delta_df.groupby(group_cols, sort=False):
        payload = dict(zip(group_cols, keys))
        payload["n_cases"] = int(group["case_name"].nunique())
        for metric in metric_cols:
            values = group[metric].to_numpy(dtype=np.float64)
            payload[metric] = float(np.nanmean(values))
            ci_low, ci_high = bootstrap_mean_ci(values, n_boot=int(bootstrap), seed=int(seed) + len(rows))
            payload["{}_ci_low".format(metric)] = ci_low
            payload["{}_ci_high".format(metric)] = ci_high
        rows.append(payload)
    return pd.DataFrame(rows).sort_values(["model", "node_order", "comparison"])


def key_readout(summary_df: pd.DataFrame, delta_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for model_name in ("baseline", "noskip_unet"):
        for node in UP_NODES:
            for role in ("main_path", "skip_input", "concat_input", "conv_output"):
                sub = summary_df[
                    (summary_df["model"] == model_name)
                    & (summary_df["node"] == node)
                    & (summary_df["source_role"] == role)
                    & (summary_df["region"].isin(["background", "edema", "necrosis", "enhancing"]))
                ]
                if sub.empty:
                    continue
                by_region = sub.set_index("region")
                tumor_total = float(
                    by_region.loc[[r for r in ("edema", "necrosis", "enhancing") if r in by_region.index],
                                  "response_mass_fraction"].sum()
                )
                rows.append(
                    {
                        "model": model_name,
                        "node": node,
                        "source_role": role,
                        "background_response_mass_fraction": float(
                            by_region.loc["background", "response_mass_fraction"]
                        )
                        if "background" in by_region.index
                        else np.nan,
                        "edema_response_mass_fraction": float(by_region.loc["edema", "response_mass_fraction"])
                        if "edema" in by_region.index
                        else np.nan,
                        "enhancing_response_mass_fraction": float(
                            by_region.loc["enhancing", "response_mass_fraction"]
                        )
                        if "enhancing" in by_region.index
                        else np.nan,
                        "tumor_total_response_mass_fraction": tumor_total,
                        "final_prob_pearson": float(by_region["final_prob_pearson"].dropna().iloc[0])
                        if by_region["final_prob_pearson"].dropna().size
                        else np.nan,
                    }
                )
    key = pd.DataFrame(rows)
    return key.sort_values(["model", "node", "source_role"]) if not key.empty else key


def plot_tumor_mass(summary_df: pd.DataFrame, out_path: str) -> None:
    configure_chinese_font()
    sub = summary_df[summary_df["region"].isin(["edema", "necrosis", "enhancing"])].copy()
    tumor = (
        sub.groupby(["model", "node", "node_order", "source_role", "role_order"], sort=False)[
            "response_mass_fraction"
        ]
        .sum()
        .reset_index(name="tumor_total_response_mass_fraction")
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.7), dpi=170, sharey=True)
    for ax, model_name in zip(axes, ("baseline", "noskip_unet")):
        msub = tumor[tumor["model"] == model_name]
        for role, group in msub.groupby("source_role", sort=False):
            group = group.sort_values("node_order")
            ax.plot(
                group["node"],
                group["tumor_total_response_mass_fraction"],
                marker="o",
                linewidth=2.0,
                label=ROLE_LABEL.get(role, role),
            )
        ax.set_title(MODEL_LABEL.get(model_name, model_name))
        ax.set_xlabel("up 节点")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("肿瘤总响应质量占比")
    fig.suptitle("up 节点内部来源的肿瘤响应质量")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_source_filtering(delta_summary: pd.DataFrame, out_path: str) -> None:
    configure_chinese_font()
    sub = delta_summary[
        (delta_summary["model"] == "baseline") & (delta_summary["comparison"] == "output_minus_concat")
    ].copy()
    if sub.empty:
        return
    metrics = {
        "response_mass_fraction_background_delta": "背景",
        "response_mass_fraction_edema_delta": "水肿",
        "response_mass_fraction_enhancing_delta": "增强肿瘤",
        "response_mass_fraction_tumor_total_delta": "肿瘤总响应",
    }
    fig, ax = plt.subplots(figsize=(9.0, 4.5), dpi=170)
    for metric, label in metrics.items():
        if metric not in sub.columns:
            continue
        group = sub.sort_values("node_order")
        ax.plot(group["node"], group[metric], marker="o", linewidth=2.0, label=label)
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.9)
    ax.set_title("有跳接 U-Net：拼接后卷积对响应落点的筛选")
    ax.set_xlabel("up 节点")
    ax.set_ylabel("卷积输出 - 拼接输入 的响应质量变化")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_up4_source_bar(summary_df: pd.DataFrame, out_path: str) -> None:
    configure_chinese_font()
    sub = summary_df[
        (summary_df["node"] == "up4")
        & (summary_df["region"].isin(["background", "edema", "necrosis", "enhancing"]))
    ].copy()
    if sub.empty:
        return
    labels = []
    values = []
    colors = []
    color_map = {"background": "#bdbdbd", "edema": "#1fb141", "necrosis": "#df1b23", "enhancing": "#ffc31a"}
    for model_name in ("baseline", "noskip_unet"):
        roles = ["main_path", "skip_input", "concat_input", "conv_output"] if model_name == "baseline" else ["main_path", "conv_output"]
        for role in roles:
            group = sub[(sub["model"] == model_name) & (sub["source_role"] == role)]
            if group.empty:
                continue
            labels.append("{}\n{}".format(MODEL_LABEL.get(model_name, model_name), ROLE_LABEL.get(role, role)))
            values.append(group.set_index("region").loc[["background", "edema", "necrosis", "enhancing"], "response_mass_fraction"].to_numpy())
    if not values:
        return
    values_arr = np.vstack(values)
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=170)
    bottom = np.zeros(values_arr.shape[0], dtype=np.float64)
    for idx, region in enumerate(["background", "edema", "necrosis", "enhancing"]):
        ax.bar(np.arange(values_arr.shape[0]), values_arr[:, idx], bottom=bottom, label=REGION_LABEL[region], color=color_map[region])
        bottom += values_arr[:, idx]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("响应质量占比")
    ax.set_title("up4 内部来源的响应落点组成")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_json(payload: Dict, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.num_cases = min(int(args.num_cases), 8)
        args.batch_size = 2
        args.bootstrap = min(int(args.bootstrap), 100)
    ensure_dir(args.output_root)
    specs = parse_model_specs(args.model_spec)
    device = resolve_device(args.device)
    _, selected_indices, selected_names, loader = build_eval_loader(
        data_dir=args.data_dir,
        split=args.eval_split,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        num_cases=int(args.num_cases),
        seed=int(args.seed),
        foreground_only=False,
        case_set_csv=args.case_set_csv,
    )
    case_frames = [analyze_model(spec, loader, args, device=device) for spec in specs]
    case_df = pd.concat(case_frames, ignore_index=True)
    summary_df = summarize(case_df, bootstrap=int(args.bootstrap), seed=int(args.seed))
    wide = wide_case_table(case_df)
    delta_df = source_delta_table(wide)
    delta_summary = summarize_delta(delta_df, bootstrap=int(args.bootstrap), seed=int(args.seed)) if not delta_df.empty else pd.DataFrame()
    key = key_readout(summary_df, delta_summary)

    case_df.to_csv(os.path.join(args.output_root, "table_up_source_case_level.csv"), index=False)
    summary_df.to_csv(os.path.join(args.output_root, "table_up_source_summary.csv"), index=False)
    wide.to_csv(os.path.join(args.output_root, "table_up_source_wide_case_level.csv"), index=False)
    delta_df.to_csv(os.path.join(args.output_root, "table_up_source_delta_case_level.csv"), index=False)
    delta_summary.to_csv(os.path.join(args.output_root, "table_up_source_delta_summary.csv"), index=False)
    key.to_csv(os.path.join(args.output_root, "table_up_source_key_readout.csv"), index=False)

    plot_tumor_mass(summary_df, os.path.join(args.output_root, "plot_up_source_tumor_mass_by_role.png"))
    plot_source_filtering(delta_summary, os.path.join(args.output_root, "plot_up_source_filtering_effect.png"))
    plot_up4_source_bar(summary_df, os.path.join(args.output_root, "plot_up4_source_response_composition.png"))

    summary = {
        "purpose": "E11 up-node source decomposition for Stage-1 structural source explanation",
        "num_cases": int(len(selected_names)),
        "models": [spec.name for spec in specs],
        "nodes": list(UP_NODES),
        "source_roles": ["main_path", "skip_input", "concat_input", "conv_output"],
        "selected_indices": selected_indices,
        "outputs": [
            "table_up_source_case_level.csv",
            "table_up_source_summary.csv",
            "table_up_source_wide_case_level.csv",
            "table_up_source_delta_case_level.csv",
            "table_up_source_delta_summary.csv",
            "table_up_source_key_readout.csv",
            "plot_up_source_tumor_mass_by_role.png",
            "plot_up_source_filtering_effect.png",
            "plot_up4_source_response_composition.png",
        ],
    }
    write_json(summary, os.path.join(args.output_root, "summary_up_source_decomposition.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
