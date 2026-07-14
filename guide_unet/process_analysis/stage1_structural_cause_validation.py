from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

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
    bootstrap_mean_ci,
    build_eval_loader,
    compute_prediction_metrics,
    ensure_dir,
    feature_response_from_activation,
    load_model,
    parse_csv_list,
    parse_model_specs,
    resize_map,
    resolve_device,
    tumor_prob_map,
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
DEFAULT_OUTPUT_ROOT = "stage1_explanation_suite_20260517/results/e12_structural_cause_validation_formal_n512"

UP_NODES = ("up1", "up2", "up3", "up4")
REGIONS = (
    ("background", 0),
    ("edema", 2),
    ("necrosis", 1),
    ("enhancing", 3),
)
REGION_ORDER = tuple(item[0] for item in REGIONS)
ROLE_ORDER = ("main_path", "skip_input", "concat_input", "conv_output")
MODEL_LABEL = {"baseline": "baseline U-Net", "noskip_unet": "no-skip U-Net"}
ROLE_LABEL = {
    "main_path": "main path",
    "skip_input": "skip input",
    "concat_input": "concat input",
    "conv_output": "conv output",
}
PERTURBATION_LABEL = {
    "skip_zero": "zero skip branch",
    "main_zero": "zero main path",
    "skip_misalign": "misalign skip branch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E12 structural-cause validation for Stage-1 phenomenon explanation."
    )
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
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--branch-perturbations", type=str, default="skip_zero,main_zero,skip_misalign")
    parser.add_argument("--frequency-threshold", type=float, default=0.30)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def configure_font() -> None:
    candidates = [
        r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for path in candidates:
        if os.path.isfile(path):
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=path).get_name()]
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
            payload: Dict[str, torch.Tensor] = {"main_path": main.detach()}
            if self.model_name == "baseline" and x_skip is not None:
                payload["skip_input"] = x_skip.detach()
                payload["concat_input"] = torch.cat([x_skip, main], dim=1).detach()
            self.sources[node_name] = payload

        return hook_fn

    def _make_forward_hook(self, node_name: str):
        def hook_fn(module, inputs, output):
            if node_name not in self.sources:
                self.sources[node_name] = {}
            self.sources[node_name]["conv_output"] = output.detach()

        return hook_fn

    def clear(self) -> None:
        self.sources = {}

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class BranchPerturbationHook:
    def __init__(self, perturbation: str):
        self.perturbation = str(perturbation)
        if self.perturbation == "skip_zero":
            self.branch = "skip"
            self.mode = "zero"
        elif self.perturbation == "main_zero":
            self.branch = "main"
            self.mode = "zero"
        elif self.perturbation == "skip_misalign":
            self.branch = "skip"
            self.mode = "misalign"
        else:
            raise ValueError("Unsupported branch perturbation: {}".format(perturbation))

    def _perturb(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.mode == "zero":
            return torch.zeros_like(tensor)
        if self.mode == "misalign":
            shift_y = max(1, int(tensor.shape[-2]) // 2)
            shift_x = max(1, int(tensor.shape[-1]) // 2)
            return torch.roll(tensor, shifts=(shift_y, shift_x), dims=(-2, -1))
        raise ValueError(self.mode)

    def __call__(self, module, inputs):
        if len(inputs) < 2:
            return inputs
        x_deep, x_skip = inputs[0], inputs[1]
        if self.branch == "main":
            return (self._perturb(x_deep), x_skip)
        if self.branch == "skip":
            return (x_deep, self._perturb(x_skip))
        return inputs


def binary_morph(mask: np.ndarray, radius: int, op: str) -> np.ndarray:
    mask_bool = np.asarray(mask).astype(bool)
    if int(radius) <= 0:
        return mask_bool.copy()
    pad = int(radius)
    padded = np.pad(mask_bool, pad_width=pad, mode="constant", constant_values=(op == "erosion"))
    if op == "dilation":
        out = np.zeros_like(mask_bool, dtype=bool)
        reducer = np.logical_or
    elif op == "erosion":
        out = np.ones_like(mask_bool, dtype=bool)
        reducer = np.logical_and
    else:
        raise ValueError(op)
    h, w = mask_bool.shape
    for dy in range(2 * pad + 1):
        for dx in range(2 * pad + 1):
            window = padded[dy : dy + h, dx : dx + w]
            out = reducer(out, window)
    return out


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    mask_bool = np.asarray(mask).astype(bool)
    if not np.any(mask_bool):
        return np.zeros_like(mask_bool, dtype=bool)
    dilated = binary_morph(mask_bool, radius=radius, op="dilation")
    eroded = binary_morph(mask_bool, radius=radius, op="erosion")
    return np.logical_xor(dilated, eroded)


def region_masks(label: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, class_id in REGIONS:
        out[name] = np.asarray(label) == int(class_id)
    return out


def safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = np.asarray(mask).astype(bool)
    if not np.any(mask_bool):
        return np.nan
    return float(np.asarray(values, dtype=np.float64)[mask_bool].mean())


def mass_fraction(values: np.ndarray, mask: np.ndarray) -> float:
    arr = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(arr.sum())
    if total <= 1e-12:
        return np.nan
    return float(arr[np.asarray(mask).astype(bool)].sum() / total)


def high_frequency_ratio(response: np.ndarray, threshold: float) -> float:
    arr = np.asarray(response, dtype=np.float64)
    arr = arr - float(np.mean(arr))
    power = np.abs(np.fft.fftshift(np.fft.fft2(arr))) ** 2
    total = float(power.sum())
    if total <= 1e-12:
        return 0.0
    yy, xx = np.indices(power.shape)
    cy = (power.shape[0] - 1) / 2.0
    cx = (power.shape[1] - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    cutoff = float(threshold) * float(radius.max())
    return float(power[radius >= cutoff].sum() / total)


def edge_map(response: np.ndarray) -> np.ndarray:
    arr = np.asarray(response, dtype=np.float64)
    gy, gx = np.gradient(arr)
    return np.sqrt(gx**2 + gy**2)


def response_from_source(tensor: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    response = feature_response_from_activation(tensor, aggregation="l2")
    return resize_map(response, output_size=output_size, mode="bilinear")


def tensor_spatial_stats(tensor: torch.Tensor) -> Dict[str, np.ndarray]:
    x = tensor.detach().float()
    abs_mean = torch.mean(torch.abs(x), dim=(1, 2, 3))
    dy = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]), dim=(1, 2, 3))
    dx = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]), dim=(1, 2, 3))
    grad_mean = 0.5 * (dx + dy)
    grad_ratio = grad_mean / (abs_mean + 1e-8)
    return {
        "tensor_abs_mean": abs_mean.detach().cpu().numpy().astype(np.float64),
        "tensor_spatial_grad_mean": grad_mean.detach().cpu().numpy().astype(np.float64),
        "tensor_spatial_gradient_ratio": grad_ratio.detach().cpu().numpy().astype(np.float64),
    }


def source_case_rows(
    model_name: str,
    batch: Dict,
    logits: torch.Tensor,
    store: UpSourceStore,
    frequency_threshold: float,
    boundary_radius: int,
) -> List[Dict]:
    labels_np = batch["label"].detach().cpu().numpy()
    case_names = list(batch["case_name"])
    output_size = tuple(int(v) for v in labels_np.shape[-2:])
    final_prob_np = tumor_prob_map(logits).detach().cpu().numpy()
    rows: List[Dict] = []

    for node_name in UP_NODES:
        node_payload = store.sources.get(node_name, {})
        for source_role, source_tensor in node_payload.items():
            tensor_stats = tensor_spatial_stats(source_tensor)
            response_tensor = response_from_source(source_tensor, output_size=output_size)
            response_np = response_tensor.detach().cpu().numpy()
            for local_idx, case_name in enumerate(case_names):
                label = labels_np[local_idx]
                response = np.maximum(response_np[local_idx].astype(np.float64), 0.0)
                final_prob = final_prob_np[local_idx].astype(np.float64)
                tumor_mask = label > 0
                boundary = boundary_band(tumor_mask, radius=int(boundary_radius))
                edge = edge_map(response)
                edge_total = float(np.maximum(edge, 0.0).sum())
                boundary_edge_mass = (
                    float(edge[boundary].sum() / edge_total) if edge_total > 1e-12 and np.any(boundary) else np.nan
                )
                boundary_mean = safe_mean(edge, boundary)
                non_boundary_mean = safe_mean(edge, ~boundary)
                masks = region_masks(label)
                row = {
                    "model": model_name,
                    "case_name": str(case_name),
                    "patient_id": str(case_name).rsplit("_", 1)[0],
                    "node": node_name,
                    "node_order": int(UP_NODES.index(node_name)),
                    "source_role": source_role,
                    "role_order": int(ROLE_ORDER.index(source_role)) if source_role in ROLE_ORDER else 99,
                    "final_prob_pearson": float(safe_corr(response, final_prob)),
                    "final_prob_cosine": float(safe_cos(response, final_prob)),
                    "high_freq_ratio": high_frequency_ratio(response, threshold=float(frequency_threshold)),
                    "edge_mean": float(np.mean(edge)),
                    "edge_std": float(np.std(edge)),
                    "tensor_abs_mean": float(tensor_stats["tensor_abs_mean"][local_idx]),
                    "tensor_spatial_grad_mean": float(tensor_stats["tensor_spatial_grad_mean"][local_idx]),
                    "tensor_spatial_gradient_ratio": float(
                        tensor_stats["tensor_spatial_gradient_ratio"][local_idx]
                    ),
                    "boundary_area_fraction": float(boundary.mean()),
                    "boundary_edge_mass_fraction": boundary_edge_mass,
                    "boundary_to_nonboundary_edge_ratio": float(boundary_mean / (non_boundary_mean + 1e-8))
                    if np.isfinite(boundary_mean) and np.isfinite(non_boundary_mean)
                    else np.nan,
                    "response_mass_fraction_tumor_total": mass_fraction(response, tumor_mask),
                    "response_mean_tumor": safe_mean(response, tumor_mask),
                    "response_mean_background": safe_mean(response, masks["background"]),
                }
                for region_name in REGION_ORDER:
                    row["response_mass_fraction_{}".format(region_name)] = mass_fraction(response, masks[region_name])
                    row["response_mean_{}".format(region_name)] = safe_mean(response, masks[region_name])
                rows.append(row)
    return rows


def analyze_sources(spec, loader, args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    model = load_model(model_name=spec.name, ckpt_path=spec.ckpt_path, device=device)
    store = UpSourceStore(model=model, model_name=spec.name, node_names=UP_NODES)
    rows: List[Dict] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="E12 source {}".format(spec.name)):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            store.clear()
            logits = unpack_logits(model(images))
            batch = dict(batch)
            batch["label"] = labels
            rows.extend(
                source_case_rows(
                    spec.name,
                    batch,
                    logits,
                    store,
                    frequency_threshold=float(args.frequency_threshold),
                    boundary_radius=int(args.boundary_radius),
                )
            )
    store.remove()
    return pd.DataFrame(rows)


def analyze_branch_perturbation(
    baseline_spec,
    loader,
    perturbations: Sequence[str],
    device: torch.device,
) -> pd.DataFrame:
    model = load_model(model_name=baseline_spec.name, ckpt_path=baseline_spec.ckpt_path, device=device)
    rows: List[Dict] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="E12 branch perturb baseline"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            case_names = list(batch["case_name"])
            base_logits = unpack_logits(model(images))
            base_pred = torch.argmax(base_logits.detach(), dim=1)
            base_metrics = compute_prediction_metrics(base_logits, labels)
            for node_name in UP_NODES:
                for perturbation in perturbations:
                    hook = BranchPerturbationHook(perturbation=perturbation)
                    handle = getattr(model, node_name).register_forward_pre_hook(hook)
                    pert_logits = unpack_logits(model(images))
                    handle.remove()
                    pert_metrics = compute_prediction_metrics(pert_logits, labels, reference_pred=base_pred)
                    for local_idx, case_name in enumerate(case_names):
                        base = base_metrics[local_idx]
                        pert = pert_metrics[local_idx]
                        rows.append(
                            {
                                "model": "baseline",
                                "case_name": str(case_name),
                                "patient_id": str(case_name).rsplit("_", 1)[0],
                                "node": node_name,
                                "node_order": int(UP_NODES.index(node_name)),
                                "perturbation": perturbation,
                                "branch": "skip" if perturbation.startswith("skip") else "main",
                                "mode": "zero" if perturbation.endswith("zero") else "misalign",
                                "baseline_final_dice": float(base["mean_dice"]),
                                "perturbed_final_dice": float(pert["mean_dice"]),
                                "final_dice_drop": float(base["mean_dice"] - pert["mean_dice"]),
                                "baseline_gt_fg_prob": float(base["gt_fg_prob"]),
                                "perturbed_gt_fg_prob": float(pert["gt_fg_prob"]),
                                "gt_fg_prob_drop": float(base["gt_fg_prob"] - pert["gt_fg_prob"]),
                                "baseline_final_fg_prob": float(base["pred_fg_prob"]),
                                "perturbed_final_fg_prob": float(pert["pred_fg_prob"]),
                                "final_fg_prob_drop": float(base["pred_fg_prob"] - pert["pred_fg_prob"]),
                                "baseline_whole_tumor_prob_mean": float(base["whole_tumor_prob_mean"]),
                                "perturbed_whole_tumor_prob_mean": float(pert["whole_tumor_prob_mean"]),
                                "whole_tumor_prob_mean_drop": float(
                                    base["whole_tumor_prob_mean"] - pert["whole_tumor_prob_mean"]
                                ),
                                "gt_fg_pixels": int(base["gt_fg_pixels"]),
                                "pred_fg_pixels": int(base["pred_fg_pixels"]),
                            }
                        )
    return pd.DataFrame(rows)


def summarize(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metrics: Sequence[str],
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict] = []
    if df.empty:
        return pd.DataFrame()
    for idx, (key, group) in enumerate(df.groupby(list(group_cols), sort=False)):
        if not isinstance(key, tuple):
            key = (key,)
        payload = dict(zip(group_cols, key))
        payload["n_cases"] = int(group["case_name"].nunique()) if "case_name" in group.columns else int(len(group))
        payload["n_rows"] = int(len(group))
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = group[metric].to_numpy(dtype=np.float64)
            payload[metric] = float(np.nanmean(values))
            ci_low, ci_high = bootstrap_mean_ci(values, n_boot=int(bootstrap), seed=int(seed) + idx)
            payload["{}_ci_low".format(metric)] = ci_low
            payload["{}_ci_high".format(metric)] = ci_high
        rows.append(payload)
    return pd.DataFrame(rows)


def fusion_delta_table(source_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "response_mass_fraction_background",
        "response_mass_fraction_edema",
        "response_mass_fraction_necrosis",
        "response_mass_fraction_enhancing",
        "response_mass_fraction_tumor_total",
        "final_prob_pearson",
        "final_prob_cosine",
        "high_freq_ratio",
        "edge_mean",
        "tensor_abs_mean",
        "tensor_spatial_grad_mean",
        "tensor_spatial_gradient_ratio",
        "boundary_edge_mass_fraction",
        "boundary_to_nonboundary_edge_ratio",
    ]
    rows: List[Dict] = []
    for (model_name, case_name, node_name), group in source_df.groupby(["model", "case_name", "node"], sort=False):
        pivot = group.set_index("source_role")
        comparisons = []
        if model_name == "baseline":
            comparisons = [
                ("concat_minus_main", "concat_input", "main_path"),
                ("concat_minus_skip", "concat_input", "skip_input"),
                ("output_minus_concat", "conv_output", "concat_input"),
                ("output_minus_main", "conv_output", "main_path"),
                ("output_minus_skip", "conv_output", "skip_input"),
            ]
        elif model_name == "noskip_unet":
            comparisons = [("output_minus_main", "conv_output", "main_path")]
        for comparison, dst, src in comparisons:
            if dst not in pivot.index or src not in pivot.index:
                continue
            row = {
                "model": model_name,
                "case_name": case_name,
                "patient_id": str(case_name).rsplit("_", 1)[0],
                "node": node_name,
                "node_order": int(UP_NODES.index(node_name)),
                "comparison": comparison,
            }
            for metric in metrics:
                if metric in pivot.columns:
                    row["{}_delta".format(metric)] = float(pivot.loc[dst, metric] - pivot.loc[src, metric])
            rows.append(row)
    return pd.DataFrame(rows)


def key_readout(branch_summary: pd.DataFrame, source_summary: pd.DataFrame, fusion_summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    if not branch_summary.empty:
        for _, row in branch_summary.iterrows():
            if row.get("node") == "up4" or row.get("perturbation") in ("skip_zero", "skip_misalign"):
                rows.append(
                    {
                        "evidence": "branch_perturbation",
                        "model": row.get("model", "baseline"),
                        "node": row.get("node", ""),
                        "item": row.get("perturbation", ""),
                        "metric": "final_fg_prob_drop",
                        "value": row.get("final_fg_prob_drop", np.nan),
                        "ci95_low": row.get("final_fg_prob_drop_ci_low", np.nan),
                        "ci95_high": row.get("final_fg_prob_drop_ci_high", np.nan),
                    }
                )
    if not source_summary.empty:
        for _, row in source_summary.iterrows():
            if row.get("model") == "baseline" and row.get("node") == "up4" and row.get("source_role") in (
                "main_path",
                "skip_input",
                "concat_input",
                "conv_output",
            ):
                for metric in (
                    "tensor_spatial_gradient_ratio",
                    "high_freq_ratio",
                    "boundary_to_nonboundary_edge_ratio",
                    "response_mass_fraction_tumor_total",
                ):
                    rows.append(
                        {
                            "evidence": "source_property",
                            "model": row.get("model", ""),
                            "node": row.get("node", ""),
                            "item": row.get("source_role", ""),
                            "metric": metric,
                            "value": row.get(metric, np.nan),
                            "ci95_low": row.get("{}_ci_low".format(metric), np.nan),
                            "ci95_high": row.get("{}_ci_high".format(metric), np.nan),
                        }
                    )
    if not fusion_summary.empty:
        sub = fusion_summary[
            (fusion_summary["model"] == "baseline")
            & (fusion_summary["node"] == "up4")
            & (fusion_summary["comparison"] == "output_minus_concat")
        ]
        for _, row in sub.iterrows():
            for metric in (
                "response_mass_fraction_background_delta",
                "response_mass_fraction_tumor_total_delta",
                "final_prob_pearson_delta",
                "boundary_to_nonboundary_edge_ratio_delta",
            ):
                rows.append(
                    {
                        "evidence": "fusion_prepost_delta",
                        "model": row.get("model", ""),
                        "node": row.get("node", ""),
                        "item": row.get("comparison", ""),
                        "metric": metric,
                        "value": row.get(metric, np.nan),
                        "ci95_low": row.get("{}_ci_low".format(metric), np.nan),
                        "ci95_high": row.get("{}_ci_high".format(metric), np.nan),
                    }
                )
    return pd.DataFrame(rows)


def plot_branch(branch_summary: pd.DataFrame, out_path: str) -> None:
    if branch_summary.empty:
        return
    configure_font()
    sub = branch_summary.copy().sort_values(["perturbation", "node_order"])
    fig, ax = plt.subplots(figsize=(9.8, 4.8), dpi=170)
    for perturbation, group in sub.groupby("perturbation", sort=False):
        group = group.sort_values("node_order")
        ax.plot(
            group["node"],
            group["final_fg_prob_drop"],
            marker="o",
            linewidth=2.0,
            label=PERTURBATION_LABEL.get(perturbation, perturbation),
        )
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.9)
    ax.set_title("Baseline U-Net: output impact of branch-level perturbations")
    ax.set_xlabel("up node")
    ax.set_ylabel("final foreground probability drop")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_frequency(source_summary: pd.DataFrame, out_path: str) -> None:
    if source_summary.empty:
        return
    configure_font()
    sub = source_summary[source_summary["model"] == "baseline"].copy()
    roles = ["main_path", "skip_input", "concat_input", "conv_output"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), dpi=170)
    for role in roles:
        group = sub[sub["source_role"] == role].sort_values("node_order")
        if group.empty:
            continue
        axes[0].plot(
            group["node"],
            group["tensor_spatial_gradient_ratio"],
            marker="o",
            linewidth=2.0,
            label=ROLE_LABEL[role],
        )
        axes[1].plot(
            group["node"],
            group["boundary_to_nonboundary_edge_ratio"],
            marker="o",
            linewidth=2.0,
            label=ROLE_LABEL[role],
        )
    axes[0].set_title("Frequency-proxy evidence")
    axes[0].set_ylabel("tensor spatial-gradient ratio")
    axes[1].set_title("Boundary evidence")
    axes[1].set_ylabel("tumor-boundary / non-boundary edge ratio")
    for ax in axes:
        ax.set_xlabel("up node")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Baseline U-Net: frequency and boundary properties inside up nodes")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_fusion_delta(fusion_summary: pd.DataFrame, out_path: str) -> None:
    if fusion_summary.empty:
        return
    configure_font()
    sub = fusion_summary[
        (fusion_summary["model"] == "baseline") & (fusion_summary["comparison"] == "output_minus_concat")
    ].copy()
    if sub.empty:
        return
    metrics = {
        "response_mass_fraction_background_delta": "background response",
        "response_mass_fraction_tumor_total_delta": "tumor response",
        "final_prob_pearson_delta": "final-probability correlation",
        "boundary_to_nonboundary_edge_ratio_delta": "boundary relative edge strength",
        "tensor_spatial_gradient_ratio_delta": "tensor spatial-gradient ratio",
    }
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=170)
    for metric, label in metrics.items():
        if metric not in sub.columns:
            continue
        group = sub.sort_values("node_order")
        ax.plot(group["node"], group[metric], marker="o", linewidth=2.0, label=label)
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.9)
    ax.set_title("Baseline U-Net: response-property change after fusion convolution")
    ax.set_xlabel("up node")
    ax.set_ylabel("conv output - concat input")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.num_cases = min(int(args.num_cases), 8)
        args.batch_size = min(int(args.batch_size), 2)
        args.num_workers = 0
        args.bootstrap = min(int(args.bootstrap), 100)

    ensure_dir(args.output_root)
    specs = parse_model_specs(args.model_spec)
    perturbations = parse_csv_list(args.branch_perturbations)
    invalid = sorted(set(perturbations) - {"skip_zero", "main_zero", "skip_misalign"})
    if invalid:
        raise ValueError("Unsupported branch perturbations: {}".format(invalid))
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

    source_frames = [analyze_sources(spec, loader, args=args, device=device) for spec in specs]
    source_df = pd.concat(source_frames, ignore_index=True)
    baseline_specs = [spec for spec in specs if spec.name == "baseline"]
    if not baseline_specs:
        raise ValueError("E12 branch perturbation requires a baseline model spec")
    branch_df = analyze_branch_perturbation(
        baseline_specs[0],
        loader=loader,
        perturbations=perturbations,
        device=device,
    )
    fusion_df = fusion_delta_table(source_df)

    source_metrics = [
        "final_prob_pearson",
        "final_prob_cosine",
        "high_freq_ratio",
        "edge_mean",
        "tensor_abs_mean",
        "tensor_spatial_grad_mean",
        "tensor_spatial_gradient_ratio",
        "boundary_edge_mass_fraction",
        "boundary_to_nonboundary_edge_ratio",
        "response_mass_fraction_background",
        "response_mass_fraction_edema",
        "response_mass_fraction_necrosis",
        "response_mass_fraction_enhancing",
        "response_mass_fraction_tumor_total",
        "response_mean_tumor",
        "response_mean_background",
    ]
    branch_metrics = [
        "final_dice_drop",
        "gt_fg_prob_drop",
        "final_fg_prob_drop",
        "whole_tumor_prob_mean_drop",
    ]
    fusion_metrics = [col for col in fusion_df.columns if col.endswith("_delta")]
    source_summary = summarize(
        source_df,
        group_cols=["model", "node", "node_order", "source_role", "role_order"],
        metrics=source_metrics,
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    ).sort_values(["model", "node_order", "role_order"])
    branch_summary = summarize(
        branch_df,
        group_cols=["model", "node", "node_order", "perturbation", "branch", "mode"],
        metrics=branch_metrics,
        bootstrap=int(args.bootstrap),
        seed=int(args.seed) + 100,
    ).sort_values(["node_order", "perturbation"])
    fusion_summary = summarize(
        fusion_df,
        group_cols=["model", "node", "node_order", "comparison"],
        metrics=fusion_metrics,
        bootstrap=int(args.bootstrap),
        seed=int(args.seed) + 200,
    ).sort_values(["model", "node_order", "comparison"])
    key = key_readout(branch_summary, source_summary, fusion_summary)

    outputs = {
        "table_e12_source_frequency_edge_case_level.csv": source_df,
        "table_e12_source_frequency_edge_summary.csv": source_summary,
        "table_e12_branch_perturbation_case_level.csv": branch_df,
        "table_e12_branch_perturbation_summary.csv": branch_summary,
        "table_e12_fusion_prepost_delta_case_level.csv": fusion_df,
        "table_e12_fusion_prepost_delta_summary.csv": fusion_summary,
        "table_e12_key_structural_readout.csv": key,
    }
    for name, frame in outputs.items():
        frame.to_csv(os.path.join(args.output_root, name), index=False)
    plot_branch(branch_summary, os.path.join(args.output_root, "plot_e12_branch_perturbation_drop.png"))
    plot_frequency(source_summary, os.path.join(args.output_root, "plot_e12_frequency_edge_by_source.png"))
    plot_fusion_delta(fusion_summary, os.path.join(args.output_root, "plot_e12_fusion_prepost_delta.png"))

    write_run_config(
        args,
        args.output_root,
        extra={"device": str(device), "selected_indices": [int(v) for v in selected_indices]},
    )
    manifest_rows = [
        {"output": name, "path": os.path.join(args.output_root, name)}
        for name in list(outputs.keys())
        + [
            "plot_e12_branch_perturbation_drop.png",
            "plot_e12_frequency_edge_by_source.png",
            "plot_e12_fusion_prepost_delta.png",
            "summary_structural_cause_validation.json",
        ]
    ]
    write_manifest(args.output_root, manifest_rows)
    summary = {
        "purpose": "E12 structural-cause validation: branch perturbation, frequency/edge evidence, and fusion pre/post filtering.",
        "num_cases": int(len(selected_names)),
        "models": [spec.name for spec in specs],
        "nodes": list(UP_NODES),
        "branch_perturbations": list(perturbations),
        "frequency_threshold": float(args.frequency_threshold),
        "boundary_radius": int(args.boundary_radius),
        "outputs": [row["output"] for row in manifest_rows],
        "guardrail": "E12 strengthens structural-source explanation for the observed Stage-1 curve. It does not claim a universal skip-connection law.",
    }
    write_json(summary, os.path.join(args.output_root, "summary_structural_cause_validation.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
