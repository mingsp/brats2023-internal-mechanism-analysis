from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
DEFAULT_CASE_SET = "stage1_explanation_suite_20260517/results/common_case_sets/formal_n512_cases.csv"
DEFAULT_OUTPUT_ROOT = "stage1_explanation_suite_20260517/results/e13_up4_skip_counterfactual_mediation_formal_n512"

REGIONS = (
    ("background", 0),
    ("edema", 2),
    ("necrosis", 1),
    ("enhancing", 3),
)
VARIANT_ORDER = (
    "boundary_cross",
    "boundary_shift",
    "boundary_background_control",
    "boundary_random_control",
    "interior_cross",
    "interior_background_control",
    "interior_random_control",
)
VARIANT_LABEL = {
    "boundary_cross": "boundary cross-case",
    "boundary_shift": "boundary spatial shift",
    "boundary_background_control": "boundary-area background control",
    "boundary_random_control": "boundary-area random control",
    "interior_cross": "interior cross-case",
    "interior_background_control": "interior-area background control",
    "interior_random_control": "interior-area random control",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E13 up4 skip counterfactual region replacement and mediation-style analysis."
    )
    parser.add_argument("--baseline-ckpt", type=str, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--eval-split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--case-set-csv", type=str, default=DEFAULT_CASE_SET)
    parser.add_argument("--num-cases", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


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
            out = reducer(out, padded[dy : dy + h, dx : dx + w])
    return out


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    mask_bool = np.asarray(mask).astype(bool)
    if not np.any(mask_bool):
        return np.zeros_like(mask_bool, dtype=bool)
    dilated = binary_morph(mask_bool, radius=radius, op="dilation")
    eroded = binary_morph(mask_bool, radius=radius, op="erosion")
    return np.logical_xor(dilated, eroded)


def region_masks(label: np.ndarray) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    for name, class_id in REGIONS:
        masks[name] = np.asarray(label) == int(class_id)
    return masks


def sample_same_area(mask_pool: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    pool = np.flatnonzero(np.asarray(mask_pool).reshape(-1).astype(bool))
    out = np.zeros(np.asarray(mask_pool).size, dtype=bool)
    if int(k) <= 0 or pool.size == 0:
        return out.reshape(np.asarray(mask_pool).shape)
    take = min(int(k), int(pool.size))
    chosen = rng.choice(pool, size=take, replace=False)
    out[chosen] = True
    return out.reshape(np.asarray(mask_pool).shape)


def build_variant_masks(labels_np: np.ndarray, boundary_radius: int, seed: int) -> Dict[str, np.ndarray]:
    masks = {name: [] for name in VARIANT_ORDER}
    for idx, label in enumerate(labels_np):
        rng = np.random.default_rng(int(seed) + idx * 9973)
        tumor = np.asarray(label) > 0
        boundary = boundary_band(tumor, radius=int(boundary_radius))
        interior = binary_morph(tumor, radius=int(boundary_radius), op="erosion")
        if not np.any(interior):
            interior = tumor.copy()
        background = np.asarray(label) == 0
        all_pixels = np.ones_like(tumor, dtype=bool)
        boundary_k = int(boundary.sum())
        interior_k = int(interior.sum())
        masks["boundary_cross"].append(boundary)
        masks["boundary_shift"].append(boundary)
        masks["boundary_background_control"].append(sample_same_area(background, boundary_k, rng))
        masks["boundary_random_control"].append(sample_same_area(all_pixels, boundary_k, rng))
        masks["interior_cross"].append(interior)
        masks["interior_background_control"].append(sample_same_area(background, interior_k, rng))
        masks["interior_random_control"].append(sample_same_area(all_pixels, interior_k, rng))
    return {name: np.stack(items, axis=0).astype(np.float32) for name, items in masks.items()}


class Up4OutputStore:
    def __init__(self, model: torch.nn.Module):
        self.output: torch.Tensor | None = None
        self.handle = model.up4.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.output = output.detach()

    def clear(self) -> None:
        self.output = None

    def remove(self) -> None:
        self.handle.remove()


class Up4SkipCounterfactualHook:
    def __init__(self, mask: torch.Tensor, variant: str):
        if mask.ndim != 3:
            raise ValueError("mask must have shape [B,H,W]")
        self.mask = mask.detach()
        self.variant = str(variant)
        self.last_mask_area_fraction: torch.Tensor | None = None

    def __call__(self, module, inputs):
        x_deep, x_skip = inputs[0], inputs[1]
        mask = self.mask.to(device=x_skip.device, dtype=x_skip.dtype).unsqueeze(1)
        if tuple(mask.shape[-2:]) != tuple(x_skip.shape[-2:]):
            mask = F.interpolate(mask, size=tuple(x_skip.shape[-2:]), mode="nearest")
        self.last_mask_area_fraction = mask.flatten(1).mean(dim=1).detach().cpu()
        if self.variant.endswith("_shift"):
            source = torch.roll(
                x_skip,
                shifts=(max(1, int(x_skip.shape[-2]) // 2), max(1, int(x_skip.shape[-1]) // 2)),
                dims=(-2, -1),
            )
        else:
            if int(x_skip.shape[0]) > 1:
                perm = torch.roll(torch.arange(int(x_skip.shape[0]), device=x_skip.device), shifts=1, dims=0)
                source = x_skip[perm]
            else:
                source = torch.roll(
                    x_skip,
                    shifts=(max(1, int(x_skip.shape[-2]) // 2), max(1, int(x_skip.shape[-1]) // 2)),
                    dims=(-2, -1),
                )
        x_skip_cf = x_skip * (1.0 - mask) + source * mask
        return (x_deep, x_skip_cf)


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


def edge_map(response: np.ndarray) -> np.ndarray:
    arr = np.asarray(response, dtype=np.float64)
    gy, gx = np.gradient(arr)
    return np.sqrt(gx**2 + gy**2)


def response_map(up4_output: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    response = feature_response_from_activation(up4_output, aggregation="l2")
    return resize_map(response, output_size=output_size, mode="bilinear")


def response_metrics(
    up4_output: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    boundary_radius: int,
) -> List[Dict[str, float]]:
    labels_np = labels.detach().cpu().numpy()
    output_size = tuple(int(v) for v in labels_np.shape[-2:])
    response_np = response_map(up4_output, output_size=output_size).detach().cpu().numpy()
    final_prob_np = tumor_prob_map(logits).detach().cpu().numpy()
    rows: List[Dict[str, float]] = []
    for idx, label in enumerate(labels_np):
        response = np.maximum(response_np[idx].astype(np.float64), 0.0)
        final_prob = final_prob_np[idx].astype(np.float64)
        tumor = label > 0
        background = label == 0
        boundary = boundary_band(tumor, radius=int(boundary_radius))
        edge = edge_map(response)
        boundary_edge_mean = safe_mean(edge, boundary)
        non_boundary_edge_mean = safe_mean(edge, ~boundary)
        row = {
            "response_mass_fraction_background": mass_fraction(response, background),
            "response_mass_fraction_tumor_total": mass_fraction(response, tumor),
            "response_mean_background": safe_mean(response, background),
            "response_mean_tumor": safe_mean(response, tumor),
            "final_prob_pearson": float(safe_corr(response, final_prob)),
            "final_prob_cosine": float(safe_cos(response, final_prob)),
            "boundary_to_nonboundary_edge_ratio": float(boundary_edge_mean / (non_boundary_edge_mean + 1e-8))
            if np.isfinite(boundary_edge_mean) and np.isfinite(non_boundary_edge_mean)
            else np.nan,
        }
        masks = region_masks(label)
        for region_name in ("edema", "necrosis", "enhancing"):
            row["response_mass_fraction_{}".format(region_name)] = mass_fraction(response, masks[region_name])
            row["response_mean_{}".format(region_name)] = safe_mean(response, masks[region_name])
        rows.append(row)
    return rows


def analyze(model: torch.nn.Module, loader, args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    rows: List[Dict] = []
    store = Up4OutputStore(model)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="E13 up4 skip counterfactual")):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            case_names = list(batch["case_name"])
            labels_np = labels.detach().cpu().numpy()
            variant_masks = build_variant_masks(
                labels_np,
                boundary_radius=int(args.boundary_radius),
                seed=int(args.seed) + batch_idx * 100000,
            )

            store.clear()
            base_logits = unpack_logits(model(images))
            if store.output is None:
                raise RuntimeError("Failed to capture baseline up4 output")
            base_up4 = store.output.detach()
            base_pred = torch.argmax(base_logits.detach(), dim=1)
            base_metrics = compute_prediction_metrics(base_logits, labels)
            base_response = response_metrics(base_up4, base_logits, labels, boundary_radius=int(args.boundary_radius))

            for variant in VARIANT_ORDER:
                mask_t = torch.from_numpy(variant_masks[variant]).to(device=device)
                hook = Up4SkipCounterfactualHook(mask=mask_t, variant=variant)
                handle = model.up4.register_forward_pre_hook(hook)
                store.clear()
                pert_logits = unpack_logits(model(images))
                handle.remove()
                if store.output is None:
                    raise RuntimeError("Failed to capture perturbed up4 output")
                pert_metrics = compute_prediction_metrics(pert_logits, labels, reference_pred=base_pred)
                pert_response = response_metrics(
                    store.output.detach(),
                    pert_logits,
                    labels,
                    boundary_radius=int(args.boundary_radius),
                )
                mask_area = hook.last_mask_area_fraction.numpy() if hook.last_mask_area_fraction is not None else None

                for local_idx, case_name in enumerate(case_names):
                    base_pred_row = base_metrics[local_idx]
                    pert_pred_row = pert_metrics[local_idx]
                    base_resp_row = base_response[local_idx]
                    pert_resp_row = pert_response[local_idx]
                    row = {
                        "model": "baseline",
                        "case_name": str(case_name),
                        "patient_id": str(case_name).rsplit("_", 1)[0],
                        "variant": variant,
                        "variant_order": int(VARIANT_ORDER.index(variant)),
                        "variant_family": "boundary" if variant.startswith("boundary") else "interior",
                        "is_lesion_region": bool("cross" in variant or variant.endswith("_shift")),
                        "is_spatial_misaligned": bool(variant.endswith("_shift")),
                        "control_type": "none"
                        if variant in ("boundary_cross", "boundary_shift", "interior_cross")
                        else ("background" if "background_control" in variant else "random"),
                        "mask_area_fraction": float(mask_area[local_idx]) if mask_area is not None else np.nan,
                        "baseline_final_dice": float(base_pred_row["mean_dice"]),
                        "perturbed_final_dice": float(pert_pred_row["mean_dice"]),
                        "final_dice_drop": float(base_pred_row["mean_dice"] - pert_pred_row["mean_dice"]),
                        "baseline_final_fg_prob": float(base_pred_row["pred_fg_prob"]),
                        "perturbed_final_fg_prob": float(pert_pred_row["pred_fg_prob"]),
                        "final_fg_prob_drop": float(
                            base_pred_row["pred_fg_prob"] - pert_pred_row["pred_fg_prob"]
                        ),
                        "baseline_gt_fg_prob": float(base_pred_row["gt_fg_prob"]),
                        "perturbed_gt_fg_prob": float(pert_pred_row["gt_fg_prob"]),
                        "gt_fg_prob_drop": float(base_pred_row["gt_fg_prob"] - pert_pred_row["gt_fg_prob"]),
                        "baseline_whole_tumor_prob_mean": float(base_pred_row["whole_tumor_prob_mean"]),
                        "perturbed_whole_tumor_prob_mean": float(pert_pred_row["whole_tumor_prob_mean"]),
                        "whole_tumor_prob_mean_drop": float(
                            base_pred_row["whole_tumor_prob_mean"] - pert_pred_row["whole_tumor_prob_mean"]
                        ),
                        "gt_fg_pixels": int(base_pred_row["gt_fg_pixels"]),
                        "pred_fg_pixels": int(base_pred_row["pred_fg_pixels"]),
                    }
                    for metric, base_value in base_resp_row.items():
                        pert_value = pert_resp_row.get(metric, np.nan)
                        row["baseline_{}".format(metric)] = float(base_value)
                        row["perturbed_{}".format(metric)] = float(pert_value)
                        row["{}_drop".format(metric)] = float(base_value - pert_value)
                        row["{}_delta".format(metric)] = float(pert_value - base_value)
                    rows.append(row)
    store.remove()
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    metrics = [
        "mask_area_fraction",
        "final_dice_drop",
        "final_fg_prob_drop",
        "gt_fg_prob_drop",
        "whole_tumor_prob_mean_drop",
        "response_mass_fraction_background_delta",
        "response_mass_fraction_tumor_total_drop",
        "response_mass_fraction_tumor_total_delta",
        "final_prob_pearson_drop",
        "boundary_to_nonboundary_edge_ratio_drop",
    ]
    rows: List[Dict] = []
    for idx, (variant, group) in enumerate(df.groupby("variant", sort=False)):
        payload = {
            "variant": variant,
            "variant_order": int(group["variant_order"].iloc[0]),
            "variant_family": str(group["variant_family"].iloc[0]),
            "control_type": str(group["control_type"].iloc[0]),
            "n_cases": int(group["case_name"].nunique()),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            payload[metric] = float(np.nanmean(values))
            lo, hi = bootstrap_mean_ci(values, n_boot=int(bootstrap), seed=int(seed) + idx)
            payload["{}_ci_low".format(metric)] = lo
            payload["{}_ci_high".format(metric)] = hi
        rows.append(payload)
    return pd.DataFrame(rows).sort_values("variant_order")


def paired_contrasts(df: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    comparisons = [
        ("boundary_cross_vs_background", "boundary_cross", "boundary_background_control"),
        ("boundary_cross_vs_random", "boundary_cross", "boundary_random_control"),
        ("boundary_shift_vs_background", "boundary_shift", "boundary_background_control"),
        ("interior_cross_vs_background", "interior_cross", "interior_background_control"),
        ("interior_cross_vs_random", "interior_cross", "interior_random_control"),
    ]
    metrics = [
        "final_fg_prob_drop",
        "final_dice_drop",
        "response_mass_fraction_tumor_total_drop",
        "response_mass_fraction_background_delta",
        "final_prob_pearson_drop",
        "boundary_to_nonboundary_edge_ratio_drop",
    ]
    rows: List[Dict] = []
    for comp_idx, (name, treatment, control) in enumerate(comparisons):
        t = df[df["variant"] == treatment].copy()
        c = df[df["variant"] == control].copy()
        merged = t.merge(c, on="case_name", suffixes=("_treatment", "_control"), how="inner")
        if merged.empty:
            continue
        payload = {
            "comparison": name,
            "treatment": treatment,
            "control": control,
            "n_cases": int(merged["case_name"].nunique()),
        }
        for metric_idx, metric in enumerate(metrics):
            diff = merged["{}_treatment".format(metric)].to_numpy(dtype=np.float64) - merged[
                "{}_control".format(metric)
            ].to_numpy(dtype=np.float64)
            payload["{}_diff".format(metric)] = float(np.nanmean(diff))
            lo, hi = bootstrap_mean_ci(
                diff,
                n_boot=int(bootstrap),
                seed=int(seed) + 100 * comp_idx + metric_idx,
            )
            payload["{}_diff_ci_low".format(metric)] = lo
            payload["{}_diff_ci_high".format(metric)] = hi
        # Paired mediator-outcome association in the contrast space.
        mediator = (
            merged["response_mass_fraction_tumor_total_drop_treatment"].to_numpy(dtype=np.float64)
            - merged["response_mass_fraction_tumor_total_drop_control"].to_numpy(dtype=np.float64)
        )
        outcome = (
            merged["final_fg_prob_drop_treatment"].to_numpy(dtype=np.float64)
            - merged["final_fg_prob_drop_control"].to_numpy(dtype=np.float64)
        )
        valid = np.isfinite(mediator) & np.isfinite(outcome)
        if valid.sum() >= 3 and np.std(mediator[valid]) > 1e-12 and np.std(outcome[valid]) > 1e-12:
            payload["paired_tumor_response_drop_vs_output_drop_corr"] = float(np.corrcoef(mediator[valid], outcome[valid])[0, 1])
        else:
            payload["paired_tumor_response_drop_vs_output_drop_corr"] = np.nan
        rows.append(payload)
    return pd.DataFrame(rows)


def fit_ols(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    yv = y[valid]
    xv = x[valid]
    if yv.size < xv.shape[1] + 2:
        return np.full(x.shape[1], np.nan)
    beta, *_ = np.linalg.lstsq(xv, yv, rcond=None)
    return beta.astype(np.float64)


def mediation_once(sub: pd.DataFrame, treatment_variant: str, control_variant: str, mediator_col: str, outcome_col: str) -> Dict[str, float]:
    work = sub[sub["variant"].isin([treatment_variant, control_variant])].copy()
    work["treatment"] = (work["variant"] == treatment_variant).astype(float)
    y = work[outcome_col].to_numpy(dtype=np.float64)
    m = work[mediator_col].to_numpy(dtype=np.float64)
    t = work["treatment"].to_numpy(dtype=np.float64)
    area = work["mask_area_fraction"].to_numpy(dtype=np.float64)
    intercept = np.ones_like(t)
    beta_m = fit_ols(m, np.column_stack([intercept, t, area]))
    beta_total = fit_ols(y, np.column_stack([intercept, t, area]))
    beta_direct = fit_ols(y, np.column_stack([intercept, t, m, area]))
    a = float(beta_m[1])
    c_total = float(beta_total[1])
    c_direct = float(beta_direct[1])
    b = float(beta_direct[2])
    indirect = float(a * b)
    mediated_fraction = float(indirect / c_total) if np.isfinite(c_total) and abs(c_total) > 1e-12 else np.nan
    return {
        "a_treatment_to_mediator": a,
        "b_mediator_to_outcome": b,
        "c_total_treatment_to_outcome": c_total,
        "c_direct_treatment_to_outcome": c_direct,
        "indirect_ab": indirect,
        "mediated_fraction": mediated_fraction,
    }


def mediation_summary(df: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    comparisons = [
        ("boundary_cross_vs_background", "boundary_cross", "boundary_background_control"),
        ("boundary_cross_vs_random", "boundary_cross", "boundary_random_control"),
        ("interior_cross_vs_background", "interior_cross", "interior_background_control"),
        ("interior_cross_vs_random", "interior_cross", "interior_random_control"),
    ]
    mediators = [
        "response_mass_fraction_tumor_total_drop",
        "response_mass_fraction_background_delta",
        "final_prob_pearson_drop",
    ]
    outcome = "final_fg_prob_drop"
    rows: List[Dict] = []
    rng = np.random.default_rng(int(seed))
    for comp_idx, (comparison, treatment, control) in enumerate(comparisons):
        sub = df[df["variant"].isin([treatment, control])].copy()
        unique_cases = sub["case_name"].astype(str).unique()
        case_to_indices = {
            case_name: sub.index[sub["case_name"].astype(str) == case_name].to_numpy(dtype=int)
            for case_name in unique_cases
        }
        for mediator in mediators:
            base = mediation_once(sub, treatment, control, mediator, outcome)
            payload = {
                "comparison": comparison,
                "treatment": treatment,
                "control": control,
                "mediator": mediator,
                "outcome": outcome,
                "n_rows": int(len(sub)),
                "n_cases": int(sub["case_name"].nunique()),
                **base,
            }
            boot_values: Dict[str, List[float]] = {key: [] for key in base}
            for _ in range(int(bootstrap)):
                sampled_cases = rng.choice(unique_cases, size=len(unique_cases), replace=True)
                sampled_indices = np.concatenate([case_to_indices[str(case)] for case in sampled_cases])
                boot = sub.loc[sampled_indices].copy()
                vals = mediation_once(boot, treatment, control, mediator, outcome)
                for key, value in vals.items():
                    if np.isfinite(value):
                        boot_values[key].append(float(value))
            for key, vals in boot_values.items():
                if vals:
                    payload["{}_ci_low".format(key)] = float(np.percentile(vals, 2.5))
                    payload["{}_ci_high".format(key)] = float(np.percentile(vals, 97.5))
                else:
                    payload["{}_ci_low".format(key)] = np.nan
                    payload["{}_ci_high".format(key)] = np.nan
            rows.append(payload)
    return pd.DataFrame(rows)


def key_readout(summary: pd.DataFrame, contrasts: pd.DataFrame, mediation: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    keep_variants = [
        "boundary_cross",
        "boundary_background_control",
        "boundary_random_control",
        "boundary_shift",
        "interior_cross",
    ]
    for _, row in summary[summary["variant"].isin(keep_variants)].iterrows():
        for metric in ("final_fg_prob_drop", "response_mass_fraction_tumor_total_drop", "final_prob_pearson_drop"):
            rows.append(
                {
                    "evidence": "counterfactual_variant",
                    "item": row["variant"],
                    "metric": metric,
                    "value": row.get(metric, np.nan),
                    "ci95_low": row.get("{}_ci_low".format(metric), np.nan),
                    "ci95_high": row.get("{}_ci_high".format(metric), np.nan),
                }
            )
    for _, row in contrasts.iterrows():
        if row["comparison"] in ("boundary_cross_vs_background", "boundary_cross_vs_random"):
            for metric in ("final_fg_prob_drop_diff", "response_mass_fraction_tumor_total_drop_diff"):
                rows.append(
                    {
                        "evidence": "paired_contrast",
                        "item": row["comparison"],
                        "metric": metric,
                        "value": row.get(metric, np.nan),
                        "ci95_low": row.get("{}_ci_low".format(metric), np.nan),
                        "ci95_high": row.get("{}_ci_high".format(metric), np.nan),
                    }
                )
    med = mediation[
        (mediation["comparison"] == "boundary_cross_vs_background")
        & (mediation["mediator"] == "final_prob_pearson_drop")
    ]
    for _, row in med.iterrows():
        for metric in ("indirect_ab", "mediated_fraction", "c_total_treatment_to_outcome", "c_direct_treatment_to_outcome"):
            rows.append(
                {
                    "evidence": "mediation_style",
                    "item": "{}|{}".format(row["comparison"], row["mediator"]),
                    "metric": metric,
                    "value": row.get(metric, np.nan),
                    "ci95_low": row.get("{}_ci_low".format(metric), np.nan),
                    "ci95_high": row.get("{}_ci_high".format(metric), np.nan),
                }
            )
    return pd.DataFrame(rows)


def plot_counterfactual(summary: pd.DataFrame, out_path: str) -> None:
    sub = summary.sort_values("variant_order").copy()
    fig, ax = plt.subplots(figsize=(11.2, 4.8), dpi=170)
    x = np.arange(len(sub))
    y = sub["final_fg_prob_drop"].to_numpy(dtype=np.float64)
    lo = sub["final_fg_prob_drop_ci_low"].to_numpy(dtype=np.float64)
    hi = sub["final_fg_prob_drop_ci_high"].to_numpy(dtype=np.float64)
    ax.bar(x, y, color=["#1f77b4" if "boundary" in v else "#ff7f0e" for v in sub["variant"]])
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor="#333333", capsize=3, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABEL.get(v, v) for v in sub["variant"]], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("final foreground probability drop")
    ax.set_title("Up4 skip counterfactual region replacement")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_mediation(contrasts: pd.DataFrame, out_path: str) -> None:
    if contrasts.empty:
        return
    sub = contrasts[contrasts["comparison"].isin(["boundary_cross_vs_background", "boundary_cross_vs_random"])].copy()
    if sub.empty:
        return
    labels = sub["comparison"].tolist()
    x = np.arange(len(sub))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=170)
    y1 = sub["final_fg_prob_drop_diff"].to_numpy(dtype=np.float64)
    y2 = sub["response_mass_fraction_tumor_total_drop_diff"].to_numpy(dtype=np.float64)
    ax.bar(x - width / 2, y1, width=width, label="extra output drop", color="#4c78a8")
    ax.bar(x + width / 2, y2, width=width, label="extra tumor-response drop", color="#f58518")
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_title("Paired counterfactual contrasts")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
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
    baseline_ckpt = str(args.baseline_ckpt)
    if not os.path.isfile(baseline_ckpt):
        raise FileNotFoundError(args.baseline_ckpt)
    device = resolve_device(args.device)
    _, indices, case_names, loader = build_eval_loader(
        data_dir=args.data_dir,
        split=args.eval_split,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        num_cases=int(args.num_cases),
        seed=int(args.seed),
        foreground_only=False,
        case_set_csv=args.case_set_csv,
    )
    model = load_model(model_name="baseline", ckpt_path=baseline_ckpt, device=device)
    case_df = analyze(model, loader, args, device=device)
    summary = summarize(case_df, bootstrap=int(args.bootstrap), seed=int(args.seed))
    contrasts = paired_contrasts(case_df, bootstrap=int(args.bootstrap), seed=int(args.seed) + 500)
    case_df.to_csv(os.path.join(args.output_root, "table_e13_counterfactual_case_level.csv"), index=False)
    summary.to_csv(os.path.join(args.output_root, "table_e13_counterfactual_summary.csv"), index=False)
    contrasts.to_csv(os.path.join(args.output_root, "table_e13_paired_contrasts.csv"), index=False)
    mediation = mediation_summary(case_df, bootstrap=int(args.bootstrap), seed=int(args.seed) + 1000)
    key = key_readout(summary, contrasts, mediation)

    outputs = {
        "table_e13_counterfactual_case_level.csv": case_df,
        "table_e13_counterfactual_summary.csv": summary,
        "table_e13_paired_contrasts.csv": contrasts,
        "table_e13_mediation_style_summary.csv": mediation,
        "table_e13_key_readout.csv": key,
    }
    for name, frame in outputs.items():
        frame.to_csv(os.path.join(args.output_root, name), index=False)
    plot_counterfactual(summary, os.path.join(args.output_root, "plot_e13_counterfactual_output_drop.png"))
    plot_mediation(contrasts, os.path.join(args.output_root, "plot_e13_paired_contrast_mediation.png"))
    write_run_config(
        args,
        args.output_root,
        extra={"device": str(device), "baseline_ckpt_resolved": baseline_ckpt, "selected_indices": [int(v) for v in indices]},
    )
    manifest_rows = [
        {"output": name, "path": os.path.join(args.output_root, name)}
        for name in list(outputs.keys())
        + [
            "plot_e13_counterfactual_output_drop.png",
            "plot_e13_paired_contrast_mediation.png",
            "summary_up4_skip_counterfactual_mediation.json",
        ]
    ]
    write_manifest(args.output_root, manifest_rows)
    summary_payload = {
        "purpose": "E13 up4 skip counterfactual region replacement and mediation-style analysis",
        "num_cases": int(len(case_names)),
        "variants": list(VARIANT_ORDER),
        "boundary_radius": int(args.boundary_radius),
        "outputs": [row["output"] for row in manifest_rows],
        "guardrail": "This is counterfactual and mediation-style evidence for the Stage-1 phenomenon, not definitive causal proof.",
    }
    write_json(summary_payload, os.path.join(args.output_root, "summary_up4_skip_counterfactual_mediation.json"))
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
