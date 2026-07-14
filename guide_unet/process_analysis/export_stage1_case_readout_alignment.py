from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


GUIDE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GUIDE_ROOT not in sys.path:
    sys.path.insert(0, GUIDE_ROOT)

from analysis_common import (  # noqa: E402
    ANALYSIS_NODE_NAMES,
    MODEL_CHOICES,
    FeatureStore,
    load_model,
    mean_region_dice,
    node_group_hint,
    patient_id_from_case_name,
    safe_corr,
    safe_cos,
    select_indices,
    set_seed,
    unpack_logits,
)
from datasets.dataset_brats import BraTSDataset  # noqa: E402


NODE_ORDER = tuple(ANALYSIS_NODE_NAMES)
TRANSITIONS = tuple(zip(NODE_ORDER[:-1], NODE_ORDER[1:]))
ALIGNMENT_METRICS = (
    "cam_final_prob_pearson",
    "cam_gt_mass_fraction",
    "cam_gt_fg_bg_contrast",
    "cam_entropy_norm",
    "feature_final_prob_pearson",
    "feature_gt_mass_fraction",
    "feature_gt_fg_bg_contrast",
    "feature_entropy_norm",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    ckpt_path: str


class NodeCAMProjector(nn.Module):
    """Same node-to-final-logit readout used by the original Stage-1 task_dice_gt."""

    def __init__(self, in_channels: int, target_channels: int, target_size: Tuple[int, int]):
        super().__init__()
        self.target_size = tuple(int(v) for v in target_size)
        self.proj = nn.Conv2d(int(in_channels), int(target_channels), kernel_size=1, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(int(target_channels), int(target_channels), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(target_channels)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(target_channels), int(target_channels), kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        if tuple(x.shape[2:]) != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
        return x + self.refine(x)


def parse_model_specs(items: Sequence[str]) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for item in items:
        if "=" not in str(item):
            raise ValueError("Expected model_name=/path/to/checkpoint, got {}".format(item))
        name, ckpt_path = str(item).split("=", 1)
        name = name.strip()
        ckpt_path = ckpt_path.strip()
        if name not in MODEL_CHOICES:
            raise ValueError("Unsupported model name: {}".format(name))
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        specs.append(ModelSpec(name=name, ckpt_path=ckpt_path))
    if not specs:
        raise ValueError("At least one --model_spec is required.")
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-case Stage-1 node readout and validate per-case transition alignment "
            "between task_dice_gt and CAM/raw-feature deltas."
        )
    )
    parser.add_argument("--model_spec", nargs="+", required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--fit_split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--eval_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--cam_case_csv", type=str, required=True)
    parser.add_argument("--stage1_reference_csv", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_fit_samples", type=int, default=256)
    parser.add_argument("--num_eval_samples", type=int, default=96)
    parser.add_argument("--fit_epochs", type=int, default=4)
    parser.add_argument("--fit_lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--foreground_only", action="store_true")
    parser.add_argument(
        "--match_cam_cases",
        action="store_true",
        help="Evaluate exactly the case names found in --cam_case_csv, so per-case deltas can be joined without sampling drift.",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def _compute_logits_loss(logits_hat: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(logits_hat, target_logits)
    cos = 1.0 - F.cosine_similarity(logits_hat.flatten(1), target_logits.flatten(1), dim=1).mean()
    return mse + 0.2 * cos


def _tumor_response(logits_chw: np.ndarray) -> np.ndarray:
    logits = torch.from_numpy(logits_chw).unsqueeze(0)
    probs = torch.softmax(logits, dim=1)[0].numpy()
    return probs[1:].sum(axis=0)


def unique_case_names_from_csv(path: str) -> List[str]:
    df = pd.read_csv(path, usecols=["case_name"])
    seen = set()
    names: List[str] = []
    for case_name in df["case_name"].astype(str).tolist():
        if case_name in seen:
            continue
        seen.add(case_name)
        names.append(case_name)
    return names


def indices_from_case_names(dataset: BraTSDataset, case_names: Sequence[str]) -> List[int]:
    name_to_index = {str(name): idx for idx, name in enumerate(dataset.sample_list)}
    missing = [str(name) for name in case_names if str(name) not in name_to_index]
    if missing:
        raise RuntimeError("Missing {} requested eval cases, first missing: {}".format(len(missing), missing[:5]))
    return [int(name_to_index[str(name)]) for name in case_names]


def build_loaders(args: argparse.Namespace) -> Tuple[List[str], DataLoader, DataLoader]:
    fit_dataset = BraTSDataset(base_dir=args.data_dir, split=args.fit_split)
    eval_dataset = fit_dataset if args.fit_split == args.eval_split else BraTSDataset(base_dir=args.data_dir, split=args.eval_split)

    fit_indices = select_indices(
        dataset=fit_dataset,
        num_samples=int(args.num_fit_samples),
        foreground_only=bool(args.foreground_only),
        seed=int(args.seed),
        patient_balanced=False,
    )
    if bool(args.match_cam_cases):
        eval_case_names = unique_case_names_from_csv(args.cam_case_csv)
        eval_indices = indices_from_case_names(eval_dataset, eval_case_names)
    else:
        eval_indices = select_indices(
            dataset=eval_dataset,
            num_samples=int(args.num_eval_samples),
            foreground_only=bool(args.foreground_only),
            seed=int(args.seed) + 1,
            patient_balanced=False,
        )
        eval_case_names = [str(eval_dataset.sample_list[idx]) for idx in eval_indices]

    fit_loader = DataLoader(
        Subset(fit_dataset, fit_indices),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        Subset(eval_dataset, eval_indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    return eval_case_names, fit_loader, eval_loader


def train_projectors(
    model: nn.Module,
    collector: FeatureStore,
    projectors: Dict[str, nn.Module],
    fit_loader: DataLoader,
    device: torch.device,
    fit_epochs: int,
    fit_lr: float,
) -> List[float]:
    params: List[torch.nn.Parameter] = []
    for projector in projectors.values():
        params.extend(list(projector.parameters()))
    optimizer = torch.optim.Adam(params, lr=float(fit_lr))
    history: List[float] = []
    for _ in range(int(fit_epochs)):
        epoch_losses: List[float] = []
        for batch in fit_loader:
            images = batch["image"].to(device)
            collector.clear()
            with torch.no_grad():
                _ = model(images)
                node_map = collector.node_map()
                final_feat = node_map["up4"].detach()
                final_logits = model.outc(final_feat).detach()

            optimizer.zero_grad(set_to_none=True)
            loss = final_logits.new_zeros(())
            for node_name, projector in projectors.items():
                feat_hat = projector(node_map[node_name].detach())
                logits_hat = model.outc(feat_hat)
                loss = loss + _compute_logits_loss(logits_hat, final_logits)
            loss = loss / float(len(projectors))
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        history.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
    return history


def collect_stage1_case_readout(
    model_name: str,
    model: nn.Module,
    collector: FeatureStore,
    projectors: Dict[str, nn.Module],
    eval_loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for projector in projectors.values():
        projector.eval()

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Stage1 case readout {}".format(model_name)):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            case_names = list(batch["case_name"])
            collector.clear()
            logits_final = unpack_logits(model(images))
            node_map = collector.node_map()
            pred_final = torch.argmax(logits_final, dim=1)

            node_logits: Dict[str, torch.Tensor] = {}
            for node_name, projector in projectors.items():
                feat_hat = projector(node_map[node_name].detach())
                node_logits[node_name] = model.outc(feat_hat)

            for local_idx, case_name in enumerate(case_names):
                label_np = labels[local_idx].detach().cpu().numpy()
                final_logits_np = logits_final[local_idx].detach().cpu().numpy()
                final_tumor = _tumor_response(final_logits_np)
                pred_final_np = pred_final[local_idx].detach().cpu().numpy()
                for node_order, node_name in enumerate(NODE_ORDER):
                    logits_hat_np = node_logits[node_name][local_idx].detach().cpu().numpy()
                    pred_hat_np = torch.argmax(node_logits[node_name][local_idx], dim=0).detach().cpu().numpy()
                    node_tumor = _tumor_response(logits_hat_np)
                    rows.append(
                        {
                            "model": model_name,
                            "case_name": str(case_name),
                            "patient_id": patient_id_from_case_name(str(case_name)),
                            "node": node_name,
                            "node_order": int(node_order),
                            "group_hint": node_group_hint(node_name),
                            "task_dice_gt_case": float(mean_region_dice(pred_hat_np, label_np)),
                            "task_corr_case": float(safe_corr(node_tumor, final_tumor)),
                            "task_cosine_case": float(safe_cos(node_tumor, final_tumor)),
                            "node_vs_final_mask_mean_dice": float(mean_region_dice(pred_hat_np, pred_final_np)),
                            "final_vs_gt_mean_dice": float(mean_region_dice(pred_final_np, label_np)),
                        }
                    )
    return pd.DataFrame(rows)


def fit_and_eval_model(spec: ModelSpec, fit_loader: DataLoader, eval_loader: DataLoader, args: argparse.Namespace, device: torch.device) -> Tuple[pd.DataFrame, List[float]]:
    model = load_model(model_name=spec.name, ckpt_path=spec.ckpt_path, device=device)
    collector = FeatureStore(model)
    warmup_batch = next(iter(fit_loader))
    with torch.no_grad():
        collector.clear()
        _ = model(warmup_batch["image"].to(device))
        node_map = collector.node_map()
        final_feat = node_map["up4"]
    target_channels = int(final_feat.shape[1])
    target_size = tuple(int(v) for v in final_feat.shape[2:])
    projectors = {
        node_name: NodeCAMProjector(
            in_channels=int(node_map[node_name].shape[1]),
            target_channels=target_channels,
            target_size=target_size,
        ).to(device)
        for node_name in NODE_ORDER
    }
    history = train_projectors(
        model=model,
        collector=collector,
        projectors=projectors,
        fit_loader=fit_loader,
        device=device,
        fit_epochs=int(args.fit_epochs),
        fit_lr=float(args.fit_lr),
    )
    readout_df = collect_stage1_case_readout(
        model_name=spec.name,
        model=model,
        collector=collector,
        projectors=projectors,
        eval_loader=eval_loader,
        device=device,
    )
    collector.remove()
    return readout_df, history


def summarize_node_readout(readout_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "task_dice_gt_case",
        "task_corr_case",
        "task_cosine_case",
        "node_vs_final_mask_mean_dice",
        "final_vs_gt_mean_dice",
    ]
    return (
        readout_df.groupby(["model", "node", "node_order", "group_hint"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["model", "node_order"])
        .reset_index(drop=True)
    )


def reference_check(node_df: pd.DataFrame, stage1_reference_csv: str) -> pd.DataFrame:
    if not stage1_reference_csv or not os.path.isfile(stage1_reference_csv):
        return pd.DataFrame()
    ref = pd.read_csv(stage1_reference_csv)
    rows: List[Dict] = []
    for _, item in node_df.iterrows():
        model = str(item["model"])
        suffix = "baseline" if model == "baseline" else "noskip" if model == "noskip_unet" else model
        col = "task_dice_gt_{}".format(suffix)
        sub = ref[ref["node"].astype(str) == str(item["node"])]
        if sub.empty or col not in sub.columns:
            continue
        ref_value = float(sub.iloc[0][col])
        value = float(item["task_dice_gt_case"])
        rows.append(
            {
                "model": model,
                "node": str(item["node"]),
                "node_order": int(item["node_order"]),
                "case_readout_mean": value,
                "reference_stage1_value": ref_value,
                "mean_minus_reference": value - ref_value,
                "abs_diff": abs(value - ref_value),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "node_order"]).reset_index(drop=True)


def build_transition_deltas(merged_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    required_metrics = ("task_dice_gt_case", *ALIGNMENT_METRICS)
    for (model, case_name), group in merged_df.groupby(["model", "case_name"], sort=False):
        pivot = group.set_index("node")
        if any(node not in pivot.index for node in NODE_ORDER):
            continue
        for src, dst in TRANSITIONS:
            row = {
                "model": model,
                "case_name": case_name,
                "patient_id": str(pivot.loc[src, "patient_id"]),
                "transition": "{}->{}".format(src, dst),
                "source_node": src,
                "target_node": dst,
                "source_order": NODE_ORDER.index(src),
                "task_dice_gt_delta": float(pivot.loc[dst, "task_dice_gt_case"] - pivot.loc[src, "task_dice_gt_case"]),
                "task_dice_gt_source": float(pivot.loc[src, "task_dice_gt_case"]),
                "task_dice_gt_target": float(pivot.loc[dst, "task_dice_gt_case"]),
            }
            missing = False
            for metric in ALIGNMENT_METRICS:
                if metric not in pivot.columns:
                    missing = True
                    break
                row["{}_delta".format(metric)] = float(pivot.loc[dst, metric] - pivot.loc[src, metric])
            if not missing and all(key in row for key in ["task_dice_gt_delta"]):
                rows.append(row)
    return pd.DataFrame(rows)


def finite_pair(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def corr_value(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    x, y = finite_pair(x, y)
    if x.size < 3 or float(np.std(x)) < 1e-10 or float(np.std(y)) < 1e-10:
        return np.nan
    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    if method == "spearman":
        return float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    raise ValueError(method)


def bootstrap_corr_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int, method: str = "pearson") -> Tuple[float, float]:
    x, y = finite_pair(x, y)
    if x.size < 6 or int(n_boot) <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    vals: List[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, x.size, size=x.size)
        val = corr_value(x[idx], y[idx], method=method)
        if np.isfinite(val):
            vals.append(float(val))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def correlation_tables(delta_df: pd.DataFrame, bootstrap: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: List[Dict] = []
    transition_rows: List[Dict] = []
    for model, group in delta_df.groupby("model", sort=False):
        for metric in ALIGNMENT_METRICS:
            metric_col = "{}_delta".format(metric)
            pearson = corr_value(group["task_dice_gt_delta"].to_numpy(), group[metric_col].to_numpy(), method="pearson")
            spearman = corr_value(group["task_dice_gt_delta"].to_numpy(), group[metric_col].to_numpy(), method="spearman")
            lo, hi = bootstrap_corr_ci(
                group["task_dice_gt_delta"].to_numpy(),
                group[metric_col].to_numpy(),
                n_boot=int(bootstrap),
                seed=int(seed),
                method="pearson",
            )
            overall_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_case_transition_pairs": int(len(group)),
                    "pearson_task_delta_vs_metric_delta": pearson,
                    "pearson_ci95_low": lo,
                    "pearson_ci95_high": hi,
                    "spearman_task_delta_vs_metric_delta": spearman,
                }
            )
        for transition, tgroup in group.groupby("transition", sort=False):
            for metric in ALIGNMENT_METRICS:
                metric_col = "{}_delta".format(metric)
                pearson = corr_value(tgroup["task_dice_gt_delta"].to_numpy(), tgroup[metric_col].to_numpy(), method="pearson")
                spearman = corr_value(tgroup["task_dice_gt_delta"].to_numpy(), tgroup[metric_col].to_numpy(), method="spearman")
                lo, hi = bootstrap_corr_ci(
                    tgroup["task_dice_gt_delta"].to_numpy(),
                    tgroup[metric_col].to_numpy(),
                    n_boot=int(bootstrap),
                    seed=int(seed) + NODE_ORDER.index(str(tgroup["source_node"].iloc[0])),
                    method="pearson",
                )
                transition_rows.append(
                    {
                        "model": model,
                        "transition": transition,
                        "source_order": int(tgroup["source_order"].iloc[0]),
                        "metric": metric,
                        "n_cases": int(len(tgroup)),
                        "pearson_task_delta_vs_metric_delta": pearson,
                        "pearson_ci95_low": lo,
                        "pearson_ci95_high": hi,
                        "spearman_task_delta_vs_metric_delta": spearman,
                        "task_delta_mean": float(tgroup["task_dice_gt_delta"].mean()),
                        "metric_delta_mean": float(tgroup[metric_col].mean()),
                    }
                )
    return pd.DataFrame(overall_rows), pd.DataFrame(transition_rows).sort_values(["model", "source_order", "metric"])


def phenomenon_support_table(readout_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    case_names = sorted(set(readout_df["case_name"].astype(str)))
    for case_name in case_names:
        sub = readout_df[readout_df["case_name"].astype(str) == case_name]
        pivot = sub.set_index(["model", "node"])
        if any((model, node) not in pivot.index for model in ("baseline", "noskip_unet") for node in NODE_ORDER):
            continue
        base = {node: float(pivot.loc[("baseline", node), "task_dice_gt_case"]) for node in NODE_ORDER}
        noskip = {node: float(pivot.loc[("noskip_unet", node), "task_dice_gt_case"]) for node in NODE_ORDER}
        base_up = [base["up2"] - base["up1"], base["up3"] - base["up2"], base["up4"] - base["up3"]]
        noskip_up = [noskip["up2"] - noskip["up1"], noskip["up3"] - noskip["up2"], noskip["up4"] - noskip["up3"]]
        rows.append(
            {
                "case_name": case_name,
                "patient_id": patient_id_from_case_name(case_name),
                "early_mean_baseline": float(np.mean([base[n] for n in ("down1", "down2", "down3")])),
                "early_mean_noskip": float(np.mean([noskip[n] for n in ("down1", "down2", "down3")])),
                "early_gap_noskip_minus_baseline": float(
                    np.mean([noskip[n] for n in ("down1", "down2", "down3")])
                    - np.mean([base[n] for n in ("down1", "down2", "down3")])
                ),
                "baseline_valley_score": float(base["down3"] - np.mean([base["down4"], base["up1"]])),
                "baseline_up1_up2_delta": base_up[0],
                "baseline_up2_up3_delta": base_up[1],
                "baseline_up3_up4_delta": base_up[2],
                "noskip_up1_up2_delta": noskip_up[0],
                "noskip_up2_up3_delta": noskip_up[1],
                "noskip_up3_up4_delta": noskip_up[2],
                "baseline_up_acceleration_pattern": bool(base_up[0] < base_up[1] < base_up[2]),
                "noskip_up_deceleration_pattern": bool(noskip_up[0] > noskip_up[1] > noskip_up[2]),
                "final_cross_baseline_gt_noskip": bool(base["up4"] > noskip["up4"]),
            }
        )
    return pd.DataFrame(rows)


def plot_key_scatter(delta_df: pd.DataFrame, out_path: str) -> None:
    metrics = [
        "cam_final_prob_pearson",
        "cam_gt_mass_fraction",
        "feature_final_prob_pearson",
        "feature_gt_fg_bg_contrast",
    ]
    colors = {"baseline": "#1f77b4", "noskip_unet": "#ff7f0e"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=160)
    axes = axes.reshape(-1)
    for ax, metric in zip(axes, metrics):
        metric_col = "{}_delta".format(metric)
        for model, group in delta_df.groupby("model", sort=False):
            ax.scatter(
                group[metric_col].astype(float),
                group["task_dice_gt_delta"].astype(float),
                s=16,
                alpha=0.55,
                label=model,
                color=colors.get(model, "#333333"),
            )
        ax.axhline(0.0, color="#888888", linewidth=0.8)
        ax.axvline(0.0, color="#888888", linewidth=0.8)
        ax.set_title(metric)
        ax.set_xlabel("metric delta")
        ax.set_ylabel("case-level task_dice_gt delta")
        ax.grid(alpha=0.2)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_transition_heatmap(corr_df: pd.DataFrame, out_path: str) -> None:
    metric_order = [
        "cam_final_prob_pearson",
        "cam_gt_mass_fraction",
        "cam_gt_fg_bg_contrast",
        "cam_entropy_norm",
        "feature_final_prob_pearson",
        "feature_gt_mass_fraction",
        "feature_gt_fg_bg_contrast",
        "feature_entropy_norm",
    ]
    models = [m for m in ("baseline", "noskip_unet") if m in set(corr_df["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(7 * max(1, len(models)), 6), dpi=160)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = corr_df[corr_df["model"] == model]
        pivot = sub.pivot(index="metric", columns="transition", values="pearson_task_delta_vs_metric_delta").reindex(metric_order)
        im = ax.imshow(pivot.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        ax.set_title("{}: case-level delta correlation".format(model))
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index))
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if np.isfinite(val):
                    ax.text(j, i, "{:.2f}".format(val), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.75, label="Pearson r")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_summary(
    args: argparse.Namespace,
    eval_case_names: Sequence[str],
    histories: Dict[str, List[float]],
    node_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    overall_corr: pd.DataFrame,
    by_transition_corr: pd.DataFrame,
    support_df: pd.DataFrame,
    out_path: str,
) -> None:
    payload = {
        "purpose": "per-case Stage-1 readout and same-case CAM/feature transition-delta alignment",
        "num_eval_cases": int(len(eval_case_names)),
        "eval_cases_from_cam_csv": bool(args.match_cam_cases),
        "cam_case_csv": str(args.cam_case_csv),
        "fit_epochs": int(args.fit_epochs),
        "fit_lr": float(args.fit_lr),
        "num_fit_samples": int(args.num_fit_samples),
        "projector_fit_history": histories,
        "node_readout_mean": {},
        "reference_abs_diff_mean": None,
        "overall_delta_correlation": {},
        "phenomenon_case_support_rate": {},
        "guardrail": (
            "This output validates whether the observed Stage-1 curve is supported at the case-transition level. "
            "It must not be used to introduce new model-improvement claims or broad skip-connection storytelling."
        ),
    }
    for model, group in node_df.groupby("model", sort=False):
        payload["node_readout_mean"][model] = {
            str(row["node"]): float(row["task_dice_gt_case"]) for _, row in group.sort_values("node_order").iterrows()
        }
    if not reference_df.empty:
        payload["reference_abs_diff_mean"] = float(reference_df["abs_diff"].mean())
    for model, group in overall_corr.groupby("model", sort=False):
        payload["overall_delta_correlation"][model] = {
            str(row["metric"]): float(row["pearson_task_delta_vs_metric_delta"])
            if np.isfinite(row["pearson_task_delta_vs_metric_delta"])
            else None
            for _, row in group.iterrows()
        }
    if not support_df.empty:
        for col in (
            "early_gap_noskip_minus_baseline",
            "baseline_valley_score",
            "baseline_up_acceleration_pattern",
            "noskip_up_deceleration_pattern",
            "final_cross_baseline_gt_noskip",
        ):
            if col in support_df.columns:
                if support_df[col].dtype == bool:
                    payload["phenomenon_case_support_rate"][col] = float(support_df[col].mean())
                else:
                    payload["phenomenon_case_support_rate"]["{}_positive_rate".format(col)] = float((support_df[col] > 0).mean())

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# 病例级 Stage 1 readout 与 CAM/feature 增量相关性结果",
        "",
        "本结果只服务于 Stage 1 曲线现象原因解释：验证同一病例、同一相邻节点转移上，`task_dice_gt` 增量是否与 CAM/raw feature response 增量同步。",
        "",
        "## 1. 输出文件",
        "",
        "- `table_stage1_case_readout.csv`：病例级 Stage 1 节点 readout。",
        "- `table_stage1_case_node_summary.csv`：病例级 readout 的节点均值。",
        "- `table_stage1_case_readout_reference_check.csv`：当前导出均值与原始 Stage 1 曲线的差异检查。",
        "- `table_case_transition_delta_alignment.csv`：每个病例、每条相邻边的 task/CAM/feature 增量。",
        "- `table_case_delta_correlation_overall.csv`：跨全部病例-边对的增量相关性。",
        "- `table_case_delta_correlation_by_transition.csv`：逐相邻边的病例级增量相关性。",
        "- `table_stage1_case_phenomenon_support.csv`：早期差距、低谷、加速/减速、末端交叉的病例级支持表。",
        "",
        "## 2. 当前约束",
        "",
        "该实验不引入新结构、不解释模型改进，只回答 Stage 1 曲线现象的病例级证据是否成立。",
    ]
    with open(os.path.join(os.path.dirname(out_path), "59_stage1_case_readout_alignment_results_cn.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(int(args.seed))
    specs = parse_model_specs(args.model_spec)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    eval_case_names, fit_loader, eval_loader = build_loaders(args)

    frames: List[pd.DataFrame] = []
    histories: Dict[str, List[float]] = {}
    for spec in specs:
        frame, history = fit_and_eval_model(spec=spec, fit_loader=fit_loader, eval_loader=eval_loader, args=args, device=device)
        frames.append(frame)
        histories[spec.name] = history
    readout_df = pd.concat(frames, ignore_index=True)
    node_df = summarize_node_readout(readout_df)
    reference_df = reference_check(node_df=node_df, stage1_reference_csv=str(args.stage1_reference_csv))

    cam_df = pd.read_csv(args.cam_case_csv)
    needed_cols = ["model", "case_name", "node", *ALIGNMENT_METRICS]
    missing = [col for col in needed_cols if col not in cam_df.columns]
    if missing:
        raise RuntimeError("CAM case CSV missing columns: {}".format(missing))
    cam_df = cam_df[needed_cols].copy()
    merged_df = pd.merge(readout_df, cam_df, on=["model", "case_name", "node"], how="inner")
    delta_df = build_transition_deltas(merged_df)
    overall_corr, by_transition_corr = correlation_tables(delta_df, bootstrap=int(args.bootstrap), seed=int(args.seed))
    support_df = phenomenon_support_table(readout_df)

    readout_df.to_csv(os.path.join(args.save_dir, "table_stage1_case_readout.csv"), index=False)
    node_df.to_csv(os.path.join(args.save_dir, "table_stage1_case_node_summary.csv"), index=False)
    reference_df.to_csv(os.path.join(args.save_dir, "table_stage1_case_readout_reference_check.csv"), index=False)
    merged_df.to_csv(os.path.join(args.save_dir, "table_stage1_case_readout_cam_feature_joined.csv"), index=False)
    delta_df.to_csv(os.path.join(args.save_dir, "table_case_transition_delta_alignment.csv"), index=False)
    overall_corr.to_csv(os.path.join(args.save_dir, "table_case_delta_correlation_overall.csv"), index=False)
    by_transition_corr.to_csv(os.path.join(args.save_dir, "table_case_delta_correlation_by_transition.csv"), index=False)
    support_df.to_csv(os.path.join(args.save_dir, "table_stage1_case_phenomenon_support.csv"), index=False)

    plot_key_scatter(delta_df, os.path.join(args.save_dir, "plot_case_delta_alignment_key_metrics.png"))
    plot_transition_heatmap(by_transition_corr, os.path.join(args.save_dir, "plot_case_delta_correlation_by_transition_heatmap.png"))

    write_summary(
        args=args,
        eval_case_names=eval_case_names,
        histories=histories,
        node_df=node_df,
        reference_df=reference_df,
        overall_corr=overall_corr,
        by_transition_corr=by_transition_corr,
        support_df=support_df,
        out_path=os.path.join(args.save_dir, "summary_stage1_case_readout_alignment.json"),
    )
    print("Saved Stage-1 case readout alignment outputs to {}".format(args.save_dir))
    print("Device:", device)
    print("Eval cases:", len(eval_case_names))


if __name__ == "__main__":
    main()
