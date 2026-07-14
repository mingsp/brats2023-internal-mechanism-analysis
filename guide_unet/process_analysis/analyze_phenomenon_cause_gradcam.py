from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


GUIDE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GUIDE_ROOT not in sys.path:
    sys.path.insert(0, GUIDE_ROOT)

from analysis_common import (  # noqa: E402
    MODEL_CHOICES,
    build_region_mask,
    load_model,
    mean_region_dice,
    safe_corr,
    safe_cos,
    set_seed,
    unpack_logits,
)
from datasets.dataset_brats import BraTSDataset  # noqa: E402


NETWORK_ORDER = ("down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4")
DECODER_NODES = ("up1", "up2", "up3", "up4")
DEFAULT_METRICS = (
    "task_dice_gt",
    "cam_final_prob_pearson",
    "cam_final_pearson",
    "cam_gt_mass_fraction",
    "cam_gt_fg_bg_contrast",
    "cam_entropy_norm",
    "cam_top10_gt_fraction",
    "feature_final_prob_pearson",
    "feature_gt_mass_fraction",
    "feature_gt_fg_bg_contrast",
    "feature_entropy_norm",
)


@dataclass
class ModelSpec:
    name: str
    ckpt_path: str


class GradCAMStore:
    def __init__(self, model: torch.nn.Module, node_names: Sequence[str]):
        self.model = model
        self.node_names = tuple(str(name) for name in node_names)
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self._handles = []
        for node_name in self.node_names:
            module = getattr(model, node_name)
            self._handles.append(module.register_forward_hook(self._make_forward_hook(node_name)))

    def _make_forward_hook(self, node_name: str):
        def hook_fn(module, inputs, output):
            tensor = output
            if isinstance(output, (tuple, list)):
                tensor = next(item for item in output if torch.is_tensor(item))
            if not torch.is_tensor(tensor):
                raise TypeError("Hook output for {} is not a tensor".format(node_name))
            self.activations[node_name] = tensor

            def grad_hook(grad):
                self.gradients[node_name] = grad

            tensor.register_hook(grad_hook)

        return hook_fn

    def clear(self):
        self.activations = {}
        self.gradients = {}

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []


def parse_model_specs(items: Sequence[str]) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for item in items:
        if "=" not in str(item):
            raise ValueError("Model spec must be model_name=/path/to/checkpoint, got {}".format(item))
        name, ckpt_path = str(item).split("=", 1)
        name = name.strip()
        ckpt_path = ckpt_path.strip()
        if name not in MODEL_CHOICES:
            raise ValueError("Unsupported model name in spec: {}".format(name))
        if not ckpt_path:
            raise ValueError("Empty checkpoint path for {}".format(name))
        specs.append(ModelSpec(name=name, ckpt_path=ckpt_path))
    if not specs:
        raise ValueError("At least one --model_spec is required.")
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phenomenon-cause analysis using segmentation Grad-CAM trajectories. "
            "The script quantifies whether full-network CAM and feature-response dynamics explain the Stage-1 "
            "acceleration/deceleration phenomenon."
        )
    )
    parser.add_argument("--model_spec", nargs="+", required=True, help="Format: model_name=/path/to/checkpoint")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--stage1_csv", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--scan_limit", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--target_mode", type=str, default="pred_fg", choices=["pred_fg", "gt_fg"])
    parser.add_argument(
        "--nodes",
        type=str,
        default="all",
        help="Node set to analyze: all, decoder, or comma-separated node names.",
    )
    parser.add_argument("--foreground_only", action="store_true")
    parser.add_argument(
        "--case_names",
        type=str,
        default="",
        help="Optional comma-separated exact case names. When set, only these slices are analyzed.",
    )
    parser.add_argument("--filmstrip_cases", type=int, default=4)
    parser.add_argument(
        "--filmstrip_case_names",
        type=str,
        default="",
        help="Optional comma-separated case names to save as filmstrips from the selected sample set.",
    )
    parser.add_argument("--save_feature_filmstrips", action="store_true")
    parser.add_argument(
        "--save_comparison_filmstrips",
        action="store_true",
        help="Save per-case baseline/no-skip comparison grids for selected case names.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def resolve_node_names(nodes_arg: str) -> Tuple[str, ...]:
    text = str(nodes_arg).strip()
    if text.lower() == "all":
        return tuple(NETWORK_ORDER)
    if text.lower() == "decoder":
        return tuple(DECODER_NODES)
    names = tuple(item.strip() for item in text.split(",") if item.strip())
    invalid = sorted(set(names) - set(NETWORK_ORDER))
    if invalid:
        raise ValueError("Invalid node names: {}".format(", ".join(invalid)))
    if not names:
        raise ValueError("--nodes resolved to an empty node list")
    return names


def parse_case_names(text: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in str(text).split(",") if item.strip())


def select_case_indices(
    dataset: BraTSDataset,
    num_samples: int,
    foreground_only: bool,
    seed: int,
    scan_limit: int,
    case_names: Sequence[str] = (),
) -> List[int]:
    requested_names = [str(item).strip() for item in case_names if str(item).strip()]
    if requested_names:
        index_by_name = {str(name): idx for idx, name in enumerate(dataset.sample_list)}
        missing = [name for name in requested_names if name not in index_by_name]
        if missing:
            raise ValueError("Requested case names not found in dataset: {}".format(", ".join(missing[:10])))
        return [int(index_by_name[name]) for name in requested_names]

    rng = np.random.default_rng(int(seed))
    all_indices = np.arange(len(dataset))
    rng.shuffle(all_indices)
    selected: List[int] = []
    checked = 0
    for idx in all_indices:
        if int(scan_limit) > 0 and checked >= int(scan_limit):
            break
        checked += 1
        if foreground_only:
            sample = dataset[int(idx)]
            if int((sample["label"] > 0).sum().item()) == 0:
                continue
        selected.append(int(idx))
        if 0 < int(num_samples) <= len(selected):
            break
    if not selected:
        raise RuntimeError("No samples selected. Try disabling --foreground_only or increasing --scan_limit.")
    return selected


def build_eval_loader(args: argparse.Namespace) -> Tuple[BraTSDataset, List[int], DataLoader]:
    dataset = BraTSDataset(base_dir=args.data_dir, split=args.split)
    indices = select_case_indices(
        dataset=dataset,
        num_samples=int(args.num_samples),
        foreground_only=bool(args.foreground_only),
        seed=int(args.seed),
        scan_limit=int(args.scan_limit),
        case_names=parse_case_names(args.case_names),
    )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, indices, loader


def tumor_logit_map(logits: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(logits[:, 1:, :, :], dim=1)


def target_mask_from_logits(logits: torch.Tensor, labels: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "gt_fg":
        mask = labels > 0
    elif mode == "pred_fg":
        pred = torch.argmax(logits.detach(), dim=1)
        mask = pred > 0
        gt_mask = labels > 0
        empty = mask.flatten(1).sum(dim=1) == 0
        if torch.any(empty):
            mask = mask.clone()
            mask[empty] = gt_mask[empty]
    else:
        raise ValueError("Unsupported target mode: {}".format(mode))
    return mask.float()


def segmentation_target_score(logits: torch.Tensor, labels: torch.Tensor, mode: str) -> torch.Tensor:
    mask = target_mask_from_logits(logits=logits, labels=labels, mode=mode)
    tumor_logits = tumor_logit_map(logits)
    denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    score_per_case = (tumor_logits * mask).flatten(1).sum(dim=1) / denom
    return score_per_case.mean()


def normalize_cam(cam: torch.Tensor) -> torch.Tensor:
    flat = cam.flatten(1)
    min_v = flat.min(dim=1).values.view(-1, 1, 1)
    max_v = flat.max(dim=1).values.view(-1, 1, 1)
    return (cam - min_v) / (max_v - min_v + 1e-8)


def compute_gradcams(store: GradCAMStore, output_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
    cams: Dict[str, torch.Tensor] = {}
    for node_name in store.node_names:
        activation = store.activations[node_name]
        gradient = store.gradients[node_name]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1))
        if tuple(cam.shape[-2:]) != tuple(output_size):
            cam = F.interpolate(cam.unsqueeze(1), size=output_size, mode="bilinear", align_corners=False).squeeze(1)
        cams[node_name] = normalize_cam(cam.detach())
    return cams


def compute_feature_response_maps(store: GradCAMStore, output_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
    maps: Dict[str, torch.Tensor] = {}
    for node_name in store.node_names:
        activation = store.activations[node_name].detach()
        response = torch.sqrt(torch.mean(activation.float() ** 2, dim=1).clamp_min(0.0))
        if tuple(response.shape[-2:]) != tuple(output_size):
            response = F.interpolate(response.unsqueeze(1), size=output_size, mode="bilinear", align_corners=False).squeeze(1)
        maps[node_name] = normalize_cam(response)
    return maps


def normalized_entropy(cam: np.ndarray) -> float:
    values = np.maximum(cam.astype(np.float64).reshape(-1), 0.0)
    total = float(values.sum())
    if total <= 1e-12:
        return 0.0
    p = values / total
    entropy = -float(np.sum(p * np.log(p + 1e-12)))
    return float(entropy / np.log(max(len(p), 2)))


def mass_fraction(cam: np.ndarray, mask: np.ndarray) -> float:
    values = np.maximum(cam.astype(np.float64), 0.0)
    total = float(values.sum())
    if total <= 1e-12:
        return 0.0
    return float(values[mask.astype(bool)].sum() / total)


def fg_bg_ratio(cam: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return 0.0
    fg_mean = float(cam[mask_bool].mean())
    bg = cam[~mask_bool]
    bg_mean = float(bg.mean()) if bg.size else 0.0
    return float((fg_mean + 1e-8) / (bg_mean + 1e-8))


def fg_bg_contrast(cam: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return 0.0
    fg_mean = float(cam[mask_bool].mean())
    bg = cam[~mask_bool]
    bg_mean = float(bg.mean()) if bg.size else 0.0
    return float((fg_mean - bg_mean) / (fg_mean + bg_mean + 1e-8))


def top_fraction_in_mask(cam: np.ndarray, mask: np.ndarray, top_fraction: float = 0.10) -> float:
    flat = cam.reshape(-1)
    mask_flat = mask.reshape(-1).astype(bool)
    k = max(1, int(round(float(top_fraction) * flat.size)))
    top_idx = np.argpartition(flat, -k)[-k:]
    return float(mask_flat[top_idx].mean())


def image_channel_for_display(image_chw: np.ndarray) -> np.ndarray:
    channel = image_chw[0].astype(np.float32)
    lo, hi = np.percentile(channel, [1, 99])
    return np.clip((channel - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def overlay_cam(base: np.ndarray, cam: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    heat = cmap(np.clip(cam, 0.0, 1.0))[..., :3]
    gray = np.stack([base, base, base], axis=-1)
    return np.clip((1.0 - alpha) * gray + alpha * heat, 0.0, 1.0)


def compute_case_rows(
    model_name: str,
    batch,
    logits: torch.Tensor,
    cams: Dict[str, torch.Tensor],
    feature_maps: Dict[str, torch.Tensor],
    node_names: Sequence[str],
    target_mode: str,
    visual_case_names: Sequence[str] = (),
    max_visual_cases: int = 0,
    visual_counter: Optional[Dict[str, int]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    images_np = batch["image"].detach().cpu().numpy()
    labels_np = batch["label"].detach().cpu().numpy()
    case_names = list(batch["case_name"])
    pred_np = torch.argmax(logits.detach(), dim=1).cpu().numpy()
    logits_np = logits.detach().cpu().numpy()
    reference_node = "up4" if "up4" in cams else str(list(cams.keys())[-1])
    up4_np = cams[reference_node].detach().cpu().numpy()
    feature_np = {node_name: feature_maps[node_name].detach().cpu().numpy() for node_name in node_names}
    rows: List[Dict] = []
    visuals: List[Dict] = []
    selected_visuals = {str(item).strip() for item in visual_case_names if str(item).strip()}
    if visual_counter is None:
        visual_counter = {}

    for local_idx, case_name in enumerate(case_names):
        label = labels_np[local_idx]
        pred = pred_np[local_idx]
        gt_fg = label > 0
        pred_fg = pred > 0
        final_dice = float(mean_region_dice(pred, label))
        final_tumor = torch.softmax(torch.from_numpy(logits_np[local_idx]).unsqueeze(0), dim=1)[0, 1:, :, :].sum(dim=0).numpy()
        case_name_text = str(case_name)
        if selected_visuals:
            keep_visual = case_name_text in selected_visuals
        else:
            keep_visual = int(max_visual_cases) > visual_counter.get(model_name, 0)
        visual_item = None
        if keep_visual:
            visual_item = {
                "model": model_name,
                "case_name": case_name_text,
                "image": images_np[local_idx],
                "label": label,
                "pred": pred,
                "final_tumor": final_tumor,
                "node_cams": {},
                "feature_maps": {},
            }

        for node_name in node_names:
            cam = cams[node_name][local_idx].detach().cpu().numpy()
            feature_response = feature_np[node_name][local_idx]
            final_cam = up4_np[local_idx]
            if visual_item is not None:
                visual_item["node_cams"][node_name] = cam
                visual_item["feature_maps"][node_name] = feature_response
            rows.append(
                {
                    "model": model_name,
                    "case_name": case_name_text,
                    "target_mode": str(target_mode),
                    "node": node_name,
                    "node_order": int(node_names.index(node_name)),
                    "final_vs_gt_mean_dice": final_dice,
                    "gt_fg_pixels": int(gt_fg.sum()),
                    "pred_fg_pixels": int(pred_fg.sum()),
                    "cam_final_pearson": float(safe_corr(cam, final_cam)),
                    "cam_final_cosine": float(safe_cos(cam, final_cam)),
                    "cam_final_prob_pearson": float(safe_corr(cam, final_tumor)),
                    "cam_final_prob_cosine": float(safe_cos(cam, final_tumor)),
                    "cam_gt_mass_fraction": float(mass_fraction(cam, gt_fg)),
                    "cam_pred_mass_fraction": float(mass_fraction(cam, pred_fg)),
                    "cam_gt_fg_bg_ratio": float(fg_bg_ratio(cam, gt_fg)),
                    "cam_pred_fg_bg_ratio": float(fg_bg_ratio(cam, pred_fg)),
                    "cam_gt_fg_bg_contrast": float(fg_bg_contrast(cam, gt_fg)),
                    "cam_pred_fg_bg_contrast": float(fg_bg_contrast(cam, pred_fg)),
                    "cam_entropy_norm": float(normalized_entropy(cam)),
                    "cam_top10_gt_fraction": float(top_fraction_in_mask(cam, gt_fg, top_fraction=0.10)),
                    "cam_top10_pred_fraction": float(top_fraction_in_mask(cam, pred_fg, top_fraction=0.10)),
                    "cam_gt_wt_mass_fraction": float(mass_fraction(cam, build_region_mask(label, "WT"))),
                    "cam_gt_tc_mass_fraction": float(mass_fraction(cam, build_region_mask(label, "TC"))),
                    "cam_gt_et_mass_fraction": float(mass_fraction(cam, build_region_mask(label, "ET"))),
                    "feature_final_prob_pearson": float(safe_corr(feature_response, final_tumor)),
                    "feature_final_prob_cosine": float(safe_cos(feature_response, final_tumor)),
                    "feature_gt_mass_fraction": float(mass_fraction(feature_response, gt_fg)),
                    "feature_pred_mass_fraction": float(mass_fraction(feature_response, pred_fg)),
                    "feature_gt_fg_bg_contrast": float(fg_bg_contrast(feature_response, gt_fg)),
                    "feature_pred_fg_bg_contrast": float(fg_bg_contrast(feature_response, pred_fg)),
                    "feature_entropy_norm": float(normalized_entropy(feature_response)),
                    "feature_top10_gt_fraction": float(top_fraction_in_mask(feature_response, gt_fg, top_fraction=0.10)),
                }
            )
        if visual_item is not None:
            visual_counter[model_name] = visual_counter.get(model_name, 0) + 1
            visuals.append(visual_item)
    return rows, visuals


def analyze_model(
    spec: ModelSpec,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    node_names: Sequence[str],
) -> Tuple[pd.DataFrame, List[Dict]]:
    model = load_model(model_name=spec.name, ckpt_path=spec.ckpt_path, device=device)
    store = GradCAMStore(model=model, node_names=node_names)
    rows: List[Dict] = []
    visuals: List[Dict] = []
    visual_counter: Dict[str, int] = {}
    selected_visual_names = parse_case_names(args.filmstrip_case_names) or parse_case_names(args.case_names)

    for batch in tqdm(loader, desc="Grad-CAM {}".format(spec.name)):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        model.zero_grad(set_to_none=True)
        store.clear()
        logits = unpack_logits(model(images))
        score = segmentation_target_score(logits=logits, labels=labels, mode=args.target_mode)
        score.backward()
        cams = compute_gradcams(store=store, output_size=tuple(logits.shape[-2:]))
        feature_maps = compute_feature_response_maps(store=store, output_size=tuple(logits.shape[-2:]))
        batch_rows, batch_visuals = compute_case_rows(
            model_name=spec.name,
            batch=batch,
            logits=logits,
            cams=cams,
            feature_maps=feature_maps,
            node_names=node_names,
            target_mode=args.target_mode,
            visual_case_names=selected_visual_names,
            max_visual_cases=int(args.filmstrip_cases),
            visual_counter=visual_counter,
        )
        rows.extend(batch_rows)
        visuals.extend(batch_visuals)

    store.remove()
    return pd.DataFrame(rows), visuals


def stage1_dynamics_from_csv(stage1_csv: str) -> pd.DataFrame:
    if not stage1_csv or not os.path.isfile(stage1_csv):
        return pd.DataFrame()
    df = pd.read_csv(stage1_csv)
    rows: List[Dict] = []
    for model_name, col in (("baseline", "task_dice_gt_baseline"), ("noskip_unet", "task_dice_gt_noskip")):
        if col not in df.columns:
            continue
        sub = df[df["node"].isin(NETWORK_ORDER)].copy()
        sub["node"] = pd.Categorical(sub["node"], categories=list(NETWORK_ORDER), ordered=True)
        sub = sub.sort_values("node")
        values = sub[col].astype(float).to_numpy()
        nodes = sub["node"].astype(str).tolist()
        for idx, node_name in enumerate(nodes):
            rows.append(
                {
                    "model": model_name,
                    "node": node_name,
                    "node_order": int(idx),
                    "metric": "task_dice_gt",
                    "value": float(values[idx]),
                    "velocity_to_next": float(values[idx + 1] - values[idx]) if idx < len(values) - 1 else np.nan,
                    "acceleration_to_next": np.nan,
                }
            )
        velocities = np.diff(values)
        accelerations = np.diff(velocities)
        for idx in range(len(accelerations)):
            rows[-len(nodes) + idx]["acceleration_to_next"] = float(accelerations[idx])
    return pd.DataFrame(rows)


def summarize_node_metrics(per_case_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "final_vs_gt_mean_dice",
        "gt_fg_pixels",
        "pred_fg_pixels",
        "cam_final_pearson",
        "cam_final_cosine",
        "cam_final_prob_pearson",
        "cam_final_prob_cosine",
        "cam_gt_mass_fraction",
        "cam_pred_mass_fraction",
        "cam_gt_fg_bg_ratio",
        "cam_pred_fg_bg_ratio",
        "cam_gt_fg_bg_contrast",
        "cam_pred_fg_bg_contrast",
        "cam_entropy_norm",
        "cam_top10_gt_fraction",
        "cam_top10_pred_fraction",
        "cam_gt_wt_mass_fraction",
        "cam_gt_tc_mass_fraction",
        "cam_gt_et_mass_fraction",
        "feature_final_prob_pearson",
        "feature_final_prob_cosine",
        "feature_gt_mass_fraction",
        "feature_pred_mass_fraction",
        "feature_gt_fg_bg_contrast",
        "feature_pred_fg_bg_contrast",
        "feature_entropy_norm",
        "feature_top10_gt_fraction",
    ]
    grouped = per_case_df.groupby(["model", "target_mode", "node", "node_order"], as_index=False)[metric_cols].mean()
    grouped = grouped.sort_values(["model", "node_order"]).reset_index(drop=True)
    return grouped


def dynamics_from_node_summary(node_df: pd.DataFrame, stage1_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for (model_name, target_mode), group in node_df.groupby(["model", "target_mode"], sort=False):
        group = group.sort_values("node_order")
        for metric in DEFAULT_METRICS:
            if metric == "task_dice_gt":
                source = stage1_df[(stage1_df["model"] == model_name) & (stage1_df["metric"] == metric)].copy()
                if source.empty:
                    continue
                values = source.sort_values("node_order")["value"].astype(float).to_numpy()
                nodes = source.sort_values("node_order")["node"].astype(str).tolist()
            else:
                values = group[metric].astype(float).to_numpy()
                nodes = group["node"].astype(str).tolist()
            velocities = np.diff(values)
            accelerations = np.diff(velocities)
            for idx, node_name in enumerate(nodes):
                rows.append(
                    {
                        "model": model_name,
                        "target_mode": target_mode,
                        "metric": metric,
                        "node": node_name,
                        "node_order": int(idx),
                        "value": float(values[idx]),
                        "velocity_to_next": float(velocities[idx]) if idx < len(velocities) else np.nan,
                        "acceleration_to_next": float(accelerations[idx]) if idx < len(accelerations) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def write_summary_json(
    args: argparse.Namespace,
    specs: Sequence[ModelSpec],
    selected_indices: Sequence[int],
    node_df: pd.DataFrame,
    dynamics_df: pd.DataFrame,
    out_path: str,
) -> None:
    payload = {
        "target_mode": args.target_mode,
        "split": args.split,
        "num_selected_cases": int(len(selected_indices)),
        "selected_indices": [int(v) for v in selected_indices],
        "model_specs": [{"model": spec.name, "ckpt_path": spec.ckpt_path} for spec in specs],
        "metric_mean": {},
        "network_dynamics": {},
        "interpretation_guardrail": (
            "The tables quantify Grad-CAM trajectories and dynamics. They do not by themselves prove a "
            "causal mechanism unless the trend is jointly supported by CAM visualization, CAM-to-final "
            "similarity, GT alignment, entropy, and Stage-1 semantic dynamics."
        ),
    }
    for model_name, group in node_df.groupby("model"):
        payload["metric_mean"][model_name] = {
            row["node"]: {
                key: float(row[key])
                for key in node_df.columns
                if key not in ("model", "target_mode", "node", "node_order")
            }
            for _, row in group.iterrows()
        }
    for (model_name, metric), group in dynamics_df.groupby(["model", "metric"]):
        payload["network_dynamics"].setdefault(model_name, {})[metric] = [
            {
                "node": row["node"],
                "value": None if pd.isna(row["value"]) else float(row["value"]),
                "velocity_to_next": None if pd.isna(row["velocity_to_next"]) else float(row["velocity_to_next"]),
                "acceleration_to_next": None
                if pd.isna(row["acceleration_to_next"])
                else float(row["acceleration_to_next"]),
            }
            for _, row in group.sort_values("node_order").iterrows()
        ]
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def plot_metric_curves(node_df: pd.DataFrame, stage1_df: pd.DataFrame, save_path: str) -> None:
    plot_metrics = [
        ("task_dice_gt", "Stage-1 semantic decodability"),
        ("cam_final_prob_pearson", "CAM-to-final-prob Pearson"),
        ("cam_gt_mass_fraction", "CAM mass inside GT"),
        ("cam_entropy_norm", "Normalized CAM entropy"),
        ("feature_gt_mass_fraction", "Feature-response mass inside GT"),
        ("feature_entropy_norm", "Feature-response entropy"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), dpi=160)
    axes = axes.reshape(-1)
    colors = {"baseline": "#1f77b4", "noskip_unet": "#ff7f0e", "skip_masked_baseline": "#2ca02c"}
    for ax, (metric, title) in zip(axes, plot_metrics):
        for model_name in sorted(node_df["model"].unique()):
            if metric == "task_dice_gt":
                src = stage1_df[(stage1_df["model"] == model_name) & (stage1_df["metric"] == metric)]
                if src.empty:
                    continue
                x = src.sort_values("node_order")["node"].astype(str).tolist()
                y = src.sort_values("node_order")["value"].astype(float).to_numpy()
            else:
                src = node_df[node_df["model"] == model_name].sort_values("node_order")
                x = src["node"].astype(str).tolist()
                y = src[metric].astype(float).to_numpy()
            ax.plot(x, y, marker="o", linewidth=2, label=model_name, color=colors.get(model_name))
        ax.set_title(title)
        ax.set_xlabel("network node")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_dynamics(dynamics_df: pd.DataFrame, save_path: str) -> None:
    metrics = [
        "task_dice_gt",
        "cam_final_prob_pearson",
        "cam_gt_mass_fraction",
        "cam_entropy_norm",
        "feature_gt_mass_fraction",
        "feature_entropy_norm",
    ]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(12, 3.2 * len(metrics)), dpi=160)
    colors = {"baseline": "#1f77b4", "noskip_unet": "#ff7f0e", "skip_masked_baseline": "#2ca02c"}
    for row_idx, metric in enumerate(metrics):
        for col_idx, field in enumerate(("velocity_to_next", "acceleration_to_next")):
            ax = axes[row_idx, col_idx]
            for model_name, group in dynamics_df[dynamics_df["metric"] == metric].groupby("model"):
                group = group.sort_values("node_order")
                valid = group.dropna(subset=[field])
                if valid.empty:
                    continue
                ax.plot(
                    valid["node"].astype(str).tolist(),
                    valid[field].astype(float).to_numpy(),
                    marker="o",
                    linewidth=2,
                    label=model_name,
                    color=colors.get(model_name),
                )
            ax.axhline(0.0, color="#777777", linewidth=1, linestyle="--")
            ax.set_title("{} {}".format(metric, "velocity" if col_idx == 0 else "acceleration"))
            ax.set_xlabel("network node")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def save_filmstrips(
    visuals: List[Dict],
    per_case_df: pd.DataFrame,
    save_dir: str,
    max_cases_per_model: int,
    node_names: Sequence[str],
    selected_case_names: Sequence[str] = (),
    save_feature_filmstrips: bool = False,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    counts: Dict[str, int] = {}
    selected_set = {str(item).strip() for item in selected_case_names if str(item).strip()}
    for item in visuals:
        model_name = item["model"]
        case_name = item["case_name"]
        if selected_set:
            if str(case_name) not in selected_set:
                continue
        elif counts.get(model_name, 0) >= int(max_cases_per_model):
            continue
        counts[model_name] = counts.get(model_name, 0) + 1
        base = image_channel_for_display(item["image"])
        label = item["label"]
        case_rows = per_case_df[(per_case_df["model"] == model_name) & (per_case_df["case_name"] == case_name)]

        fig, axes = plt.subplots(1, len(node_names) + 2, figsize=(2.4 * (len(node_names) + 2), 3.2), dpi=160)
        axes[0].imshow(base, cmap="gray")
        axes[0].imshow(np.ma.masked_where(label <= 0, label), cmap="Set1", alpha=0.42, vmin=0, vmax=3)
        axes[0].set_title("MRI + GT")
        axes[0].axis("off")

        axes[1].imshow(base, cmap="gray")
        axes[1].imshow(np.ma.masked_where(item["pred"] <= 0, item["pred"]), cmap="Set1", alpha=0.42, vmin=0, vmax=3)
        axes[1].set_title("MRI + Pred")
        axes[1].axis("off")

        for idx, node_name in enumerate(node_names):
            ax = axes[idx + 2]
            cam = item["node_cams"][node_name]
            ax.imshow(overlay_cam(base, cam))
            row = case_rows[case_rows["node"] == node_name]
            if not row.empty:
                row = row.iloc[0]
                subtitle = "{}\nProbR={:.2f}, GTmass={:.2f}, H={:.2f}".format(
                    node_name,
                    float(row["cam_final_prob_pearson"]),
                    float(row["cam_gt_mass_fraction"]),
                    float(row["cam_entropy_norm"]),
                )
            else:
                subtitle = node_name
            ax.set_title(subtitle, fontsize=8)
            ax.axis("off")
        fig.suptitle("{} | {}".format(model_name, case_name), fontsize=11)
        fig.tight_layout()
        out_path = os.path.join(save_dir, "{}__{}.png".format(model_name, case_name))
        fig.savefig(out_path)
        plt.close(fig)

        if not save_feature_filmstrips:
            continue

        fig, axes = plt.subplots(1, len(node_names) + 2, figsize=(2.4 * (len(node_names) + 2), 3.2), dpi=160)
        axes[0].imshow(base, cmap="gray")
        axes[0].imshow(np.ma.masked_where(label <= 0, label), cmap="Set1", alpha=0.42, vmin=0, vmax=3)
        axes[0].set_title("MRI + GT")
        axes[0].axis("off")

        axes[1].imshow(base, cmap="gray")
        axes[1].imshow(np.ma.masked_where(item["pred"] <= 0, item["pred"]), cmap="Set1", alpha=0.42, vmin=0, vmax=3)
        axes[1].set_title("MRI + Pred")
        axes[1].axis("off")

        for idx, node_name in enumerate(node_names):
            ax = axes[idx + 2]
            feature_response = item["feature_maps"][node_name]
            ax.imshow(overlay_cam(base, feature_response))
            row = case_rows[case_rows["node"] == node_name]
            if not row.empty:
                row = row.iloc[0]
                subtitle = "{}\nFProbR={:.2f}, GTmass={:.2f}, H={:.2f}".format(
                    node_name,
                    float(row["feature_final_prob_pearson"]),
                    float(row["feature_gt_mass_fraction"]),
                    float(row["feature_entropy_norm"]),
                )
            else:
                subtitle = node_name
            ax.set_title(subtitle, fontsize=8)
            ax.axis("off")
        fig.suptitle("{} feature response | {}".format(model_name, case_name), fontsize=11)
        fig.tight_layout()
        out_path = os.path.join(save_dir, "{}__{}__feature_response.png".format(model_name, case_name))
        fig.savefig(out_path)
        plt.close(fig)


def save_comparison_filmstrips(
    visuals: List[Dict],
    per_case_df: pd.DataFrame,
    save_dir: str,
    node_names: Sequence[str],
    save_feature_filmstrips: bool = False,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    by_case: Dict[str, Dict[str, Dict]] = {}
    for item in visuals:
        by_case.setdefault(str(item["case_name"]), {})[str(item["model"])] = item

    model_order = ("baseline", "noskip_unet")
    for case_name, model_items in by_case.items():
        if not all(model_name in model_items for model_name in model_order):
            continue

        for mode_name in (("cam",) + (("feature",) if save_feature_filmstrips else ())):
            n_rows = len(model_order)
            n_cols = len(node_names) + 2
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(2.45 * n_cols, 4.1 * n_rows),
                dpi=180,
            )
            for row_idx, model_name in enumerate(model_order):
                item = model_items[model_name]
                base = image_channel_for_display(item["image"])
                label = item["label"]
                model_rows = per_case_df[(per_case_df["model"] == model_name) & (per_case_df["case_name"] == case_name)]

                axes[row_idx, 0].imshow(base, cmap="gray")
                axes[row_idx, 0].imshow(np.ma.masked_where(label <= 0, label), cmap="Set1", alpha=0.44, vmin=0, vmax=3)
                axes[row_idx, 0].set_title("{}\nMRI + GT".format(model_name), fontsize=9)
                axes[row_idx, 0].axis("off")

                axes[row_idx, 1].imshow(base, cmap="gray")
                axes[row_idx, 1].imshow(np.ma.masked_where(item["pred"] <= 0, item["pred"]), cmap="Set1", alpha=0.44, vmin=0, vmax=3)
                dice_rows = model_rows["final_vs_gt_mean_dice"].dropna()
                dice_text = "Dice={:.3f}".format(float(dice_rows.iloc[0])) if not dice_rows.empty else ""
                axes[row_idx, 1].set_title("MRI + Pred\n{}".format(dice_text), fontsize=9)
                axes[row_idx, 1].axis("off")

                for col_idx, node_name in enumerate(node_names, start=2):
                    ax = axes[row_idx, col_idx]
                    row = model_rows[model_rows["node"] == node_name]
                    if mode_name == "cam":
                        response = item["node_cams"][node_name]
                        prefix = "CAM"
                        metric_names = ("cam_final_prob_pearson", "cam_gt_mass_fraction", "cam_entropy_norm")
                    else:
                        response = item["feature_maps"][node_name]
                        prefix = "Raw"
                        metric_names = ("feature_final_prob_pearson", "feature_gt_mass_fraction", "feature_entropy_norm")
                    ax.imshow(overlay_cam(base, response))
                    if not row.empty:
                        row = row.iloc[0]
                        title = "{}\n{}: P={:.2f}, G={:.2f}, H={:.2f}".format(
                            node_name,
                            prefix,
                            float(row[metric_names[0]]),
                            float(row[metric_names[1]]),
                            float(row[metric_names[2]]),
                        )
                    else:
                        title = "{}\n{}".format(node_name, prefix)
                    ax.set_title(title, fontsize=8)
                    ax.axis("off")
            fig.suptitle("{} | {} response comparison".format(case_name, "CAM" if mode_name == "cam" else "raw feature"), fontsize=13, y=0.985)
            fig.text(
                0.5,
                0.018,
                "P: Pearson with final foreground probability | G: GT foreground mass fraction | H: normalized entropy",
                ha="center",
                fontsize=9,
            )
            fig.subplots_adjust(left=0.01, right=0.995, top=0.90, bottom=0.06, wspace=0.06, hspace=0.36)
            out_name = "comparison__{}__{}.png".format(case_name, "cam" if mode_name == "cam" else "feature_response")
            fig.savefig(os.path.join(save_dir, out_name))
            plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(int(args.seed))
    specs = parse_model_specs(args.model_spec)
    node_names = resolve_node_names(args.nodes)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    _, selected_indices, loader = build_eval_loader(args)

    per_model_frames = []
    all_visuals: List[Dict] = []
    for spec in specs:
        frame, visuals = analyze_model(spec=spec, loader=loader, args=args, device=device, node_names=node_names)
        per_model_frames.append(frame)
        all_visuals.extend(visuals)
    per_case_df = pd.concat(per_model_frames, ignore_index=True)
    node_df = summarize_node_metrics(per_case_df)
    stage1_df = stage1_dynamics_from_csv(args.stage1_csv)
    dynamics_df = dynamics_from_node_summary(node_df=node_df, stage1_df=stage1_df)

    per_case_path = os.path.join(args.save_dir, "table_phase1_2_gradcam_per_case.csv")
    node_path = os.path.join(args.save_dir, "table_phase1_2_gradcam_node_summary.csv")
    dynamics_path = os.path.join(args.save_dir, "table_phase0_3_semantic_cam_dynamics.csv")
    stage1_path = os.path.join(args.save_dir, "table_phase0_stage1_semantic_dynamics.csv")
    summary_path = os.path.join(args.save_dir, "summary_phase0_3_phenomenon_cause_gradcam.json")

    per_case_df.to_csv(per_case_path, index=False)
    node_df.to_csv(node_path, index=False)
    dynamics_df.to_csv(dynamics_path, index=False)
    if not stage1_df.empty:
        stage1_df.to_csv(stage1_path, index=False)
    write_summary_json(
        args=args,
        specs=specs,
        selected_indices=selected_indices,
        node_df=node_df,
        dynamics_df=dynamics_df,
        out_path=summary_path,
    )

    plot_metric_curves(
        node_df=node_df,
        stage1_df=stage1_df,
        save_path=os.path.join(args.save_dir, "plot_phase0_2_semantic_cam_metric_curves.png"),
    )
    plot_dynamics(
        dynamics_df=dynamics_df,
        save_path=os.path.join(args.save_dir, "plot_phase0_3_semantic_cam_velocity_acceleration.png"),
    )
    if int(args.filmstrip_cases) > 0 or str(args.filmstrip_case_names).strip():
        selected_case_names = [item.strip() for item in str(args.filmstrip_case_names).split(",") if item.strip()]
        save_filmstrips(
            visuals=all_visuals,
            per_case_df=per_case_df,
            save_dir=os.path.join(args.save_dir, "filmstrips"),
            max_cases_per_model=int(args.filmstrip_cases),
            node_names=node_names,
            selected_case_names=selected_case_names,
            save_feature_filmstrips=bool(args.save_feature_filmstrips),
        )
    if bool(args.save_comparison_filmstrips):
        save_comparison_filmstrips(
            visuals=all_visuals,
            per_case_df=per_case_df,
            save_dir=os.path.join(args.save_dir, "comparison_filmstrips"),
            node_names=node_names,
            save_feature_filmstrips=bool(args.save_feature_filmstrips),
        )

    print("Saved per-case CAM metrics:", per_case_path)
    print("Saved node summary:", node_path)
    print("Saved dynamics:", dynamics_path)
    print("Saved summary:", summary_path)
    print("Device:", device)
    print("Selected cases:", len(selected_indices))
    print("Analyzed nodes:", ",".join(node_names))


if __name__ == "__main__":
    main()
