from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


GUIDE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(GUIDE_ROOT)
if GUIDE_ROOT not in sys.path:
    sys.path.insert(0, GUIDE_ROOT)

from analysis_common import (  # noqa: E402
    ANALYSIS_NODE_NAMES,
    MODEL_CHOICES,
    FeatureStore,
    build_region_mask,
    load_model,
    mean_region_dice,
    patient_id_from_case_name,
    safe_corr,
    safe_cos,
    select_indices,
    set_seed,
    unpack_logits,
)
from datasets.dataset_brats import BraTSDataset  # noqa: E402


NETWORK_ORDER = tuple(ANALYSIS_NODE_NAMES)
DECODER_NODES = ("up1", "up2", "up3", "up4")
TRANSITIONS = tuple(zip(NETWORK_ORDER[:-1], NETWORK_ORDER[1:]))
MODEL_ALIASES = {"noskip": "noskip_unet", "no_skip": "noskip_unet"}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    ckpt_path: str


class GradCAMStore:
    def __init__(self, model: torch.nn.Module, node_names: Sequence[str]):
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

            if tensor.requires_grad:
                tensor.register_hook(grad_hook)

        return hook_fn

    def clear(self) -> None:
        self.activations = {}
        self.gradients = {}

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class SpatialMaskPerturbationHook:
    def __init__(self, mask: torch.Tensor, gamma: float):
        if mask.ndim != 3:
            raise ValueError("mask must have shape [B,H,W], got {}".format(tuple(mask.shape)))
        self.mask = mask.detach()
        self.gamma = float(gamma)
        self.last_output: Optional[torch.Tensor] = None

    def _apply(self, tensor: torch.Tensor) -> torch.Tensor:
        mask = self.mask.to(device=tensor.device, dtype=tensor.dtype).unsqueeze(1)
        if tuple(mask.shape[-2:]) != tuple(tensor.shape[-2:]):
            mask = F.interpolate(mask, size=tuple(tensor.shape[-2:]), mode="nearest")
        out = tensor * (1.0 - float(self.gamma) * mask)
        self.last_output = out
        return out

    def __call__(self, module, inputs, output):
        if torch.is_tensor(output):
            return self._apply(output)
        if isinstance(output, tuple):
            items = list(output)
            for idx, item in enumerate(items):
                if torch.is_tensor(item):
                    items[idx] = self._apply(item)
                    return tuple(items)
        if isinstance(output, list):
            items = list(output)
            for idx, item in enumerate(items):
                if torch.is_tensor(item):
                    items[idx] = self._apply(item)
                    return items
        raise TypeError("Unsupported hook output type for spatial perturbation")


def workspace_path(*parts: str) -> str:
    return os.path.join(WORKSPACE_ROOT, *parts)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_csv_list(text: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in str(text).split(",") if item.strip())


def parse_model_specs(items: Sequence[str]) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for item in items:
        if "=" not in str(item):
            raise ValueError("Expected model_name=/path/to/checkpoint, got {}".format(item))
        name, ckpt_path = str(item).split("=", 1)
        name = MODEL_ALIASES.get(name.strip(), name.strip())
        ckpt_path = ckpt_path.strip()
        if name not in MODEL_CHOICES:
            raise ValueError("Unsupported model name: {}".format(name))
        if not ckpt_path:
            raise ValueError("Empty checkpoint path for {}".format(name))
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        specs.append(ModelSpec(name=name, ckpt_path=ckpt_path))
    if not specs:
        raise ValueError("At least one model spec is required")
    return specs


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
        raise ValueError("Empty node list")
    return names


def resolve_device(device_arg: str) -> torch.device:
    if str(device_arg) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device_arg))


def write_json(payload: Dict, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_run_config(args, out_dir: str, extra: Optional[Dict] = None) -> None:
    payload = vars(args).copy() if hasattr(args, "__dict__") else dict(args)
    if extra:
        payload.update(extra)
    write_json(payload, os.path.join(out_dir, "run_config.json"))


def write_manifest(out_dir: str, rows: Sequence[Dict]) -> None:
    pd.DataFrame(list(rows)).to_csv(os.path.join(out_dir, "run_manifest.csv"), index=False)


def load_case_names_from_csv(path: str, limit: int = 0) -> List[str]:
    if not path:
        return []
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "case_name" not in df.columns:
        raise ValueError("Case-set CSV must contain case_name: {}".format(path))
    names: List[str] = []
    seen = set()
    for name in df["case_name"].astype(str).tolist():
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
        if int(limit) > 0 and len(names) >= int(limit):
            break
    return names


def dataset_indices_from_case_names(dataset: BraTSDataset, case_names: Sequence[str]) -> List[int]:
    by_name = {str(name): idx for idx, name in enumerate(dataset.sample_list)}
    missing = [str(name) for name in case_names if str(name) not in by_name]
    if missing:
        raise ValueError("Missing {} requested cases, first: {}".format(len(missing), missing[:5]))
    return [int(by_name[str(name)]) for name in case_names]


def build_eval_loader(
    data_dir: str,
    split: str,
    batch_size: int,
    num_workers: int,
    num_cases: int,
    seed: int,
    foreground_only: bool,
    case_set_csv: str = "",
    case_names: Sequence[str] = (),
    patient_balanced: bool = False,
) -> Tuple[BraTSDataset, List[int], List[str], DataLoader]:
    dataset = BraTSDataset(base_dir=data_dir, split=split)
    selected_names = list(case_names) or load_case_names_from_csv(case_set_csv, limit=int(num_cases))
    if selected_names:
        indices = dataset_indices_from_case_names(dataset, selected_names)
        if int(num_cases) > 0:
            indices = indices[: int(num_cases)]
    else:
        indices = select_indices(
            dataset=dataset,
            num_samples=int(num_cases),
            foreground_only=bool(foreground_only),
            seed=int(seed),
            patient_balanced=bool(patient_balanced),
        )
    names = [str(dataset.sample_list[idx]) for idx in indices]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, indices, names, loader


def tumor_prob_map(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 1:, :, :].sum(dim=1)


def tumor_logit_map(logits: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(logits[:, 1:, :, :], dim=1)


def target_mask_from_logits(logits: torch.Tensor, labels: torch.Tensor, mode: str) -> torch.Tensor:
    mode = str(mode)
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
    return ((tumor_logits * mask).flatten(1).sum(dim=1) / denom).mean()


def normalize_tensor_map(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(1)
    min_v = flat.min(dim=1).values.view(-1, 1, 1)
    max_v = flat.max(dim=1).values.view(-1, 1, 1)
    return (x - min_v) / (max_v - min_v + 1e-8)


def resize_map(x: torch.Tensor, output_size: Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(output_size):
        return x
    return F.interpolate(x.unsqueeze(1), size=tuple(output_size), mode=mode, align_corners=False if mode == "bilinear" else None).squeeze(1)


def feature_response_from_activation(
    activation: torch.Tensor,
    aggregation: str,
    topk_channel_fraction: float = 0.10,
) -> torch.Tensor:
    x = activation.detach().float()
    aggregation = str(aggregation)
    if aggregation == "l2":
        response = torch.sqrt(torch.mean(x ** 2, dim=1).clamp_min(0.0))
    elif aggregation == "mean_abs":
        response = torch.mean(torch.abs(x), dim=1)
    elif aggregation == "max_abs":
        response = torch.max(torch.abs(x), dim=1).values
    elif aggregation == "positive_only":
        response = torch.mean(torch.relu(x), dim=1)
    elif aggregation == "topk_channel":
        bsz, channels = int(x.shape[0]), int(x.shape[1])
        k = max(1, int(round(float(topk_channel_fraction) * channels)))
        energy = torch.mean(torch.abs(x), dim=(2, 3))
        idx = torch.topk(energy, k=k, dim=1).indices
        rows = []
        for batch_idx in range(bsz):
            rows.append(torch.mean(torch.abs(x[batch_idx, idx[batch_idx], :, :]), dim=0))
        response = torch.stack(rows, dim=0)
    else:
        raise ValueError("Unsupported aggregation: {}".format(aggregation))
    return normalize_tensor_map(response)


def compute_cam_variant_native(store: GradCAMStore, node_name: str, variant: str) -> torch.Tensor:
    activation = store.activations[node_name]
    gradient = store.gradients[node_name]
    variant = str(variant)
    if variant == "gradcam":
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1))
    elif variant in ("hirescam", "hirescam_elementwise"):
        cam = torch.relu((gradient * activation).sum(dim=1))
    elif variant == "gradcam_abs":
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.abs((weights * activation).sum(dim=1))
    else:
        raise ValueError("Unsupported CAM variant: {}".format(variant))
    return normalize_tensor_map(cam.detach())


def normalized_entropy(arr: np.ndarray) -> float:
    values = np.maximum(np.asarray(arr, dtype=np.float64).reshape(-1), 0.0)
    total = float(values.sum())
    if total <= 1e-12:
        return 0.0
    p = values / total
    entropy = -float(np.sum(p * np.log(p + 1e-12)))
    return float(entropy / np.log(max(len(p), 2)))


def mass_fraction(arr: np.ndarray, mask: np.ndarray) -> float:
    values = np.maximum(np.asarray(arr, dtype=np.float64), 0.0)
    total = float(values.sum())
    if total <= 1e-12:
        return 0.0
    return float(values[np.asarray(mask).astype(bool)].sum() / total)


def fg_bg_contrast(arr: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(arr, dtype=np.float64)
    mask_bool = np.asarray(mask).astype(bool)
    if not np.any(mask_bool):
        return 0.0
    fg_mean = float(values[mask_bool].mean())
    bg = values[~mask_bool]
    bg_mean = float(bg.mean()) if bg.size else 0.0
    return float((fg_mean - bg_mean) / (fg_mean + bg_mean + 1e-8))


def top_fraction_mask(arr: np.ndarray, fraction: float, largest: bool = True) -> np.ndarray:
    flat = np.asarray(arr).reshape(-1)
    k = max(1, int(round(float(fraction) * flat.size)))
    if largest:
        idx = np.argpartition(flat, -k)[-k:]
    else:
        idx = np.argpartition(flat, k - 1)[:k]
    mask = np.zeros(flat.size, dtype=bool)
    mask[idx] = True
    return mask.reshape(np.asarray(arr).shape)


def top_fraction_in_mask(arr: np.ndarray, mask: np.ndarray, fraction: float = 0.10) -> float:
    top_mask = top_fraction_mask(arr, fraction=fraction, largest=True)
    return float(np.asarray(mask).astype(bool)[top_mask].mean())


def mask_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = np.asarray(a).astype(bool)
    b_bool = np.asarray(b).astype(bool)
    union = np.logical_or(a_bool, b_bool).sum()
    if int(union) == 0:
        return 1.0
    inter = np.logical_and(a_bool, b_bool).sum()
    return float(inter / union)


def map_metric_row(response: np.ndarray, final_prob: np.ndarray, label: np.ndarray, pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    gt_fg = label > 0
    pred_fg = pred > 0
    row = {
        "{}final_prob_pearson".format(prefix): float(safe_corr(response, final_prob)),
        "{}final_prob_cosine".format(prefix): float(safe_cos(response, final_prob)),
        "{}gt_mass_fraction".format(prefix): float(mass_fraction(response, gt_fg)),
        "{}pred_mass_fraction".format(prefix): float(mass_fraction(response, pred_fg)),
        "{}gt_fg_bg_contrast".format(prefix): float(fg_bg_contrast(response, gt_fg)),
        "{}pred_fg_bg_contrast".format(prefix): float(fg_bg_contrast(response, pred_fg)),
        "{}entropy_norm".format(prefix): float(normalized_entropy(response)),
        "{}top10_gt_fraction".format(prefix): float(top_fraction_in_mask(response, gt_fg, fraction=0.10)),
        "{}gt_wt_mass_fraction".format(prefix): float(mass_fraction(response, build_region_mask(label, "WT"))),
        "{}gt_tc_mass_fraction".format(prefix): float(mass_fraction(response, build_region_mask(label, "TC"))),
        "{}gt_et_mass_fraction".format(prefix): float(mass_fraction(response, build_region_mask(label, "ET"))),
    }
    return row


def bootstrap_mean_ci(values: Sequence[float], n_boot: int, seed: int) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2 or int(n_boot) <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    means = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, arr.size, size=arr.size)
        means.append(float(arr[idx].mean()))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def finite_pair(x: Sequence[float], y: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def corr_value(x: Sequence[float], y: Sequence[float], method: str = "pearson") -> float:
    a, b = finite_pair(x, y)
    if a.size < 3 or float(np.std(a)) < 1e-10 or float(np.std(b)) < 1e-10:
        return np.nan
    if method == "pearson":
        return float(np.corrcoef(a, b)[0, 1])
    if method == "spearman":
        return float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    raise ValueError(method)


def bootstrap_corr_ci(x: Sequence[float], y: Sequence[float], n_boot: int, seed: int, method: str = "pearson") -> Tuple[float, float]:
    a, b = finite_pair(x, y)
    if a.size < 6 or int(n_boot) <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    vals: List[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, a.size, size=a.size)
        val = corr_value(a[idx], b[idx], method=method)
        if np.isfinite(val):
            vals.append(float(val))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def add_node_order(df: pd.DataFrame, node_col: str = "node") -> pd.DataFrame:
    out = df.copy()
    order = {node: idx for idx, node in enumerate(NETWORK_ORDER)}
    out["node_order"] = out[node_col].astype(str).map(order).astype(int)
    return out


def transition_delta_table(df: pd.DataFrame, value_cols: Sequence[str], group_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    for group_key, group in df.groupby(list(group_cols), sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_payload = dict(zip(group_cols, group_key))
        pivot = group.set_index("node")
        available_nodes = [node for node in NETWORK_ORDER if node in set(pivot.index.astype(str))]
        if len(available_nodes) < 2:
            continue
        for src, dst in zip(available_nodes[:-1], available_nodes[1:]):
            row = dict(group_payload)
            row.update(
                {
                    "transition": "{}->{}".format(src, dst),
                    "source_node": src,
                    "target_node": dst,
                    "source_order": int(NETWORK_ORDER.index(src)),
                }
            )
            for col in value_cols:
                row["{}_source".format(col)] = float(pivot.loc[src, col])
                row["{}_target".format(col)] = float(pivot.loc[dst, col])
                row["{}_delta".format(col)] = float(pivot.loc[dst, col] - pivot.loc[src, col])
            rows.append(row)
    return pd.DataFrame(rows)


def compute_prediction_metrics(logits: torch.Tensor, labels: torch.Tensor, reference_pred: Optional[torch.Tensor] = None) -> List[Dict]:
    probs = tumor_prob_map(logits).detach().cpu().numpy()
    pred = torch.argmax(logits.detach(), dim=1).cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    ref_pred_np = reference_pred.detach().cpu().numpy() if reference_pred is not None else pred
    rows: List[Dict] = []
    for idx in range(int(logits.shape[0])):
        label = labels_np[idx]
        gt_fg = label > 0
        ref_fg = ref_pred_np[idx] > 0
        final_prob = probs[idx]
        rows.append(
            {
                "mean_dice": float(mean_region_dice(pred[idx], label)),
                "gt_fg_prob": float(final_prob[gt_fg].mean()) if np.any(gt_fg) else 0.0,
                "pred_fg_prob": float(final_prob[ref_fg].mean()) if np.any(ref_fg) else 0.0,
                "whole_tumor_prob_mean": float(final_prob.mean()),
                "pred_fg_pixels": int(ref_fg.sum()),
                "gt_fg_pixels": int(gt_fg.sum()),
            }
        )
    return rows
