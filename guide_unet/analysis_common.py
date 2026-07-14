from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


GUIDE_ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(GUIDE_ROOT)
if GUIDE_ROOT not in sys.path:
    sys.path.insert(0, GUIDE_ROOT)

from Architecture.unet_baseline import UNetBaseline
from Architecture.unet_noskip import UNetNoSkip
from datasets.dataset_brats import BraTSDataset


MODEL_CHOICES = ("baseline", "noskip_unet", "skip_masked_baseline")
ANALYSIS_NODE_NAMES = ("down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4")
STAGE_NAMES = ("shallow", "deep_bridge", "mid_fusion", "late")
STAGE_TO_MODULES = {
    "shallow": ("down1", "down2"),
    "deep_bridge": ("down3", "down4"),
    "mid_fusion": ("up1", "up2", "up3"),
    "late": ("up4",),
}
NODE_GROUP_HINTS = {
    "down1": "shallow",
    "down2": "shallow",
    "down3": "deep_bridge",
    "down4": "deep_bridge",
    "up1": "mid_fusion",
    "up2": "mid_fusion",
    "up3": "mid_fusion",
    "up4": "late",
}
EDGE_NAMES = (
    "down1_to_down2",
    "down2_to_down3",
    "down3_to_down4",
    "down4_to_up1",
    "up1_to_up2",
    "up2_to_up3",
    "up3_to_up4",
)
EDGE_TO_MODULES = {
    "down1_to_down2": {"source_node_name": "down1", "target_node_name": "down2"},
    "down2_to_down3": {"source_node_name": "down2", "target_node_name": "down3"},
    "down3_to_down4": {"source_node_name": "down3", "target_node_name": "down4"},
    "down4_to_up1": {"source_node_name": "down4", "target_node_name": "up1"},
    "up1_to_up2": {"source_node_name": "up1", "target_node_name": "up2"},
    "up2_to_up3": {"source_node_name": "up2", "target_node_name": "up3"},
    "up3_to_up4": {"source_node_name": "up3", "target_node_name": "up4"},
}
REGION_TO_CLASSES = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": (3,),
}


def default_dir(name: str) -> str:
    return os.path.join(WORKSPACE_ROOT, name)


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))


def patient_id_from_case_name(case_name: str) -> str:
    parts = str(case_name).split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return str(case_name)


def _normalize_hook_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    raise TypeError("Expected hook output to contain a tensor.")


def unpack_logits(model_outputs):
    if isinstance(model_outputs, (tuple, list)) and len(model_outputs) >= 1:
        return model_outputs[0]
    raise TypeError("Unexpected model outputs; expected tuple/list with logits in the first slot.")


def load_state_dict_compatible(ckpt_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    payload = torch.load(ckpt_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict):
        return payload
    raise TypeError("Unsupported checkpoint payload type: {}".format(type(payload)))


def build_model(model_name: str, n_channels: int = 4, n_classes: int = 4, skip_mask_layers: str = "all") -> nn.Module:
    if model_name == "baseline":
        return UNetBaseline(n_channels=n_channels, n_classes=n_classes)
    if model_name == "noskip_unet":
        return UNetNoSkip(n_channels=n_channels, n_classes=n_classes)
    if model_name == "skip_masked_baseline":
        return UNetBaseline(
            n_channels=n_channels,
            n_classes=n_classes,
            skip_mask_mode="zero",
            skip_mask_layers=skip_mask_layers,
        )
    raise ValueError("Unsupported model_name: {}".format(model_name))


def load_model(model_name: str, ckpt_path: str, device: torch.device, n_channels: int = 4, n_classes: int = 4):
    model = build_model(model_name=model_name, n_channels=n_channels, n_classes=n_classes).to(device)
    state_dict = load_state_dict_compatible(ckpt_path=ckpt_path, device=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


class FeatureStore:
    def __init__(self, model: nn.Module):
        self.cache: Dict[str, torch.Tensor] = {}
        self._handles = []
        for module_name in ("inc", "down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4"):
            module = getattr(model, module_name)
            self._handles.append(module.register_forward_hook(self._make_hook(module_name)))

    def _make_hook(self, module_name: str):
        def hook_fn(module, inputs, output):
            self.cache[module_name] = _normalize_hook_output(output)

        return hook_fn

    def clear(self):
        self.cache = {}

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def node_map(self) -> Dict[str, torch.Tensor]:
        return {node_name: self.cache[node_name] for node_name in ANALYSIS_NODE_NAMES}


class StagePerturbationHook:
    def __init__(self, mode: str, value: float, seed: int = 0):
        self.mode = str(mode)
        self.value = float(value)
        self.seed = int(seed)
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))
        self._cached_mask = None
        self._cached_shape = None

    def _apply_to_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "scale":
            return x * float(self.value)

        if self.mode == "channel_drop":
            drop_ratio = min(max(float(self.value), 0.0), 1.0)
            channel_count = int(x.shape[1])
            keep_count = max(1, int(round(channel_count * (1.0 - drop_ratio))))
            shape_key = (channel_count, x.device.type, str(x.device))
            if self._cached_mask is None or self._cached_shape != shape_key:
                perm = torch.randperm(channel_count, generator=self.generator)
                keep_idx = perm[:keep_count]
                mask = torch.zeros(channel_count, dtype=x.dtype)
                mask[keep_idx] = 1.0
                self._cached_mask = mask
                self._cached_shape = shape_key
            mask = self._cached_mask.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
            return x * mask

        raise ValueError("Unsupported perturbation mode: {}".format(self.mode))

    def __call__(self, module, inputs, output):
        if torch.is_tensor(output):
            return self._apply_to_tensor(output)
        if isinstance(output, tuple):
            items = list(output)
            for idx, item in enumerate(items):
                if torch.is_tensor(item):
                    items[idx] = self._apply_to_tensor(item)
                    return tuple(items)
        if isinstance(output, list):
            items = list(output)
            for idx, item in enumerate(items):
                if torch.is_tensor(item):
                    items[idx] = self._apply_to_tensor(item)
                    return items
        raise TypeError("Unsupported hook output type for perturbation.")


def select_indices(
    dataset: BraTSDataset,
    num_samples: int,
    foreground_only: bool,
    seed: int = 0,
    patient_balanced: bool = False,
) -> List[int]:
    eligible_indices: List[int] = []
    per_patient_indices: Dict[str, List[int]] = {}

    for idx in range(len(dataset)):
        sample = dataset[idx]
        if foreground_only and int(sample["label"].sum().item()) == 0:
            continue
        eligible_indices.append(idx)
        if patient_balanced:
            patient_id = patient_id_from_case_name(sample["case_name"])
            per_patient_indices.setdefault(patient_id, []).append(idx)

    if int(num_samples) <= 0 or int(num_samples) >= len(eligible_indices):
        return eligible_indices

    rng = np.random.default_rng(int(seed))
    if not patient_balanced:
        shuffled = list(eligible_indices)
        rng.shuffle(shuffled)
        return shuffled[: int(num_samples)]

    patient_ids = list(per_patient_indices.keys())
    rng.shuffle(patient_ids)
    patient_queues = {patient_id: list(indices) for patient_id, indices in per_patient_indices.items()}
    for patient_id in patient_ids:
        rng.shuffle(patient_queues[patient_id])

    selected: List[int] = []
    while len(selected) < int(num_samples):
        progressed = False
        for patient_id in patient_ids:
            queue = patient_queues[patient_id]
            if not queue:
                continue
            selected.append(queue.pop())
            progressed = True
            if len(selected) >= int(num_samples):
                break
        if not progressed:
            break
    return selected


def split_indices(indices: Sequence[int], fit_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    if len(indices) < 2:
        raise ValueError("Need at least 2 indices to split fit/eval within the same split.")
    fit_ratio = float(fit_ratio)
    if not 0.0 < fit_ratio < 1.0:
        raise ValueError("fit_ratio must be within (0, 1).")
    rng = np.random.default_rng(int(seed))
    shuffled = list(indices)
    rng.shuffle(shuffled)
    fit_count = min(max(1, int(round(len(shuffled) * fit_ratio))), len(shuffled) - 1)
    return shuffled[:fit_count], shuffled[fit_count:]


def build_loader(dataset: BraTSDataset, indices: Sequence[int], batch_size: int, shuffle: bool, num_workers: int):
    subset = Subset(dataset, list(indices))
    return DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def prepare_fit_eval_loaders(
    data_dir: str,
    fit_split: str,
    eval_split: str,
    num_fit_samples: int,
    num_eval_samples: int,
    foreground_only: bool,
    seed: int,
    fit_ratio: float,
    batch_size: int,
    num_workers: int,
    patient_balanced: bool = False,
):
    fit_dataset = BraTSDataset(base_dir=data_dir, split=fit_split)
    eval_dataset = fit_dataset if fit_split == eval_split else BraTSDataset(base_dir=data_dir, split=eval_split)

    if fit_split == eval_split:
        base_indices = select_indices(
            dataset=fit_dataset,
            num_samples=max(int(num_fit_samples), int(num_eval_samples)) if (int(num_fit_samples) > 0 and int(num_eval_samples) > 0) else -1,
            foreground_only=bool(foreground_only),
            seed=int(seed),
            patient_balanced=bool(patient_balanced),
        )
        fit_indices, eval_indices = split_indices(indices=base_indices, fit_ratio=float(fit_ratio), seed=int(seed))
        if int(num_fit_samples) > 0:
            fit_indices = fit_indices[: int(num_fit_samples)]
        if int(num_eval_samples) > 0:
            eval_indices = eval_indices[: int(num_eval_samples)]
    else:
        fit_indices = select_indices(
            dataset=fit_dataset,
            num_samples=int(num_fit_samples),
            foreground_only=bool(foreground_only),
            seed=int(seed),
            patient_balanced=bool(patient_balanced),
        )
        eval_indices = select_indices(
            dataset=eval_dataset,
            num_samples=int(num_eval_samples),
            foreground_only=bool(foreground_only),
            seed=int(seed) + 1,
            patient_balanced=bool(patient_balanced),
        )

    fit_loader = build_loader(
        dataset=fit_dataset,
        indices=fit_indices,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
    )
    eval_loader = build_loader(
        dataset=eval_dataset,
        indices=eval_indices,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
    )
    return {
        "fit_dataset": fit_dataset,
        "eval_dataset": eval_dataset,
        "fit_indices": fit_indices,
        "eval_indices": eval_indices,
        "fit_loader": fit_loader,
        "eval_loader": eval_loader,
    }


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std < 1e-8 or b_std < 1e-8:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def safe_cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denom)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    inter = int(np.logical_and(a_bool, b_bool).sum())
    union = int(np.logical_or(a_bool, b_bool).sum())
    return float(inter / union) if union > 0 else 1.0


def mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    inter = int(np.logical_and(a_bool, b_bool).sum())
    total = int(a_bool.sum() + b_bool.sum())
    return float((2 * inter) / total) if total > 0 else 1.0


def build_region_mask(mask: np.ndarray, region_name: str) -> np.ndarray:
    mask = mask.astype(np.uint8)
    classes = REGION_TO_CLASSES[str(region_name)]
    region_mask = np.zeros_like(mask, dtype=np.uint8)
    for class_id in classes:
        region_mask = np.logical_or(region_mask, mask == int(class_id))
    return region_mask.astype(np.uint8)


def region_dice(pred_mask: np.ndarray, label_mask: np.ndarray, region_name: str) -> float:
    return mask_dice(build_region_mask(pred_mask, region_name), build_region_mask(label_mask, region_name))


def mean_region_dice(pred_mask: np.ndarray, label_mask: np.ndarray) -> float:
    return float(np.mean([region_dice(pred_mask, label_mask, region_name) for region_name in REGION_TO_CLASSES]))


def class_transition_counts(source_mask: np.ndarray, target_mask: np.ndarray, num_classes: int = 4) -> np.ndarray:
    source_flat = source_mask.reshape(-1).astype(np.int64)
    target_flat = target_mask.reshape(-1).astype(np.int64)
    valid = (
        (source_flat >= 0)
        & (source_flat < int(num_classes))
        & (target_flat >= 0)
        & (target_flat < int(num_classes))
    )
    counts = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    if not np.any(valid):
        return counts
    source_valid = source_flat[valid]
    target_valid = target_flat[valid]
    encoded = source_valid * int(num_classes) + target_valid
    bincount = np.bincount(encoded, minlength=int(num_classes) * int(num_classes))
    return bincount.reshape(int(num_classes), int(num_classes))


def foreground_transition_rate(source_mask: np.ndarray, target_mask: np.ndarray) -> float:
    source_fg = source_mask.reshape(-1) > 0
    target_fg = target_mask.reshape(-1) > 0
    total = int(source_fg.size)
    if total == 0:
        return 0.0
    changed = np.not_equal(source_fg, target_fg).sum()
    return float(changed / total)


def soft_dice_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 4, eps: float = 1e-5):
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(labels.long(), num_classes=int(num_classes)).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = torch.sum(probs * one_hot, dim=dims)
    denom = torch.sum(probs + one_hot, dim=dims)
    dice = (2.0 * inter + float(eps)) / (denom + float(eps))
    return 1.0 - dice.mean()


@dataclass
class StageConfig:
    name: str
    module_names: Tuple[str, ...]


def iter_stage_configs() -> Iterable[StageConfig]:
    for stage_name in STAGE_NAMES:
        yield StageConfig(name=stage_name, module_names=tuple(STAGE_TO_MODULES[stage_name]))


@dataclass
class NodeConfig:
    name: str
    group_hint: str


def iter_node_configs() -> Iterable[NodeConfig]:
    for node_name in ANALYSIS_NODE_NAMES:
        yield NodeConfig(name=node_name, group_hint=str(NODE_GROUP_HINTS[node_name]))


def node_group_hint(node_name: str) -> str:
    return str(NODE_GROUP_HINTS[str(node_name)])


@dataclass
class EdgeConfig:
    name: str
    source_node_name: str
    target_node_name: str
    source_group_hint: str
    target_group_hint: str


def iter_edge_configs() -> Iterable[EdgeConfig]:
    for edge_name in EDGE_NAMES:
        payload = EDGE_TO_MODULES[edge_name]
        source_node_name = str(payload["source_node_name"])
        target_node_name = str(payload["target_node_name"])
        yield EdgeConfig(
            name=edge_name,
            source_node_name=source_node_name,
            target_node_name=target_node_name,
            source_group_hint=node_group_hint(source_node_name),
            target_group_hint=node_group_hint(target_node_name),
        )


def iter_transfer_configs() -> Iterable[EdgeConfig]:
    return iter_edge_configs()
