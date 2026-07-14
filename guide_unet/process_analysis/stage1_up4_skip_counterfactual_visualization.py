from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.nn.functional as F


GUIDE_ROOT = Path(__file__).resolve().parents[1]
if str(GUIDE_ROOT) not in sys.path:
    sys.path.insert(0, str(GUIDE_ROOT))

from datasets.dataset_brats import BraTSDataset  # noqa: E402
from process_analysis.stage1_explanation_utils import (  # noqa: E402
    compute_prediction_metrics,
    dataset_indices_from_case_names,
    feature_response_from_activation,
    load_model,
    resize_map,
    tumor_prob_map,
    unpack_logits,
)
from process_analysis.stage1_up4_skip_counterfactual_mediation import (  # noqa: E402
    Up4SkipCounterfactualHook,
    boundary_band,
    binary_morph,
    sample_same_area,
)


DEFAULT_BASELINE_CKPT = (
    "checkpoints/baseline_brats2023_scratch100_seed42/"
    "baseline/seed_42/baseline_seed42_best_val_loss.pth"
)
DEFAULT_CASE_SET = "stage1_explanation_suite_20260517/results/common_case_sets/formal_n512_cases.csv"
DEFAULT_OUTPUT_ROOT = "stage1_explanation_suite_20260517/results/e13_up4_skip_counterfactual_mediation_formal_n512/visuals"

CLASS_COLORS = {
    1: np.array([0.86, 0.08, 0.12], dtype=np.float32),
    2: np.array([0.10, 0.70, 0.22], dtype=np.float32),
    3: np.array([1.00, 0.78, 0.08], dtype=np.float32),
}
VARIANT_LABEL = {
    "boundary_cross": "boundary cross",
    "boundary_background_control": "boundary bg control",
    "boundary_random_control": "boundary random",
    "interior_cross": "interior cross",
    "interior_background_control": "interior bg control",
}
PAPER_LABEL = {
    "boundary_cross": "Boundary CF",
    "boundary_background_control": "Boundary bg control",
    "boundary_random_control": "Boundary random",
    "interior_cross": "Interior CF",
    "interior_background_control": "Interior bg control",
}
PLOT_VARIANTS = (
    "boundary_cross",
    "boundary_background_control",
    "boundary_random_control",
    "interior_cross",
    "interior_background_control",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate E13 up4-skip counterfactual visualization panels.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--case-name", type=str, default="BraTS-GLI-00568-000_78")
    parser.add_argument("--source-case-name", type=str, default="")
    parser.add_argument("--case-set-csv", type=str, default=DEFAULT_CASE_SET)
    parser.add_argument("--baseline-ckpt", type=str, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--zoom-pad", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if str(device_arg) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device_arg))


def image_channel_for_display(image_chw: np.ndarray) -> np.ndarray:
    channel = image_chw[0].astype(np.float32)
    lo, hi = np.percentile(channel, [1, 99])
    return np.clip((channel - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def overlay_label(base: np.ndarray, label: np.ndarray, alpha: float = 0.58) -> np.ndarray:
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    for class_id, color in CLASS_COLORS.items():
        mask = np.asarray(label) == int(class_id)
        if np.any(mask):
            rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color
    return np.clip(rgb, 0.0, 1.0)


def overlay_binary_mask(base: np.ndarray, mask: np.ndarray, color: Tuple[float, float, float], alpha: float = 0.72) -> np.ndarray:
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    mask_bool = np.asarray(mask).astype(bool)
    if np.any(mask_bool):
        color_arr = np.asarray(color, dtype=np.float32)
        rgb[mask_bool] = (1.0 - alpha) * rgb[mask_bool] + alpha * color_arr
    return np.clip(rgb, 0.0, 1.0)


def overlay_heat(base: np.ndarray, heat: np.ndarray, alpha: float = 0.52, cmap_name: str = "magma") -> np.ndarray:
    heat = np.asarray(heat, dtype=np.float32)
    heat = np.clip(heat, 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    color = cmap(heat)[..., :3]
    gray = np.stack([base, base, base], axis=-1)
    return np.clip((1.0 - alpha) * gray + alpha * color, 0.0, 1.0)


def crop(arr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return np.asarray(arr)[y0:y1, x0:x1]


def tumor_bbox(label: np.ndarray, pad: int) -> Tuple[int, int, int, int]:
    mask = np.asarray(label) > 0
    if not np.any(mask):
        return 0, label.shape[0], 0, label.shape[1]
    ys, xs = np.where(mask)
    y0 = max(0, int(ys.min()) - int(pad))
    y1 = min(label.shape[0], int(ys.max()) + int(pad) + 1)
    x0 = max(0, int(xs.min()) - int(pad))
    x1 = min(label.shape[1], int(xs.max()) + int(pad) + 1)
    return y0, y1, x0, x1


def add_bbox(ax, bbox: Tuple[int, int, int, int], color: str = "white") -> None:
    y0, y1, x0, x1 = bbox
    rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.4)
    ax.add_patch(rect)


def contour_mask(ax, mask: np.ndarray, bbox: Tuple[int, int, int, int], color: str, linewidth: float = 1.6) -> None:
    cropped = crop(mask, bbox).astype(float)
    if np.min(cropped) < 0.5 < np.max(cropped):
        ax.contour(cropped, levels=[0.5], colors=[color], linewidths=linewidth)


def plot_base(ax, base: np.ndarray, bbox: Tuple[int, int, int, int], title: str = "") -> None:
    ax.imshow(crop(base, bbox), cmap="gray", vmin=0.0, vmax=1.0)
    if title:
        ax.set_title(title, fontsize=9.5, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_heat_overlay(
    ax,
    base: np.ndarray,
    heat: np.ndarray,
    bbox: Tuple[int, int, int, int],
    title: str,
    cmap: str,
    alpha: float,
    vmin: float,
    vmax: float,
):
    plot_base(ax, base, bbox)
    image = ax.imshow(crop(heat, bbox), cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8.4, pad=3)
    return image


def load_cases(args: argparse.Namespace, device: torch.device):
    dataset = BraTSDataset(base_dir=args.data_dir, split=args.split)
    target_idx = dataset_indices_from_case_names(dataset, [args.case_name])[0]
    source_name = args.source_case_name.strip() or choose_source_case(dataset, args.case_name, args.case_set_csv)
    source_idx = dataset_indices_from_case_names(dataset, [source_name])[0]
    target = dataset[target_idx]
    source = dataset[source_idx]
    image = torch.stack([target["image"], source["image"]], dim=0).to(device)
    label = torch.stack([target["label"], source["label"]], dim=0).to(device)
    base = image_channel_for_display(target["image"].numpy())
    label_np = target["label"].numpy()
    return image, label, base, label_np, source_name


def choose_source_case(dataset: BraTSDataset, target_name: str, case_set_csv: str) -> str:
    candidate_names: List[str] = []
    if case_set_csv and os.path.isfile(case_set_csv):
        with open(case_set_csv, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            candidate_names = [row["case_name"] for row in reader if row.get("case_name")]
    if not candidate_names:
        candidate_names = list(dataset.sample_list)
    for name in candidate_names:
        if str(name) == str(target_name):
            continue
        idx = dataset_indices_from_case_names(dataset, [str(name)])[0]
        sample = dataset[idx]
        label = sample["label"].numpy()
        if np.any(label > 0):
            return str(name)
    raise RuntimeError("Could not find a source case with foreground")


def variant_masks(label: np.ndarray, radius: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    tumor = np.asarray(label) > 0
    boundary = boundary_band(tumor, radius=int(radius))
    interior = binary_morph(tumor, radius=int(radius), op="erosion")
    if not np.any(interior):
        interior = tumor.copy()
    background = np.asarray(label) == 0
    all_pixels = np.ones_like(tumor, dtype=bool)
    boundary_k = int(boundary.sum())
    interior_k = int(interior.sum())
    return {
        "boundary_cross": boundary,
        "boundary_background_control": sample_same_area(background, boundary_k, rng),
        "boundary_random_control": sample_same_area(all_pixels, boundary_k, rng),
        "interior_cross": interior,
        "interior_background_control": sample_same_area(background, interior_k, rng),
    }


class Up4Capture:
    def __init__(self, model: torch.nn.Module):
        self.skip_input: torch.Tensor | None = None
        self.conv_output: torch.Tensor | None = None
        self.handles = [
            model.up4.register_forward_pre_hook(self._pre_hook),
            model.up4.register_forward_hook(self._post_hook),
        ]

    def _pre_hook(self, module, inputs):
        self.skip_input = inputs[1].detach()

    def _post_hook(self, module, inputs, output):
        self.conv_output = output.detach()

    def clear(self) -> None:
        self.skip_input = None
        self.conv_output = None

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def map_from_tensor(tensor: torch.Tensor, output_size: Tuple[int, int], index: int = 0) -> np.ndarray:
    response = feature_response_from_activation(tensor, aggregation="l2")
    response = resize_map(response, output_size=output_size, mode="bilinear")
    return response.detach().cpu().numpy()[index]


def run_variant(model, capture: Up4Capture, image, label, mask_np: np.ndarray | None, variant: str | None):
    if mask_np is None:
        capture.clear()
        logits = unpack_logits(model(image))
        return logits, capture.skip_input.detach(), capture.conv_output.detach()
    mask_batch = np.stack([mask_np, mask_np], axis=0).astype(np.float32)
    mask_t = torch.from_numpy(mask_batch).to(device=image.device)
    hook = Up4SkipCounterfactualHook(mask=mask_t, variant=str(variant))
    handle = model.up4.register_forward_pre_hook(hook, prepend=True)
    capture.clear()
    logits = unpack_logits(model(image))
    handle.remove()
    return logits, capture.skip_input.detach(), capture.conv_output.detach()


def collect_visual_data(args: argparse.Namespace):
    device = resolve_device(args.device)
    baseline_ckpt = args.baseline_ckpt
    if not os.path.isfile(baseline_ckpt):
        raise FileNotFoundError(args.baseline_ckpt)
    image, label, base, label_np, source_name = load_cases(args, device)
    output_size = tuple(int(v) for v in label_np.shape)
    masks = variant_masks(label_np, radius=int(args.boundary_radius), seed=int(args.seed))
    model = load_model(model_name="baseline", ckpt_path=baseline_ckpt, device=device)
    capture = Up4Capture(model)

    with torch.no_grad():
        base_logits, base_skip, base_up4 = run_variant(model, capture, image, label, None, None)
        base_metrics = compute_prediction_metrics(base_logits[:1], label[:1])[0]
        base_prob = tumor_prob_map(base_logits).detach().cpu().numpy()[0]
        data = {
            "original": {
                "logits": base_logits.detach(),
                "skip_map": map_from_tensor(base_skip, output_size, index=0),
                "up4_map": map_from_tensor(base_up4, output_size, index=0),
                "prob_map": base_prob,
                "metrics": base_metrics,
            }
        }
        for variant, mask in masks.items():
            logits, skip, up4 = run_variant(model, capture, image, label, mask, variant)
            metrics = compute_prediction_metrics(logits[:1], label[:1], reference_pred=torch.argmax(base_logits[:1], dim=1))[0]
            prob = tumor_prob_map(logits).detach().cpu().numpy()[0]
            data[variant] = {
                "skip_map": map_from_tensor(skip, output_size, index=0),
                "up4_map": map_from_tensor(up4, output_size, index=0),
                "prob_map": prob,
                "metrics": metrics,
                "mask": mask,
            }
    capture.remove()
    return data, masks, base, label_np, source_name


def save_region_mask_figure(out_dir: Path, case_name: str, source_name: str, base, label, masks, bbox) -> None:
    panels = [
        ("MRI + label", overlay_label(base, label), None),
        ("boundary", overlay_binary_mask(base, masks["boundary_cross"], (0.00, 0.48, 0.95)), masks["boundary_cross"]),
        ("interior", overlay_binary_mask(base, masks["interior_cross"], (1.00, 0.55, 0.00)), masks["interior_cross"]),
        ("bg control", overlay_binary_mask(base, masks["boundary_background_control"], (0.30, 0.75, 0.30)), masks["boundary_background_control"]),
        ("random control", overlay_binary_mask(base, masks["boundary_random_control"], (0.70, 0.30, 0.95)), masks["boundary_random_control"]),
    ]
    fig, axes = plt.subplots(2, len(panels), figsize=(12.4, 5.2), dpi=220)
    for col, (title, image, _) in enumerate(panels):
        axes[0, col].imshow(image)
        add_bbox(axes[0, col], bbox)
        axes[0, col].set_title(title, fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(crop(image, bbox))
        axes[1, col].axis("off")
    fig.suptitle(f"{case_name} | replacement regions (source: {source_name})", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"fig_e13_region_replacement_masks_{case_name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"fig_e13_region_replacement_masks_{case_name}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_map_figure(
    out_dir: Path,
    case_name: str,
    data: Dict,
    base: np.ndarray,
    bbox: Tuple[int, int, int, int],
    map_key: str,
    file_stem: str,
    title: str,
    include_drop: bool,
) -> None:
    keys = ["original", "boundary_cross", "boundary_background_control", "boundary_random_control", "interior_cross", "interior_background_control"]
    fig, axes = plt.subplots(1, len(keys), figsize=(14.2, 2.9), dpi=240)
    for col, key in enumerate(keys):
        heat = data[key][map_key]
        image = overlay_heat(base, heat)
        axes[col].imshow(crop(image, bbox))
        if key == "original":
            panel_title = "original"
        else:
            label = VARIANT_LABEL.get(key, key)
            if include_drop:
                drop = data["original"]["metrics"]["pred_fg_prob"] - data[key]["metrics"]["pred_fg_prob"]
                panel_title = f"{label}\ndrop={drop:.3f}"
            else:
                panel_title = label
        axes[col].set_title(panel_title, fontsize=8.2)
        axes[col].axis("off")
    fig.suptitle(f"{case_name} | {title}", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_dir / f"{file_stem}_{case_name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{file_stem}_{case_name}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_paper_region_figure(out_dir: Path, case_name: str, source_name: str, base, label, masks, bbox) -> None:
    panels = [
        ("MRI + labels", overlay_label(base, label), None, "white"),
        ("Boundary mask", overlay_binary_mask(base, masks["boundary_cross"], (0.00, 0.48, 0.95)), masks["boundary_cross"], "#0077cc"),
        ("Interior mask", overlay_binary_mask(base, masks["interior_cross"], (1.00, 0.55, 0.00)), masks["interior_cross"], "#ff9900"),
    ]
    fig, axes = plt.subplots(2, len(panels), figsize=(6.9, 5.2), dpi=320)
    for col, (title, image, mask, color) in enumerate(panels):
        axes[0, col].imshow(image)
        add_bbox(axes[0, col], bbox, color="white")
        axes[0, col].set_title(title, fontsize=8.8, pad=3)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        axes[1, col].imshow(crop(image, bbox))
        if mask is not None:
            contour_mask(axes[1, col], mask, bbox, color=color, linewidth=1.5)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
    axes[0, 0].set_ylabel("Full", fontsize=8.8)
    axes[1, 0].set_ylabel("Zoom", fontsize=8.8)
    legend_items = [
        Patch(facecolor=CLASS_COLORS[2], edgecolor="none", label="ED"),
        Patch(facecolor=CLASS_COLORS[1], edgecolor="none", label="NCR/NET"),
        Patch(facecolor=CLASS_COLORS[3], edgecolor="none", label="ET"),
        Patch(facecolor="#0077cc", edgecolor="none", label="Boundary"),
        Patch(facecolor="#ff9900", edgecolor="none", label="Interior"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=5, frameon=False, fontsize=7.8, bbox_to_anchor=(0.5, -0.012))
    fig.tight_layout(rect=[0, 0.045, 1, 1], w_pad=0.45, h_pad=0.35)
    fig.savefig(out_dir / f"fig_e13_paper_replacement_regions_{case_name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"fig_e13_paper_replacement_regions_{case_name}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_paper_probability_figure(out_dir: Path, case_name: str, data: Dict, base: np.ndarray, label: np.ndarray, masks, bbox) -> None:
    rows = [
        ("Boundary", "boundary_cross", "boundary_background_control", "#0077cc"),
        ("Interior", "interior_cross", "interior_background_control", "#ff9900"),
    ]
    original_prob = data["original"]["prob_map"]
    loss_maps = [np.clip(original_prob - data[row[1]]["prob_map"], 0.0, 1.0) for row in rows]
    loss_vmax = float(max(0.10, np.percentile(np.concatenate([crop(m, bbox).ravel() for m in loss_maps]), 99.5)))
    fig, axes = plt.subplots(len(rows), 4, figsize=(8.9, 4.7), dpi=320)
    for row_idx, (row_label, cf_key, control_key, mask_color) in enumerate(rows):
        row_axes = axes[row_idx]
        plot_base(row_axes[0], base, bbox, title="Original")
        row_axes[0].imshow(crop(original_prob, bbox), cmap="magma", alpha=0.72, vmin=0.0, vmax=1.0)
        contour_mask(row_axes[0], label > 0, bbox, color="white", linewidth=1.2)

        drop = data["original"]["metrics"]["pred_fg_prob"] - data[cf_key]["metrics"]["pred_fg_prob"]
        plot_heat_overlay(
            row_axes[1],
            base,
            data[cf_key]["prob_map"],
            bbox,
            f"{PAPER_LABEL[cf_key]}\ndrop={drop:.3f}",
            "magma",
            0.72,
            0.0,
            1.0,
        )
        contour_mask(row_axes[1], masks[cf_key], bbox, color=mask_color, linewidth=1.5)
        contour_mask(row_axes[1], label > 0, bbox, color="white", linewidth=0.9)

        control_drop = data["original"]["metrics"]["pred_fg_prob"] - data[control_key]["metrics"]["pred_fg_prob"]
        plot_heat_overlay(
            row_axes[2],
            base,
            data[control_key]["prob_map"],
            bbox,
            f"{PAPER_LABEL[control_key]}\ndrop={control_drop:.3f}",
            "magma",
            0.72,
            0.0,
            1.0,
        )
        contour_mask(row_axes[2], label > 0, bbox, color="white", linewidth=0.9)

        loss = np.clip(original_prob - data[cf_key]["prob_map"], 0.0, 1.0)
        plot_heat_overlay(
            row_axes[3],
            base,
            loss,
            bbox,
            "Probability loss\nOriginal - CF",
            "inferno",
            0.82,
            0.0,
            loss_vmax,
        )
        contour_mask(row_axes[3], masks[cf_key], bbox, color=mask_color, linewidth=1.5)
        contour_mask(row_axes[3], label > 0, bbox, color="white", linewidth=0.9)
        row_axes[0].set_ylabel(row_label, fontsize=8.8)

    fig.tight_layout(rect=[0, 0, 1, 1], w_pad=0.35, h_pad=0.55)
    fig.savefig(out_dir / f"fig_e13_paper_final_probability_effect_{case_name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"fig_e13_paper_final_probability_effect_{case_name}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_paper_response_cascade_figure(out_dir: Path, case_name: str, data: Dict, base: np.ndarray, label: np.ndarray, masks, bbox) -> None:
    rows = [("Skip input", "skip_map"), ("Fusion output", "up4_map")]
    columns = [
        ("Original", "original", None, "response", "white"),
        ("Boundary CF", "boundary_cross", "boundary_cross", "response", "#0077cc"),
        ("Boundary change", "boundary_cross", "boundary_cross", "change", "#0077cc"),
        ("Interior CF", "interior_cross", "interior_cross", "response", "#ff9900"),
        ("Interior change", "interior_cross", "interior_cross", "change", "#ff9900"),
    ]
    change_maps = []
    for _, map_key in rows:
        original = data["original"][map_key]
        for variant in ("boundary_cross", "interior_cross"):
            change_maps.append(np.abs(original - data[variant][map_key]))
    change_vmax = float(max(0.10, np.percentile(np.concatenate([crop(m, bbox).ravel() for m in change_maps]), 99.5)))
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(10.8, 4.6), dpi=320)
    for row_idx, (row_label, map_key) in enumerate(rows):
        original = data["original"][map_key]
        for col_idx, (title, variant, mask_key, mode, color) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            if mode == "response":
                plot_heat_overlay(ax, base, data[variant][map_key], bbox, title, "magma", 0.62, 0.0, 1.0)
            else:
                change = np.abs(original - data[variant][map_key])
                plot_heat_overlay(ax, base, change, bbox, title, "viridis", 0.78, 0.0, change_vmax)
            contour_mask(ax, label > 0, bbox, color="white", linewidth=0.9)
            if mask_key is not None:
                contour_mask(ax, masks[mask_key], bbox, color=color, linewidth=1.35)
        axes[row_idx, 0].set_ylabel(row_label, fontsize=8.8)
    fig.tight_layout(rect=[0, 0, 1, 1], w_pad=0.35, h_pad=0.55)
    fig.savefig(out_dir / f"fig_e13_paper_up4_response_cascade_{case_name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"fig_e13_paper_up4_response_cascade_{case_name}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_case_readout(out_dir: Path, case_name: str, data: Dict, masks: Dict[str, np.ndarray]) -> None:
    rows = []
    original = data["original"]
    for key, payload in data.items():
        if key == "original":
            rows.append(
                {
                    "case_name": case_name,
                    "variant": key,
                    "mask_area_fraction": 0.0,
                    "final_fg_prob_drop": 0.0,
                    "final_dice_drop": 0.0,
                    "baseline_final_fg_prob": original["metrics"]["pred_fg_prob"],
                    "variant_final_fg_prob": original["metrics"]["pred_fg_prob"],
                    "baseline_final_dice": original["metrics"]["mean_dice"],
                    "variant_final_dice": original["metrics"]["mean_dice"],
                }
            )
            continue
        rows.append(
            {
                "case_name": case_name,
                "variant": key,
                "mask_area_fraction": float(np.mean(masks[key])),
                "final_fg_prob_drop": float(original["metrics"]["pred_fg_prob"] - payload["metrics"]["pred_fg_prob"]),
                "final_dice_drop": float(original["metrics"]["mean_dice"] - payload["metrics"]["mean_dice"]),
                "baseline_final_fg_prob": original["metrics"]["pred_fg_prob"],
                "variant_final_fg_prob": payload["metrics"]["pred_fg_prob"],
                "baseline_final_dice": original["metrics"]["mean_dice"],
                "variant_final_dice": payload["metrics"]["mean_dice"],
            }
        )
    pd = __import__("pandas")
    pd.DataFrame(rows).to_csv(out_dir / f"table_e13_visual_case_readout_{case_name}.csv", index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    data, masks, base, label, source_name = collect_visual_data(args)
    bbox = tumor_bbox(label, pad=int(args.zoom_pad))
    case_name = str(args.case_name)
    save_region_mask_figure(out_dir, case_name, source_name, base, label, masks, bbox)
    save_paper_region_figure(out_dir, case_name, source_name, base, label, masks, bbox)
    save_map_figure(
        out_dir,
        case_name,
        data,
        base,
        bbox,
        map_key="skip_map",
        file_stem="fig_e13_up4_skip_feature_replacement_zoom",
        title="up4 skip feature response after region replacement",
        include_drop=False,
    )
    save_map_figure(
        out_dir,
        case_name,
        data,
        base,
        bbox,
        map_key="up4_map",
        file_stem="fig_e13_up4_fusion_response_counterfactual_zoom",
        title="up4 fusion response after counterfactual replacement",
        include_drop=True,
    )
    save_map_figure(
        out_dir,
        case_name,
        data,
        base,
        bbox,
        map_key="prob_map",
        file_stem="fig_e13_final_probability_counterfactual_zoom",
        title="final tumor probability after counterfactual replacement",
        include_drop=True,
    )
    save_paper_probability_figure(out_dir, case_name, data, base, label, masks, bbox)
    save_paper_response_cascade_figure(out_dir, case_name, data, base, label, masks, bbox)
    save_case_readout(out_dir, case_name, data, masks)
    print(f"Saved E13 counterfactual visuals for {case_name} to {out_dir}")


if __name__ == "__main__":
    main()
