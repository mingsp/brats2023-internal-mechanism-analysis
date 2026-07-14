from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
import torch.nn.functional as F


GUIDE_ROOT = Path(__file__).resolve().parents[1]
if str(GUIDE_ROOT) not in sys.path:
    sys.path.insert(0, str(GUIDE_ROOT))

from datasets.dataset_brats import BraTSDataset  # noqa: E402
from process_analysis.stage1_explanation_utils import (  # noqa: E402
    GradCAMStore,
    compute_cam_variant_native,
    dataset_indices_from_case_names,
    load_model,
    segmentation_target_score,
    top_fraction_mask,
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
DEFAULT_OUTPUT = (
    "stage1_explanation_suite_20260517/results/e8_paper_figures/main/"
    "fig6b_cam_guided_perturbation_mask_allnodes_BraTS-GLI-00568-000_78.png"
)

MODEL_ORDER = ("baseline", "noskip_unet")
MODEL_LABEL = {"baseline": "有跳接 U-Net", "noskip_unet": "无跳接 U-Net"}
CLASS_COLORS = {
    1: np.array([0.86, 0.08, 0.12], dtype=np.float32),  # necrosis: red
    2: np.array([0.10, 0.70, 0.22], dtype=np.float32),  # edema: green
    3: np.array([1.00, 0.78, 0.08], dtype=np.float32),  # enhancing: yellow
}


def configure_chinese_font() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Source Han Sans CN",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a visual readout of CAM-guided perturbation mask locations for one Stage-1 case."
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--case-name", type=str, default="BraTS-GLI-00568-000_78")
    parser.add_argument("--baseline-ckpt", type=str, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--noskip-ckpt", type=str, default=DEFAULT_NOSKIP_CKPT)
    parser.add_argument("--nodes", type=str, default="down1,down2,down3,down4,up1,up2,up3,up4")
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--target-mode", type=str, default="pred_fg", choices=["pred_fg", "gt_fg"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if str(device_arg) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device_arg))


def image_channel_for_display(image_chw: np.ndarray) -> np.ndarray:
    channel = image_chw[0].astype(np.float32)
    lo, hi = np.percentile(channel, [1, 99])
    return np.clip((channel - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def resize_binary_mask(mask: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    resized = F.interpolate(tensor, size=tuple(output_shape), mode="nearest")[0, 0].numpy()
    return resized > 0.5


def overlay_label(base: np.ndarray, label: np.ndarray, alpha: float = 0.58) -> np.ndarray:
    rgb = np.stack([base, base, base], axis=-1)
    for class_id, color in CLASS_COLORS.items():
        mask = np.asarray(label) == int(class_id)
        if np.any(mask):
            rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color
    return np.clip(rgb, 0.0, 1.0)


def class_contour(mask: np.ndarray, class_id: int) -> np.ndarray:
    class_mask = np.asarray(mask) == int(class_id)
    if not np.any(class_mask):
        return np.zeros_like(class_mask, dtype=bool)
    padded = np.pad(class_mask, 1, mode="constant", constant_values=False)
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return class_mask & ~eroded


def overlay_perturbation_mask(base: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = np.stack([base, base, base], axis=-1)
    perturb_color = np.array([0.00, 0.48, 0.95], dtype=np.float32)
    out = rgb.copy()
    mask_bool = mask.astype(bool)
    out[mask_bool] = 0.25 * out[mask_bool] + 0.75 * perturb_color
    return np.clip(out, 0.0, 1.0)


def case_batch(data_dir: str, split: str, case_name: str, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray]:
    dataset = BraTSDataset(base_dir=data_dir, split=split)
    idx = dataset_indices_from_case_names(dataset, [case_name])[0]
    sample = dataset[idx]
    image = sample["image"].unsqueeze(0).to(device)
    label = sample["label"].unsqueeze(0).to(device)
    batch = {"image": image, "label": label, "case_name": [case_name]}
    base = image_channel_for_display(sample["image"].numpy())
    label_np = sample["label"].numpy()
    return batch, base, label_np


def cam_masks_for_model(
    model_name: str,
    ckpt_path: str,
    batch: dict,
    nodes: list[str],
    top_fraction: float,
    target_mode: str,
    device: torch.device,
    output_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    model = load_model(model_name=model_name, ckpt_path=ckpt_path, device=device)
    store = GradCAMStore(model=model, node_names=nodes)
    model.zero_grad(set_to_none=True)
    store.clear()
    logits = unpack_logits(model(batch["image"]))
    score = segmentation_target_score(logits=logits, labels=batch["label"], mode=target_mode)
    score.backward()

    masks: dict[str, np.ndarray] = {}
    for node_name in nodes:
        cam = compute_cam_variant_native(store=store, node_name=node_name, variant="gradcam")
        cam_np = cam.detach().cpu().numpy()[0]
        native_mask = top_fraction_mask(cam_np, fraction=float(top_fraction), largest=True)
        masks[node_name] = resize_binary_mask(native_mask, output_shape=output_shape)
    store.remove()
    return masks


def save_figure(
    output: Path,
    case_name: str,
    base: np.ndarray,
    label: np.ndarray,
    masks_by_model: dict[str, dict[str, np.ndarray]],
    nodes: list[str],
    top_fraction: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    configure_chinese_font()
    n_cols = len(nodes) + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(2.05 * n_cols, 6.4), dpi=220)

    for row_idx, model_name in enumerate(MODEL_ORDER):
        axes[row_idx, 0].imshow(overlay_label(base, label))
        if row_idx == 0:
            axes[row_idx, 0].set_title("MRI + 真实标签", fontsize=9, pad=6)
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].text(
            -0.10,
            0.50,
            MODEL_LABEL[model_name],
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=9,
        )

        for col_idx, node_name in enumerate(nodes, start=1):
            image = overlay_perturbation_mask(base, masks_by_model[model_name][node_name])
            axes[row_idx, col_idx].imshow(image)
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(node_name, fontsize=9, pad=6)
            axes[row_idx, col_idx].axis("off")

    fig.suptitle(
        f"{case_name} | CAM 高响应遮挡区域位置（前 {int(round(top_fraction * 100))}%）",
        fontsize=11,
        y=0.99,
    )
    fig.text(
        0.5,
        0.015,
        "蓝色表示被遮挡的 CAM 高响应区域；肿瘤类别只在 MRI + 真实标签列显示。",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.035, right=0.995, top=0.84, bottom=0.13, wspace=0.04, hspace=0.34)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    nodes = [item.strip() for item in str(args.nodes).split(",") if item.strip()]
    device = resolve_device(args.device)
    batch, base, label = case_batch(args.data_dir, args.split, args.case_name, device)
    output_shape = tuple(int(v) for v in label.shape)
    specs = {
        "baseline": args.baseline_ckpt,
        "noskip_unet": args.noskip_ckpt,
    }
    masks_by_model = {
        model_name: cam_masks_for_model(
            model_name=model_name,
            ckpt_path=ckpt,
            batch=batch,
            nodes=nodes,
            top_fraction=float(args.top_fraction),
            target_mode=str(args.target_mode),
            device=device,
            output_shape=output_shape,
        )
        for model_name, ckpt in specs.items()
    }
    save_figure(
        output=Path(args.output),
        case_name=str(args.case_name),
        base=base,
        label=label,
        masks_by_model=masks_by_model,
        nodes=nodes,
        top_fraction=float(args.top_fraction),
    )
    print("Saved perturbation mask visual to {}".format(args.output))


if __name__ == "__main__":
    main()
