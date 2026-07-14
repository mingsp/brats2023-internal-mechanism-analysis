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
    feature_response_from_activation,
    load_model,
    segmentation_target_score,
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
DEFAULT_OUTPUT_ROOT = "stage1_explanation_suite_20260517/results/e8_paper_figures/main"

ALL_NODES = ["down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4"]
KEY_NODES = ["down4", "up1", "up3", "up4"]
MODEL_ORDER = ("baseline", "noskip_unet")
MODEL_LABEL = {"baseline": "有跳接 U-Net", "noskip_unet": "无跳接 U-Net"}

# Processed BraTS labels use 1/2/3 after remapping original label 4 to 3.
# 1: necrotic/non-enhancing tumor core, 2: edema, 3: enhancing tumor.
CLASS_COLORS = {
    1: np.array([0.86, 0.08, 0.12], dtype=np.float32),  # necrosis: red
    2: np.array([0.10, 0.70, 0.22], dtype=np.float32),  # edema: green
    3: np.array([1.00, 0.78, 0.08], dtype=np.float32),  # enhancing: yellow
}
CLASS_NAMES = {
    1: "坏死/核心",
    2: "水肿",
    3: "增强肿瘤",
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
    parser = argparse.ArgumentParser(description="Generate class-colored Stage-1 visual case figures.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--case-name", type=str, default="BraTS-GLI-00568-000_78")
    parser.add_argument("--baseline-ckpt", type=str, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--noskip-ckpt", type=str, default=DEFAULT_NOSKIP_CKPT)
    parser.add_argument("--target-mode", type=str, default="pred_fg", choices=["pred_fg", "gt_fg"])
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


def resize_response(response: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(response, dtype=np.float32)
    if tuple(arr.shape) == tuple(output_shape):
        return arr
    tensor = torch.from_numpy(arr)[None, None]
    resized = F.interpolate(tensor, size=tuple(output_shape), mode="bilinear", align_corners=False)[0, 0].numpy()
    low, high = float(resized.min()), float(resized.max())
    return (resized - low) / (high - low + 1e-8)


def overlay_response(base: np.ndarray, response: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    response = resize_response(response, output_shape=tuple(base.shape))
    cmap = plt.get_cmap("jet")
    heat = cmap(np.clip(response, 0.0, 1.0))[..., :3]
    gray = np.stack([base, base, base], axis=-1)
    return np.clip((1.0 - alpha) * gray + alpha * heat, 0.0, 1.0)


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


def add_class_contours(ax, label: np.ndarray, linewidth: float = 0.9) -> None:
    for class_id, color in CLASS_COLORS.items():
        contour = class_contour(label, class_id)
        if np.any(contour):
            ax.contour(
                contour.astype(float),
                levels=[0.5],
                colors=[tuple(color.tolist())],
                linewidths=linewidth,
            )


def add_class_legend(fig: plt.Figure) -> None:
    x0 = 0.36
    y = 0.018
    gap = 0.11
    labels = [(2, "水肿"), (1, "坏死/核心"), (3, "增强肿瘤")]
    for idx, (class_id, name) in enumerate(labels):
        color = CLASS_COLORS[class_id]
        fig.text(x0 + idx * gap, y, "■", color=tuple(color.tolist()), fontsize=9, ha="right", va="center")
        fig.text(x0 + idx * gap + 0.006, y, name, color="#222222", fontsize=8.5, ha="left", va="center")


def load_case(data_dir: str, split: str, case_name: str, device: torch.device):
    dataset = BraTSDataset(base_dir=data_dir, split=split)
    idx = dataset_indices_from_case_names(dataset, [case_name])[0]
    sample = dataset[idx]
    image = sample["image"].unsqueeze(0).to(device)
    label = sample["label"].unsqueeze(0).to(device)
    base = image_channel_for_display(sample["image"].numpy())
    label_np = sample["label"].numpy()
    return image, label, base, label_np


def collect_model_outputs(
    model_name: str,
    ckpt_path: str,
    image: torch.Tensor,
    label: torch.Tensor,
    target_mode: str,
    device: torch.device,
) -> dict:
    model = load_model(model_name=model_name, ckpt_path=ckpt_path, device=device)
    store = GradCAMStore(model=model, node_names=ALL_NODES)
    model.zero_grad(set_to_none=True)
    store.clear()
    logits = unpack_logits(model(image))
    score = segmentation_target_score(logits=logits, labels=label, mode=target_mode)
    score.backward()
    pred = torch.argmax(logits.detach(), dim=1).cpu().numpy()[0]

    cams = {}
    features = {}
    for node in ALL_NODES:
        cam = compute_cam_variant_native(store=store, node_name=node, variant="gradcam")
        cams[node] = cam.detach().cpu().numpy()[0]
        feat = feature_response_from_activation(store.activations[node], aggregation="l2")
        features[node] = feat.detach().cpu().numpy()[0]
    store.remove()
    return {"pred": pred, "cam": cams, "feature": features}


def save_visual(
    output: Path,
    case_name: str,
    base: np.ndarray,
    label: np.ndarray,
    by_model: dict,
    nodes: list[str],
    response_key: str,
    title_suffix: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    configure_chinese_font()
    n_cols = len(nodes) + 2
    fig, axes = plt.subplots(2, n_cols, figsize=(2.05 * n_cols, 5.55), dpi=220)
    for row_idx, model_name in enumerate(MODEL_ORDER):
        model_data = by_model[model_name]
        axes[row_idx, 0].imshow(overlay_label(base, label))
        axes[row_idx, 0].set_title("MRI + 真实标签", fontsize=9, pad=5)
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

        axes[row_idx, 1].imshow(overlay_label(base, model_data["pred"]))
        axes[row_idx, 1].set_title("MRI + 预测结果", fontsize=9, pad=5)
        axes[row_idx, 1].axis("off")

        for col_idx, node in enumerate(nodes, start=2):
            ax = axes[row_idx, col_idx]
            ax.imshow(overlay_response(base, model_data[response_key][node]))
            ax.set_title(node, fontsize=9, pad=5)
            ax.axis("off")

    fig.suptitle(f"{case_name} | {title_suffix}", fontsize=11, y=0.985)
    add_class_legend(fig)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.87, bottom=0.08, wspace=0.05, hspace=0.34)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    image, label, base, label_np = load_case(args.data_dir, args.split, args.case_name, device)
    specs = {
        "baseline": args.baseline_ckpt,
        "noskip_unet": args.noskip_ckpt,
    }
    by_model = {
        model_name: collect_model_outputs(
            model_name=model_name,
            ckpt_path=ckpt,
            image=image,
            label=label,
            target_mode=args.target_mode,
            device=device,
        )
        for model_name, ckpt in specs.items()
    }
    out_dir = Path(args.output_root)
    case_name = str(args.case_name)
    save_visual(
        output=out_dir / f"fig3_cam_fullnode_trajectory_{case_name}.png",
        case_name=case_name,
        base=base,
        label=label_np,
        by_model=by_model,
        nodes=ALL_NODES,
        response_key="cam",
        title_suffix="CAM 全节点响应轨迹",
    )
    save_visual(
        output=out_dir / f"fig3_cam_trajectory_{case_name}.png",
        case_name=case_name,
        base=base,
        label=label_np,
        by_model=by_model,
        nodes=KEY_NODES,
        response_key="cam",
        title_suffix="CAM 关键节点响应轨迹",
    )
    save_visual(
        output=out_dir / f"fig4_feature_fullnode_trajectory_{case_name}.png",
        case_name=case_name,
        base=base,
        label=label_np,
        by_model=by_model,
        nodes=ALL_NODES,
        response_key="feature",
        title_suffix="原始特征响应全节点轨迹",
    )
    save_visual(
        output=out_dir / f"fig4_feature_trajectory_{case_name}.png",
        case_name=case_name,
        base=base,
        label=label_np,
        by_model=by_model,
        nodes=KEY_NODES,
        response_key="feature",
        title_suffix="原始特征响应关键节点轨迹",
    )
    print(f"Saved class-colored case visuals for {case_name} to {out_dir}")


if __name__ == "__main__":
    main()
