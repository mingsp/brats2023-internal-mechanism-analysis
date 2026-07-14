from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from surface_distance import compute_robust_hausdorff, compute_surface_distances
from tqdm import tqdm


GUIDE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(GUIDE_ROOT)
if GUIDE_ROOT not in sys.path:
    sys.path.insert(0, GUIDE_ROOT)

from Architecture.unet_baseline import UNetBaseline
from Architecture.unet_noskip import UNetNoSkip
from datasets.dataset_brats import BraTSDataset


MODEL_CHOICES = ("baseline", "noskip_unet", "skip_masked_baseline")
CC3D_MIN_SIZE_GRID = (0, 50, 100, 200, 500)


def load_checkpoint_compatible(checkpoint_path: str, map_location=None):
    payload = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict):
        return payload
    raise TypeError("Unsupported checkpoint payload type: {}".format(type(payload)))


def build_model(model_name: str, n_classes: int, skip_mask_layers: str = "all"):
    if model_name == "baseline":
        return UNetBaseline(n_channels=4, n_classes=int(n_classes))
    if model_name == "noskip_unet":
        return UNetNoSkip(n_channels=4, n_classes=int(n_classes))
    if model_name == "skip_masked_baseline":
        return UNetBaseline(
            n_channels=4,
            n_classes=int(n_classes),
            skip_mask_mode="zero",
            skip_mask_layers=skip_mask_layers,
        )
    raise ValueError("Unsupported model: {}".format(model_name))


def _label_connected_components(binary_mask: np.ndarray):
    try:
        import cc3d as cc3d_module

        labels_out, n_components = cc3d_module.connected_components(
            binary_mask,
            connectivity=26,
            return_N=True,
        )
        stats = cc3d_module.statistics(labels_out)
        return labels_out, n_components, stats["voxel_counts"]
    except (ImportError, ValueError) as exc:
        warnings.warn(
            "cc3d unavailable ({}); falling back to scipy.ndimage.label.".format(exc),
            RuntimeWarning,
        )
        from scipy import ndimage

        labels_out, n_components = ndimage.label(
            binary_mask,
            structure=np.ones((3, 3, 3), dtype=np.uint8),
        )
        return labels_out, n_components, np.bincount(labels_out.ravel())


def remove_small_objects(pred_volume: np.ndarray, min_size: int = 100) -> np.ndarray:
    if int(min_size) <= 0:
        return pred_volume.copy()

    clean_pred = np.zeros_like(pred_volume)
    for class_id in (1, 2, 3):
        binary_mask = pred_volume == int(class_id)
        if not np.any(binary_mask):
            continue
        labels_out, n_components, voxel_counts = _label_connected_components(binary_mask)
        for label_id in range(1, int(n_components) + 1):
            if voxel_counts[label_id] >= int(min_size):
                clean_pred[labels_out == label_id] = int(class_id)
    return clean_pred


def apply_postproc(pred_volume: np.ndarray, mode: str = "off", min_size: int = 100) -> np.ndarray:
    if mode == "off":
        return pred_volume.copy()
    if mode == "cc3d":
        return remove_small_objects(pred_volume, min_size=int(min_size))
    raise ValueError("Unsupported postproc mode: {}".format(mode))


def compute_dice_3d(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    inter = float((pred * target).sum())
    return float((2.0 * inter + 1e-5) / (float(pred.sum()) + float(target.sum()) + 1e-5))


def compute_volume_diagonal_mm(shape_zyx, spacing_zyx) -> float:
    shape_zyx = np.asarray(shape_zyx, dtype=np.float64)
    spacing_zyx = np.asarray(spacing_zyx, dtype=np.float64)
    extent_zyx = np.maximum(shape_zyx - 1.0, 0.0)
    return float(np.sqrt(np.sum((extent_zyx * spacing_zyx) ** 2)))


def compute_hd95_3d(pred: np.ndarray, target: np.ndarray, spacing_zyx=(1.0, 1.0, 1.0)) -> float:
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    if not np.any(pred_bool) and not np.any(target_bool):
        return 0.0
    if not np.any(pred_bool) or not np.any(target_bool):
        return compute_volume_diagonal_mm(pred_bool.shape, spacing_zyx)
    surface_distances = compute_surface_distances(target_bool, pred_bool, spacing_mm=spacing_zyx)
    return float(compute_robust_hausdorff(surface_distances, 95))


def build_brats_region_masks(volume: np.ndarray):
    return {
        "ET": (volume == 3).astype(np.uint8),
        "TC": ((volume == 1) | (volume == 3)).astype(np.uint8),
        "WT": ((volume == 1) | (volume == 2) | (volume == 3)).astype(np.uint8),
    }


def evaluate_volume_3d(pred_volume: np.ndarray, target_volume: np.ndarray, n_classes: int = 4, spacing_zyx=(1.0, 1.0, 1.0)):
    class_names = {1: "NCR", 2: "Edema", 3: "ET"}
    metrics = {}
    dice_label_scores = []
    hd95_label_scores = []

    for class_id in range(1, int(n_classes)):
        pred_c = (pred_volume == int(class_id)).astype(np.uint8)
        target_c = (target_volume == int(class_id)).astype(np.uint8)
        class_name = class_names.get(class_id, "Class_{}".format(class_id))
        dice = compute_dice_3d(pred_c, target_c)
        hd95 = compute_hd95_3d(pred_c, target_c, spacing_zyx=spacing_zyx)
        metrics["Dice_{}".format(class_name)] = float(dice)
        metrics["HD95_{}".format(class_name)] = float(hd95)
        dice_label_scores.append(float(dice))
        hd95_label_scores.append(float(hd95))

    pred_regions = build_brats_region_masks(pred_volume)
    target_regions = build_brats_region_masks(target_volume)
    dice_region_scores = []
    hd95_region_scores = []
    for region in ("WT", "TC", "ET"):
        dice = compute_dice_3d(pred_regions[region], target_regions[region])
        hd95 = compute_hd95_3d(pred_regions[region], target_regions[region], spacing_zyx=spacing_zyx)
        metrics["Dice_{}".format(region)] = float(dice)
        metrics["HD95_{}".format(region)] = float(hd95)
        dice_region_scores.append(float(dice))
        hd95_region_scores.append(float(hd95))

    metrics["Dice_Label_Mean"] = float(np.mean(dice_label_scores))
    metrics["HD95_Label_Mean"] = float(np.mean(hd95_label_scores))
    metrics["Dice_Mean"] = float(np.mean(dice_region_scores))
    metrics["HD95_Mean"] = float(np.mean(hd95_region_scores))
    return metrics


def parse_patient_id(case_name: str) -> str:
    return "_".join(str(case_name).split("_")[:-1])


def parse_slice_idx(case_name: str) -> int:
    return int(str(case_name).split("_")[-1])


def load_patient_spacing_zyx(data_dir: str, patient_id: str):
    meta_path = os.path.join(data_dir, "meta", "{}.json".format(patient_id))
    if not os.path.exists(meta_path):
        warnings.warn(
            "Spacing metadata missing for {}; defaulting to (1.0, 1.0, 1.0)".format(patient_id),
            RuntimeWarning,
        )
        return (1.0, 1.0, 1.0)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    spacing_zyx = meta.get("spacing_zyx")
    if not isinstance(spacing_zyx, (list, tuple)) or len(spacing_zyx) != 3:
        return (1.0, 1.0, 1.0)
    return tuple(float(v) for v in spacing_zyx)


def evaluate_model_3d(model, dataloader, device, model_name: str, data_dir: str, n_classes: int, postproc_mode: str, cc3d_min_size: int):
    model.eval()
    patient_data = defaultdict(lambda: {"preds": {}, "gts": {}})

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating {}".format(model_name)):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            case_name = batch["case_name"][0]
            outputs = model(images)
            if not isinstance(outputs, (tuple, list)) or len(outputs) < 1:
                raise TypeError("Model forward must return tuple/list with logits as the first item.")
            logits = outputs[0]
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            patient_data[parse_patient_id(case_name)]["preds"][parse_slice_idx(case_name)] = pred[0].cpu().numpy()
            patient_data[parse_patient_id(case_name)]["gts"][parse_slice_idx(case_name)] = labels[0].cpu().numpy()

    results = []
    for patient_id in tqdm(patient_data.keys(), desc="3D metrics"):
        slice_indices = sorted(patient_data[patient_id]["preds"].keys())
        pred_volume = np.stack([patient_data[patient_id]["preds"][idx] for idx in slice_indices], axis=0)
        gt_volume = np.stack([patient_data[patient_id]["gts"][idx] for idx in slice_indices], axis=0)
        pred_volume = apply_postproc(pred_volume, mode=postproc_mode, min_size=int(cc3d_min_size))
        metrics = evaluate_volume_3d(
            pred_volume,
            gt_volume,
            n_classes=int(n_classes),
            spacing_zyx=load_patient_spacing_zyx(data_dir, patient_id),
        )
        metrics["Patient"] = patient_id
        metrics["Num_Slices"] = int(len(slice_indices))
        results.append(metrics)

    df = pd.DataFrame(results)
    summary = {"Num_Patients": int(len(results))}
    for region in ("WT", "TC", "ET"):
        for metric_type in ("Dice", "HD95"):
            key = "{}_{}".format(metric_type, region)
            summary[key] = float(df[key].mean())
            summary["{}_std".format(key)] = float(df[key].std())
            summary["{}_median".format(key)] = float(df[key].median())
    for label in ("NCR", "Edema", "ET"):
        for metric_type in ("Dice", "HD95"):
            key = "{}_{}".format(metric_type, label)
            summary[key] = float(df[key].mean())
            summary["{}_std".format(key)] = float(df[key].std())
    summary["Dice_Mean"] = float(df["Dice_Mean"].mean())
    summary["Dice_Mean_std"] = float(df["Dice_Mean"].std())
    summary["HD95_Mean"] = float(df["HD95_Mean"].mean())
    summary["HD95_Mean_std"] = float(df["HD95_Mean"].std())
    return results, summary


def print_summary(summary, model_name: str, postproc_mode: str, cc3d_min_size: int):
    print("\n" + "=" * 60)
    print("Model: {} | postproc={} | cc3d_min_size={}".format(model_name, postproc_mode, int(cc3d_min_size)))
    print("Number of patients: {}".format(summary.get("Num_Patients", "N/A")))
    print("-" * 60)
    for region in ("WT", "TC", "ET", "Mean"):
        print(
            "Dice_{:<4s}: {:.4f} +/- {:.4f}".format(
                region,
                float(summary["Dice_{}".format(region)]),
                float(summary.get("Dice_{}_std".format(region), 0.0)),
            )
        )
    for region in ("WT", "TC", "ET", "Mean"):
        print(
            "HD95_{:<4s}: {:.4f} +/- {:.4f}".format(
                region,
                float(summary["HD95_{}".format(region)]),
                float(summary.get("HD95_{}_std".format(region), 0.0)),
            )
        )


def select_cc3d_min_size(model, dataloader, device, model_name: str, data_dir: str, n_classes: int):
    best_score = -float("inf")
    best_size = 0
    rows = []
    for min_size in CC3D_MIN_SIZE_GRID:
        _, summary = evaluate_model_3d(
            model,
            dataloader,
            device,
            model_name=model_name,
            data_dir=data_dir,
            n_classes=int(n_classes),
            postproc_mode="cc3d",
            cc3d_min_size=int(min_size),
        )
        score = float(summary["Dice_Mean"])
        rows.append({"cc3d_min_size": int(min_size), "val_dice_mean": score})
        if score > best_score:
            best_score = score
            best_size = int(min_size)
    return best_size, rows


def _default_dir(name: str) -> str:
    return os.path.join(WORKSPACE_ROOT, name)


def parse_args():
    parser = argparse.ArgumentParser(description="BraTS 3D evaluation for current U-Net mechanism experiments.")
    parser.add_argument("--model", type=str, choices=MODEL_CHOICES, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=_default_dir("data"))
    parser.add_argument("--result_dir", "--save_dir", dest="result_dir", type=str, default=_default_dir("results"))
    parser.add_argument("--split", type=str, choices=("val", "test"), default="test")
    parser.add_argument("--postproc", "--postproc_mode", dest="postproc", type=str, choices=("off", "cc3d"), default="off")
    parser.add_argument("--cc3d_min_size", type=int, default=100)
    parser.add_argument("--tune_cc3d_min_size", action="store_true")
    parser.add_argument("--report_both_postproc", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1, help="Accepted for compatibility; evaluation uses patient-wise batch size 1.")
    parser.add_argument(
        "--skip_mask_layers",
        type=str,
        default="all",
        help="Comma-separated skip layers to zero for skip_masked_baseline: up1,up2,up3,up4, all, or none.",
    )
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if int(args.batch_size) != 1:
        warnings.warn("3D patient reconstruction uses batch_size=1; ignoring --batch_size.", RuntimeWarning)
    if int(args.cc3d_min_size) < 0:
        raise ValueError("--cc3d_min_size must be >= 0")
    if args.tune_cc3d_min_size and args.split != "val":
        raise ValueError("--tune_cc3d_min_size only supports --split val")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.result_dir, exist_ok=True)

    dataset = BraTSDataset(base_dir=args.data_dir, split=args.split)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(args.num_workers))
    model = build_model(args.model, n_classes=int(args.num_classes), skip_mask_layers=args.skip_mask_layers).to(device)
    model.load_state_dict(load_checkpoint_compatible(args.checkpoint, map_location=device))

    tuned_cc3d_min_size = None
    if args.tune_cc3d_min_size:
        tuned_cc3d_min_size, tuning_rows = select_cc3d_min_size(
            model,
            loader,
            device,
            model_name=args.model,
            data_dir=args.data_dir,
            n_classes=int(args.num_classes),
        )
        pd.DataFrame(tuning_rows).to_csv(
            os.path.join(args.result_dir, "{}_tuning_{}.csv".format(args.model, args.split)),
            index=False,
        )

    cc3d_min_size = int(tuned_cc3d_min_size if tuned_cc3d_min_size is not None else args.cc3d_min_size)
    configs = [{"postproc_mode": args.postproc, "cc3d_min_size": cc3d_min_size}]
    if args.report_both_postproc:
        alt_mode = "cc3d" if args.postproc == "off" else "off"
        configs.append({"postproc_mode": alt_mode, "cc3d_min_size": cc3d_min_size})

    summary_rows = []
    for cfg in configs:
        results, summary = evaluate_model_3d(
            model,
            loader,
            device,
            model_name=args.model,
            data_dir=args.data_dir,
            n_classes=int(args.num_classes),
            postproc_mode=cfg["postproc_mode"],
            cc3d_min_size=int(cfg["cc3d_min_size"]),
        )
        print_summary(summary, args.model, cfg["postproc_mode"], int(cfg["cc3d_min_size"]))

        suffix = "{}_min{}".format(cfg["postproc_mode"], int(cfg["cc3d_min_size"]))
        pd.DataFrame(results).to_csv(
            os.path.join(args.result_dir, "{}_results_3d_{}_{}.csv".format(args.model, args.split, suffix)),
            index=False,
        )
        summary_rows.append(
            {
                "Model": args.model,
                "Split": args.split,
                "Postproc_Mode": cfg["postproc_mode"],
                "CC3D_Min_Size": int(cfg["cc3d_min_size"]),
                "Dice_WT": float(summary["Dice_WT"]),
                "Dice_TC": float(summary["Dice_TC"]),
                "Dice_ET": float(summary["Dice_ET"]),
                "Dice_Mean": float(summary["Dice_Mean"]),
                "HD95_WT": float(summary["HD95_WT"]),
                "HD95_TC": float(summary["HD95_TC"]),
                "HD95_ET": float(summary["HD95_ET"]),
                "HD95_Mean": float(summary["HD95_Mean"]),
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.result_dir, "{}_summary_3d_{}.csv".format(args.model, args.split)),
        index=False,
    )
    print("\nResults saved to {}".format(args.result_dir))


if __name__ == "__main__":
    main()
