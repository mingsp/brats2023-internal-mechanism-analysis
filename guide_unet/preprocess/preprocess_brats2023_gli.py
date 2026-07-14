import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = ROOT_DIR.parent
PARENT_WORKSPACE_DIR = WORKSPACE_DIR.parent
DEFAULT_TRAIN_INPUT_DIR = str(
    PARENT_WORKSPACE_DIR / "brats2023_dataset" / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)
DEFAULT_VAL_INPUT_DIR = str(
    PARENT_WORKSPACE_DIR / "brats2023_dataset" / "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
)
DEFAULT_OUTPUT_DIR = str(WORKSPACE_DIR / "data")


def normalize(volume: np.ndarray, top_pct: float = 99, bottom_pct: float = 1) -> np.ndarray:
    volume = volume.astype(np.float32)
    brain_mask = volume > 0
    if not np.any(brain_mask):
        return volume

    brain_pixels = volume[brain_mask]
    hi = np.percentile(brain_pixels, top_pct)
    lo = np.percentile(brain_pixels, bottom_pct)

    volume_clipped = volume.copy()
    volume_clipped[brain_mask] = np.clip(volume[brain_mask], lo, hi)

    clipped_pixels = volume_clipped[brain_mask]
    mean = float(np.mean(clipped_pixels))
    std = float(np.std(clipped_pixels))
    if std == 0:
        return volume_clipped

    output = np.zeros_like(volume_clipped, dtype=np.float32)
    output[brain_mask] = (volume_clipped[brain_mask] - mean) / (std + 1e-8)
    return output


def crop_center(volume_dhw: np.ndarray, crop_hw: int) -> np.ndarray:
    d, h, w = volume_dhw.shape
    if crop_hw > h or crop_hw > w:
        raise ValueError(f"crop_hw={crop_hw} larger than volume HW={h}x{w}")
    start_h = h // 2 - crop_hw // 2
    start_w = w // 2 - crop_hw // 2
    return volume_dhw[:, start_h : start_h + crop_hw, start_w : start_w + crop_hw]


def expected_case_files(case_dir: str, case_name: str, with_seg: bool = True) -> Dict[str, str]:
    suffix_map = {
        "t1c": "-t1c.nii.gz",
        "t1n": "-t1n.nii.gz",
        "t2f": "-t2f.nii.gz",
        "t2w": "-t2w.nii.gz",
    }
    file_map = {mod: os.path.join(case_dir, case_name + suffix) for mod, suffix in suffix_map.items()}
    if with_seg:
        file_map["seg"] = os.path.join(case_dir, case_name + "-seg.nii.gz")
    return file_map


def validate_case_layout(case_dir: str, case_name: str, with_seg: bool = True) -> None:
    file_map = expected_case_files(case_dir=case_dir, case_name=case_name, with_seg=with_seg)
    missing = [path for path in file_map.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing expected files for case {}:\n{}".format(case_name, "\n".join(missing)))


def build_internal_split_manifest(
    case_names: List[str],
    seed: int,
    test_ratio: float,
    val_ratio_in_train_pool: float,
) -> Dict:
    if len(case_names) < 3:
        raise ValueError("Need at least 3 cases to create internal train/val/test split")

    shuffled = list(case_names)
    rng = random.Random(int(seed))
    rng.shuffle(shuffled)

    total = len(shuffled)
    test_count = max(1, int(round(total * float(test_ratio))))
    if test_count >= total:
        test_count = total - 1

    remaining = total - test_count
    val_count = max(1, int(round(remaining * float(val_ratio_in_train_pool))))
    if val_count >= remaining:
        val_count = remaining - 1

    test_cases = shuffled[:test_count]
    val_cases = shuffled[test_count : test_count + val_count]
    train_cases = shuffled[test_count + val_count :]
    if not train_cases:
        raise ValueError("Split produced empty train set, please adjust ratios")

    return {
        "seed": int(seed),
        "dataset": "BraTS2023-GLI",
        "split_strategy": {
            "test_ratio": float(test_ratio),
            "val_ratio_in_train_pool": float(val_ratio_in_train_pool),
            "overall_target": "70/10/20 (train/val/test) within training data",
        },
        "counts": {
            "total": int(total),
            "train": int(len(train_cases)),
            "val": int(len(val_cases)),
            "test": int(len(test_cases)),
        },
        "train": sorted(train_cases),
        "val": sorted(val_cases),
        "test": sorted(test_cases),
    }


def load_or_create_manifest(
    manifest_path: str,
    case_names: List[str],
    seed: int,
    test_ratio: float,
    val_ratio_in_train_pool: float,
) -> Tuple[Dict, bool]:
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return manifest, False

    parent = os.path.dirname(manifest_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    manifest = build_internal_split_manifest(
        case_names,
        seed=seed,
        test_ratio=test_ratio,
        val_ratio_in_train_pool=val_ratio_in_train_pool,
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest, True


def validate_manifest(manifest: Dict) -> None:
    for split in ("train", "val", "test"):
        if split not in manifest:
            raise KeyError(f"Split '{split}' missing from manifest")
        if not isinstance(manifest[split], list):
            raise TypeError(f"Split '{split}' must be a list")

    train_set = set(manifest["train"])
    val_set = set(manifest["val"])
    test_set = set(manifest["test"])
    overlap = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
    if overlap:
        raise ValueError(f"Manifest contains overlapping case IDs: {sorted(overlap)}")


def save_meta(meta_dir: str, case_name: str, spacing_xyz, shape_dhw) -> None:
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    meta_path = os.path.join(meta_dir, f"{case_name}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "spacing_xyz": [float(v) for v in spacing_xyz],
                "spacing_zyx": [float(v) for v in spacing_zyx],
                "shape_dhw": [int(v) for v in shape_dhw],
                "dataset": "BraTS2023-GLI",
            },
            f,
            indent=2,
        )


def process_training_case(
    case_dir: str,
    case_name: str,
    output_image_dir: str,
    output_mask_dir: str,
    meta_dir: str,
    crop_size: int,
    filter_empty_slices: bool,
) -> int:
    validate_case_layout(case_dir, case_name, with_seg=True)
    file_map = expected_case_files(case_dir, case_name, with_seg=True)

    spacing_xyz = None
    volumes = {}
    for mod in ("t2f", "t1n", "t1c", "t2w"):
        img = sitk.ReadImage(file_map[mod], sitk.sitkFloat32)
        if spacing_xyz is None:
            spacing_xyz = tuple(float(v) for v in img.GetSpacing())
        volumes[mod] = sitk.GetArrayFromImage(img)

    mask = sitk.GetArrayFromImage(sitk.ReadImage(file_map["seg"], sitk.sitkUInt8)).astype(np.uint8)
    mask[mask == 4] = 3

    for mod in volumes:
        volumes[mod] = normalize(volumes[mod])
        volumes[mod] = crop_center(volumes[mod], crop_size)
    mask = crop_center(mask, crop_size)

    save_meta(meta_dir, case_name, spacing_xyz=spacing_xyz, shape_dhw=mask.shape)

    slice_count = 0
    for slice_idx in range(int(mask.shape[0])):
        if filter_empty_slices and np.max(mask[slice_idx]) == 0:
            continue

        multi_channel = np.stack(
            [
                volumes["t2f"][slice_idx],
                volumes["t1n"][slice_idx],
                volumes["t1c"][slice_idx],
                volumes["t2w"][slice_idx],
            ],
            axis=-1,
        ).astype(np.float32)
        mask_slice = mask[slice_idx].astype(np.uint8)

        file_name = f"{case_name}_{slice_idx}.npy"
        np.save(os.path.join(output_image_dir, file_name), multi_channel)
        np.save(os.path.join(output_mask_dir, file_name), mask_slice)
        slice_count += 1

    return int(slice_count)


def process_validation_case(
    case_dir: str,
    case_name: str,
    output_image_dir: str,
    meta_dir: str,
    crop_size: int,
) -> int:
    validate_case_layout(case_dir, case_name, with_seg=False)
    file_map = expected_case_files(case_dir, case_name, with_seg=False)

    spacing_xyz = None
    volumes = {}
    for mod in ("t2f", "t1n", "t1c", "t2w"):
        img = sitk.ReadImage(file_map[mod], sitk.sitkFloat32)
        if spacing_xyz is None:
            spacing_xyz = tuple(float(v) for v in img.GetSpacing())
        volumes[mod] = sitk.GetArrayFromImage(img)

    for mod in volumes:
        volumes[mod] = normalize(volumes[mod])
        volumes[mod] = crop_center(volumes[mod], crop_size)

    shape_dhw = next(iter(volumes.values())).shape
    save_meta(meta_dir, case_name, spacing_xyz=spacing_xyz, shape_dhw=shape_dhw)

    slice_count = 0
    depth = int(shape_dhw[0])
    for slice_idx in range(depth):
        multi_channel = np.stack(
            [
                volumes["t2f"][slice_idx],
                volumes["t1n"][slice_idx],
                volumes["t1c"][slice_idx],
                volumes["t2w"][slice_idx],
            ],
            axis=-1,
        ).astype(np.float32)
        file_name = f"{case_name}_{slice_idx}.npy"
        np.save(os.path.join(output_image_dir, file_name), multi_channel)
        slice_count += 1

    return int(slice_count)


def list_case_names(input_dir: str) -> List[str]:
    case_names = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    case_names.sort()
    return case_names


def select_cases(case_names: List[str], explicit_cases: List[str], max_cases: int) -> List[str]:
    if explicit_cases:
        explicit_set = set(explicit_cases)
        selected = [case for case in case_names if case in explicit_set]
        missing = sorted(explicit_set - set(selected))
        if missing:
            raise ValueError("Requested cases not found: {}".format(", ".join(missing)))
        return selected
    if int(max_cases) > 0:
        return case_names[: int(max_cases)]
    return case_names


def write_report_csv(report_csv: str, rows: List[Dict]) -> None:
    if not report_csv:
        return
    parent = os.path.dirname(report_csv)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fieldnames = ["split", "case_name", "slice_count"]
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="BraTS2023 GLI preprocessing to 2D .npy slices")
    parser.add_argument(
        "--train_input_dir",
        type=str,
        default=DEFAULT_TRAIN_INPUT_DIR,
        help="BraTS2023 GLI official training directory",
    )
    parser.add_argument(
        "--val_input_dir",
        type=str,
        default=DEFAULT_VAL_INPUT_DIR,
        help="BraTS2023 GLI official validation directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory under brats2023_workspace/data",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test", "challenge_val"],
        help="Internal split from official training data, or official unlabeled challenge validation",
    )
    parser.add_argument("--crop_size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--val_ratio_in_train_pool", type=float, default=0.125)
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument("--report_csv", type=str, default="")
    parser.add_argument("--keep_empty_slices", action="store_true")
    parser.add_argument("--cases", type=str, nargs="*", default=[])
    parser.add_argument("--max_cases", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_meta_dir = os.path.join(args.output_dir, "meta")
    os.makedirs(output_meta_dir, exist_ok=True)

    if args.split == "challenge_val":
        output_image_dir = os.path.join(args.output_dir, "challenge_val_Image")
        os.makedirs(output_image_dir, exist_ok=True)
        case_names = list_case_names(args.val_input_dir)
        selected_cases = select_cases(case_names, args.cases, args.max_cases)
        report_rows = []
        for case_name in tqdm(selected_cases, desc="Processing challenge_val"):
            case_dir = os.path.join(args.val_input_dir, case_name)
            slice_count = process_validation_case(
                case_dir=case_dir,
                case_name=case_name,
                output_image_dir=output_image_dir,
                meta_dir=output_meta_dir,
                crop_size=int(args.crop_size),
            )
            report_rows.append({"split": args.split, "case_name": case_name, "slice_count": int(slice_count)})
        write_report_csv(args.report_csv, report_rows)
        print("Preprocessing complete for challenge_val")
        print(f"Cases processed: {len(selected_cases)}")
        print(f"Total slices saved: {sum(row['slice_count'] for row in report_rows)}")
        print(f"Output Image: {output_image_dir}")
        print(f"Output Meta: {output_meta_dir}")
        return

    train_case_names = list_case_names(args.train_input_dir)
    if not args.manifest_path:
        args.manifest_path = os.path.join(args.output_dir, "splits", f"brats2023_gli_seed{args.seed}.json")
    manifest, created = load_or_create_manifest(
        manifest_path=args.manifest_path,
        case_names=train_case_names,
        seed=int(args.seed),
        test_ratio=float(args.test_ratio),
        val_ratio_in_train_pool=float(args.val_ratio_in_train_pool),
    )
    validate_manifest(manifest)

    split_case_names = list(manifest[args.split])
    selected_cases = select_cases(split_case_names, args.cases, args.max_cases)

    output_image_dir = os.path.join(args.output_dir, f"{args.split}_Image")
    output_mask_dir = os.path.join(args.output_dir, f"{args.split}_Mask")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    report_rows = []
    filter_empty = not bool(args.keep_empty_slices)

    print(f"Found {len(train_case_names)} training cases in {args.train_input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split: {args.split}")
    print(f"Manifest: {args.manifest_path} ({'created' if created else 'loaded'})")
    print(f"Cases selected for split '{args.split}': {len(selected_cases)}")
    print(f"Filter empty slices: {filter_empty}")
    print("-" * 60)

    for case_name in tqdm(selected_cases, desc=f"Processing {args.split}"):
        case_dir = os.path.join(args.train_input_dir, case_name)
        slice_count = process_training_case(
            case_dir=case_dir,
            case_name=case_name,
            output_image_dir=output_image_dir,
            output_mask_dir=output_mask_dir,
            meta_dir=output_meta_dir,
            crop_size=int(args.crop_size),
            filter_empty_slices=filter_empty,
        )
        report_rows.append({"split": args.split, "case_name": case_name, "slice_count": int(slice_count)})

    write_report_csv(args.report_csv, report_rows)
    print("-" * 60)
    print("Preprocessing complete!")
    print(f"Total cases in split '{args.split}': {len(selected_cases)}")
    print(f"Total slices saved: {sum(row['slice_count'] for row in report_rows)}")
    print(f"Output Image: {output_image_dir}")
    print(f"Output Mask: {output_mask_dir}")
    print(f"Output Meta: {output_meta_dir}")


if __name__ == "__main__":
    main()
