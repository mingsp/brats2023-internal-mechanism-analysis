from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_HWC_SHAPE = (160, 160, 4)
IMAGE_CHW_SHAPE = (4, 160, 160)
LABEL_SHAPE = (160, 160)
_ALLOWED_SOURCE_LABELS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class SliceRecord:
    """Paths and identifier for one paired BraTS 2D slice."""

    slice_id: str
    image_path: Path
    mask_path: Path


def ensure_chw(image: np.ndarray) -> np.ndarray:
    """Validate a BraTS image and return contiguous float32 CHW data."""
    array = np.asarray(image)
    if array.shape == IMAGE_HWC_SHAPE:
        chw = array.transpose(2, 0, 1)
    elif array.shape == IMAGE_CHW_SHAPE:
        chw = array
    else:
        raise ValueError(
            "Invalid image shape: expected (160, 160, 4) or "
            f"(4, 160, 160), got {array.shape}"
        )

    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError("image values must be real, numeric, and finite")
    if not np.isfinite(array).all():
        raise ValueError("image values must be finite")

    with np.errstate(over="ignore", invalid="ignore"):
        result = np.ascontiguousarray(chw, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("image values must remain finite as float32")
    return result


def normalize_label(label: np.ndarray) -> np.ndarray:
    """Validate a BraTS mask and map source label 4 to class 3."""
    array = np.asarray(label)
    if array.shape != LABEL_SHAPE:
        raise ValueError(
            f"Invalid label shape: expected (160, 160), got {array.shape}"
        )

    if np.issubdtype(array.dtype, np.integer):
        if np.any(array < 0):
            raise ValueError("label values must not be negative")
    elif np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("label values must be finite")
        if np.any(array < 0):
            raise ValueError("label values must not be negative")
        if not np.equal(array, np.trunc(array)).all():
            raise ValueError("label values must be integer-valued")
    else:
        raise ValueError("label dtype must be integer or floating point")

    unique_values = np.unique(array)
    unexpected = unique_values[
        ~np.isin(unique_values, _ALLOWED_SOURCE_LABELS)
    ]
    if unexpected.size:
        unexpected_labels = [int(value) for value in unexpected.tolist()]
        raise ValueError(f"Unexpected labels: {unexpected_labels}")

    result = np.array(array, dtype=np.uint8, order="C", copy=True)
    result[result == 4] = 3
    return result


def _require_directory(path: str | Path, label: str) -> Path:
    directory = Path(path)
    if not directory.exists():
        raise FileNotFoundError(f"{label} does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {directory}")
    return directory


def _direct_npy_files(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix == ".npy"
    }


def discover_slice_records(
    image_dir: str | Path, mask_dir: str | Path
) -> list[SliceRecord]:
    """Discover direct image/mask pairs in stable slice identifier order."""
    image_root = _require_directory(image_dir, "image_dir")
    mask_root = _require_directory(mask_dir, "mask_dir")
    image_files = _direct_npy_files(image_root)
    mask_files = _direct_npy_files(mask_root)

    image_stems = set(image_files)
    mask_stems = set(mask_files)
    if image_stems != mask_stems:
        missing_masks = sorted(image_stems - mask_stems)
        missing_images = sorted(mask_stems - image_stems)
        raise ValueError(
            "Unpaired .npy stems: "
            f"missing_masks={missing_masks[:5]} (total={len(missing_masks)}), "
            f"missing_images={missing_images[:5]} "
            f"(total={len(missing_images)})"
        )

    return [
        SliceRecord(
            slice_id=stem,
            image_path=image_files[stem],
            mask_path=mask_files[stem],
        )
        for stem in sorted(image_stems)
    ]


class BraTS2DDataset(Dataset):
    """Load paired BraTS 2D NumPy slices as contiguous torch tensors."""

    def __init__(self, image_dir: str | Path, mask_dir: str | Path) -> None:
        self.records = discover_slice_records(image_dir, mask_dir)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        from pptt.data.patient_splits import patient_id_from_slice

        record = self.records[index]
        image = np.load(record.image_path, allow_pickle=False)
        label = np.load(record.mask_path, allow_pickle=False)
        image_tensor = torch.from_numpy(ensure_chw(image)).contiguous()
        label_tensor = torch.from_numpy(normalize_label(label)).to(
            dtype=torch.int64
        )
        return {
            "image": image_tensor,
            "label": label_tensor.contiguous(),
            "slice_id": record.slice_id,
            "patient_id": patient_id_from_slice(record.slice_id),
        }
