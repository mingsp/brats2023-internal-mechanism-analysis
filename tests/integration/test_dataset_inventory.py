import json
import os
from pathlib import Path

import numpy as np
import pytest

from pptt.data.brats2d import discover_slice_records
from pptt.data.patient_splits import (
    assert_disjoint_patient_splits,
    patient_ids_from_records,
)


DEFAULT_ASSET_ROOT = Path("/root/autodl-tmp/A_scheme_workspace/brats2023_data")
EXPECTED_SLICE_COUNTS = {"train": 56937, "val": 8366, "test": 16118}
EXPECTED_PATIENT_COUNTS = {"train": 876, "val": 125, "test": 250}
EXPECTED_IMAGE_SHAPE = (160, 160, 4)
EXPECTED_MASK_SHAPE = (160, 160)
ALLOWED_MASK_LABELS = {0, 1, 2, 3}


def _resolve_asset_root() -> Path:
    configured_root = os.environ.get("PPTT_ASSET_ROOT")
    if configured_root is not None:
        if not configured_root.strip():
            pytest.fail("PPTT_ASSET_ROOT is set but empty")
        root = Path(configured_root).expanduser()
        assert root.is_dir(), f"PPTT_ASSET_ROOT is not a directory: {root}"
        return root
    if DEFAULT_ASSET_ROOT.is_dir():
        return DEFAULT_ASSET_ROOT
    pytest.skip(
        "PPTT_ASSET_ROOT is unset and the default server asset root does not "
        f"exist: {DEFAULT_ASSET_ROOT}"
    )


def test_complete_brats2023_2d_inventory():
    asset_root = _resolve_asset_root()
    processed_root = asset_root / "processed_2d"
    patient_splits: dict[str, set[str]] = {}

    for split, expected_slices in EXPECTED_SLICE_COUNTS.items():
        records = discover_slice_records(
            processed_root / f"{split}_Image",
            processed_root / f"{split}_Mask",
        )
        assert len(records) == expected_slices, (
            f"{split} slice count mismatch: expected={expected_slices}, "
            f"actual={len(records)}"
        )

        patients = patient_ids_from_records(records)
        patient_splits[split] = patients
        assert len(patients) == EXPECTED_PATIENT_COUNTS[split], (
            f"{split} patient count mismatch: "
            f"expected={EXPECTED_PATIENT_COUNTS[split]}, actual={len(patients)}"
        )

        for record in records:
            image = np.load(record.image_path, mmap_mode="r", allow_pickle=False)
            assert image.shape == EXPECTED_IMAGE_SHAPE, record.image_path
            assert image.dtype == np.float32, record.image_path
            del image

            mask = np.load(record.mask_path, allow_pickle=False)
            assert mask.shape == EXPECTED_MASK_SHAPE, record.mask_path
            assert mask.dtype == np.uint8, record.mask_path
            labels = {int(value) for value in np.unique(mask)}
            assert labels <= ALLOWED_MASK_LABELS, (
                f"Unexpected mask labels in {record.mask_path}: {sorted(labels)}"
            )

    assert_disjoint_patient_splits(patient_splits)

    split_path = processed_root / "splits" / "brats2023_gli_seed42.json"
    if split_path.is_file():
        with split_path.open("r", encoding="utf-8") as handle:
            declared_splits = json.load(handle)
        assert set(EXPECTED_PATIENT_COUNTS).issubset(declared_splits)
        for split, expected_count in EXPECTED_PATIENT_COUNTS.items():
            declared = declared_splits[split]
            assert isinstance(declared, list)
            assert all(isinstance(patient, str) for patient in declared)
            assert len(declared) == expected_count
            assert len(set(declared)) == len(declared)
            assert set(declared) == patient_splits[split]
