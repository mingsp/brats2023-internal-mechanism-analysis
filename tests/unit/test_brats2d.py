from pathlib import Path

import numpy as np
import pytest
import torch

from pptt.data.brats2d import (
    BraTS2DDataset,
    discover_slice_records,
    ensure_chw,
    normalize_label,
)
from pptt.data.patient_splits import (
    assert_disjoint_patient_splits,
    patient_id_from_slice,
    patient_ids_from_directory,
    patient_ids_from_records,
)


IMAGE_HWC_SHAPE = (160, 160, 4)
IMAGE_CHW_SHAPE = (4, 160, 160)
LABEL_SHAPE = (160, 160)


def _write_pair(
    image_dir: Path,
    mask_dir: Path,
    stem: str,
    image: np.ndarray | None = None,
    label: np.ndarray | None = None,
) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    if image is None:
        image = np.zeros(IMAGE_HWC_SHAPE, dtype=np.float32)
    if label is None:
        label = np.zeros(LABEL_SHAPE, dtype=np.uint8)
    np.save(image_dir / f"{stem}.npy", image, allow_pickle=False)
    np.save(mask_dir / f"{stem}.npy", label, allow_pickle=False)


@pytest.mark.parametrize("layout", ["hwc", "chw"])
def test_ensure_chw_accepts_both_layouts_without_changing_values(layout: str):
    hwc = np.arange(np.prod(IMAGE_HWC_SHAPE), dtype=np.float64).reshape(
        IMAGE_HWC_SHAPE
    )
    image = hwc if layout == "hwc" else hwc.transpose(2, 0, 1)

    result = ensure_chw(image)

    assert result.shape == IMAGE_CHW_SHAPE
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, hwc.transpose(2, 0, 1).astype(np.float32))


@pytest.mark.parametrize("shape", [(160, 160), (160, 160, 3), (4, 159, 160)])
def test_ensure_chw_rejects_invalid_shapes(shape: tuple[int, ...]):
    with pytest.raises(ValueError, match="image shape"):
        ensure_chw(np.zeros(shape, dtype=np.float32))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_ensure_chw_rejects_non_finite_values(value: float):
    image = np.zeros(IMAGE_HWC_SHAPE, dtype=np.float32)
    image.flat[0] = value

    with pytest.raises(ValueError, match="finite"):
        ensure_chw(image)


def test_normalize_label_maps_four_and_returns_new_uint8_array():
    label = np.zeros(LABEL_SHAPE, dtype=np.uint8)
    label[0, :5] = [0, 1, 2, 3, 4]
    original = label.copy()

    result = normalize_label(label)

    assert result.dtype == np.uint8
    assert result is not label
    assert not np.shares_memory(result, label)
    assert result[0, :5].tolist() == [0, 1, 2, 3, 3]
    np.testing.assert_array_equal(label, original)


def test_normalize_label_accepts_integral_floats():
    label = np.zeros(LABEL_SHAPE, dtype=np.float32)
    label[0, 0] = 4.0
    assert normalize_label(label)[0, 0] == 3


@pytest.mark.parametrize("shape", [(160,), (160, 160, 1), (159, 160)])
def test_normalize_label_rejects_invalid_shapes(shape: tuple[int, ...]):
    with pytest.raises(ValueError, match="label shape"):
        normalize_label(np.zeros(shape, dtype=np.uint8))


def test_normalize_label_rejects_negative_non_integer_and_unknown_labels():
    negative = np.zeros(LABEL_SHAPE, dtype=np.int16)
    negative[0, 0] = -1
    with pytest.raises(ValueError, match="negative"):
        normalize_label(negative)

    fractional = np.zeros(LABEL_SHAPE, dtype=np.float32)
    fractional[0, 0] = 1.5
    with pytest.raises(ValueError, match="integer"):
        normalize_label(fractional)

    unknown = np.zeros(LABEL_SHAPE, dtype=np.uint8)
    unknown[0, 0] = 5
    with pytest.raises(ValueError, match="Unexpected labels.*5"):
        normalize_label(unknown)


@pytest.mark.parametrize(
    ("slice_name", "expected"),
    [
        ("BraTS-GLI-00000-000_49", "BraTS-GLI-00000-000"),
        ("BraTS-GLI-00000-000_49.npy", "BraTS-GLI-00000-000"),
        (Path("nested") / "patient_with_underscores_007.npy", "patient_with_underscores"),
    ],
)
def test_patient_id_from_slice_uses_last_underscore(
    slice_name: str | Path, expected: str
):
    assert patient_id_from_slice(slice_name) == expected


@pytest.mark.parametrize(
    "slice_name",
    ["patient", "patient_", "patient_.npy", "patient_slice.npy", "patient_1.npz"],
)
def test_patient_id_from_slice_rejects_missing_or_non_numeric_suffix(
    slice_name: str,
):
    with pytest.raises(ValueError, match="slice"):
        patient_id_from_slice(slice_name)


def test_assert_disjoint_patient_splits_reports_every_overlapping_pair():
    splits = {
        "train": {"p0", "p1", "p2", "p3", "p4", "p5", "train-test"},
        "val": {"p0", "p1", "p2", "p3", "p4", "p5", "val-test"},
        "test": {"train-test", "val-test"},
    }

    with pytest.raises(ValueError) as error:
        assert_disjoint_patient_splits(splits)

    message = str(error.value)
    assert "train vs val" in message
    assert "train vs test" in message
    assert "val vs test" in message
    assert "p0" in message and "p4" in message
    assert "p5" not in message
    assert "total=6" in message


def test_discovery_is_stable_and_patient_sets_can_be_built(tmp_path: Path):
    image_dir = tmp_path / "Image"
    mask_dir = tmp_path / "Mask"
    stems = [
        "BraTS-GLI-00001-000_2",
        "BraTS-GLI-00000-000_2",
        "BraTS-GLI-00000-000_10",
    ]
    for stem in stems:
        _write_pair(image_dir, mask_dir, stem)

    first = discover_slice_records(image_dir, mask_dir)
    second = discover_slice_records(image_dir, mask_dir)

    assert first == second
    assert [record.slice_id for record in first] == sorted(stems)
    expected_patients = {"BraTS-GLI-00000-000", "BraTS-GLI-00001-000"}
    assert patient_ids_from_records(first) == expected_patients
    assert patient_ids_from_directory(image_dir) == expected_patients


def test_discovery_requires_exact_stem_pairing(tmp_path: Path):
    image_dir = tmp_path / "Image"
    mask_dir = tmp_path / "Mask"
    image_dir.mkdir()
    mask_dir.mkdir()
    np.save(image_dir / "image_only_1.npy", np.zeros(1), allow_pickle=False)
    np.save(mask_dir / "mask_only_1.npy", np.zeros(1), allow_pickle=False)

    with pytest.raises(ValueError) as error:
        discover_slice_records(image_dir, mask_dir)

    message = str(error.value)
    assert "missing_masks" in message and "image_only_1" in message
    assert "missing_images" in message and "mask_only_1" in message


def test_dataset_returns_contiguous_tensors_and_identifiers(tmp_path: Path):
    image_dir = tmp_path / "Image"
    mask_dir = tmp_path / "Mask"
    stem = "BraTS-GLI-00000-000_49"
    image = np.arange(np.prod(IMAGE_HWC_SHAPE), dtype=np.float32).reshape(
        IMAGE_HWC_SHAPE
    )
    label = np.zeros(LABEL_SHAPE, dtype=np.uint8)
    label[0, :5] = [0, 1, 2, 3, 4]
    _write_pair(image_dir, mask_dir, stem, image, label)

    dataset = BraTS2DDataset(image_dir, mask_dir)
    sample = dataset[0]

    assert len(dataset) == 1
    assert set(sample) == {"image", "label", "slice_id", "patient_id"}
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].dtype == torch.float32
    assert sample["image"].shape == IMAGE_CHW_SHAPE
    assert sample["image"].is_contiguous()
    assert torch.equal(sample["image"], torch.from_numpy(image.transpose(2, 0, 1).copy()))
    assert isinstance(sample["label"], torch.Tensor)
    assert sample["label"].dtype == torch.int64
    assert sample["label"].shape == LABEL_SHAPE
    assert sample["label"].is_contiguous()
    assert sample["label"][0, :5].tolist() == [0, 1, 2, 3, 3]
    assert sample["slice_id"] == stem
    assert sample["patient_id"] == "BraTS-GLI-00000-000"


def test_dataset_disables_pickle_loading(tmp_path: Path):
    image_dir = tmp_path / "Image"
    mask_dir = tmp_path / "Mask"
    image_dir.mkdir()
    mask_dir.mkdir()
    stem = "BraTS-GLI-00000-000_1"
    np.save(
        image_dir / f"{stem}.npy",
        np.empty(IMAGE_HWC_SHAPE, dtype=object),
        allow_pickle=True,
    )
    np.save(
        mask_dir / f"{stem}.npy",
        np.zeros(LABEL_SHAPE, dtype=np.uint8),
        allow_pickle=False,
    )

    with pytest.raises(ValueError, match="allow_pickle=False"):
        BraTS2DDataset(image_dir, mask_dir)[0]
