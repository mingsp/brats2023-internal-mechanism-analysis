import numpy as np

from pptt.observers.full_cache import (
    IndexedPatientCache,
    load_balanced_node_cache,
    load_indexed_patient_cache,
    save_indexed_patient_cache,
)


def _cache(patient_id: str, rows: int) -> IndexedPatientCache:
    labels = np.arange(rows) % 4
    targets = np.eye(4, dtype=np.float32)[labels]
    return IndexedPatientCache(
        patient_id=patient_id,
        features={"down1": np.arange(rows * 3).reshape(rows, 3).astype(np.float32)},
        target_probabilities=targets,
        spatial_target_probabilities=np.roll(targets, 1, axis=0),
        truth=labels.astype(np.int64),
        slice_ids=np.asarray([f"{patient_id}_{index:03d}" for index in range(rows)]),
        y=np.arange(rows, dtype=np.int16),
        x=np.arange(rows, dtype=np.int16) + 1,
        predicted_class=labels.astype(np.uint8),
        sampling_seeds=np.arange(rows, dtype=np.uint64) + 100,
    )


def test_indexed_patient_cache_roundtrip_preserves_coordinates(tmp_path):
    expected = _cache("case-a", 7)
    path = tmp_path / "case-a.npz"
    save_indexed_patient_cache(expected, path)
    actual = load_indexed_patient_cache(path)

    assert actual.patient_id == expected.patient_id
    np.testing.assert_array_equal(actual.features["down1"], expected.features["down1"])
    np.testing.assert_array_equal(actual.slice_ids, expected.slice_ids)
    np.testing.assert_array_equal(actual.y, expected.y)
    np.testing.assert_array_equal(actual.x, expected.x)
    np.testing.assert_array_equal(actual.sampling_seeds, expected.sampling_seeds)


def test_balanced_node_cache_uses_equal_rows_per_patient(tmp_path):
    paths = []
    for patient_id, rows in (("case-b", 7), ("case-a", 5), ("case-c", 9)):
        path = tmp_path / f"{patient_id}.npz"
        save_indexed_patient_cache(_cache(patient_id, rows), path)
        paths.append(path)

    merged = load_balanced_node_cache(
        paths,
        node="down1",
        capacity=12,
        seed=42,
    )

    assert merged.patient_ids == ("case-a", "case-b", "case-c")
    assert merged.rows_per_patient == 4
    assert merged.features.shape == (12, 3)
    np.testing.assert_array_equal(merged.patient_indices, [0] * 4 + [1] * 4 + [2] * 4)
