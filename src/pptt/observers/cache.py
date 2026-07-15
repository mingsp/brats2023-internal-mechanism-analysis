from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pptt.observers.objective import class_balanced_pixel_weights


@dataclass(frozen=True)
class ObserverCache:
    features: np.ndarray
    target_probabilities: np.ndarray
    pixel_weights: np.ndarray
    patient_indices: np.ndarray

    @property
    def in_channels(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_classes(self) -> int:
        return int(self.target_probabilities.shape[1])

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            features=self.features,
            target_probabilities=self.target_probabilities,
            pixel_weights=self.pixel_weights,
            patient_indices=self.patient_indices,
        )


def select_patient_slice_indices(
    lesion_areas: npt.ArrayLike,
    *,
    max_lesion_slices: int = 6,
    max_nonlesion_slices: int = 2,
) -> np.ndarray:
    areas = np.asarray(lesion_areas)
    if areas.ndim != 1 or areas.size == 0:
        raise ValueError("lesion_areas must be a nonempty one-dimensional array")
    if not np.issubdtype(areas.dtype, np.number) or not np.isfinite(areas).all():
        raise ValueError("lesion_areas must be finite numeric values")
    if np.any(areas < 0):
        raise ValueError("lesion_areas must be nonnegative")
    if max_lesion_slices <= 0 or max_nonlesion_slices < 0:
        raise ValueError("slice limits must be positive for lesion and nonnegative otherwise")

    def quantile_indices(candidates: np.ndarray, limit: int) -> list[int]:
        if limit == 0 or candidates.size == 0:
            return []
        if candidates.size <= limit:
            return candidates.tolist()
        targets = np.quantile(areas[candidates], np.linspace(0.0, 1.0, limit))
        selected: list[int] = []
        for target in targets:
            remaining = [int(index) for index in candidates if int(index) not in selected]
            chosen = min(
                remaining,
                key=lambda index: (abs(float(areas[index]) - float(target)), index),
            )
            selected.append(chosen)
        return selected

    lesion = np.flatnonzero(areas > 0)
    nonlesion = np.flatnonzero(areas == 0)
    selected_lesion = quantile_indices(lesion, max_lesion_slices)
    selected_nonlesion = quantile_indices(nonlesion, max_nonlesion_slices)
    return np.asarray(selected_lesion + selected_nonlesion, dtype=np.int64)


def build_observer_cache(
    features: npt.ArrayLike,
    target_probabilities: npt.ArrayLike,
    *,
    patient_indices: npt.ArrayLike | None = None,
    pixel_weights: npt.ArrayLike | None = None,
) -> ObserverCache:
    feature_array = np.asarray(features)
    target_array = np.asarray(target_probabilities)
    if feature_array.ndim != 2 or feature_array.shape[0] == 0:
        raise ValueError(f"features must be a nonempty NxC array, got {feature_array.shape}")
    if target_array.ndim != 2 or target_array.shape[0] != feature_array.shape[0]:
        raise ValueError("target_probabilities must be NxK with the same N as features")
    if not np.issubdtype(feature_array.dtype, np.floating):
        raise ValueError("features must have floating dtype")
    if not np.isfinite(feature_array).all():
        raise ValueError("features must be finite")
    target_array = target_array.astype(np.float32, copy=False)
    if np.any(target_array < 0) or not np.isfinite(target_array).all():
        raise ValueError("target probabilities must be finite and nonnegative")
    if not np.allclose(target_array.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("target probabilities must sum to one")

    if patient_indices is None:
        patients = np.zeros(feature_array.shape[0], dtype=np.int32)
    else:
        patients = np.asarray(patient_indices)
        if patients.shape != (feature_array.shape[0],) or not np.issubdtype(
            patients.dtype,
            np.integer,
        ):
            raise ValueError("patient_indices must be an integer vector with one entry per pixel")
        patients = patients.astype(np.int32, copy=False)

    if pixel_weights is None:
        weights = class_balanced_pixel_weights(target_array)
    else:
        weights = np.asarray(pixel_weights, dtype=np.float32)
        if weights.shape != (feature_array.shape[0],) or np.any(weights <= 0):
            raise ValueError("pixel_weights must be a positive vector with one entry per pixel")
        weights = weights / weights.mean()
    return ObserverCache(
        features=feature_array.astype(np.float16),
        target_probabilities=target_array.astype(np.float32, copy=False),
        pixel_weights=weights.astype(np.float32, copy=False),
        patient_indices=patients,
    )


def load_observer_cache(path: str | Path) -> ObserverCache:
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        return build_observer_cache(
            archive["features"],
            archive["target_probabilities"],
            patient_indices=archive["patient_indices"],
            pixel_weights=archive["pixel_weights"],
        )


def sample_slice_pixels(
    feature_map: npt.ArrayLike,
    target_probabilities: npt.ArrayLike,
    *,
    patient_index: int,
    max_per_class: int = 32,
    seed: int,
) -> ObserverCache:
    features = np.asarray(feature_map)
    targets = np.asarray(target_probabilities)
    if features.ndim != 3 or targets.ndim != 3:
        raise ValueError("feature_map and target_probabilities must be CHW arrays")
    if features.shape[1:] != targets.shape[1:]:
        raise ValueError("feature and target maps must already share spatial resolution")
    if max_per_class <= 0:
        raise ValueError("max_per_class must be positive")
    labels = targets.argmax(axis=0).reshape(-1)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_index in range(targets.shape[0]):
        candidates = np.flatnonzero(labels == class_index)
        if candidates.size:
            selected.append(
                rng.choice(
                    candidates,
                    size=min(max_per_class, candidates.size),
                    replace=False,
                )
            )
    if not selected:
        raise ValueError("No pixels were available for observer caching")
    indices = np.concatenate(selected)
    flat_features = np.moveaxis(features, 0, -1).reshape(-1, features.shape[0])
    flat_targets = np.moveaxis(targets, 0, -1).reshape(-1, targets.shape[0])
    return build_observer_cache(
        flat_features[indices],
        flat_targets[indices],
        patient_indices=np.full(indices.size, patient_index, dtype=np.int32),
    )


def round_robin_merge(
    caches: list[ObserverCache],
    *,
    capacity: int = 262_144,
    seed: int,
) -> ObserverCache:
    if not caches or capacity <= 0:
        raise ValueError("At least one cache and positive capacity are required")
    features = np.concatenate([cache.features for cache in caches])
    targets = np.concatenate([cache.target_probabilities for cache in caches])
    patients = np.concatenate([cache.patient_indices for cache in caches])
    rng = np.random.default_rng(seed)
    queues: dict[int, np.ndarray] = {}
    for patient in sorted(np.unique(patients).tolist()):
        indices = np.flatnonzero(patients == patient)
        queues[int(patient)] = rng.permutation(indices)
    selected: list[int] = []
    offset = 0
    while len(selected) < min(capacity, features.shape[0]):
        added = False
        for patient in sorted(queues):
            queue = queues[patient]
            if offset < queue.size and len(selected) < capacity:
                selected.append(int(queue[offset]))
                added = True
        if not added:
            break
        offset += 1
    indices = np.asarray(selected, dtype=np.int64)
    return build_observer_cache(
        features[indices],
        targets[indices],
        patient_indices=patients[indices],
    )
