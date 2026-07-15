from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from pptt.data.brats2d import SliceRecord, ensure_chw, normalize_label
from pptt.experiments.pilot import (
    sample_feature_vectors,
    sample_pixel_indices,
    stable_seed,
)
from pptt.observers.cache import select_patient_slice_indices
from pptt.observers.controls import spatially_shifted_targets


@dataclass(frozen=True)
class IndexedPatientCache:
    patient_id: str
    features: dict[str, np.ndarray]
    target_probabilities: np.ndarray
    spatial_target_probabilities: np.ndarray
    truth: np.ndarray
    slice_ids: np.ndarray
    y: np.ndarray
    x: np.ndarray
    predicted_class: np.ndarray
    sampling_seeds: np.ndarray

    @property
    def row_count(self) -> int:
        return int(self.truth.size)


@dataclass(frozen=True)
class BalancedNodeCache:
    patient_ids: tuple[str, ...]
    features: np.ndarray
    target_probabilities: np.ndarray
    spatial_target_probabilities: np.ndarray
    truth: np.ndarray
    patient_indices: np.ndarray
    rows_per_patient: int


def _validate(cache: IndexedPatientCache) -> None:
    rows = cache.row_count
    if rows <= 0 or not cache.features:
        raise ValueError("Patient cache must contain rows and node features")
    if cache.target_probabilities.ndim != 2:
        raise ValueError("target_probabilities must be a row-by-class matrix")
    classes = cache.target_probabilities.shape[1]
    expected_vectors = (
        cache.truth,
        cache.slice_ids,
        cache.y,
        cache.x,
        cache.predicted_class,
        cache.sampling_seeds,
    )
    if any(vector.shape != (rows,) for vector in expected_vectors):
        raise ValueError("Patient cache index arrays must align with cached rows")
    if cache.spatial_target_probabilities.shape != (rows, classes):
        raise ValueError("Spatial targets do not align with real targets")
    if any(feature.ndim != 2 or feature.shape[0] != rows for feature in cache.features.values()):
        raise ValueError("Every node feature must share the cache row axis")
    if not np.allclose(cache.target_probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Target probabilities must sum to one")
    if not np.allclose(cache.spatial_target_probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Spatial target probabilities must sum to one")
    if np.any(cache.predicted_class != cache.target_probabilities.argmax(axis=1)):
        raise ValueError("Cached predicted classes must match target probabilities")


def save_indexed_patient_cache(cache: IndexedPatientCache, path: str | Path) -> None:
    _validate(cache)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    node_names = tuple(cache.features)
    payload: dict[str, np.ndarray] = {
        "patient_id": np.asarray(cache.patient_id),
        "node_names": np.asarray(node_names),
        "target_probabilities": cache.target_probabilities.astype(np.float32),
        "spatial_target_probabilities": cache.spatial_target_probabilities.astype(
            np.float32
        ),
        "truth": cache.truth.astype(np.int64),
        "slice_ids": cache.slice_ids.astype(str),
        "y": cache.y.astype(np.int16),
        "x": cache.x.astype(np.int16),
        "predicted_class": cache.predicted_class.astype(np.uint8),
        "sampling_seeds": cache.sampling_seeds.astype(np.uint64),
    }
    payload.update(
        {
            f"feature_{index}": cache.features[node].astype(np.float16)
            for index, node in enumerate(node_names)
        }
    )
    np.savez_compressed(destination, **payload)


def load_indexed_patient_cache(path: str | Path) -> IndexedPatientCache:
    with np.load(Path(path), allow_pickle=False) as archive:
        nodes = tuple(str(value) for value in archive["node_names"].tolist())
        cache = IndexedPatientCache(
            patient_id=str(archive["patient_id"].item()),
            features={
                node: archive[f"feature_{index}"].astype(np.float16, copy=False)
                for index, node in enumerate(nodes)
            },
            target_probabilities=archive["target_probabilities"].astype(
                np.float32,
                copy=False,
            ),
            spatial_target_probabilities=archive[
                "spatial_target_probabilities"
            ].astype(np.float32, copy=False),
            truth=archive["truth"].astype(np.int64, copy=False),
            slice_ids=archive["slice_ids"].astype(str, copy=False),
            y=archive["y"].astype(np.int16, copy=False),
            x=archive["x"].astype(np.int16, copy=False),
            predicted_class=archive["predicted_class"].astype(np.uint8, copy=False),
            sampling_seeds=archive["sampling_seeds"].astype(np.uint64, copy=False),
        )
    _validate(cache)
    return cache


def extract_indexed_patient_cache(
    adapter: torch.nn.Module,
    records: list[SliceRecord],
    *,
    patient_id: str,
    nodes: tuple[str, ...],
    num_classes: int,
    max_lesion_slices: int,
    max_nonlesion_slices: int,
    max_pixels_per_class: int,
    spatial_shift_y: int,
    spatial_shift_x: int,
    seed: int,
    device: torch.device,
) -> tuple[IndexedPatientCache, dict[str, Any]]:
    ordered = sorted(records, key=lambda item: item.slice_id)
    lesion_areas = np.asarray(
        [
            np.count_nonzero(
                normalize_label(np.load(record.mask_path, allow_pickle=False))
            )
            for record in ordered
        ]
    )
    selected_indices = select_patient_slice_indices(
        lesion_areas,
        max_lesion_slices=max_lesion_slices,
        max_nonlesion_slices=max_nonlesion_slices,
    )
    selected = [ordered[int(index)] for index in selected_indices]
    feature_parts: dict[str, list[np.ndarray]] = {node: [] for node in nodes}
    target_parts: list[np.ndarray] = []
    spatial_parts: list[np.ndarray] = []
    truth_parts: list[np.ndarray] = []
    slice_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    x_parts: list[np.ndarray] = []
    class_parts: list[np.ndarray] = []
    seed_parts: list[np.ndarray] = []
    inference_seconds = 0.0

    for record in selected:
        image = torch.from_numpy(
            ensure_chw(np.load(record.image_path, allow_pickle=False))
        )[None].to(device)
        truth = normalize_label(np.load(record.mask_path, allow_pickle=False))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        with torch.inference_mode():
            trace = adapter.trace(image)
            probabilities = torch.softmax(trace.logits, dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += perf_counter() - start
        sampling_seed = stable_seed(seed, patient_id, record.slice_id, "pixels")
        flat_indices = sample_pixel_indices(
            probabilities[0],
            max_per_class=max_pixels_per_class,
            seed=sampling_seed,
        )
        flat_numpy = flat_indices.cpu().numpy()
        probability_map = probabilities[0].cpu().numpy()
        spatial_map = spatially_shifted_targets(
            probability_map,
            shift_y=spatial_shift_y,
            shift_x=spatial_shift_x,
        )
        targets = probability_map.transpose(1, 2, 0).reshape(-1, num_classes)[
            flat_numpy
        ]
        target_parts.append(targets)
        spatial_parts.append(
            spatial_map.transpose(1, 2, 0).reshape(-1, num_classes)[flat_numpy]
        )
        truth_parts.append(truth.reshape(-1)[flat_numpy])
        y = flat_numpy // truth.shape[1]
        x = flat_numpy % truth.shape[1]
        y_parts.append(y.astype(np.int16))
        x_parts.append(x.astype(np.int16))
        slice_parts.append(np.full(flat_numpy.size, record.slice_id))
        class_parts.append(targets.argmax(axis=1).astype(np.uint8))
        seed_parts.append(np.full(flat_numpy.size, sampling_seed, dtype=np.uint64))
        for node in nodes:
            feature_parts[node].append(
                sample_feature_vectors(
                    trace.activations[node],
                    flat_indices,
                    output_size=truth.shape,
                )
                .cpu()
                .numpy()
                .astype(np.float16)
            )
        del trace, probabilities, image

    cache = IndexedPatientCache(
        patient_id=patient_id,
        features={node: np.concatenate(parts) for node, parts in feature_parts.items()},
        target_probabilities=np.concatenate(target_parts).astype(np.float32),
        spatial_target_probabilities=np.concatenate(spatial_parts).astype(np.float32),
        truth=np.concatenate(truth_parts).astype(np.int64),
        slice_ids=np.concatenate(slice_parts),
        y=np.concatenate(y_parts),
        x=np.concatenate(x_parts),
        predicted_class=np.concatenate(class_parts),
        sampling_seeds=np.concatenate(seed_parts),
    )
    _validate(cache)
    return cache, {
        "patient_id": patient_id,
        "selected_slices": len(selected),
        "rows": cache.row_count,
        "inference_seconds": inference_seconds,
    }


def load_balanced_node_cache(
    paths: list[Path],
    *,
    node: str,
    capacity: int,
    seed: int,
) -> BalancedNodeCache:
    if len(paths) < 2 or capacity < len(paths):
        raise ValueError("Balanced cache requires at least two patients and capacity")
    ordered = sorted(paths, key=lambda path: path.stem)
    counts: list[int] = []
    patient_ids: list[str] = []
    for path in ordered:
        with np.load(path, allow_pickle=False) as archive:
            patient_ids.append(str(archive["patient_id"].item()))
            counts.append(int(archive["truth"].size))
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("Balanced cache paths contain duplicate patients")
    rows_per_patient = min(min(counts), capacity // len(paths))
    if rows_per_patient <= 0:
        raise ValueError("Capacity cannot retain one row per patient")
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    spatial_targets: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    for path, patient_id, count in zip(ordered, patient_ids, counts, strict=True):
        cache = load_indexed_patient_cache(path)
        if node not in cache.features:
            raise KeyError(f"Node {node!r} is absent from {path}")
        rng = np.random.default_rng(stable_seed(seed, patient_id, node))
        selected = rng.permutation(count)[:rows_per_patient]
        features.append(cache.features[node][selected])
        targets.append(cache.target_probabilities[selected])
        spatial_targets.append(cache.spatial_target_probabilities[selected])
        truth.append(cache.truth[selected])
    return BalancedNodeCache(
        patient_ids=tuple(patient_ids),
        features=np.concatenate(features),
        target_probabilities=np.concatenate(targets),
        spatial_target_probabilities=np.concatenate(spatial_targets),
        truth=np.concatenate(truth),
        patient_indices=np.repeat(
            np.arange(len(paths), dtype=np.int32),
            rows_per_patient,
        ),
        rows_per_patient=rows_per_patient,
    )


def write_split_cache_index(
    cache_paths: list[Path],
    *,
    output: str | Path,
    split: str,
    model: str,
    model_seed: int,
    checkpoint_sha256: str,
) -> int:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        for path in sorted(cache_paths):
            cache = load_indexed_patient_cache(path)
            for node in cache.features:
                frame = pd.DataFrame(
                    {
                        "model": model,
                        "model_seed": model_seed,
                        "checkpoint_sha256": checkpoint_sha256,
                        "split": split,
                        "patient_id": cache.patient_id,
                        "slice_id": cache.slice_ids,
                        "y": cache.y,
                        "x": cache.x,
                        "predicted_class": cache.predicted_class,
                        "node": node,
                        "sampling_seed": cache.sampling_seeds,
                    }
                )
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
                writer.write_table(table)
                row_count += len(frame)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("No cache rows were available for the split index")
    return row_count
