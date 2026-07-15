from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class OODReference:
    feature_mean: np.ndarray
    components: np.ndarray
    projected_mean: np.ndarray
    precision: np.ndarray
    reference_projected: np.ndarray
    neighbors: int
    mahalanobis_threshold: float
    knn_threshold: float


def save_ood_reference(path: str | Path, reference: OODReference) -> None:
    np.savez_compressed(
        path,
        feature_mean=reference.feature_mean,
        components=reference.components,
        projected_mean=reference.projected_mean,
        precision=reference.precision,
        reference_projected=reference.reference_projected,
        neighbors=np.asarray(reference.neighbors, dtype=np.int64),
        mahalanobis_threshold=np.asarray(reference.mahalanobis_threshold),
        knn_threshold=np.asarray(reference.knn_threshold),
    )


def load_ood_reference(path: str | Path) -> OODReference:
    with np.load(Path(path), allow_pickle=False) as archive:
        return OODReference(
            feature_mean=archive["feature_mean"],
            components=archive["components"],
            projected_mean=archive["projected_mean"],
            precision=archive["precision"],
            reference_projected=archive["reference_projected"],
            neighbors=int(archive["neighbors"].item()),
            mahalanobis_threshold=float(
                archive["mahalanobis_threshold"].item()
            ),
            knn_threshold=float(archive["knn_threshold"].item()),
        )


@dataclass(frozen=True)
class OODScores:
    mahalanobis: np.ndarray
    knn: np.ndarray


def local_context_descriptor(activation: np.ndarray) -> np.ndarray:
    """Describe each location by its center and 3x3 channel-wise context."""
    values = np.asarray(activation, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("activation must be a finite CxHxW tensor")
    local_mean = ndimage.uniform_filter(
        values,
        size=(1, 3, 3),
        mode="nearest",
    )
    local_second_moment = ndimage.uniform_filter(
        np.square(values),
        size=(1, 3, 3),
        mode="nearest",
    )
    local_variance = np.maximum(local_second_moment - np.square(local_mean), 0.0)
    return np.concatenate(
        [values, local_mean, np.sqrt(local_variance)],
        axis=0,
    ).astype(np.float32, copy=False)


def _project(reference: OODReference, vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != reference.feature_mean.size:
        raise ValueError("vectors do not match the OOD reference feature dimension")
    return (values - reference.feature_mean) @ reference.components.T


def fit_ood_reference(
    vectors: np.ndarray,
    *,
    components: int,
    neighbors: int,
    max_samples: int,
    seed: int,
    threshold_quantile: float = 0.99,
) -> OODReference:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 8 or values.shape[1] < 2:
        raise ValueError("OOD fitting requires a nontrivial row-by-feature matrix")
    if not 0 < threshold_quantile < 1 or neighbors < 1 or max_samples < 8:
        raise ValueError("Invalid OOD fitting configuration")
    if values.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        values = values[rng.choice(values.shape[0], max_samples, replace=False)]
    component_count = min(int(components), values.shape[1], values.shape[0] - 1)
    pca = PCA(n_components=component_count, svd_solver="full")
    projected = pca.fit_transform(values)
    covariance = LedoitWolf().fit(projected)
    centered = projected - covariance.location_
    mahalanobis = np.sqrt(
        np.einsum("ni,ij,nj->n", centered, covariance.precision_, centered)
    )
    neighbor_count = min(int(neighbors), projected.shape[0] - 1)
    nearest = NearestNeighbors(n_neighbors=neighbor_count + 1).fit(projected)
    distances = nearest.kneighbors(projected, return_distance=True)[0][:, 1:]
    knn = distances.mean(axis=1)
    return OODReference(
        feature_mean=pca.mean_,
        components=pca.components_,
        projected_mean=covariance.location_,
        precision=covariance.precision_,
        reference_projected=projected.astype(np.float32),
        neighbors=neighbor_count,
        mahalanobis_threshold=float(np.quantile(mahalanobis, threshold_quantile)),
        knn_threshold=float(np.quantile(knn, threshold_quantile)),
    )


def score_ood(reference: OODReference, vectors: np.ndarray) -> OODScores:
    projected = _project(reference, vectors)
    centered = projected - reference.projected_mean
    mahalanobis = np.sqrt(
        np.einsum("ni,ij,nj->n", centered, reference.precision, centered)
    )
    nearest = NearestNeighbors(n_neighbors=reference.neighbors).fit(
        reference.reference_projected
    )
    distances = nearest.kneighbors(projected, return_distance=True)[0]
    return OODScores(
        mahalanobis=mahalanobis,
        knn=distances.mean(axis=1),
    )
