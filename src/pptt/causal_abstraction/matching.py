from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = (
    "row_id",
    "patient_id",
    "node",
    "truth_class",
    "state",
    "boundary_distance",
    "spatial_scale",
    "feature_norm",
)
_OPTIONAL_IDENTITY_COLUMNS = ("split", "architecture", "model_seed")


@dataclass(frozen=True)
class MatchingRules:
    """Locked nuisance-matching and patient-cap rules."""

    same_patient_first: bool
    max_pixels_per_patient_node_state: int
    boundary_edges: tuple[float, ...]
    feature_norm_quantile_bins: int
    seed: int
    feature_norm_edges: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.max_pixels_per_patient_node_state <= 0:
            raise ValueError("matching cap must be positive")
        if self.feature_norm_quantile_bins < 2:
            raise ValueError("feature_norm_quantile_bins must be at least two")
        _validated_edges(self.boundary_edges, name="boundary_edges")
        if self.feature_norm_edges is not None:
            _validated_edges(self.feature_norm_edges, name="feature_norm_edges")


def _validated_edges(values: tuple[float, ...], *, name: str) -> np.ndarray:
    edges = np.asarray(values, dtype=np.float64)
    if edges.ndim != 1 or not edges.size or not np.isfinite(edges).all():
        raise ValueError(f"{name} must be a nonempty finite sequence")
    if np.any(np.diff(edges) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return edges


def _validate_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = tuple(column for column in _REQUIRED_COLUMNS if column not in frame)
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name} must be nonempty")
    result = frame.copy()
    if result["row_id"].isna().any() or not result["row_id"].is_unique:
        raise ValueError(f"{name} row_id values must be nonempty and unique")
    result["row_id"] = result["row_id"].astype(str)
    if (result["row_id"].str.len() == 0).any():
        raise ValueError(f"{name} row_id values must be nonempty and unique")
    for column in ("patient_id", "node", "spatial_scale"):
        if result[column].isna().any():
            raise ValueError(f"{name}.{column} contains missing values")
    for column in ("truth_class", "state"):
        values = result[column].to_numpy()
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name}.{column} must have integer dtype")
        if (values < 0).any():
            raise ValueError(f"{name}.{column} must be nonnegative")
    for column in ("boundary_distance", "feature_norm"):
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise ValueError(f"{name}.{column} must be finite")
        result[column] = values
    return result


def _stable_hash(seed: int, role: str, row_id: str) -> int:
    payload = f"{int(seed)}|{role}|{row_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _quantile_edges(values: np.ndarray, bin_count: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, int(bin_count) + 1)[1:-1]
    return np.unique(np.quantile(values, quantiles))


def _prepare_strata(
    base: pd.DataFrame,
    source: pd.DataFrame,
    rules: MatchingRules,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = base.copy()
    right = source.copy()
    boundary_edges = _validated_edges(rules.boundary_edges, name="boundary_edges")
    if rules.feature_norm_edges is None:
        norm_edges = _quantile_edges(
            right["feature_norm"].to_numpy(dtype=np.float64),
            rules.feature_norm_quantile_bins,
        )
    else:
        norm_edges = _validated_edges(
            rules.feature_norm_edges,
            name="feature_norm_edges",
        )
    for frame in (left, right):
        frame["_boundary_stratum"] = np.digitize(
            frame["boundary_distance"].to_numpy(dtype=np.float64),
            boundary_edges,
            right=False,
        )
        frame["_norm_stratum"] = np.digitize(
            frame["feature_norm"].to_numpy(dtype=np.float64),
            norm_edges,
            right=False,
        )
        frame["_stable_id"] = frame["row_id"].map(
            lambda row_id: _stable_hash(rules.seed, "row", row_id)
        )
    return left, right


def _same_value(left: Any, right: Any) -> bool:
    if pd.isna(left) or pd.isna(right):
        return False
    return bool(left == right)


def match_natural_sources(
    base: pd.DataFrame,
    source_bank: pd.DataFrame,
    rules: MatchingRules,
) -> pd.DataFrame:
    """Return deterministic natural source/base pairs without replacement."""

    left = _validate_frame(base, name="base")
    right = _validate_frame(source_bank, name="source_bank")
    identity_columns = tuple(
        column
        for column in _OPTIONAL_IDENTITY_COLUMNS
        if column in left.columns and column in right.columns
    )
    asymmetric_identity = tuple(
        column
        for column in _OPTIONAL_IDENTITY_COLUMNS
        if (column in left.columns) != (column in right.columns)
    )
    if asymmetric_identity:
        raise ValueError(
            "matching identity columns must be present in both frames: "
            f"{asymmetric_identity}"
        )
    if "requested_source_state" in left:
        requested = left["requested_source_state"].to_numpy()
        if not np.issubdtype(requested.dtype, np.integer) or (requested < 0).any():
            raise ValueError("requested_source_state must have nonnegative integer dtype")
        if np.any(requested == left["state"].to_numpy()):
            raise ValueError("requested_source_state must differ from base state")

    left, right = _prepare_strata(left, right, rules)
    left["_base_order"] = left["row_id"].map(
        lambda row_id: _stable_hash(rules.seed, "base", row_id)
    )
    left = left.sort_values(["_base_order", "row_id"], kind="mergesort")

    used_sources: set[str] = set()
    patient_state_counts: dict[tuple[str, str, int], int] = {}
    output_rows: list[dict[str, Any]] = []
    cap = int(rules.max_pixels_per_patient_node_state)

    for _, base_row in left.iterrows():
        candidates = right[
            (right["node"] == base_row["node"])
            & (right["truth_class"] == base_row["truth_class"])
            & (right["state"] != base_row["state"])
            & (right["_boundary_stratum"] == base_row["_boundary_stratum"])
            & (right["_norm_stratum"] == base_row["_norm_stratum"])
            & (right["spatial_scale"] == base_row["spatial_scale"])
        ].copy()
        for column in identity_columns:
            candidates = candidates[candidates[column] == base_row[column]]
        requested_state = base_row.get("requested_source_state", None)
        if requested_state is not None and not pd.isna(requested_state):
            candidates = candidates[candidates["state"] == int(requested_state)]
        candidates = candidates[~candidates["row_id"].isin(used_sources)]

        if not candidates.empty:
            candidates["_same_patient_rank"] = (
                candidates["patient_id"] != base_row["patient_id"]
            ).astype(np.int8)
            if not rules.same_patient_first:
                candidates["_same_patient_rank"] = 0
            candidates["_boundary_distance"] = np.abs(
                candidates["boundary_distance"] - base_row["boundary_distance"]
            )
            candidates["_norm_distance"] = np.abs(
                candidates["feature_norm"] - base_row["feature_norm"]
            )
            candidates = candidates.sort_values(
                [
                    "_same_patient_rank",
                    "_boundary_distance",
                    "_norm_distance",
                    "_stable_id",
                    "row_id",
                ],
                kind="mergesort",
            )

        selected = None
        for _, candidate in candidates.iterrows():
            count_key = (
                str(base_row["patient_id"]),
                str(base_row["node"]),
                int(candidate["state"]),
            )
            if patient_state_counts.get(count_key, 0) < cap:
                selected = candidate
                patient_state_counts[count_key] = (
                    patient_state_counts.get(count_key, 0) + 1
                )
                break

        common = {
            "status": "MATCHED" if selected is not None else "NO_MATCH",
            "patient_id": str(base_row["patient_id"]),
            "node": str(base_row["node"]),
            "base_row_id": str(base_row["row_id"]),
            "base_patient_id": str(base_row["patient_id"]),
            "base_truth_class": int(base_row["truth_class"]),
            "base_state": int(base_row["state"]),
            "requested_source_state": (
                int(requested_state)
                if requested_state is not None and not pd.isna(requested_state)
                else pd.NA
            ),
            "base_boundary_stratum": int(base_row["_boundary_stratum"]),
            "base_norm_stratum": int(base_row["_norm_stratum"]),
        }
        if selected is None:
            output_rows.append(
                {
                    **common,
                    "source_row_id": pd.NA,
                    "source_patient_id": pd.NA,
                    "source_truth_class": pd.NA,
                    "source_state": pd.NA,
                    "source_boundary_stratum": pd.NA,
                    "source_norm_stratum": pd.NA,
                    "same_patient": pd.NA,
                }
            )
            continue

        source_id = str(selected["row_id"])
        used_sources.add(source_id)
        output_rows.append(
            {
                **common,
                "source_row_id": source_id,
                "source_patient_id": str(selected["patient_id"]),
                "source_truth_class": int(selected["truth_class"]),
                "source_state": int(selected["state"]),
                "source_boundary_stratum": int(selected["_boundary_stratum"]),
                "source_norm_stratum": int(selected["_norm_stratum"]),
                "same_patient": bool(
                    _same_value(selected["patient_id"], base_row["patient_id"])
                ),
            }
        )

    result = pd.DataFrame(output_rows)
    result = result.sort_values("base_row_id", kind="mergesort").reset_index(drop=True)
    return result


__all__ = ["MatchingRules", "match_natural_sources"]
