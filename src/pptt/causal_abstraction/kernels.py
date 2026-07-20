from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransitionProcess:
    """A non-homogeneous finite-state process over registered network nodes."""

    node_names: tuple[str, ...]
    kernels: np.ndarray
    coverage: np.ndarray
    alpha: float
    source: str

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.node_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("node_names must be nonempty and unique")
        kernels = np.asarray(self.kernels, dtype=np.float64)
        if (
            kernels.ndim != 3
            or kernels.shape[0] != len(names)
            or kernels.shape[1] != kernels.shape[2]
            or kernels.shape[1] < 2
        ):
            raise ValueError(
                "kernels must have shape node_count x state_count x state_count"
            )
        if not np.isfinite(kernels).all() or np.any(kernels < 0):
            raise ValueError("kernels must be finite and nonnegative")
        if not np.allclose(kernels.sum(axis=2), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("every transition-kernel row must sum to one")
        coverage = np.asarray(self.coverage, dtype=np.int64)
        if coverage.shape != kernels.shape[:2] or np.any(coverage < 0):
            raise ValueError("coverage must be a nonnegative node-by-state matrix")
        if not np.isfinite(float(self.alpha)) or float(self.alpha) < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if not str(self.source):
            raise ValueError("source must be nonempty")
        kernels = kernels.copy()
        coverage = coverage.copy()
        kernels.setflags(write=False)
        coverage.setflags(write=False)
        object.__setattr__(self, "node_names", names)
        object.__setattr__(self, "kernels", kernels)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "source", str(self.source))

    @property
    def state_count(self) -> int:
        return int(self.kernels.shape[1])

    def rollout(
        self,
        *,
        intervention_node: int,
        source_state: int,
    ) -> np.ndarray:
        """Return distributions after every transition from a node through Y."""

        node = int(intervention_node)
        state = int(source_state)
        if node < 0 or node >= len(self.node_names):
            raise ValueError(f"intervention_node is outside [0, {len(self.node_names)})")
        if state < 0 or state >= self.state_count:
            raise ValueError(f"source_state is outside [0, {self.state_count})")
        distribution = np.zeros(self.state_count, dtype=np.float64)
        distribution[state] = 1.0
        downstream: list[np.ndarray] = []
        for transition_index in range(node, len(self.node_names)):
            distribution = distribution @ self.kernels[transition_index]
            downstream.append(distribution.copy())
        return np.stack(downstream)

    def to_payload(self) -> dict[str, Any]:
        return {
            "node_names": list(self.node_names),
            "kernels": self.kernels.tolist(),
            "coverage": self.coverage.tolist(),
            "alpha": self.alpha,
            "source": self.source,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TransitionProcess":
        return cls(
            node_names=tuple(str(value) for value in payload["node_names"]),
            kernels=np.asarray(payload["kernels"], dtype=np.float64),
            coverage=np.asarray(payload["coverage"], dtype=np.int64),
            alpha=float(payload["alpha"]),
            source=str(payload["source"]),
        )


_TRANSITION_COLUMNS = {
    "patient_id",
    "model_seed",
    "transition_index",
    "state_from",
    "state_to",
    "count",
}


def _validated_transition_rows(
    rows: pd.DataFrame,
    *,
    transition_count: int,
    state_count: int,
) -> pd.DataFrame:
    missing = _TRANSITION_COLUMNS - set(rows.columns)
    if missing:
        raise ValueError(f"transition rows are missing columns: {sorted(missing)}")
    frame = rows.loc[:, sorted(_TRANSITION_COLUMNS)].copy()
    if frame.empty:
        raise ValueError("transition rows must be nonempty")
    if frame["patient_id"].isna().any() or (frame["patient_id"].astype(str) == "").any():
        raise ValueError("patient identifiers must be nonempty")
    for column in ("model_seed", "transition_index", "state_from", "state_to"):
        numeric = pd.to_numeric(frame[column], errors="raise")
        if not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{column} must contain integers")
        frame[column] = numeric.astype(np.int64)
    counts = pd.to_numeric(frame["count"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(counts).all() or np.any(counts <= 0):
        raise ValueError("transition counts must be finite and positive")
    frame["count"] = counts
    if not frame["transition_index"].between(0, transition_count - 1).all():
        raise ValueError("transition_index is outside the registered process")
    for column in ("state_from", "state_to"):
        if not frame[column].between(0, state_count - 1).all():
            raise ValueError(f"{column} is outside the registered state space")
    return frame


def estimate_patient_equal_process(
    rows: pd.DataFrame,
    *,
    node_names: Sequence[str],
    state_count: int = 5,
    alpha: float = 0.5,
    source: str = "unspecified",
) -> TransitionProcess:
    """Estimate source-conditional kernels with patient and seed equal weighting."""

    names = tuple(str(name) for name in node_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("node_names must be nonempty and unique")
    if int(state_count) < 2:
        raise ValueError("state_count must be at least two")
    if not np.isfinite(float(alpha)) or float(alpha) < 0:
        raise ValueError("alpha must be finite and nonnegative")
    frame = _validated_transition_rows(
        rows,
        transition_count=len(names),
        state_count=int(state_count),
    )
    kernels = np.full(
        (len(names), int(state_count), int(state_count)),
        1.0 / int(state_count),
        dtype=np.float64,
    )
    coverage = np.zeros((len(names), int(state_count)), dtype=np.int64)
    for transition_index in range(len(names)):
        for state_from in range(int(state_count)):
            selected = frame[
                (frame["transition_index"] == transition_index)
                & (frame["state_from"] == state_from)
            ]
            patient_probabilities: list[tuple[int, np.ndarray]] = []
            for (seed, _patient_id), group in selected.groupby(
                ["model_seed", "patient_id"],
                sort=True,
            ):
                counts = np.zeros(int(state_count), dtype=np.float64)
                np.add.at(
                    counts,
                    group["state_to"].to_numpy(dtype=np.int64),
                    group["count"].to_numpy(dtype=np.float64),
                )
                denominator = float(counts.sum() + float(alpha) * int(state_count))
                probabilities = (counts + float(alpha)) / denominator
                patient_probabilities.append((int(seed), probabilities))
            if not patient_probabilities:
                continue
            by_seed: list[np.ndarray] = []
            for seed in sorted({item[0] for item in patient_probabilities}):
                values = [item[1] for item in patient_probabilities if item[0] == seed]
                by_seed.append(np.mean(values, axis=0))
            kernels[transition_index, state_from] = np.mean(by_seed, axis=0)
            coverage[transition_index, state_from] = len(patient_probabilities)
    return TransitionProcess(
        node_names=names,
        kernels=kernels,
        coverage=coverage,
        alpha=float(alpha),
        source=source,
    )


def pool_architecture_processes(
    processes: Mapping[str, TransitionProcess],
) -> TransitionProcess:
    """Pool processes with one equal contribution from each architecture."""

    if len(processes) < 2:
        raise ValueError("at least two architecture processes are required")
    ordered = sorted((str(name), process) for name, process in processes.items())
    reference = ordered[0][1]
    for name, process in ordered[1:]:
        if process.node_names != reference.node_names:
            raise ValueError(f"node names differ for architecture {name}")
        if process.kernels.shape != reference.kernels.shape:
            raise ValueError(f"state dimensions differ for architecture {name}")
        if process.alpha != reference.alpha:
            raise ValueError(f"Dirichlet alpha differs for architecture {name}")
    kernels = np.mean([process.kernels for _, process in ordered], axis=0)
    coverage = np.min([process.coverage for _, process in ordered], axis=0)
    return TransitionProcess(
        node_names=reference.node_names,
        kernels=kernels,
        coverage=coverage,
        alpha=reference.alpha,
        source="shared:" + "+".join(name for name, _ in ordered),
    )


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a nonempty finite vector")
    if int(iterations) < 100:
        raise ValueError("bootstrap_iterations must be at least 100")
    if array.size == 1:
        return float(array[0]), float(array[0])
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, array.size, size=(int(iterations), array.size))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def evaluate_history_dependence(
    comparison_rows: pd.DataFrame,
    *,
    tolerance: float,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 20260720,
) -> dict[str, Any]:
    """Gate first-order adequacy using held-out first- versus second-order TV."""

    required = {"patient_id", "node", "first_order_tv", "second_order_tv"}
    missing = required - set(comparison_rows.columns)
    if missing:
        raise ValueError(f"history comparison rows are missing columns: {sorted(missing)}")
    if not np.isfinite(float(tolerance)) or float(tolerance) < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    frame = comparison_rows.loc[:, sorted(required)].copy()
    if frame.empty or frame[["patient_id", "node"]].isna().any().any():
        raise ValueError("history comparison rows must be nonempty and identified")
    first = pd.to_numeric(frame["first_order_tv"], errors="raise")
    second = pd.to_numeric(frame["second_order_tv"], errors="raise")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("history comparison errors must be finite")
    frame["tv_gain"] = first - second
    nodes: dict[str, Any] = {}
    failed: list[str] = []
    for index, (node, group) in enumerate(frame.groupby("node", sort=True)):
        gains = group["tv_gain"].to_numpy(dtype=np.float64)
        low, high = _bootstrap_mean_interval(
            gains,
            iterations=int(bootstrap_iterations),
            seed=int(bootstrap_seed) + index,
        )
        mean = float(gains.mean())
        node_failed = mean > float(tolerance) and low > 0.0
        if node_failed:
            failed.append(str(node))
        nodes[str(node)] = {
            "patient_count": int(group["patient_id"].nunique()),
            "mean_tv_gain": mean,
            "ci_low": low,
            "ci_high": high,
            "tolerance": float(tolerance),
            "failed": node_failed,
        }
    return {
        "status": "HIGH_LEVEL_MODEL_MISSPECIFIED" if failed else "PASS",
        "failed_nodes": failed,
        "nodes": nodes,
    }


_HISTORY_COLUMNS = {
    "patient_id",
    "model_seed",
    "transition_index",
    "previous_state",
    "state_from",
    "state_to",
    "count",
}


def _history_fold(patient_id: str, *, seed: int, fold_count: int) -> int:
    encoded = f"{int(seed)}|{patient_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % int(
        fold_count
    )


def _conditional_kernel(
    rows: pd.DataFrame,
    *,
    transition_count: int,
    state_count: int,
    alpha: float,
    second_order: bool,
) -> np.ndarray:
    condition_shape = (
        (transition_count, state_count, state_count, state_count)
        if second_order
        else (transition_count, state_count, state_count)
    )
    kernel = np.full(condition_shape, 1.0 / state_count, dtype=np.float64)
    condition_columns = (
        ("previous_state", "state_from") if second_order else ("state_from",)
    )
    group_columns = ["transition_index", *condition_columns]
    for condition, selected in rows.groupby(group_columns, sort=True):
        values = condition if isinstance(condition, tuple) else (condition,)
        patient_probabilities: list[tuple[int, np.ndarray]] = []
        for (model_seed, _patient_id), patient_rows in selected.groupby(
            ["model_seed", "patient_id"],
            sort=True,
        ):
            counts = np.zeros(state_count, dtype=np.float64)
            np.add.at(
                counts,
                patient_rows["state_to"].to_numpy(dtype=np.int64),
                patient_rows["count"].to_numpy(dtype=np.float64),
            )
            probabilities = (counts + alpha) / (counts.sum() + alpha * state_count)
            patient_probabilities.append((int(model_seed), probabilities))
        seed_probabilities = []
        for model_seed in sorted({value[0] for value in patient_probabilities}):
            seed_probabilities.append(
                np.mean(
                    [
                        probability
                        for seed, probability in patient_probabilities
                        if seed == model_seed
                    ],
                    axis=0,
                )
            )
        kernel[tuple(int(value) for value in values)] = np.mean(
            seed_probabilities,
            axis=0,
        )
    return kernel


def cross_validated_history_comparison(
    history_rows: pd.DataFrame,
    *,
    node_names: Sequence[str],
    state_count: int = 5,
    alpha: float = 0.5,
    fold_count: int = 5,
    fold_seed: int = 20260720,
) -> pd.DataFrame:
    """Compare held-out first- and second-order predictions by patient."""

    missing = _HISTORY_COLUMNS - set(history_rows.columns)
    if missing:
        raise ValueError(f"history rows are missing columns: {sorted(missing)}")
    names = tuple(str(value) for value in node_names)
    if len(names) != 8 or len(set(names)) != 8:
        raise ValueError("history admission requires the eight registered nodes")
    if int(state_count) < 2 or float(alpha) <= 0 or int(fold_count) < 2:
        raise ValueError("history model dimensions and smoothing must be positive")
    frame = history_rows.loc[:, sorted(_HISTORY_COLUMNS)].copy()
    if frame.empty:
        raise ValueError("history rows must be nonempty")
    for column in (
        "model_seed",
        "transition_index",
        "previous_state",
        "state_from",
        "state_to",
    ):
        numeric = pd.to_numeric(frame[column], errors="raise")
        if not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{column} must contain integers")
        frame[column] = numeric.astype(np.int64)
    counts = pd.to_numeric(frame["count"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(counts).all() or np.any(counts <= 0):
        raise ValueError("history counts must be finite and positive")
    frame["count"] = counts
    if not frame["transition_index"].between(1, len(names) - 1).all():
        raise ValueError("history transition_index must be in [1, 7]")
    for column in ("previous_state", "state_from", "state_to"):
        if not frame[column].between(0, int(state_count) - 1).all():
            raise ValueError(f"{column} is outside the registered state space")
    frame["patient_id"] = frame["patient_id"].astype(str)
    if (frame["patient_id"].str.len() == 0).any():
        raise ValueError("history patient identifiers must be nonempty")
    frame["fold"] = frame["patient_id"].map(
        lambda value: _history_fold(
            value,
            seed=int(fold_seed),
            fold_count=int(fold_count),
        )
    )

    evaluations: list[dict[str, Any]] = []
    for fold in range(int(fold_count)):
        train = frame[frame["fold"] != fold]
        held_out = frame[frame["fold"] == fold]
        if train.empty or held_out.empty:
            raise ValueError("history folds must contain training and held-out patients")
        first = _conditional_kernel(
            train,
            transition_count=len(names),
            state_count=int(state_count),
            alpha=float(alpha),
            second_order=False,
        )
        second = _conditional_kernel(
            train,
            transition_count=len(names),
            state_count=int(state_count),
            alpha=float(alpha),
            second_order=True,
        )
        for (
            patient_id,
            model_seed,
            transition_index,
        ), patient_rows in held_out.groupby(
            ["patient_id", "model_seed", "transition_index"],
            sort=True,
        ):
            first_total = 0.0
            second_total = 0.0
            weight_total = 0.0
            for (previous_state, state_from), condition_rows in patient_rows.groupby(
                ["previous_state", "state_from"],
                sort=True,
            ):
                empirical_counts = np.zeros(int(state_count), dtype=np.float64)
                np.add.at(
                    empirical_counts,
                    condition_rows["state_to"].to_numpy(dtype=np.int64),
                    condition_rows["count"].to_numpy(dtype=np.float64),
                )
                weight = float(empirical_counts.sum())
                empirical = empirical_counts / weight
                first_error = 0.5 * np.abs(
                    empirical - first[int(transition_index), int(state_from)]
                ).sum()
                second_error = 0.5 * np.abs(
                    empirical
                    - second[
                        int(transition_index),
                        int(previous_state),
                        int(state_from),
                    ]
                ).sum()
                first_total += weight * float(first_error)
                second_total += weight * float(second_error)
                weight_total += weight
            evaluations.append(
                {
                    "patient_id": str(patient_id),
                    "model_seed": int(model_seed),
                    "node": names[int(transition_index)],
                    "first_order_tv": first_total / weight_total,
                    "second_order_tv": second_total / weight_total,
                }
            )
    per_seed = pd.DataFrame(evaluations)
    return (
        per_seed.groupby(["patient_id", "node"], as_index=False, sort=True)[
            ["first_order_tv", "second_order_tv"]
        ]
        .mean()
        .sort_values(["node", "patient_id"], kind="mergesort")
        .reset_index(drop=True)
    )


__all__ = [
    "cross_validated_history_comparison",
    "TransitionProcess",
    "estimate_patient_equal_process",
    "evaluate_history_dependence",
    "pool_architecture_processes",
]
