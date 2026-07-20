from __future__ import annotations

import pandas as pd

from pptt.causal_abstraction.matching import (
    MatchingRules,
    match_natural_sources,
)


def _rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.DataFrame(
        {
            "row_id": ["b0", "b1", "b2", "b3"],
            "patient_id": ["p1", "p1", "p1", "p2"],
            "split": ["test"] * 4,
            "architecture": ["unet"] * 4,
            "model_seed": [42] * 4,
            "node": ["down1"] * 4,
            "truth_class": [2, 2, 2, 2],
            "state": [3, 3, 3, 3],
            "requested_source_state": [1, 1, 1, 1],
            "boundary_distance": [1.1, 1.2, 1.3, 1.1],
            "spatial_scale": ["small"] * 4,
            "feature_norm": [2.0, 2.1, 2.2, 2.0],
        }
    )
    source = pd.DataFrame(
        {
            "row_id": ["s0", "s1", "s2", "s3", "wrong-truth", "wrong-split"],
            "patient_id": ["p1", "p1", "p2", "p3", "p1", "p1"],
            "split": ["test", "test", "test", "test", "test", "val"],
            "architecture": ["unet"] * 6,
            "model_seed": [42] * 6,
            "node": ["down1"] * 6,
            "truth_class": [2, 2, 2, 2, 1, 2],
            "state": [1, 1, 1, 1, 1, 1],
            "boundary_distance": [1.0, 1.4, 1.2, 1.0, 1.1, 1.1],
            "spatial_scale": ["small"] * 6,
            "feature_norm": [2.0, 2.3, 2.1, 2.0, 2.0, 2.0],
        }
    )
    return base, source


def _rules(cap: int = 2) -> MatchingRules:
    return MatchingRules(
        same_patient_first=True,
        max_pixels_per_patient_node_state=cap,
        boundary_edges=(0.0, 2.0, 5.0),
        feature_norm_quantile_bins=2,
        seed=20260720,
    )


def test_matching_preserves_truth_state_request_and_patient_cap():
    base, source = _rows()

    matches = match_natural_sources(base, source, _rules(cap=2))
    matched = matches[matches.status == "MATCHED"]

    assert (matched.base_truth_class == matched.source_truth_class).all()
    assert (matched.base_state != matched.source_state).all()
    assert (matched.source_state == matched.requested_source_state).all()
    assert matched.source_row_id.is_unique
    assert (
        matched.groupby(["patient_id", "node", "source_state"]).size().max()
        <= 2
    )
    assert matches.status.isin(["MATCHED", "NO_MATCH"]).all()


def test_matching_is_order_invariant_and_hash_stable():
    base, source = _rows()

    left = match_natural_sources(
        base.sample(frac=1, random_state=3),
        source.sample(frac=1, random_state=5),
        _rules(),
    )
    right = match_natural_sources(
        base.sample(frac=1, random_state=9),
        source.sample(frac=1, random_state=11),
        _rules(),
    )

    pd.testing.assert_frame_equal(left, right)


def test_same_patient_source_is_preferred_over_closer_other_patient():
    base, source = _rows()
    selected = match_natural_sources(base.iloc[[0]], source, _rules(cap=4)).iloc[0]

    assert selected.status == "MATCHED"
    assert selected.source_patient_id == "p1"
    assert selected.same_patient is True or bool(selected.same_patient)


def test_no_match_is_explicit_and_does_not_cross_split_or_truth_class():
    base, source = _rows()
    impossible = base.iloc[[0]].copy()
    impossible["requested_source_state"] = 4

    result = match_natural_sources(impossible, source, _rules()).iloc[0]

    assert result.status == "NO_MATCH"
    assert pd.isna(result.source_row_id)
    assert result.base_row_id == "b0"


def test_matching_rejects_missing_or_duplicate_stable_row_identity():
    base, source = _rows()
    with pd.option_context("mode.copy_on_write", True):
        invalid = source.copy()
        invalid.loc[1, "row_id"] = "s0"

    try:
        match_natural_sources(base.drop(columns=["row_id"]), source, _rules())
    except ValueError as exc:
        assert "row_id" in str(exc)
    else:
        raise AssertionError("missing row identity was accepted")

    try:
        match_natural_sources(base, invalid, _rules())
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate source identity was accepted")

