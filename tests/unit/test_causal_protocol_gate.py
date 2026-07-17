from pptt.statistics.process_effects import candidate_specific_causal_gate


def _row(seed: int, *, effect: float, ci_low: float, holm_p: float, d: float):
    return {
        "transition_index": 4,
        "transition": "up1->up2",
        "model_seed": seed,
        "mean_difference": effect,
        "ci_low": ci_low,
        "holm_p": holm_p,
        "paired_cohens_d": d,
    }


def test_candidate_specific_gate_requires_every_registered_seed():
    rows = [
        _row(42, effect=0.11, ci_low=0.10, holm_p=0.001, d=2.2),
        _row(123, effect=0.10, ci_low=0.09, holm_p=0.001, d=2.0),
        _row(3407, effect=0.08, ci_low=0.07, holm_p=0.001, d=2.1),
    ]

    result = candidate_specific_causal_gate(
        rows,
        transition="up1->up2",
        required_model_seeds=(42, 123, 3407),
        alpha=0.05,
        minimum_standardized_effect=0.8,
    )

    assert result["passed"] is True
    assert result["supported_seed_count"] == 3


def test_candidate_specific_gate_fails_without_large_effect_in_one_seed():
    rows = [
        _row(42, effect=0.11, ci_low=0.10, holm_p=0.001, d=2.2),
        _row(123, effect=0.10, ci_low=0.09, holm_p=0.001, d=2.0),
        _row(3407, effect=0.08, ci_low=0.07, holm_p=0.001, d=0.6),
    ]

    result = candidate_specific_causal_gate(
        rows,
        transition="up1->up2",
        required_model_seeds=(42, 123, 3407),
        alpha=0.05,
        minimum_standardized_effect=0.8,
    )

    assert result["passed"] is False
    assert result["supported_seed_count"] == 2

