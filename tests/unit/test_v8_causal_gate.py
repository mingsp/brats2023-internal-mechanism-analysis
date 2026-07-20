from pptt.statistics.causal_replication import evaluate_replication_gate


def _rows(*, failed_endpoint: str | None = None):
    rows = []
    for endpoint in ("necessity", "restoration", "specificity"):
        for seed in (42, 123, 3407):
            supported = not (endpoint == failed_endpoint and seed in (42, 123))
            rows.append(
                {
                    "endpoint": endpoint,
                    "model_seed": seed,
                    "patient_count": 40,
                    "mean_difference": 0.15 if supported else 0.01,
                    "ci_low": 0.10 if supported else -0.01,
                    "holm_p": 0.01 if supported else 0.5,
                    "paired_cohens_d": 1.2 if supported else 0.2,
                }
            )
    return rows


def _dose_rows(*, failed_seeds=()):
    return [
        {
            "model_seed": seed,
            "patient_count": 40,
            "monotonic": seed not in failed_seeds,
        }
        for seed in (42, 123, 3407)
    ]


def _operator_rows(*, failed_seeds=()):
    return [
        {"model_seed": seed, "passed": seed not in failed_seeds}
        for seed in (42, 123, 3407)
    ]


def test_replication_gate_requires_all_registered_components():
    result = evaluate_replication_gate(
        _rows(),
        dose_rows=_dose_rows(),
        operator_rows=_operator_rows(),
        minimum_supported_seeds=2,
        minimum_patients_per_seed=20,
        alpha=0.05,
        minimum_standardized_effect=0.8,
    )

    assert result["passed"]
    assert result["status"] == "INTERVENTIONALLY_FAITHFUL_TRANSUNET_REPLICATION"
    assert all(result["gates"].values())


def test_failed_specificity_cannot_authorize_mechanism_claim():
    result = evaluate_replication_gate(
        _rows(failed_endpoint="specificity"),
        dose_rows=_dose_rows(),
        operator_rows=_operator_rows(),
        minimum_supported_seeds=2,
        minimum_patients_per_seed=20,
        alpha=0.05,
        minimum_standardized_effect=0.8,
    )

    assert not result["passed"]
    assert not result["gates"]["specificity"]
    assert result["status"] == "TRANSUNET_PROCESS_CANDIDATE_NOT_CAUSALLY_CONFIRMED"


def test_operator_or_dose_failure_is_not_hidden():
    result = evaluate_replication_gate(
        _rows(),
        dose_rows=_dose_rows(failed_seeds=(42, 123)),
        operator_rows=_operator_rows(failed_seeds=(42, 123)),
        minimum_supported_seeds=2,
        minimum_patients_per_seed=20,
        alpha=0.05,
        minimum_standardized_effect=0.8,
    )

    assert not result["passed"]
    assert not result["gates"]["dose_response"]
    assert not result["gates"]["operator_audit"]


def test_small_evaluable_sample_cannot_pass_even_with_large_effect():
    rows = _rows()
    for row in rows:
        if row["model_seed"] in (42, 123):
            row["patient_count"] = 5
    result = evaluate_replication_gate(
        rows,
        dose_rows=_dose_rows(),
        operator_rows=_operator_rows(),
        minimum_supported_seeds=2,
        minimum_patients_per_seed=20,
        alpha=0.05,
        minimum_standardized_effect=0.8,
    )

    assert not result["passed"]
    assert not result["gates"]["necessity"]
    assert not result["gates"]["restoration"]
    assert not result["gates"]["specificity"]
