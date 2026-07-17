from pptt.statistics.causal_effects import evaluate_causal_conclusion_gate


def _rows(specificity_supported: bool = True):
    rows = []
    for endpoint in ("necessity", "restoration", "specificity"):
        for seed in (42, 123, 3407):
            supported = specificity_supported or endpoint != "specificity" or seed == 42
            rows.append(
                {
                    "endpoint": endpoint,
                    "model_seed": seed,
                    "mean_difference": 0.2 if supported else 0.01,
                    "ci_low": 0.1 if supported else -0.01,
                    "holm_p": 0.001 if supported else 0.2,
                    "paired_cohens_d": 1.2 if supported else 0.2,
                }
            )
    return rows


def test_causal_gate_requires_all_registered_evidence_families():
    gate = evaluate_causal_conclusion_gate(
        _rows(),
        dose_rows=[
            {"model_seed": 42, "monotonic": True},
            {"model_seed": 123, "monotonic": True},
            {"model_seed": 3407, "monotonic": False},
        ],
        negative_control_pass=True,
        restoration_audit_pass=True,
        minimum_supported_seeds=2,
        alpha=0.05,
        minimum_standardized_effect=0.8,
    )

    assert gate["passed"] is True
    assert gate["endpoint_audit"]["specificity"]["supported_seed_count"] == 3


def test_causal_gate_fails_when_specificity_is_not_replicated():
    gate = evaluate_causal_conclusion_gate(
        _rows(specificity_supported=False),
        dose_rows=[
            {"model_seed": 42, "monotonic": True},
            {"model_seed": 123, "monotonic": True},
            {"model_seed": 3407, "monotonic": True},
        ],
        negative_control_pass=True,
        restoration_audit_pass=True,
        minimum_supported_seeds=2,
        alpha=0.05,
        minimum_standardized_effect=0.8,
    )

    assert gate["passed"] is False
    assert gate["endpoint_audit"]["specificity"]["supported_seed_count"] == 1
