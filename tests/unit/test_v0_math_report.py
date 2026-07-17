from scripts.run_v0_math import run


def test_v0_report_contains_theory_strengthening_results():
    report = run(
        {
            "seed": 17,
            "num_classes": 4,
            "stages": 8,
            "height": 12,
            "width": 10,
            "tolerance": 1.0e-12,
        }
    )

    assert report["additive_functional_exact"]["passed"]
    assert report["endpoint_hidden_dimension"]["dimension"] == 36
    assert report["endpoint_hidden_dimension"]["interior_point_assumption"]
    assert report["pairwise_not_lineage_sufficient"]["passed"]
    assert report["passed"]
