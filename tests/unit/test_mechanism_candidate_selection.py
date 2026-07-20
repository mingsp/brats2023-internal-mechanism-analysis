import pandas as pd

from pptt.statistics.mechanism_replication import (
    select_registered_candidate,
    summarize_validation_candidates,
)


SEEDS = (42, 123, 3407)


def _statistics_row(
    path_id: str,
    topology_order: int,
    seed: int,
    *,
    mean: float,
    ci_low: float,
    holm_p: float,
    effect: float,
):
    return {
        "path_id": path_id,
        "topology_order": topology_order,
        "model_seed": seed,
        "mean_difference": mean,
        "ci_low": ci_low,
        "ci_high": mean + 0.02,
        "wilcoxon_p": holm_p,
        "holm_p": holm_p,
        "paired_cohens_d": effect,
    }


def _coverage_row(path_id: str, topology_order: int, seed: int):
    return {
        "path_id": path_id,
        "topology_order": topology_order,
        "model_seed": seed,
        "patient_count": 125,
        "eligible_patient_count": 40,
        "aggregate_target_pixel_count": 2400,
    }


def test_selector_uses_cross_seed_minimum_effect_then_topology():
    statistics = []
    coverage = []
    for path_id, order, effects in (
        ("bottleneck_to_up1", 0, (1.1, 1.0, 0.9)),
        ("skip_down2_to_up2", 2, (1.3, 1.2, 1.0)),
        ("skip_down1_to_up3", 3, (1.0, 1.0, 1.0)),
    ):
        for seed, effect in zip(SEEDS, effects, strict=True):
            statistics.append(
                _statistics_row(
                    path_id,
                    order,
                    seed,
                    mean=0.08,
                    ci_low=0.04,
                    holm_p=0.01,
                    effect=effect,
                )
            )
            coverage.append(_coverage_row(path_id, order, seed))

    selected = select_registered_candidate(
        statistics,
        coverage,
        required_seeds=SEEDS,
        minimum_supported_seeds=2,
        minimum_patients_per_seed=20,
        minimum_target_pixels_per_seed=512,
        alpha=0.05,
        minimum_standardized_effect=0.8,
    )

    assert selected["status"] == "REGISTERED_TRANSUNET_CANDIDATE"
    assert selected["test_intervention_authorized"]
    assert selected["candidate"]["path_id"] == "skip_down2_to_up2"
    assert selected["candidate"]["minimum_seed_effect"] == 1.0


def test_selector_uses_topology_only_after_effect_ties():
    statistics = []
    coverage = []
    for path_id, order in (("earlier", 1), ("later", 2)):
        for seed in SEEDS:
            statistics.append(
                _statistics_row(
                    path_id,
                    order,
                    seed,
                    mean=0.08,
                    ci_low=0.04,
                    holm_p=0.01,
                    effect=1.0,
                )
            )
            coverage.append(_coverage_row(path_id, order, seed))

    selected = select_registered_candidate(
        statistics,
        coverage,
        required_seeds=SEEDS,
        minimum_supported_seeds=2,
        minimum_patients_per_seed=20,
        minimum_target_pixels_per_seed=512,
        alpha=0.05,
        minimum_standardized_effect=0.8,
    )

    assert selected["candidate"]["path_id"] == "earlier"


def test_selector_returns_explicit_no_candidate_status():
    statistics = [
        _statistics_row(
            "path",
            0,
            seed,
            mean=(-0.01 if seed == 3407 else 0.08),
            ci_low=(-0.03 if seed == 3407 else 0.04),
            holm_p=0.01,
            effect=(-0.2 if seed == 3407 else 1.0),
        )
        for seed in SEEDS
    ]
    coverage = [_coverage_row("path", 0, seed) for seed in SEEDS]

    selected = select_registered_candidate(
        statistics,
        coverage,
        required_seeds=SEEDS,
        minimum_supported_seeds=2,
        minimum_patients_per_seed=20,
        minimum_target_pixels_per_seed=512,
        alpha=0.05,
        minimum_standardized_effect=0.8,
    )

    assert selected["status"] == "NO_REGISTERED_TRANSUNET_CANDIDATE"
    assert not selected["test_intervention_authorized"]
    assert selected["candidate"] is None


def test_validation_summary_uses_all_patients_and_global_holm_correction():
    rows = []
    for path_order, path_id in enumerate(("path_a", "path_b")):
        for seed in SEEDS:
            for patient_index, value in enumerate((0.02, 0.04, 0.06, 0.08)):
                rows.append(
                    {
                        "path_id": path_id,
                        "topology_order": path_order,
                        "model_seed": seed,
                        "patient_id": f"p{patient_index}",
                        "net_recovery": value + 0.01 * path_order,
                        "selected_slice_target_pixel_count": 40,
                        "eligible": True,
                    }
                )

    statistics, coverage = summarize_validation_candidates(
        pd.DataFrame(rows),
        required_seeds=SEEDS,
        bootstrap_iterations=200,
        bootstrap_seed=19,
    )

    assert len(statistics) == 6
    assert len(coverage) == 6
    assert set(statistics.patient_count) == {4}
    assert (statistics.holm_p >= statistics.wilcoxon_p).all()
    assert set(coverage.aggregate_target_pixel_count) == {160}
