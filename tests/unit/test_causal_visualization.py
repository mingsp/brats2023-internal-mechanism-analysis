import numpy as np
import pandas as pd

from pptt.visualization.causal import (
    bootstrap_mean_interval,
    causal_event_map,
    display_channel,
    render_causal_result_figure,
)
from scripts.summarize_v6_causal import _dose_summary


def test_bootstrap_mean_interval_is_reproducible_and_contains_mean():
    values = np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float64)

    first = bootstrap_mean_interval(values, iterations=1000, seed=17)
    second = bootstrap_mean_interval(values, iterations=1000, seed=17)

    assert first == second
    assert first[0] <= values.mean() <= first[1]


def test_causal_event_map_separates_retained_and_lost_target_pixels():
    truth = np.asarray([[1, 1], [2, 0]], dtype=np.uint8)
    target = np.asarray([[True, True], [True, False]])
    states = np.asarray(
        [
            [[1, 0], [2, 0]],
            [[1, 1], [0, 0]],
        ],
        dtype=np.uint8,
    )

    events = causal_event_map(
        states,
        truth,
        target,
        transition_index=0,
    )

    # 0: outside the fixed clean target set; 1: persistently retained;
    # 2: target correction lost under the intervention.
    np.testing.assert_array_equal(events, [[1, 1], [2, 0]])


def test_dose_summary_reports_bootstrap_interval_for_each_seed_and_dose():
    rows = []
    for patient_index, value in enumerate((0.2, 0.4, 0.6, 0.8)):
        rows.extend(
            [
                {
                    "model_seed": 42,
                    "patient_id": f"p{patient_index}",
                    "condition": "corrupt",
                    "persistent_retention": value,
                },
                {
                    "model_seed": 42,
                    "patient_id": f"p{patient_index}",
                    "condition": "restore_target_1.00",
                    "persistent_retention": min(1.0, value + 0.1),
                },
            ]
        )

    summary = _dose_summary(
        pd.DataFrame(rows),
        seeds=(42,),
        alphas=(1.0,),
        iterations=1000,
        bootstrap_seed=23,
    )

    assert {"ci_low", "ci_high"}.issubset(summary.columns)
    assert (summary.ci_low <= summary.mean_persistent_retention).all()
    assert (summary.mean_persistent_retention <= summary.ci_high).all()


def test_causal_result_figure_renders_case_dose_and_effects(tmp_path):
    image = np.zeros((32, 32), dtype=np.float32)
    image[6:26, 6:26] = np.linspace(0.0, 1.0, 400).reshape(20, 20)
    truth = np.zeros((32, 32), dtype=np.uint8)
    truth[10:22, 10:22] = 2
    target = np.zeros((32, 32), dtype=bool)
    target[12:20, 12:20] = True
    clean = np.zeros((3, 32, 32), dtype=np.uint8)
    clean[1:, target] = 2
    corrupt = clean.copy()
    corrupt[1:, 12:18, 12:18] = 0
    restored = clean.copy()
    control = corrupt.copy()
    control[1:, 12:14, 12:14] = 2
    dose = pd.DataFrame(
        [
            {
                "model_seed": seed,
                "alpha": alpha,
                "mean_persistent_retention": 0.4 + 0.4 * alpha,
                "ci_low": 0.36 + 0.4 * alpha,
                "ci_high": 0.44 + 0.4 * alpha,
            }
            for seed in (42, 123, 3407)
            for alpha in (0.0, 0.5, 1.0)
        ]
    )
    statistics = pd.DataFrame(
        [
            {
                "endpoint": endpoint,
                "model_seed": seed,
                "mean_difference": mean,
                "ci_low": mean - 0.04,
                "ci_high": mean + 0.04,
                "paired_cohens_d": 1.2,
            }
            for endpoint, mean in (
                ("necessity", 0.5),
                ("restoration", 0.3),
                ("specificity", 0.1),
            )
            for seed in (42, 123, 3407)
        ]
    )

    outputs = render_causal_result_figure(
        output_base=tmp_path / "causal",
        language="en",
        image=image,
        truth=truth,
        target_mask=target,
        condition_states={
            "clean": clean,
            "corrupt": corrupt,
            "restore_target": restored,
            "restore_control": control,
        },
        transition_index=0,
        dose=dose,
        statistics=statistics,
        dpi=300,
    )

    assert outputs[0].is_file() and outputs[0].stat().st_size > 5000
    assert outputs[1].is_file() and outputs[1].stat().st_size > 1000


def test_display_channel_extracts_flair_from_batched_four_channel_input():
    image = np.arange(1 * 4 * 3 * 2, dtype=np.float32).reshape(1, 4, 3, 2)

    selected = display_channel(image, channel_index=0)

    assert selected.shape == (3, 2)
    np.testing.assert_array_equal(selected, image[0, 0])
