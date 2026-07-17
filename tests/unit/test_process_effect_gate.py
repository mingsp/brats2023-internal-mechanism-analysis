import numpy as np

from pptt.statistics.process_effects import (
    class_balanced_damage_rate,
    class_balanced_path_tv,
    class_balanced_persistent_net_by_transition,
    class_balanced_persistent_net_recovery,
    evaluate_causal_entry_gate,
    holm_adjust,
    select_robust_transition_candidate,
)


def test_class_balanced_path_tv_preserves_joint_transition_difference():
    left = np.zeros((2, 3, 3, 3), dtype=np.int64)
    right = np.zeros_like(left)
    left[:, 1, 1, 1] = 10
    right[:, 1, 1, 2] = 10
    left[:, 2, 2, 2] = 10
    right[:, 2, 2, 2] = 10

    value = class_balanced_path_tv(left, right, truth_classes=(1, 2))

    assert value == 0.5


def test_class_balanced_damage_rate_counts_destruction_and_wrong_reencoding():
    tensor = np.zeros((2, 3, 3, 3), dtype=np.int64)
    tensor[0, 1, 1, 0] = 4
    tensor[0, 1, 0, 2] = 2
    tensor[0, 1, 0, 0] = 4
    tensor[1, 2, 2, 2] = 10

    value = class_balanced_damage_rate(tensor, truth_classes=(1, 2))

    assert value == 0.3


def test_persistent_net_recovery_is_class_balanced_and_transition_additive():
    truth = np.array([1, 1, 2, 2], dtype=np.uint8)
    states = np.array(
        [
            [0, 1, 0, 2],
            [1, 0, 2, 2],
            [1, 0, 2, 1],
            [1, 0, 2, 1],
        ],
        dtype=np.uint8,
    )

    value = class_balanced_persistent_net_recovery(
        states,
        truth,
        transition_indices=(0, 1, 2),
        truth_classes=(1, 2),
    )

    assert value == 0.0

    np.testing.assert_allclose(
        class_balanced_persistent_net_by_transition(
            states,
            truth,
            truth_classes=(1, 2),
        ),
        [0.25, -0.25, 0.0],
    )


def test_persistent_net_recovery_skips_requested_classes_absent_in_patient():
    truth = np.array([1, 1], dtype=np.uint8)
    states = np.array([[0, 1], [1, 1]], dtype=np.uint8)

    value = class_balanced_persistent_net_recovery(
        states,
        truth,
        transition_indices=(0,),
        truth_classes=(1, 2, 3),
    )

    assert value == 0.5


def test_holm_adjust_is_monotone_in_sorted_p_value_order():
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03, 0.20]))

    np.testing.assert_allclose(adjusted, [0.04, 0.09, 0.09, 0.20])


def test_causal_entry_gate_requires_process_effect_not_terminal_dice():
    rows = [
        {
            "metric": "middle_damage",
            "model_seed": seed,
            "mean_difference": 0.12,
            "ci_low": 0.08,
            "holm_p": 0.01,
            "paired_cohens_d": 1.2,
        }
        for seed in (42, 123, 3407)
    ]
    rows += [
        {
            "metric": "late_persistent_net_recovery",
            "model_seed": seed,
            "mean_difference": 0.09,
            "ci_low": 0.05,
            "holm_p": 0.02,
            "paired_cohens_d": 1.0,
        }
        for seed in (42, 123, 3407)
    ]
    rows += [
        {
            "metric": "terminal_dice",
            "model_seed": seed,
            "mean_difference": 0.0,
            "ci_low": -0.01,
            "holm_p": 1.0,
            "paired_cohens_d": 0.0,
        }
        for seed in (42, 123, 3407)
    ]

    decision = evaluate_causal_entry_gate(rows)

    assert decision["passed"] is True
    assert decision["terminal_dice_required"] is False


def test_transition_candidate_uses_positive_validation_maximin_rule():
    rows = []
    effects = {
        3: (0.04, 0.05, 0.06),
        4: (0.08, 0.07, 0.06),
        5: (0.12, -0.01, 0.10),
        6: (0.05, 0.04, 0.03),
    }
    for transition_index, seed_effects in effects.items():
        for model_seed, effect in zip((42, 123, 3407), seed_effects, strict=True):
            rows.append(
                {
                    "transition_index": transition_index,
                    "transition": f"t{transition_index}",
                    "model_seed": model_seed,
                    "mean_difference": effect,
                }
            )

    candidate = select_robust_transition_candidate(
        rows,
        allowed_transition_indices=(3, 4, 5, 6),
        required_model_seeds=(42, 123, 3407),
    )

    assert candidate["transition_index"] == 4
    assert candidate["minimum_seed_effect"] == 0.06
    assert candidate["selection_rule"] == "positive_all_seeds_then_maximize_minimum"
