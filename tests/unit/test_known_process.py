import numpy as np

from pptt.synthetic.known_process import (
    build_known_process_batch,
    decode_known_process_features,
)


def test_known_process_is_deterministic_linearly_readable_and_full_rank():
    first = build_known_process_batch(
        batch_size=3,
        height=64,
        width=64,
        seed=20260715,
    )
    second = build_known_process_batch(
        batch_size=3,
        height=64,
        width=64,
        seed=20260715,
    )

    np.testing.assert_array_equal(first.truth, second.truth)
    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_allclose(first.features, second.features, rtol=0.0, atol=0.0)
    assert first.states.shape == (3, 8, 64, 64)
    assert first.features.shape[:2] == (3, 8)
    assert first.features.shape[-2:] == (64, 64)
    assert all(np.linalg.matrix_rank(matrix) == 4 for matrix in first.mixing_matrices)

    probabilities = decode_known_process_features(first)
    decoded = probabilities.argmax(axis=2)
    np.testing.assert_array_equal(decoded, first.states)
    np.testing.assert_allclose(probabilities.sum(axis=2), 1.0, atol=1e-6)


def test_known_process_places_each_event_at_the_declared_stage_and_region():
    batch = build_known_process_batch(
        batch_size=2,
        height=64,
        width=64,
        seed=20260715,
    )
    expected = batch.expected_changed_masks

    np.testing.assert_array_equal(expected[:, 1], batch.region_masks["A"])
    np.testing.assert_array_equal(expected[:, 2], batch.region_masks["D"])
    np.testing.assert_array_equal(expected[:, 3], batch.region_masks["B"])
    np.testing.assert_array_equal(expected[:, 5], batch.region_masks["C"])
    np.testing.assert_array_equal(expected[:, 6], batch.region_masks["B"])
    assert batch.active_transition_indices == (1, 2, 3, 5, 6)
    assert batch.unique_branch_transition_index == 5
    assert batch.branch_names[batch.unique_effective_branch_index] == "corrective_branch"
    natural = batch.states[:, 6]
    effects = np.sum(batch.branch_ablation_states != natural[None], axis=(1, 2, 3))
    assert np.flatnonzero(effects).tolist() == [batch.unique_effective_branch_index]
    assert np.all(batch.states[:, -1][batch.region_masks["D"]] != batch.truth[batch.region_masks["D"]])
