import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pptt.transitions.fields import states_and_margins
from pptt.transitions.metrics import confusion_from_labels, metrics_from_confusion
from pptt.transitions.tensors import (
    build_transition_tensor,
    confusions_from_transition,
    state_flows_from_transition,
)


@settings(max_examples=80, deadline=None)
@given(
    num_classes=st.integers(min_value=2, max_value=6),
    triples=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=100),
            st.integers(min_value=0, max_value=100),
            st.integers(min_value=0, max_value=100),
        ),
        min_size=1,
        max_size=200,
    ),
)
def test_transition_conservation_and_confusion_reconstruction_property(
    num_classes,
    triples,
):
    values = np.asarray(triples, dtype=np.int64) % num_classes
    y, z0, z1 = values.T
    tensor = build_transition_tensor(y, z0, z1, num_classes=num_classes)
    c0, c1 = confusions_from_transition(tensor)
    flow = state_flows_from_transition(tensor)

    assert int(tensor.sum()) == len(triples)
    np.testing.assert_array_equal(c0, confusion_from_labels(y, z0, num_classes))
    np.testing.assert_array_equal(c1, confusion_from_labels(y, z1, num_classes))
    np.testing.assert_array_equal(flow.occupancy_delta, flow.inflow - flow.outflow)


@settings(max_examples=40, deadline=None)
@given(
    num_classes=st.integers(min_value=2, max_value=5),
    stage_count=st.integers(min_value=2, max_value=8),
    pixel_count=st.integers(min_value=1, max_value=100),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_metric_path_obeys_telescoping_identity(
    num_classes,
    stage_count,
    pixel_count,
    seed,
):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, num_classes, size=pixel_count, dtype=np.int64)
    states = rng.integers(
        0,
        num_classes,
        size=(stage_count, pixel_count),
        dtype=np.int64,
    )
    metric_path = [
        metrics_from_confusion(confusion_from_labels(y, stage, num_classes)).dice
        for stage in states
    ]
    deltas = [right - left for left, right in zip(metric_path[:-1], metric_path[1:])]
    np.testing.assert_allclose(
        np.sum(deltas, axis=0),
        metric_path[-1] - metric_path[0],
        rtol=0.0,
        atol=1e-12,
    )


def test_prediction_margin_bound_preserves_argmax_state():
    probabilities = np.array(
        [
            [
                [[0.62, 0.55], [0.71, 0.51]],
                [[0.25, 0.31], [0.20, 0.29]],
                [[0.13, 0.14], [0.09, 0.20]],
            ],
        ],
        dtype=np.float64,
    )
    states, margins = states_and_margins(probabilities, class_axis=1)
    perturbation = np.array(
        [
            [
                [[0.02, -0.01], [0.01, -0.02]],
                [[0.00, 0.01], [0.00, 0.01]],
                [[-0.02, 0.00], [-0.01, 0.01]],
            ],
        ],
        dtype=np.float64,
    )
    assert np.max(np.abs(perturbation), axis=1).max() < (margins / 2).min()
    perturbed_states, _ = states_and_margins(
        probabilities + perturbation,
        class_axis=1,
    )
    np.testing.assert_array_equal(perturbed_states, states)


def test_state_extraction_rejects_unnormalized_scores():
    with pytest.raises(ValueError, match="sum to one"):
        states_and_margins(
            np.array([[[[2.0]], [[1.0]]]], dtype=np.float64),
            class_axis=1,
        )
