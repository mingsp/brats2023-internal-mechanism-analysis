import numpy as np
import pytest

from pptt.transitions.tensors import (
    build_transition_matrix,
    build_transition_tensor,
    confusions_from_transition,
    state_flows_from_transition,
)


def test_transition_tensor_reconstructs_both_confusions():
    y = np.array([[0, 1], [2, 3]], dtype=np.int64)
    z0 = np.array([[0, 0], [2, 1]], dtype=np.int64)
    z1 = np.array([[0, 1], [3, 3]], dtype=np.int64)
    tensor = build_transition_tensor(y, z0, z1, num_classes=4)
    c0, c1 = confusions_from_transition(tensor)

    assert tensor.shape == (4, 4, 4)
    assert tensor.dtype == np.int64
    assert tensor.sum() == 4
    assert c0[1, 0] == 1
    assert c1[1, 1] == 1
    assert c0[3, 1] == 1
    assert c1[3, 3] == 1


def test_unlabeled_transition_matrix_and_mask_preserve_selected_count():
    z0 = np.array([[0, 0, 2], [1, 3, 3]], dtype=np.int64)
    z1 = np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int64)
    mask = np.array([[True, False, True], [True, False, True]])
    matrix = build_transition_matrix(z0, z1, num_classes=4, mask=mask)

    assert matrix.sum() == int(mask.sum())
    assert matrix[0, 0] == 1
    assert matrix[2, 3] == 1
    assert matrix[1, 1] == 1
    assert matrix[3, 3] == 1


def test_state_flow_exactly_matches_occupancy_change():
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    z0 = np.array([0, 1, 0, 1, 2, 3, 0, 3], dtype=np.int64)
    z1 = np.array([0, 0, 1, 2, 2, 2, 3, 0], dtype=np.int64)
    tensor = build_transition_tensor(y, z0, z1, num_classes=4)
    flow = state_flows_from_transition(tensor)

    np.testing.assert_array_equal(
        flow.occupancy_delta,
        flow.inflow - flow.outflow,
    )


@pytest.mark.parametrize(
    "y,z0,z1,error",
    [
        (np.zeros((2, 2), int), np.zeros(4, int), np.zeros((2, 2), int), "shape"),
        (np.array([0.0, 1.0]), np.array([0, 1]), np.array([0, 1]), "integer"),
        (np.array([0, 4]), np.array([0, 1]), np.array([0, 1]), "outside"),
    ],
)
def test_transition_tensor_rejects_invalid_inputs(y, z0, z1, error):
    with pytest.raises(ValueError, match=error):
        build_transition_tensor(y, z0, z1, num_classes=4)
