from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from pptt.causal_abstraction.interventions import (
    apply_flat_feature_edit,
    bilinear_resize_matrix,
    bilinear_resize_rows,
    equal_norm_nullspace_control,
    minimum_norm_state_exchange,
    project_feature_edit,
    stack_observer_contrast_weights,
    stack_observer_weights,
    stack_restart_logit_contrasts,
    stack_restart_logit_deltas,
)


def test_bilinear_matrix_matches_torch_align_corners_false():
    source = torch.arange(12, dtype=torch.float64).reshape(1, 1, 3, 4)
    matrix = bilinear_resize_matrix(
        (3, 4),
        (5, 7),
        dtype=torch.float64,
    )

    expected = F.interpolate(
        source,
        size=(5, 7),
        mode="bilinear",
        align_corners=False,
    )
    actual = (matrix @ source.flatten()).reshape(1, 1, 5, 7)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1.0e-12)
    torch.testing.assert_close(
        matrix.sum(dim=1),
        torch.ones(35, dtype=torch.float64),
        rtol=0,
        atol=1.0e-12,
    )


def test_selected_resize_rows_equal_full_matrix_rows():
    indices = torch.tensor([0, 6, 17, 34], dtype=torch.int64)
    full = bilinear_resize_matrix((3, 4), (5, 7), dtype=torch.float64)

    selected = bilinear_resize_rows(
        (3, 4),
        (5, 7),
        indices,
        dtype=torch.float64,
    )

    torch.testing.assert_close(selected, full[indices], rtol=0, atol=0)


def test_exchange_is_feasible_and_minimum_norm():
    resize_rows = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]],
        dtype=torch.float64,
    )
    observer_weight = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    delta_logits = torch.tensor(
        [[2.0, -1.0], [0.5, 1.5]],
        dtype=torch.float64,
    )

    edit = minimum_norm_state_exchange(
        resize_rows,
        observer_weight,
        delta_logits,
        rcond=1.0e-12,
    )
    null = torch.zeros_like(edit.delta_h)
    null[:, 2] = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64)

    torch.testing.assert_close(
        resize_rows @ edit.delta_h @ observer_weight.T,
        delta_logits,
        rtol=0,
        atol=1.0e-10,
    )
    assert edit.target_max_abs_error <= 1.0e-10
    assert edit.frobenius_norm < float(torch.linalg.vector_norm(edit.delta_h + null))


def test_exchange_keeps_ill_conditioned_directions_above_locked_cutoff():
    resize_rows = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0e-4]],
        dtype=torch.float64,
    )
    observer_weight = torch.eye(2, dtype=torch.float64)
    delta_logits = torch.tensor(
        [[1.0, -2.0], [3.0, 4.0]],
        dtype=torch.float64,
    )

    edit = minimum_norm_state_exchange(
        resize_rows,
        observer_weight,
        delta_logits,
        rcond=1.0e-7,
    )

    torch.testing.assert_close(
        resize_rows @ edit.delta_h @ observer_weight.T,
        delta_logits,
        rtol=0,
        atol=1.0e-10,
    )
    assert edit.effective_rank_spatial == 2
    assert edit.target_max_abs_error <= 1.0e-10


def test_equal_norm_nullspace_control_preserves_observer_logits():
    observer_weight = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    task_edit = torch.tensor(
        [[1.0, 2.0, 0.0, 0.0], [0.5, -0.5, 0.0, 0.0]],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(17)

    control = equal_norm_nullspace_control(
        observer_weight,
        task_edit,
        generator=generator,
        rcond=1.0e-12,
    )

    torch.testing.assert_close(
        control @ observer_weight.T,
        torch.zeros((2, 2), dtype=torch.float64),
        rtol=0,
        atol=1.0e-10,
    )
    assert float(torch.linalg.vector_norm(control)) == pytest.approx(
        float(torch.linalg.vector_norm(task_edit)),
        abs=1.0e-10,
    )


def test_stacked_restart_operator_reconstructs_every_observer_logit_delta():
    restart_weights = (
        torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
    )
    source_logits = torch.tensor(
        [
            [[2.0, -2.0], [1.0, -1.0]],
            [[1.5, -1.5], [0.5, -0.5]],
            [[1.0, -1.0], [0.25, -0.25]],
        ],
        dtype=torch.float64,
    )
    base_logits = torch.zeros_like(source_logits)
    resize_rows = torch.eye(2, dtype=torch.float64)
    stacked_weight = stack_observer_weights(restart_weights)
    stacked_delta = stack_restart_logit_deltas(source_logits, base_logits)

    exchange = minimum_norm_state_exchange(
        resize_rows,
        stacked_weight,
        stacked_delta,
        rcond=1.0e-12,
    )

    reconstructed = (
        resize_rows @ exchange.delta_h @ stacked_weight.T
    ).reshape(2, 3, 2).permute(1, 0, 2)
    torch.testing.assert_close(reconstructed, source_logits, rtol=0, atol=1.0e-10)
    assert exchange.target_max_abs_error <= 1.0e-10


def test_contrast_exchange_preserves_source_decisions_without_common_mode():
    restart_weights = (
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
        torch.tensor(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            dtype=torch.float64,
        ),
    )
    base_logits = torch.tensor(
        [[[0.0, 1.0, 2.0]], [[2.0, 1.0, 0.0]]],
        dtype=torch.float64,
    )
    source_logits = torch.tensor(
        [[[101.0, 99.0, 105.0]], [[-46.0, -53.0, -44.0]]],
        dtype=torch.float64,
    )
    weight = stack_observer_contrast_weights(restart_weights)
    target = stack_restart_logit_contrasts(source_logits, base_logits)

    exchange = minimum_norm_state_exchange(
        torch.ones((1, 1), dtype=torch.float64),
        weight,
        target,
        rcond=1.0e-7,
    )
    reconstructed = (
        base_logits
        + exchange.reconstructed_delta_logits.reshape(1, 2, 3)
        .permute(1, 0, 2)
    )

    torch.testing.assert_close(
        target.mean(dim=1),
        torch.zeros(1, dtype=torch.float64),
        rtol=0,
        atol=1.0e-12,
    )
    assert torch.equal(reconstructed.argmax(dim=2), source_logits.argmax(dim=2))
    assert int(torch.linalg.matrix_rank(weight)) == 2
    assert exchange.target_max_abs_error <= 1.0e-10


def test_projected_edit_matches_observer_then_interpolation():
    delta_h = torch.arange(24, dtype=torch.float64).reshape(6, 4) / 10.0
    observer_weight = torch.tensor(
        [[1.0, 0.0, -1.0, 0.0], [0.0, 2.0, 0.0, 1.0]],
        dtype=torch.float64,
    )

    projected = project_feature_edit(
        delta_h,
        native_shape=(2, 3),
        observer_weight=observer_weight,
        output_shape=(4, 5),
    )
    native = (delta_h @ observer_weight.T).T.reshape(1, 2, 2, 3)
    expected = F.interpolate(
        native,
        size=(4, 5),
        mode="bilinear",
        align_corners=False,
    )[0]

    torch.testing.assert_close(projected, expected, rtol=0, atol=1.0e-12)


def test_apply_flat_feature_edit_only_changes_requested_batch():
    activation = torch.zeros((2, 3, 2, 2), dtype=torch.float32)
    edit = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    changed = apply_flat_feature_edit(
        activation,
        edit,
        doses=torch.tensor([0.0, 0.5]),
    )

    torch.testing.assert_close(changed[0], activation[0])
    torch.testing.assert_close(
        changed[1],
        0.5 * edit.T.reshape(3, 2, 2),
    )


@pytest.mark.parametrize(
    ("native_shape", "output_shape", "indices"),
    [
        ((0, 2), (4, 4), torch.tensor([0])),
        ((2, 2), (4, 4), torch.tensor([], dtype=torch.int64)),
        ((2, 2), (4, 4), torch.tensor([16], dtype=torch.int64)),
    ],
)
def test_resize_rows_reject_invalid_geometry(native_shape, output_shape, indices):
    with pytest.raises(ValueError):
        bilinear_resize_rows(native_shape, output_shape, indices)


def test_nullspace_control_rejects_full_column_rank_observer():
    weight = torch.eye(3, dtype=torch.float64)
    edit = torch.ones((2, 3), dtype=torch.float64)

    with pytest.raises(ValueError, match="nullspace"):
        equal_norm_nullspace_control(
            weight,
            edit,
            generator=torch.Generator().manual_seed(1),
        )
