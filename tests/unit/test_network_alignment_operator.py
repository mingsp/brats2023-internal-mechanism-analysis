from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from pptt.interventions.network_alignment import (
    NodeRestoreMasks,
    map_transition_mask,
    match_node_restore_masks,
    restore_clean_positions,
    spatial_roll_module_output,
)
from pptt.interventions.network_runtime import (
    module_output_transforms,
    run_network_alignment_forwards,
)
from pptt.models.adapters import build_adapter


def test_spatial_roll_preserves_each_channel_multiset_exactly():
    activation = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(
        2, 3, 4, 5
    )

    shifted = spatial_roll_module_output(activation, shift_yx=(1, -2))

    for batch_index in range(activation.shape[0]):
        for channel_index in range(activation.shape[1]):
            torch.testing.assert_close(
                torch.sort(activation[batch_index, channel_index].flatten()).values,
                torch.sort(shifted[batch_index, channel_index].flatten()).values,
                rtol=0,
                atol=0,
            )


def test_full_root_restore_reproduces_clean_logits_within_tolerance():
    torch.manual_seed(19)
    adapter = build_adapter("unet_baseline").eval()
    image = torch.randn(1, 4, 32, 32)
    with torch.inference_mode():
        trace = adapter.trace(image)
    shape = trace.activations["down2"].shape[-2:]
    target = torch.zeros(shape, dtype=torch.bool)
    target[0, 0] = True
    control = torch.zeros_like(target)
    control[-1, -1] = True

    result = run_network_alignment_forwards(
        adapter,
        image,
        root_node="down1",
        restore_nodes=("down2",),
        masks={
            "down2": {
                0: NodeRestoreMasks(
                    target=target,
                    control=control,
                    target_count=1,
                    control_count=1,
                )
            }
        },
        input_equivalent_shift_yx=(8, 8),
        logit_tolerance=1.0e-6,
    )

    np.testing.assert_allclose(
        result.root_restored_logits,
        result.clean_logits,
        rtol=0,
        atol=1.0e-6,
    )
    assert result.audits["root_restore_pass"] is True


def test_target_restore_changes_only_registered_spatial_positions():
    clean = torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3)
    corrupted = torch.full_like(clean, -1)
    mask = torch.zeros((3, 3), dtype=torch.bool)
    mask[0, 1] = True
    mask[2, 2] = True

    restored = restore_clean_positions(corrupted, clean, mask)

    expanded = mask[None, None].expand_as(clean)
    torch.testing.assert_close(restored[expanded], clean[expanded])
    torch.testing.assert_close(restored[~expanded], corrupted[~expanded])


def test_hooks_are_removed_after_success_and_exception():
    module = nn.Identity()
    before = len(module._forward_hooks)

    with module_output_transforms({module: lambda output: output + 1}):
        assert torch.equal(module(torch.tensor([1])), torch.tensor([2]))
    assert len(module._forward_hooks) == before

    with pytest.raises(RuntimeError, match="forced failure"):
        with module_output_transforms({module: lambda output: output}):
            raise RuntimeError("forced failure")
    assert len(module._forward_hooks) == before


def test_each_transition_mask_is_mapped_without_interpolation():
    source = np.zeros((4, 4), dtype=bool)
    source[0, 0] = True
    source[3, 3] = True

    mapped = map_transition_mask(source, output_shape=(2, 2))

    np.testing.assert_array_equal(
        mapped,
        np.array([[True, False], [False, True]]),
    )
    assert mapped.dtype == np.bool_


def test_control_positions_are_disjoint_equal_count_and_stratum_matched():
    target = np.array([[True, False, False, False, False, False, False, True]])
    control_pool = np.array(
        [[False, True, False, False, False, False, True, False]]
    )
    truth_class = np.array([[1, 1, 0, 0, 0, 0, 2, 2]], dtype=np.uint8)
    boundary = np.array([[1.0, 1.0, 9.0, 9.0, 9.0, 9.0, 4.0, 4.0]])
    activation_norm = np.array([[1.0, 1.0, 9.0, 9.0, 9.0, 9.0, 2.0, 2.0]])

    masks = match_node_restore_masks(
        target_mask=target,
        control_pool=control_pool,
        truth_class=truth_class,
        boundary_distance=boundary,
        activation_norm=activation_norm,
        boundary_bin_edges=(1.5, 3.5, 7.5),
        activation_norm_quantile_bins=4,
    )

    assert masks.control is not None
    assert masks.target_count == masks.control_count == 2
    assert not torch.any(masks.target & masks.control)
    target_indices = torch.nonzero(
        masks.target.flatten(), as_tuple=False
    ).flatten().tolist()
    control_indices = torch.nonzero(
        masks.control.flatten(), as_tuple=False
    ).flatten().tolist()
    assert target_indices == [0, 7]
    assert control_indices == [1, 6]
