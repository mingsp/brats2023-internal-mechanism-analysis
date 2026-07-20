import torch
from torch import nn

from pptt.interventions.activation_swap import (
    replace_activation_vectors,
    transform_module_input,
)
from pptt.interventions.transunet_paths import (
    audit_path_activation_identity,
    declared_transunet_decoder_paths,
)
from pptt.models.adapters import build_adapter


def test_temporary_skip_intervention_restores_native_forward_exactly():
    torch.manual_seed(5)
    adapter = build_adapter("unet_baseline").eval()
    image = torch.randn(1, 4, 32, 32)
    with torch.inference_mode():
        native = adapter(image)
        with transform_module_input(
            adapter.model.up4,
            argument_index=1,
            transform=torch.zeros_like,
        ):
            intervened = adapter(image)
        restored = adapter(image)

    assert not torch.equal(native, intervened)
    assert torch.equal(native, restored)
    assert len(adapter.model.up4._forward_pre_hooks) == 0


def test_natural_vector_replacement_changes_only_selected_spatial_vectors():
    activation = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    destinations = torch.tensor([1, 7], dtype=torch.int64)
    sources = torch.tensor([[100.0, 200.0], [300.0, 400.0]])

    replaced = replace_activation_vectors(
        activation,
        destinations,
        sources,
        alpha=1.0,
    )

    flat = replaced.flatten(2)
    torch.testing.assert_close(flat[0, :, 1], sources[0])
    torch.testing.assert_close(flat[0, :, 7], sources[1])
    torch.testing.assert_close(flat[0, :, 0], activation.flatten(2)[0, :, 0])


class _KeywordSkipBlock(nn.Module):
    def forward(self, x: torch.Tensor, *, skip: torch.Tensor) -> torch.Tensor:
        return x + skip


def test_temporary_intervention_supports_keyword_tensor_inputs():
    block = _KeywordSkipBlock()
    x = torch.ones(1, 1, 2, 2)
    skip = torch.full_like(x, 2.0)

    native = block(x, skip=skip)
    with transform_module_input(
        block,
        argument_name="skip",
        transform=torch.zeros_like,
    ):
        intervened = block(x, skip=skip)
    restored = block(x, skip=skip)

    torch.testing.assert_close(native, torch.full_like(native, 3.0))
    torch.testing.assert_close(intervened, torch.ones_like(intervened))
    torch.testing.assert_close(restored, native)
    assert len(block._forward_pre_hooks) == 0


def test_transunet_declared_paths_are_exact_forward_tensor_identities():
    torch.manual_seed(17)
    adapter = build_adapter("transunet_r50_vit_b16", img_size=160).eval()
    image = torch.randn(1, 4, 160, 160)

    for path in declared_transunet_decoder_paths():
        audit = audit_path_activation_identity(adapter, path, image)
        assert audit["source_capture_count"] == 1
        assert audit["receiver_capture_count"] == 1
        assert audit["shape_equal"]
        assert audit["exact_equal"]
        assert audit["max_abs_error"] == 0.0
        assert audit["source_hook_count_after"] == 0
        assert audit["receiver_hook_count_after"] == 0
