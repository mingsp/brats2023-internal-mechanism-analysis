import torch

from pptt.interventions.activation_swap import (
    replace_activation_vectors,
    transform_module_input,
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
