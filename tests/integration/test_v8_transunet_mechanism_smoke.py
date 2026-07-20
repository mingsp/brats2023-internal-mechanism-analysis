import torch

from pptt.interventions.activation_swap import transform_module_input
from pptt.interventions.causal_tracing import spatial_corrupt_restore
from pptt.interventions.transunet_paths import (
    declared_transunet_decoder_paths,
    resolve_module,
)
from pptt.models.adapters import build_adapter


def test_registered_transunet_keyword_path_changes_and_restores_native_forward():
    torch.manual_seed(31)
    adapter = build_adapter("transunet_r50_vit_b16", img_size=160).eval()
    image = torch.randn(1, 4, 160, 160)
    path = next(
        value
        for value in declared_transunet_decoder_paths()
        if value.path_id == "skip_down2_to_up2"
    )
    receiver = resolve_module(adapter.model, path.receiver_module_path)
    mask = torch.zeros(40, 40, dtype=torch.bool)

    with torch.inference_mode():
        native = adapter(image)
        with transform_module_input(
            receiver,
            argument_name=path.argument_name,
            transform=lambda activation: spatial_corrupt_restore(
                activation,
                restore_mask=mask,
                shift_yx=(2, 2),
                alpha=0.0,
            ),
        ):
            corrupted = adapter(image)
        restored = adapter(image)

    assert not torch.equal(native, corrupted)
    assert torch.equal(native, restored)
    assert len(receiver._forward_pre_hooks) == 0
