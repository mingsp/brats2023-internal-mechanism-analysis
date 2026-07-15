import pytest
import torch

from pptt.models.adapters import build_adapter


CHECKPOINT_NAMES = (
    "down1",
    "down2",
    "down3",
    "down4",
    "up1",
    "up2",
    "up3",
    "up4",
)
EXPECTED_NODE_SHAPES = {
    "down1": (2, 64, 80, 80),
    "down2": (2, 256, 40, 40),
    "down3": (2, 512, 20, 20),
    "down4": (2, 512, 10, 10),
    "up1": (2, 256, 20, 20),
    "up2": (2, 128, 40, 40),
    "up3": (2, 64, 80, 80),
    "up4": (2, 16, 160, 160),
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration test")
def test_transunet_returns_ordered_spatial_trace_on_bra_ts_shape():
    adapter = build_adapter(
        "transunet_r50_vit_b16",
        n_channels=4,
        num_classes=4,
        img_size=160,
    ).cuda()
    adapter.eval()
    generator = torch.Generator(device="cpu").manual_seed(42)
    x = torch.randn(2, 4, 160, 160, generator=generator).cuda()

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        trace = adapter.trace(x)

    assert trace.logits.shape == (2, 4, 160, 160)
    assert tuple(trace.activations) == CHECKPOINT_NAMES
    assert {
        name: tuple(value.shape)
        for name, value in trace.activations.items()
    } == EXPECTED_NODE_SHAPES
    assert torch.isfinite(trace.logits).all()
    assert all(torch.isfinite(value).all() for value in trace.activations.values())
    root = adapter.model.transformer.embeddings.hybrid_model.root.conv
    assert root.in_channels == 4
    assert torch.cuda.max_memory_allocated() > 0


def test_transunet_requires_input_size_divisible_by_sixteen():
    with pytest.raises(ValueError, match="divisible by 16"):
        build_adapter("transunet_r50_vit_b16", img_size=158)
