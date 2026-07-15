import os
from pathlib import Path

import pytest
import torch

from pptt.models.transunet import build_transunet, load_pretrained_npz


DEFAULT_PRETRAINED = Path(
    "/root/autodl-tmp/A_scheme_workspace/brats2023_data/pretrained/"
    "R50+ViT-B_16.npz"
)


def _resolve_pretrained() -> Path:
    configured = os.environ.get("PPTT_TRANSUNET_PRETRAINED")
    path = Path(configured).expanduser() if configured else DEFAULT_PRETRAINED
    if not path.is_file():
        pytest.skip(f"Official TransUNet initialization is unavailable: {path}")
    return path


def test_official_transunet_pretrained_load_is_auditable():
    model = build_transunet(n_channels=4, num_classes=4, img_size=160)
    report = load_pretrained_npz(model, _resolve_pretrained())

    assert report.source_key_count >= report.matched_source_key_count > 0
    assert report.parameter_coverage > 0.9
    assert report.unexplained_shape_skips == ()
    assert set(report.intentionally_random_modules) == {
        "decoder.blocks",
        "decoder.conv_more",
        "segmentation_head",
    }
    root_weight = model.transformer.embeddings.hybrid_model.root.conv.weight
    assert root_weight.shape[1] == 4
    for channel in range(1, 4):
        torch.testing.assert_close(root_weight[:, 0], root_weight[:, channel])
