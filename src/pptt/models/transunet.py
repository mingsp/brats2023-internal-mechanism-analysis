from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from pptt.models._vendor.transunet.vit_seg_configs import get_r50_b16_config
from pptt.models._vendor.transunet.vit_seg_modeling import VisionTransformer
from pptt.models._vendor.transunet.vit_seg_modeling_resnet_skip import (
    adapt_rgb_kernel_to_four_channels,
)


@dataclass(frozen=True)
class PretrainedLoadReport:
    source_path: str
    source_key_count: int
    matched_source_key_count: int
    unmatched_source_keys: tuple[str, ...]
    model_parameter_count: int
    initialized_parameter_count: int
    parameter_coverage: float
    input_adaptation: str
    intentionally_random_modules: tuple[str, ...]
    unexplained_shape_skips: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_transunet(
    *,
    n_channels: int = 4,
    num_classes: int = 4,
    img_size: int = 160,
) -> VisionTransformer:
    if n_channels != 4:
        raise ValueError(
            f"The locked BraTS TransUNet configuration requires 4 channels, got {n_channels}"
        )
    if img_size <= 0 or img_size % 16 != 0:
        raise ValueError(f"img_size must be positive and divisible by 16, got {img_size}")

    config = get_r50_b16_config()
    config.n_classes = int(num_classes)
    config.n_skip = 3
    grid = img_size // 16
    config.patches.grid = (grid, grid)
    return VisionTransformer(
        config,
        img_size=img_size,
        num_classes=num_classes,
        in_channels=n_channels,
        vis=False,
    )


def checkpoint_modules(model: VisionTransformer) -> OrderedDict[str, nn.Module]:
    hybrid = model.transformer.embeddings.hybrid_model
    blocks = model.decoder.blocks
    return OrderedDict(
        (
            ("down1", hybrid.feature_taps[0]),
            ("down2", hybrid.feature_taps[1]),
            ("down3", hybrid.feature_taps[2]),
            ("down4", model.decoder.conv_more),
            ("up1", blocks[0]),
            ("up2", blocks[1]),
            ("up3", blocks[2]),
            ("up4", blocks[3]),
        )
    )


def _parameter_count(parameters: Mapping[str, torch.Tensor]) -> int:
    return sum(value.numel() for value in parameters.values())


class _TrackingWeights:
    def __init__(self, archive: np.lib.npyio.NpzFile) -> None:
        self._archive = archive
        self.accessed: set[str] = set()

    def __getitem__(self, key: str) -> np.ndarray:
        self.accessed.add(key)
        return self._archive[key]


def load_pretrained_npz(
    model: VisionTransformer,
    path: str | Path,
) -> PretrainedLoadReport:
    """Load the official R50-ViT-B/16 NPZ without silent shape skipping."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with np.load(source_path, allow_pickle=False) as weights:
        source_key_count = len(weights.files)
        source_keys = set(weights.files)
        tracked = _TrackingWeights(weights)
        try:
            model.load_from(tracked)
        except (KeyError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Official TransUNet initialization is incompatible with {source_path}: {exc}"
            ) from exc
        matched_source_keys = tracked.accessed

    state = model.state_dict()
    total_parameters = _parameter_count(state)
    intentionally_random = (
        "decoder.blocks",
        "decoder.conv_more",
        "segmentation_head",
    )
    random_parameters = _parameter_count(
        {
            name: value
            for name, value in state.items()
            if name.startswith(intentionally_random)
        }
    )
    initialized_parameters = total_parameters - random_parameters
    return PretrainedLoadReport(
        source_path=str(source_path.resolve()),
        source_key_count=source_key_count,
        matched_source_key_count=len(matched_source_keys),
        unmatched_source_keys=tuple(sorted(source_keys - matched_source_keys)),
        model_parameter_count=total_parameters,
        initialized_parameter_count=initialized_parameters,
        parameter_coverage=initialized_parameters / total_parameters,
        input_adaptation="RGB root kernel -> four symmetric channels with 3/4 scaling",
        intentionally_random_modules=intentionally_random,
        unexplained_shape_skips=(),
    )
