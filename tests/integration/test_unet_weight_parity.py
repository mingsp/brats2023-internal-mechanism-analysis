from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch
from torch import nn

from pptt.hooks.checkpoints import load_checkpoint
from pptt.models.adapters import build_adapter


DEFAULT_ASSET_ROOT = Path("/root/autodl-tmp/A_scheme_workspace/brats2023_data")
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
MAX_ABS_ERROR_LIMIT = 1e-7
MODEL_CASES = (
    pytest.param(
        "unet_baseline",
        "baseline_seed42_best_val_loss.pth",
        "baseline",
        "guide_unet.Architecture.unet_baseline",
        "UNetBaseline",
        id="baseline",
    ),
    pytest.param(
        "unet_noskip",
        "noskip_unet_seed42_best_val_loss.pth",
        "noskip",
        "guide_unet.Architecture.unet_noskip",
        "UNetNoSkip",
        id="noskip",
    ),
)


def _resolve_asset_root() -> Path:
    configured_root = os.environ.get("PPTT_ASSET_ROOT")
    if configured_root is not None:
        if not configured_root.strip():
            pytest.fail("PPTT_ASSET_ROOT is set but empty")
        root = Path(configured_root).expanduser()
        assert root.is_dir(), f"PPTT_ASSET_ROOT is not a directory: {root}"
        return root
    if DEFAULT_ASSET_ROOT.is_dir():
        return DEFAULT_ASSET_ROOT
    pytest.skip(
        "U-Net parity assets are unavailable: PPTT_ASSET_ROOT is unset and "
        f"the default asset root does not exist: {DEFAULT_ASSET_ROOT}"
    )


def _clear_guide_unet_modules() -> None:
    for module_name in tuple(sys.modules):
        if module_name == "guide_unet" or module_name.startswith("guide_unet."):
            del sys.modules[module_name]


@contextmanager
def _temporary_legacy_module(
    source_root: Path,
    module_name: str,
) -> Iterator[ModuleType]:
    _clear_guide_unet_modules()
    source_text = str(source_root)
    sys.path.insert(0, source_text)
    try:
        yield importlib.import_module(module_name)
    finally:
        sys.path.remove(source_text)
        _clear_guide_unet_modules()


def _legacy_trace(
    model: nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def capture(name: str):
        def hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured[name] = output

        return hook

    for name in CHECKPOINT_NAMES:
        handles.append(getattr(model, name).register_forward_hook(capture(name)))
    try:
        result = model(x)
    finally:
        for handle in handles:
            handle.remove()

    assert isinstance(result, tuple), "Legacy U-Net forward must return a tuple"
    assert isinstance(result[0], torch.Tensor)
    assert tuple(captured) == CHECKPOINT_NAMES
    return result[0], OrderedDict((name, captured[name]) for name in CHECKPOINT_NAMES)


def _max_abs_error(left: torch.Tensor, right: torch.Tensor) -> float:
    assert left.shape == right.shape
    return torch.max(torch.abs(left - right)).item()


@pytest.mark.parametrize(
    (
        "model_name",
        "checkpoint_filename",
        "source_directory",
        "legacy_module_name",
        "legacy_class_name",
    ),
    MODEL_CASES,
)
def test_seed42_unet_weights_match_legacy_logits_and_nodes(
    model_name: str,
    checkpoint_filename: str,
    source_directory: str,
    legacy_module_name: str,
    legacy_class_name: str,
    record_property,
):
    asset_root = _resolve_asset_root()
    checkpoint_path = asset_root / "validation_checkpoints" / checkpoint_filename
    source_root = asset_root / "validation_sources" / source_directory
    assert checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}"
    assert source_root.is_dir(), f"Missing legacy source directory: {source_root}"

    adapter = build_adapter(
        model_name,
        n_channels=4,
        num_classes=4,
        bilinear=False,
    )
    checkpoint_sha256 = adapter.load_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )

    with _temporary_legacy_module(source_root, legacy_module_name) as legacy_module:
        legacy_class = getattr(legacy_module, legacy_class_name)
        legacy_model = legacy_class(n_channels=4, n_classes=4, bilinear=False)
        legacy_sha256 = load_checkpoint(
            legacy_model,
            checkpoint_path,
            map_location="cpu",
        )

        assert legacy_sha256 == checkpoint_sha256
        adapter.eval()
        legacy_model.eval()
        generator = torch.Generator(device="cpu").manual_seed(42)
        x = torch.randn(
            1,
            4,
            160,
            160,
            generator=generator,
            dtype=torch.float32,
        )

        with torch.no_grad():
            legacy_logits, legacy_activations = _legacy_trace(legacy_model, x)
            current_trace = adapter.trace(x)

    assert tuple(current_trace.activations) == CHECKPOINT_NAMES
    errors = {
        "logits": _max_abs_error(legacy_logits, current_trace.logits),
        **{
            name: _max_abs_error(
                legacy_activations[name],
                current_trace.activations[name],
            )
            for name in CHECKPOINT_NAMES
        },
    }
    record_property("checkpoint_sha256", checkpoint_sha256)
    record_property("max_abs_errors", errors)
    print(
        f"model={model_name} checkpoint_sha256={checkpoint_sha256} "
        f"max_abs_errors={errors}"
    )

    for node_name, error in errors.items():
        assert error < MAX_ABS_ERROR_LIMIT, (
            f"{model_name} {node_name} max_abs_error={error} is not below "
            f"{MAX_ABS_ERROR_LIMIT}; checkpoint_sha256={checkpoint_sha256}"
        )
