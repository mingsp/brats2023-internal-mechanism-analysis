from collections import OrderedDict
from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

from pptt.hooks.checkpoints import load_checkpoint
from pptt.models.adapters import build_adapter
from pptt.types import ForwardTrace


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
    "down1": (1, 128, 16, 16),
    "down2": (1, 256, 8, 8),
    "down3": (1, 512, 4, 4),
    "down4": (1, 1024, 2, 2),
    "up1": (1, 512, 4, 4),
    "up2": (1, 256, 8, 8),
    "up3": (1, 128, 16, 16),
    "up4": (1, 64, 32, 32),
}


def _checkpoint_hook_count(adapter: nn.Module) -> int:
    model = adapter.model
    return sum(
        len(getattr(model, name)._forward_hooks)
        for name in CHECKPOINT_NAMES
    )


def test_forward_trace_is_frozen_and_preserves_tensor_references():
    logits = torch.randn(1, 4, 8, 8)
    activation = torch.randn(1, 8, 4, 4)
    trace = ForwardTrace(logits=logits, activations={"node": activation})

    assert trace.logits is logits
    assert trace.activations["node"] is activation
    with pytest.raises(FrozenInstanceError):
        setattr(trace, "logits", activation)


@pytest.mark.parametrize("model_name", ["unet_baseline", "unet_noskip"])
def test_unet_adapters_return_logits_and_stable_checkpoint_shapes(
    model_name: str,
):
    adapter = build_adapter(model_name, n_channels=4, num_classes=4)
    adapter.eval()
    x = torch.randn(1, 4, 32, 32, dtype=torch.float32)

    with torch.no_grad():
        logits = adapter(x)
        trace = adapter.trace(x)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, 4, 32, 32)
    assert torch.equal(trace.logits, logits)
    assert tuple(trace.activations) == CHECKPOINT_NAMES
    assert {
        name: tuple(value.shape)
        for name, value in trace.activations.items()
    } == EXPECTED_NODE_SHAPES


def test_trace_does_not_detach_logits_or_activations():
    adapter = build_adapter("unet_noskip")
    adapter.eval()
    x = torch.randn(1, 4, 16, 16, requires_grad=True)

    trace = adapter.trace(x)

    assert trace.logits.requires_grad
    assert all(value.requires_grad for value in trace.activations.values())


def test_repeated_trace_calls_do_not_leak_hooks_or_retain_old_activations():
    adapter = build_adapter("unet_baseline")
    adapter.eval()
    before = _checkpoint_hook_count(adapter)

    with torch.no_grad():
        first = adapter.trace(torch.zeros(1, 4, 16, 16))
        after_first = _checkpoint_hook_count(adapter)
        second = adapter.trace(torch.ones(1, 4, 16, 16))
        after_second = _checkpoint_hook_count(adapter)

    assert before == after_first == after_second == 0
    assert first.activations is not second.activations
    for name in CHECKPOINT_NAMES:
        assert first.activations[name] is not second.activations[name]


def test_trace_removes_hooks_when_forward_raises():
    adapter = build_adapter("unet_noskip")
    adapter.eval()

    def fail_forward(_module: nn.Module, _inputs: tuple[torch.Tensor, ...]) -> None:
        raise RuntimeError("intentional forward failure")

    failure_handle = adapter.model.register_forward_pre_hook(fail_forward)
    try:
        with pytest.raises(RuntimeError, match="intentional forward failure"):
            adapter.trace(torch.zeros(1, 4, 16, 16))
        assert _checkpoint_hook_count(adapter) == 0
    finally:
        failure_handle.remove()


def test_build_adapter_rejects_unknown_model_name():
    with pytest.raises(
        ValueError,
        match=r"Unknown model adapter.*unet_baseline.*unet_noskip",
    ):
        build_adapter("unknown_unet")


def test_adapter_exposes_only_declared_checkpoint_modules():
    adapter = build_adapter("unet_baseline")

    assert adapter.checkpoint_module("down1") is adapter.model.down1
    assert adapter.randomization_module("down1") is adapter.model.down1
    with pytest.raises(KeyError, match="Unknown checkpoint module"):
        adapter.checkpoint_module("inc")
    with pytest.raises(KeyError, match="Unknown randomization module"):
        adapter.randomization_module("inc")


def test_checkpoint_loader_accepts_raw_ordered_dict_and_returns_sha256(
    tmp_path: Path,
):
    source = nn.Linear(3, 2)
    target = nn.Linear(3, 2)
    state_dict = OrderedDict(
        (name, value.detach().clone())
        for name, value in source.state_dict().items()
    )
    checkpoint_path = tmp_path / "raw_state_dict.pth"
    torch.save(state_dict, checkpoint_path)
    expected_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    actual_sha256 = load_checkpoint(
        target,
        checkpoint_path,
        map_location="cpu",
    )

    assert actual_sha256 == expected_sha256
    for source_value, target_value in zip(
        source.state_dict().values(),
        target.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(source_value, target_value)


def test_checkpoint_loader_rejects_non_tensor_state_dict(tmp_path: Path):
    checkpoint_path = tmp_path / "wrapped_state_dict.pth"
    torch.save({"state_dict": nn.Linear(2, 2).state_dict()}, checkpoint_path)

    with pytest.raises(ValueError, match="tensor state_dict"):
        load_checkpoint(nn.Linear(2, 2), checkpoint_path, map_location="cpu")


def test_adapter_checkpoint_loading_is_strict(tmp_path: Path):
    adapter = build_adapter("unet_noskip")
    checkpoint_path = tmp_path / "incomplete_state_dict.pth"
    torch.save(
        OrderedDict([("unexpected.weight", torch.zeros(1))]),
        checkpoint_path,
    )

    with pytest.raises(
        RuntimeError,
        match=r"(?s)Missing key\(s\).*Unexpected key\(s\)",
    ):
        adapter.load_checkpoint(checkpoint_path, map_location="cpu")
