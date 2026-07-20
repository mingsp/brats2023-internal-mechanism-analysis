from __future__ import annotations

from collections import OrderedDict
import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from pptt.causal_abstraction.runtime import run_state_exchange
from pptt.models.adapters import HookedModelAdapter
from pptt.observers.linear import LinearObserver


NODE_NAMES = (
    "down1",
    "down2",
    "down3",
    "down4",
    "up1",
    "up2",
    "up3",
    "up4",
)


class _EightNodeIdentityNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in NODE_NAMES:
            layer = nn.Conv2d(2, 2, kernel_size=1, bias=False)
            with torch.no_grad():
                layer.weight.copy_(torch.eye(2).reshape(2, 2, 1, 1))
            setattr(self, name, layer)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        state = image
        for name in NODE_NAMES:
            state = getattr(self, name)(state)
        return state


def _adapter() -> HookedModelAdapter:
    model = _EightNodeIdentityNet()
    return HookedModelAdapter(
        model,
        OrderedDict((name, getattr(model, name)) for name in NODE_NAMES),
    ).eval()


def _observer() -> LinearObserver:
    observer = LinearObserver(2, 2)
    with torch.no_grad():
        observer.projection.weight.copy_(torch.eye(2).reshape(2, 2, 1, 1))
        observer.projection.bias.zero_()
    return observer.eval()


def _observers() -> dict[str, dict[int, LinearObserver]]:
    return {
        node: {seed: _observer() for seed in (17, 29, 43)}
        for node in NODE_NAMES
    }


def _parameter_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _hook_count(adapter: HookedModelAdapter) -> int:
    return sum(
        len(adapter.checkpoint_module(name)._forward_hooks)
        for name in adapter.checkpoint_names
    )


def _image() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [[2.0, -1.0], [3.0, -2.0]],
                [[-2.0, 1.0], [-3.0, 2.0]],
            ]
        ],
        dtype=torch.float32,
    )


def test_state_exchange_traces_complete_registered_downstream_path():
    adapter = _adapter()
    image = _image()
    truth = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    delta_h = torch.zeros((4, 2), dtype=torch.float32)
    delta_h[0] = torch.tensor([-4.0, 4.0])
    clean_logits = adapter(image).detach().cpu().numpy()[0]
    before_hash = _parameter_hash(adapter)

    result = run_state_exchange(
        adapter,
        image,
        node="down3",
        delta_h=delta_h,
        doses=(0.0, 0.5, 1.0),
        observers=_observers(),
        truth=truth,
        reliability_threshold=0.0,
    )

    expected_path = ("down3", "down4", "up1", "up2", "up3", "up4", "Y")
    assert result.node == "down3"
    assert result.doses == (0.0, 0.5, 1.0)
    assert tuple(result.downstream_states[0.0]) == expected_path
    assert tuple(result.downstream_states[1.0]) == expected_path
    np.testing.assert_allclose(result.final_logits[0.0], clean_logits, atol=1e-6)
    assert result.downstream_states[0.0]["Y"][0, 0] == 0
    assert result.downstream_states[1.0]["Y"][0, 0] == 2
    assert result.audits["upstream_max_abs_error"] == pytest.approx(0.0)
    assert result.audits["dose_zero_max_abs_logit_error"] <= 1.0e-6
    assert result.audits["parameter_storage_and_version_unchanged"] is True
    assert _parameter_hash(adapter) == before_hash
    assert _hook_count(adapter) == 0


def test_state_exchange_dose_batching_preserves_complete_results():
    common = {
        "adapter": _adapter(),
        "image": _image(),
        "node": "down2",
        "delta_h": torch.tensor(
            [[-4.0, 4.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=torch.float32,
        ),
        "doses": (0.0, 0.25, 0.5, 0.75, 1.0),
        "observers": _observers(),
        "truth": np.array([[0, 1], [0, 1]], dtype=np.uint8),
        "reliability_threshold": 0.0,
    }
    unbatched = run_state_exchange(**common)
    batched = run_state_exchange(**common, dose_batch_size=2)

    assert batched.audits["dose_batch_count"] == 3
    for dose in common["doses"]:
        for depth in batched.downstream_states[dose]:
            np.testing.assert_array_equal(
                batched.downstream_states[dose][depth],
                unbatched.downstream_states[dose][depth],
            )
            np.testing.assert_array_equal(
                batched.downstream_reliable[dose][depth],
                unbatched.downstream_reliable[dose][depth],
            )
        np.testing.assert_allclose(
            batched.final_logits[dose],
            unbatched.final_logits[dose],
            atol=1.0e-6,
        )


def test_state_exchange_accepts_one_shared_clean_trace():
    adapter = _adapter()
    image = _image()
    clean = adapter.trace(image)
    result = run_state_exchange(
        adapter,
        image,
        node="up2",
        delta_h=torch.tensor(
            [[-4.0, 4.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=torch.float32,
        ),
        doses=(0.0, 1.0),
        observers=_observers(),
        truth=np.array([[0, 1], [0, 1]], dtype=np.uint8),
        reliability_threshold=0.0,
        clean_trace=clean,
    )

    assert result.audits["dose_zero_max_abs_logit_error"] <= 1.0e-6
    assert _hook_count(adapter) == 0


def test_state_exchange_removes_transform_hook_after_forward_failure():
    adapter = _adapter()
    bad_delta = torch.zeros((3, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="delta_h shape"):
        run_state_exchange(
            adapter,
            _image(),
            node="down3",
            delta_h=bad_delta,
            doses=(0.0, 1.0),
            observers=_observers(),
            truth=np.zeros((2, 2), dtype=np.uint8),
            reliability_threshold=0.0,
        )

    assert _hook_count(adapter) == 0


def test_checkpoint_transform_context_rejects_shape_change_and_cleans_up():
    adapter = _adapter()

    with pytest.raises(ValueError, match="changed tensor shape"):
        with adapter.transform_checkpoint_output(
            "down2",
            lambda output: output[..., :-1],
        ):
            adapter(_image())

    assert _hook_count(adapter) == 0


def test_state_exchange_rejects_unknown_or_duplicate_protocol_values():
    adapter = _adapter()
    common = {
        "adapter": adapter,
        "image": _image(),
        "delta_h": torch.zeros((4, 2), dtype=torch.float32),
        "observers": _observers(),
        "truth": np.zeros((2, 2), dtype=np.uint8),
        "reliability_threshold": 0.0,
    }
    with pytest.raises(ValueError, match="unknown checkpoint"):
        run_state_exchange(node="missing", doses=(0.0, 1.0), **common)
    with pytest.raises(ValueError, match="unique"):
        run_state_exchange(node="down3", doses=(0.0, 1.0, 1.0), **common)
    with pytest.raises(ValueError, match="dose zero"):
        run_state_exchange(node="down3", doses=(0.5, 1.0), **common)
