import numpy as np
import pytest
import torch

from pptt.models.adapters import build_adapter
from pptt.observers.randomization import (
    module_state_sha256,
    randomize_checkpoint_module,
)


def _outside_state(adapter, node):
    prefix = f"{node}."
    return {
        name: value.detach().clone()
        for name, value in adapter.model.state_dict().items()
        if not name.startswith(prefix)
    }


@pytest.mark.parametrize("model", ["unet_baseline", "unet_noskip"])
def test_checkpoint_randomization_changes_only_selected_module(model):
    torch.manual_seed(3)
    adapter = build_adapter(model)
    before_outside = _outside_state(adapter, "down4")
    before_hash = module_state_sha256(adapter.checkpoint_module("down4"))

    recorded_before, after_hash = randomize_checkpoint_module(
        adapter,
        "down4",
        seed=17,
    )

    assert recorded_before == before_hash
    assert after_hash != before_hash
    for name, expected in before_outside.items():
        np.testing.assert_array_equal(adapter.model.state_dict()[name], expected)


def test_checkpoint_randomization_is_deterministic():
    first = build_adapter("unet_baseline")
    second = build_adapter("unet_baseline")
    second.load_state_dict(first.state_dict())

    _, first_hash = randomize_checkpoint_module(first, "up4", seed=29)
    _, second_hash = randomize_checkpoint_module(second, "up4", seed=29)

    assert first_hash == second_hash
