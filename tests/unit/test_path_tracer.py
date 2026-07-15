from pathlib import Path

import numpy as np
import torch
from torch import nn

from pptt.models.protocol import ModelAdapter
from pptt.observers.linear import LinearObserver
from pptt.pipeline.trace_model import ObserverPathTracer
from pptt.types import ForwardTrace


class _TwoNodeAdapter(ModelAdapter):
    checkpoint_names = ("n1", "n2")

    def __init__(self):
        super().__init__()
        self.first = nn.Identity()
        self.second = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, :1], -x[:, :1]], dim=1)

    def trace(self, x: torch.Tensor) -> ForwardTrace:
        return ForwardTrace(
            logits=self.forward(x),
            activations={"n1": self.first(x), "n2": self.second(-x)},
        )

    def checkpoint_module(self, name: str) -> nn.Module:
        return {"n1": self.first, "n2": self.second}[name]

    def randomization_module(self, name: str) -> nn.Module:
        return self.checkpoint_module(name)

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        map_location=None,
    ) -> str:
        raise NotImplementedError


def _identity_observer() -> LinearObserver:
    observer = LinearObserver(2, 2)
    with torch.no_grad():
        observer.projection.weight.copy_(
            torch.tensor([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]])
        )
        observer.projection.bias.zero_()
    return observer


def test_path_tracer_forms_canonical_states_and_restart_reliability():
    observers = {
        node: {seed: _identity_observer() for seed in (17, 29, 43)}
        for node in ("n1", "n2")
    }
    tracer = ObserverPathTracer(
        _TwoNodeAdapter(),
        observers,
        nodes=("n1", "n2"),
        seeds=(17, 29, 43),
        reliability_threshold=0.1,
        device="cpu",
    )
    image = torch.tensor(
        [[[[2.0, -2.0], [3.0, -3.0]], [[-2.0, 2.0], [-3.0, 3.0]]]]
    )

    result = tracer.trace_slice(image, np.zeros((2, 2), dtype=np.uint8))

    np.testing.assert_array_equal(result.states[0], [[0, 1], [0, 1]])
    np.testing.assert_array_equal(result.states[1], [[1, 0], [1, 0]])
    np.testing.assert_array_equal(result.reliable, np.ones((1, 2, 2), dtype=bool))
    np.testing.assert_array_equal(result.final_model_state, [[0, 1], [0, 1]])
