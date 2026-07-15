import numpy as np
import pytest
import torch

from pptt.observers.linear import LinearObserver
from pptt.observers.objective import (
    class_balanced_pixel_weights,
    temperature_scale_probabilities,
)


def test_observer_contains_only_pointwise_linear_projection():
    observer = LinearObserver(in_channels=64, num_classes=4)
    modules = list(observer.modules())
    convolutions = [module for module in modules if isinstance(module, torch.nn.Conv2d)]

    assert len(convolutions) == 1
    assert convolutions[0].kernel_size == (1, 1)
    assert not any(
        isinstance(module, (torch.nn.ReLU, torch.nn.BatchNorm2d))
        for module in modules
    )


def test_observer_resizes_logits_deterministically():
    observer = LinearObserver(in_channels=3, num_classes=4)
    feature = torch.randn(2, 3, 5, 7)
    first = observer(feature, output_size=(20, 28))
    second = observer(feature, output_size=(20, 28))

    assert first.shape == (2, 4, 20, 28)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_class_balance_weights_depend_only_on_final_predicted_classes():
    probabilities = np.array(
        [
            [0.9, 0.05, 0.03, 0.02],
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
        ],
        dtype=np.float32,
    )
    weights = class_balanced_pixel_weights(probabilities, smoothing=1.0)

    assert weights.shape == (4,)
    assert np.isfinite(weights).all()
    assert np.all(weights > 0)
    assert weights[0] < weights[2]
    assert weights.mean() == pytest.approx(1.0)


def test_temperature_scaling_preserves_normalization_and_rejects_invalid_tau():
    probabilities = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float32)
    scaled = temperature_scale_probabilities(probabilities, temperature=2.0)

    torch.testing.assert_close(scaled.sum(dim=1), torch.ones(1))
    assert scaled.max() < probabilities.max()
    with pytest.raises(ValueError, match="positive"):
        temperature_scale_probabilities(probabilities, temperature=0.0)
