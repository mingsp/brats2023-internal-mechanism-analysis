import pytest
import torch
import torch.nn.functional as F

from pptt.models.transunet import adapt_rgb_kernel_to_four_channels


def test_four_channel_adaptation_preserves_equal_channel_response():
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(8, 3, 7, 7, generator=generator, dtype=torch.float64)
    adapted = adapt_rgb_kernel_to_four_channels(weight)
    x3 = torch.ones(1, 3, 32, 32, dtype=torch.float64)
    x4 = torch.ones(1, 4, 32, 32, dtype=torch.float64)

    assert adapted.shape == (8, 4, 7, 7)
    torch.testing.assert_close(
        F.conv2d(x3, weight),
        F.conv2d(x4, adapted),
        atol=1e-6,
        rtol=1e-6,
    )


def test_four_channel_adaptation_is_channel_symmetric():
    weight = torch.arange(2 * 3 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3, 3)
    adapted = adapt_rgb_kernel_to_four_channels(weight)

    for channel in range(1, 4):
        torch.testing.assert_close(adapted[:, 0], adapted[:, channel])


@pytest.mark.parametrize(
    "shape",
    [(3, 7, 7), (8, 1, 7, 7), (8, 4, 7, 7)],
)
def test_four_channel_adaptation_rejects_non_rgb_oihw_kernels(shape):
    with pytest.raises(ValueError, match="OIHW kernel with three input channels"):
        adapt_rgb_kernel_to_four_channels(torch.zeros(shape))
