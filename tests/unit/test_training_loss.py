import torch
from torch.nn import functional as F

from pptt.training.losses import (
    multiclass_dice_loss,
    segmentation_loss,
    segmentation_loss_parts,
)


def test_segmentation_loss_is_finite_and_differentiable():
    logits = torch.randn(2, 4, 16, 16, requires_grad=True)
    target = torch.randint(0, 4, (2, 16, 16))
    loss = segmentation_loss(logits, target)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_segmentation_loss_uses_locked_weights():
    logits = torch.randn(2, 4, 8, 8)
    target = torch.randint(0, 4, (2, 8, 8))
    parts = segmentation_loss_parts(logits, target)

    torch.testing.assert_close(parts.total, 0.2 * parts.cross_entropy + 0.8 * parts.dice)


def test_manual_cross_entropy_matches_pytorch_reference():
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    target = torch.randint(0, 4, (2, 8, 8))
    parts = segmentation_loss_parts(logits, target)

    torch.testing.assert_close(parts.cross_entropy, F.cross_entropy(logits, target))


def test_multiclass_dice_is_small_for_confident_correct_logits():
    target = torch.tensor([[[0, 1], [2, 3]]])
    logits = torch.full((1, 4, 2, 2), -12.0)
    logits.scatter_(1, target[:, None], 12.0)

    assert float(multiclass_dice_loss(logits, target)) < 1e-6
