import numpy as np

from pptt.visualization.flagship import (
    BRATS_CLASS_COLORS,
    segmentation_overlay,
    select_representative_slice,
)


def test_representative_slice_is_fixed_by_all_classes_then_largest_area():
    truth = np.zeros((3, 4, 4), dtype=np.uint8)
    truth[0, 0, :2] = [1, 2]
    truth[1, 0, :3] = [1, 2, 3]
    truth[2, 0, :4] = [1, 2, 3, 1]

    selected = select_representative_slice(truth, truth_classes=(1, 2, 3))

    assert selected == 2


def test_representative_slice_falls_back_to_largest_tumor_area():
    truth = np.zeros((2, 3, 3), dtype=np.uint8)
    truth[0, 0, 0] = 1
    truth[1, 0, :2] = 2

    selected = select_representative_slice(truth, truth_classes=(1, 2, 3))

    assert selected == 1


def test_segmentation_overlay_preserves_background_and_blends_classes():
    image = np.zeros((2, 2), dtype=np.float32)
    labels = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    overlay = segmentation_overlay(
        image,
        labels,
        colors=BRATS_CLASS_COLORS,
        alpha=0.5,
    )

    np.testing.assert_allclose(overlay[0, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(overlay[0, 1], [0.5, 0.0, 0.0])
    np.testing.assert_allclose(overlay[1, 0], [0.0, 0.5, 0.0])
    np.testing.assert_allclose(overlay[1, 1], [0.5, 0.5, 0.0])
