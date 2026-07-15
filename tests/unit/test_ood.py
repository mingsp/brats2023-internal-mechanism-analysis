import numpy as np

from pptt.interventions.ood import (
    fit_ood_reference,
    local_context_descriptor,
    score_ood,
)


def test_ood_reference_assigns_larger_scores_to_shifted_vectors():
    rng = np.random.default_rng(17)
    native = rng.normal(size=(256, 12))
    reference = fit_ood_reference(
        native,
        components=6,
        neighbors=5,
        max_samples=256,
        seed=17,
    )

    in_distribution = score_ood(reference, native[:32])
    shifted = score_ood(reference, native[:32] + 8.0)

    assert shifted.mahalanobis.mean() > in_distribution.mahalanobis.mean()
    assert shifted.knn.mean() > in_distribution.knn.mean()


def test_local_context_descriptor_preserves_center_and_adds_context():
    activation = np.arange(18, dtype=np.float32).reshape(2, 3, 3)

    descriptor = local_context_descriptor(activation)

    assert descriptor.shape == (6, 3, 3)
    np.testing.assert_array_equal(descriptor[:2], activation)
    assert np.all(descriptor[4:] >= 0)
