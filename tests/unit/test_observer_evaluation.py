import numpy as np
import torch

from pptt.observers.evaluation import (
    patient_probability_metrics,
    predict_observer_probabilities,
)
from pptt.observers.linear import LinearObserver


def test_observer_prediction_returns_canonical_untempered_probabilities():
    observer = LinearObserver(2, 2)
    with torch.no_grad():
        observer.projection.weight.copy_(
            torch.tensor([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]])
        )
        observer.projection.bias.zero_()
    features = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    actual = predict_observer_probabilities(
        observer,
        features,
        device="cpu",
    )
    expected = torch.softmax(torch.tensor([[2.0, 0.0], [0.0, 2.0]]), dim=1).numpy()

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0)


def test_patient_probability_metrics_keep_patient_as_unit():
    target = np.eye(2, dtype=np.float32)[[0, 0, 1, 1]]
    prediction = target.copy()
    patients = np.array([0, 0, 1, 1])
    rows = patient_probability_metrics(target, prediction, patients)

    assert len(rows) == 2
    assert all(row["mean_js_divergence"] == 0 for row in rows)
    assert all(row["hard_agreement"] == 1 for row in rows)
