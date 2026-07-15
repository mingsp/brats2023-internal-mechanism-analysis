import numpy as np
import torch

from pptt.observers.cache import build_observer_cache
from pptt.observers.controls import patient_permuted_targets
from pptt.observers.objective import mean_js_divergence
from pptt.observers.trainer import ObserverTrainingConfig, train_observer


def _toy_cache(seed: int = 20260715):
    rng = np.random.default_rng(seed)
    patient_indices = np.repeat(np.arange(8), 32)
    features = rng.normal(size=(patient_indices.size, 5)).astype(np.float32)
    true_weight = rng.normal(size=(4, 5)).astype(np.float32)
    true_bias = rng.normal(size=(4,)).astype(np.float32)
    logits = features @ true_weight.T + true_bias
    logits -= logits.max(axis=1, keepdims=True)
    targets = np.exp(logits)
    targets /= targets.sum(axis=1, keepdims=True)
    return build_observer_cache(
        features,
        targets.astype(np.float32),
        patient_indices=patient_indices,
    )


def _predict(observer, features):
    tensor = torch.from_numpy(features.astype(np.float32))[:, :, None, None]
    with torch.no_grad():
        logits = observer(tensor, output_size=(1, 1))[:, :, 0, 0]
        return torch.softmax(logits, dim=1).cpu().numpy()


def test_convex_observer_converges_across_initializations_and_beats_controls():
    cache = _toy_cache()
    config = ObserverTrainingConfig(
        l2=1e-3,
        temperature=1.0,
        max_iter=120,
        tolerance_grad=1e-8,
        tolerance_change=1e-12,
        history_size=50,
    )
    predictions = []
    reports = []
    for seed in (17, 29, 43):
        observer, report = train_observer(cache, config=config, seed=seed)
        predictions.append(_predict(observer, cache.features))
        reports.append(report)

    for prediction in predictions[1:]:
        np.testing.assert_allclose(prediction, predictions[0], atol=2e-4, rtol=2e-4)
    assert all(report.final_gradient_norm < 1e-4 for report in reports)
    real_jsd = mean_js_divergence(cache.target_probabilities, predictions[0])

    patient_targets, _ = patient_permuted_targets(
        cache.target_probabilities,
        cache.patient_indices,
        seed=20260715,
    )
    patient_cache = build_observer_cache(
        cache.features,
        patient_targets,
        patient_indices=cache.patient_indices,
    )
    patient_observer, _ = train_observer(patient_cache, config=config, seed=17)
    patient_jsd = mean_js_divergence(
        patient_targets,
        _predict(patient_observer, cache.features),
    )

    spatial_targets = np.roll(cache.target_probabilities, 47, axis=0)
    spatial_cache = build_observer_cache(
        cache.features,
        spatial_targets,
        patient_indices=cache.patient_indices,
    )
    spatial_observer, _ = train_observer(spatial_cache, config=config, seed=17)
    spatial_jsd = mean_js_divergence(
        spatial_targets,
        _predict(spatial_observer, cache.features),
    )

    assert real_jsd < patient_jsd
    assert real_jsd < spatial_jsd
