from __future__ import annotations

import numpy as np
import torch

from pptt.observers.linear import LinearObserver
from pptt.observers.objective import mean_js_divergence


def predict_observer_probabilities(
    observer: LinearObserver,
    features: np.ndarray,
    *,
    temperature: float,
    device: str | torch.device,
    batch_size: int = 65536,
) -> np.ndarray:
    if temperature <= 0 or batch_size <= 0:
        raise ValueError("temperature and batch_size must be positive")
    feature_array = np.asarray(features)
    if feature_array.ndim != 2 or feature_array.shape[1] != observer.in_channels:
        raise ValueError("features do not match the observer input channels")
    target_device = torch.device(device)
    observer = observer.to(target_device).eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, feature_array.shape[0], batch_size):
            batch = torch.from_numpy(
                feature_array[start : start + batch_size].astype(np.float32, copy=False)
            ).to(target_device)
            logits = observer(batch[:, :, None, None], output_size=(1, 1))[:, :, 0, 0]
            output.append(torch.softmax(logits / float(temperature), dim=1).cpu().numpy())
    return np.concatenate(output).astype(np.float32, copy=False)


def patient_probability_metrics(
    target_probabilities: np.ndarray,
    predicted_probabilities: np.ndarray,
    patient_indices: np.ndarray,
) -> list[dict[str, float | int]]:
    target = np.asarray(target_probabilities)
    prediction = np.asarray(predicted_probabilities)
    patients = np.asarray(patient_indices)
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("target and prediction must be equal row-by-class matrices")
    if patients.shape != (target.shape[0],):
        raise ValueError("patient_indices must contain one entry per row")
    rows: list[dict[str, float | int]] = []
    for patient_index in sorted(np.unique(patients).tolist()):
        selected = patients == patient_index
        rows.append(
            {
                "patient_index": int(patient_index),
                "row_count": int(selected.sum()),
                "mean_js_divergence": mean_js_divergence(
                    target[selected],
                    prediction[selected],
                ),
                "hard_agreement": float(
                    np.mean(
                        target[selected].argmax(axis=1)
                        == prediction[selected].argmax(axis=1)
                    )
                ),
            }
        )
    return rows
