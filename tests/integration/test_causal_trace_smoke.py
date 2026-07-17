import numpy as np
import torch

from pptt.interventions.causal_runtime import trace_spatial_intervention
from pptt.models.adapters import build_adapter
from pptt.observers.linear import LinearObserver
from pptt.pipeline.trace_model import ObserverPathTracer


def _tracer(model_name: str, image: torch.Tensor) -> ObserverPathTracer:
    adapter = build_adapter(model_name).eval()
    with torch.inference_mode():
        traced = adapter.trace(image)
    seeds = (17, 29)
    observers = {
        node: {
            seed: LinearObserver(activation.shape[1], 4).eval()
            for seed in seeds
        }
        for node, activation in traced.activations.items()
    }
    return ObserverPathTracer(
        adapter,
        observers,
        nodes=tuple(traced.activations),
        seeds=seeds,
        reliability_threshold=0.0,
        device="cpu",
    )


def test_spatial_path_intervention_changes_baseline_and_cleans_hook():
    torch.manual_seed(7)
    image = torch.randn(1, 4, 32, 32)
    truth = np.zeros((32, 32), dtype=np.uint8)
    tracer = _tracer("unet_baseline", image)
    module = tracer.adapter.model.up2
    clean = tracer.trace_slice(image, truth)
    mask = torch.ones((8, 8), dtype=torch.bool)

    changed = trace_spatial_intervention(
        tracer,
        image,
        truth,
        module=module,
        argument_index=1,
        restore_mask=mask,
        shift_yx=(2, 2),
        alpha=0.0,
    )

    assert clean.final_logits is not None
    assert changed.final_logits is not None
    assert not np.array_equal(clean.final_logits, changed.final_logits)
    assert len(module._forward_pre_hooks) == 0


def test_no_skip_same_named_input_is_an_exact_negative_control():
    torch.manual_seed(11)
    image = torch.randn(1, 4, 32, 32)
    truth = np.zeros((32, 32), dtype=np.uint8)
    tracer = _tracer("unet_noskip", image)
    module = tracer.adapter.model.up2
    clean = tracer.trace_slice(image, truth)
    mask = torch.ones((8, 8), dtype=torch.bool)

    changed = trace_spatial_intervention(
        tracer,
        image,
        truth,
        module=module,
        argument_index=1,
        restore_mask=mask,
        shift_yx=(2, 2),
        alpha=0.0,
    )

    np.testing.assert_array_equal(clean.final_logits, changed.final_logits)
    np.testing.assert_array_equal(clean.states, changed.states)
    np.testing.assert_array_equal(clean.reliable, changed.reliable)
    assert len(module._forward_pre_hooks) == 0
