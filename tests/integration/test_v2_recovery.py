from pptt.synthetic.benchmarks import evaluate_known_process
from pptt.synthetic.known_process import build_known_process_batch


def test_v2_recovers_known_transition_stage_direction_and_region():
    batch = build_known_process_batch(
        batch_size=8,
        height=64,
        width=64,
        seed=20260715,
    )
    result = evaluate_known_process(batch)

    assert result.stage_accuracy == 1.0
    assert result.direction_macro_f1 >= 0.95
    assert result.region_iou >= 0.90
    assert result.origin_depth_accuracy >= 0.95
    assert result.branch_attribution_accuracy >= 0.95
    assert result.max_metric_reconstruction_error < 1e-10
    assert 0.0 <= result.node_dice_stage_accuracy <= 1.0
    assert 0.0 <= result.gradcam_region_iou <= 1.0
    assert 0.0 <= result.layercam_region_iou <= 1.0
