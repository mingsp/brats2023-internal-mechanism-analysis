import unittest

import numpy as np

from guide_unet.process_analysis.internal_inference_framework import (
    NodeState,
    analyze_internal_path,
    build_path_transitions,
)


class InternalInferenceFrameworkTest(unittest.TestCase):
    def setUp(self):
        self.states = [
            NodeState(
                "up2",
                1,
                semantic_score=0.50,
                response_alignment={"cam": 0.20, "raw": 0.30},
                region_allocation={"cam.tumor": 0.25},
                output_influence={"cam.random_adjusted": 0.02},
            ),
            NodeState(
                "up3",
                2,
                semantic_score=0.62,
                response_alignment={"cam": 0.35, "raw": 0.42},
                region_allocation={"cam.tumor": 0.40},
                output_influence={"cam.random_adjusted": 0.04},
            ),
            NodeState(
                "up4",
                3,
                semantic_score=0.81,
                response_alignment={"cam": 0.73, "raw": 0.68},
                region_allocation={"cam.tumor": 0.76},
                output_influence={},
            ),
        ]

    def test_f_computes_named_transition_metrics(self):
        transitions = build_path_transitions(
            self.states,
            structural_by_transition={"up3->up4": {"skip_zero.drop": 0.77}},
        )
        self.assertEqual([item.transition_id for item in transitions], ["up2->up3", "up3->up4"])
        self.assertAlmostEqual(transitions[0].values["semantic.delta"], 0.12)
        self.assertAlmostEqual(transitions[1].values["response.cam.delta"], 0.38)
        self.assertAlmostEqual(transitions[1].values["structure.skip_zero.drop"], 0.77)

    def test_g_preserves_missing_evidence(self):
        output = analyze_internal_path(
            self.states,
            structural_by_transition={"up3->up4": {"skip_zero.drop": 0.77}},
        )
        trajectory = output.metric_trajectory("function.cam.random_adjusted.delta")
        self.assertAlmostEqual(trajectory[0], 0.02)
        self.assertIsNone(trajectory[1])
        self.assertAlmostEqual(output.coverage("function.cam.random_adjusted.delta"), 0.5)

    def test_g_has_no_aggregate_performance_score(self):
        output = analyze_internal_path(self.states)
        self.assertEqual(output.values.shape[1], 2)
        self.assertFalse(np.any(np.isinf(output.values)))
        self.assertEqual(output.key_transition_by_metric["semantic.delta"], "up3->up4")
        self.assertFalse(hasattr(output, "score"))

    def test_duplicate_order_is_rejected(self):
        invalid = [self.states[0], NodeState("duplicate", 1, semantic_score=0.6)]
        with self.assertRaises(ValueError):
            build_path_transitions(invalid)


if __name__ == "__main__":
    unittest.main()
