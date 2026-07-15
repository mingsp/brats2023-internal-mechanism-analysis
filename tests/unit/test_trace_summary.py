import numpy as np

from pptt.io.artifacts import CaseTrace
from pptt.pipeline.summarize_trace import summarize_case_trace


def test_case_summary_recovers_persistent_events_and_exact_metrics():
    truth = np.array([[[1, 1, 2, 2, 3, 3]]], dtype=np.uint8)
    states = np.array(
        [
            [[[0, 1, 2, 2, 0, 3]]],
            [[[1, 1, 2, 0, 0, 3]]],
            [[[1, 1, 2, 0, 3, 3]]],
        ],
        dtype=np.uint8,
    )
    trace = CaseTrace(
        states=states,
        reliable=np.ones((2, 1, 1, 6), dtype=bool),
        truth=truth,
    )

    summary = summarize_case_trace(
        trace,
        nodes=("n1", "n2", "n3"),
        num_classes=4,
    )

    first = summary.transition_rows[0]
    second = summary.transition_rows[1]
    assert first["persistent_correction_count"] == 1
    assert first["persistent_destruction_count"] == 1
    assert second["persistent_correction_count"] == 1
    assert all(row["metric_max_abs_error"] == 0 for row in summary.reconstruction_rows)
    assert all(
        row["left_confusion_max_abs_error"] == 0
        and row["right_confusion_max_abs_error"] == 0
        for row in summary.reconstruction_rows
    )
    assert sum(
        row["count"]
        for row in summary.tensor_rows
        if row["transition_index"] == 0 and row["scope"] == "full"
    ) == truth.size
    assert all(
        row["occupancy_delta"] == row["inflow"] - row["outflow"]
        for row in summary.flow_rows
    )
    assert len(summary.metric_delta_rows) == 2 * 5
    assert sum(
        row["count"]
        for row in summary.confusion_rows
        if row["node"] == "n1"
    ) == truth.size
