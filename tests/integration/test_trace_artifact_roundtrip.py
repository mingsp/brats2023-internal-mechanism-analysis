import numpy as np

from pptt.io.artifacts import load_case_trace, save_case_trace


def test_case_trace_roundtrip(tmp_path):
    states = np.arange(8 * 3 * 5 * 7, dtype=np.uint16).reshape(8, 3, 5, 7) % 4
    reliable = np.ones((7, 3, 5, 7), dtype=bool)
    truth = np.zeros((3, 5, 7), dtype=np.uint8)
    final_model_state = np.ones((3, 5, 7), dtype=np.uint8)
    path = tmp_path / "case_trace.npz"

    save_case_trace(
        path,
        states=states,
        reliable=reliable,
        truth=truth,
        final_model_state=final_model_state,
        slice_ids=("case_001", "case_002", "case_003"),
    )
    loaded = load_case_trace(path)

    np.testing.assert_array_equal(loaded.states, states)
    np.testing.assert_array_equal(loaded.reliable, reliable)
    np.testing.assert_array_equal(loaded.truth, truth)
    np.testing.assert_array_equal(loaded.final_model_state, final_model_state)
    assert loaded.slice_ids == ("case_001", "case_002", "case_003")


def test_minimal_case_trace_roundtrip(tmp_path):
    states = np.zeros((8, 5, 7), dtype=np.uint8)
    reliable = np.ones((7, 5, 7), dtype=bool)
    path = tmp_path / "minimal.npz"

    save_case_trace(path, states=states, reliable=reliable)
    loaded = load_case_trace(path)

    np.testing.assert_array_equal(loaded.states, states)
    np.testing.assert_array_equal(loaded.reliable, reliable)
    assert loaded.truth is None
