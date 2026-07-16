import numpy as np

from pptt.io.artifacts import CaseTrace
from scripts.run_v3_unet_pair import _trace_storage_payload, _v1_admission_status


def _trace() -> CaseTrace:
    return CaseTrace(
        states=np.zeros((3, 2, 4, 4), dtype=np.uint8),
        reliable=np.ones((2, 2, 4, 4), dtype=np.bool_),
        margins=np.full((3, 2, 4, 4), 0.25, dtype=np.float16),
        truth=np.ones((2, 4, 4), dtype=np.uint8),
        final_model_state=np.full((2, 4, 4), 2, dtype=np.uint8),
        slice_ids=("patient_0", "patient_1"),
    )


def test_v3_compact_storage_omits_unused_margins_but_keeps_auditable_states():
    trace = _trace()

    payload = _trace_storage_payload(
        trace,
        {"retain_margins": False, "retain_final_model_state": True},
    )

    assert payload["margins"] is None
    assert payload["final_model_state"] is trace.final_model_state
    assert payload["states"] is trace.states
    assert payload["reliable"] is trace.reliable
    assert payload["truth"] is trace.truth
    assert payload["slice_ids"] == trace.slice_ids


def test_v3_trace_storage_can_retain_optional_fields_explicitly():
    trace = _trace()

    payload = _trace_storage_payload(
        trace,
        {"retain_margins": True, "retain_final_model_state": False},
    )

    assert payload["margins"] is trace.margins
    assert payload["final_model_state"] is None


def test_v3_requires_an_explicit_passed_v1_status(tmp_path):
    status_path = tmp_path / "v1_status.json"

    assert _v1_admission_status(status_path) == "MISSING_V1"

    status_path.write_text('{"status": "PENDING_RANDOMIZATION"}', encoding="utf-8")
    assert _v1_admission_status(status_path) == "V1_NOT_PASSED"

    status_path.write_text('{"status": "FAIL"}', encoding="utf-8")
    assert _v1_admission_status(status_path) == "V1_NOT_PASSED"

    status_path.write_text('{"status": "PASS"}', encoding="utf-8")
    assert _v1_admission_status(status_path) == "PASS"
