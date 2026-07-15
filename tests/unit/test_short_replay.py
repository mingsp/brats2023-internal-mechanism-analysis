from pathlib import Path

import pytest

from pptt.reproducibility.replay import compare_json_outputs, guarded_output_directory


def test_compare_json_outputs_checks_bytes_and_numeric_payload(tmp_path: Path):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text('{"passed": true, "value": 1.0}\n', encoding="utf-8")
    after.write_text('{"passed": true, "value": 1.0}\n', encoding="utf-8")

    report = compare_json_outputs(before, after)

    assert report["byte_identical"] is True
    assert report["payload_identical"] is True
    assert report["before_sha256"] == report["after_sha256"]


def test_guarded_output_directory_rejects_paths_outside_results(tmp_path: Path):
    workspace = tmp_path / "workspace"
    results = workspace / "results"
    results.mkdir(parents=True)
    output = results / "v0" / "report.json"

    assert guarded_output_directory(output, results) == output.parent.resolve()
    with pytest.raises(ValueError, match="outside the guarded results root"):
        guarded_output_directory(workspace / "config.json", results)
