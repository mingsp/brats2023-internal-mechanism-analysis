from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pptt.statistics.process_effects import candidate_specific_causal_gate


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_locked(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"existing protocol lock differs: {path}")
        return
    patient_outputs = list(path.parent.glob("**/patient_results/*.json"))
    if patient_outputs:
        raise ValueError("cannot create protocol lock after intervention outputs exist")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lock the candidate-specific V6 causal protocol before intervention."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/v6_pixel_transition_causal_tracing.yaml"),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument(
        "--lock-filename",
        default="v6_protocol_lock.json",
    )
    parser.add_argument("--supersedes-lock", type=Path)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    config_path = args.config.resolve()
    config = _load_yaml(config_path)
    effect_root = workspace / str(config["process_effect_root"])
    output_root = workspace / str(config["output_root"])
    candidate_path = effect_root / "candidate_transition_statistics.parquet"
    global_gate_path = effect_root / "causal_entry_gate.json"
    representative_path = effect_root / "representative_case.json"
    for path in (candidate_path, global_gate_path, representative_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    candidate_rows = pd.read_parquet(candidate_path)
    global_gate = json.loads(global_gate_path.read_text(encoding="utf-8"))
    target = config["target"]
    statistics = config["statistics"]
    candidate_gate = candidate_specific_causal_gate(
        candidate_rows.to_dict("records"),
        transition=str(target["transition"]),
        required_model_seeds=tuple(int(value) for value in config["model_seeds"]),
        alpha=float(statistics["alpha"]),
        minimum_standardized_effect=float(
            statistics["minimum_standardized_effect"]
        ),
    )
    selected = global_gate.get("candidate", {})
    selected_path = selected.get("path", {})
    expected_identity = {
        "transition": str(target["transition"]),
        "transition_index": int(target["transition_index"]),
        "module": str(target["module"]),
        "argument_index": int(target["argument_index"]),
        "source": str(target["source"]),
    }
    actual_identity = {
        "transition": str(selected.get("transition")),
        "transition_index": int(selected.get("transition_index")),
        "module": str(selected_path.get("module")),
        "argument_index": int(selected_path.get("argument_index")),
        "source": str(selected_path.get("source")),
    }
    if actual_identity != expected_identity:
        raise ValueError(
            f"configured V6 path differs from validation-selected path: {actual_identity}"
        )
    if not candidate_gate["passed"]:
        raise ValueError("candidate-specific validation gate did not pass")

    payload = {
        "status": "LOCKED_BEFORE_FIRST_INTERVENTION",
        "intervention_authorized": True,
        "sequential_design": True,
        "global_two_metric_gate_passed": bool(global_gate.get("passed", False)),
        "global_two_metric_gate_must_remain_reported": True,
        "candidate_identity": expected_identity,
        "candidate_specific_validation_gate": candidate_gate,
        "candidate_selection_split": str(config["discovery_split"]),
        "intervention_evaluation_split": str(config["primary_split"]),
        "registered_model_seeds": [int(value) for value in config["model_seeds"]],
        "thresholds_may_not_be_changed_after_lock": True,
        "configuration_sha256": _sha256(config_path),
        "source_sha256": {
            "candidate_transition_statistics": _sha256(candidate_path),
            "global_causal_entry_gate": _sha256(global_gate_path),
            "representative_case": _sha256(representative_path),
        },
        "configuration": config,
    }
    if args.supersedes_lock is not None:
        superseded = args.supersedes_lock.resolve()
        if not superseded.is_file():
            raise FileNotFoundError(superseded)
        if list(output_root.glob("**/patient_results/*.json")):
            raise ValueError("protocol identity cannot be amended after intervention output")
        payload["supersedes"] = {
            "path": str(superseded),
            "sha256": _sha256(superseded),
            "reason": str(config["protocol"]["amendment_reason"]),
            "scope": "observer_seed_identity_only",
            "before_first_intervention": True,
        }
    lock_path = output_root / str(args.lock_filename)
    _write_locked(lock_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"V6 protocol locked: {lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
