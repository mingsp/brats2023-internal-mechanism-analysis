#!/usr/bin/env python3
"""Profile V11 peak GPU memory on one locked validation patient."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from pptt.data.brats2d import discover_slice_records
from pptt.experiments.model_matrix import load_model_matrix
from scripts.lock_v11_causal_abstraction_protocol import (
    sha256_file,
    sha256_named_values,
    source_tree_sha256,
    validate_v11_configuration,
)
from scripts.run_v11_causal_abstraction import (
    FormalPatientProcessor,
    _group_records,
    _job_for,
    _load_formal_runtime,
    _load_json,
    _load_yaml,
    _write_json_atomic,
)


def _select_profile_patient(
    plan_root: Path,
    patient_ids: Sequence[str],
) -> tuple[str, dict[str, Any]]:
    ranked = []
    for patient_id in sorted(str(value) for value in patient_ids):
        payload = _load_json(plan_root / f"{patient_id}.json")
        pairs_by_node = {
            str(node["node"]): len(node.get("pairs", []))
            for node in payload.get("nodes", [])
        }
        matched_nodes = sum(value > 0 for value in pairs_by_node.values())
        pair_count = sum(pairs_by_node.values())
        ranked.append((-matched_nodes, -pair_count, patient_id, payload))
    if not ranked:
        raise ValueError("validation planning contains no patients")
    _, negative_pairs, patient_id, payload = min(ranked)
    if negative_pairs == 0:
        raise ValueError("validation planning contains no matched intervention pairs")
    return patient_id, payload


def profile_v11_memory(
    *,
    workspace: Path,
    config_path: Path,
    protocol_lock_path: Path,
    asset_root: Path,
    model: str,
    model_seed: int,
    device: torch.device,
    dose_batch_size: int,
    output_path: Path,
    resume: bool,
) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V11 memory profiling requires an available CUDA device")
    configuration = _load_yaml(config_path)
    validated = validate_v11_configuration(configuration)
    if model not in validated.main_models:
        raise ValueError("memory profiles are registered only for the two main architectures")
    if int(model_seed) not in validated.model_seeds:
        raise ValueError("memory-profile seed is outside the registered matrix")
    lock = _load_json(protocol_lock_path)
    if (
        lock.get("status") != "LOCKED_BEFORE_FORMAL_INTERVENTION"
        or lock.get("formal_intervention_authorized") is not True
        or lock.get("configuration_sha256") != sha256_file(config_path)
        or lock.get("source_identity", {}).get("source_tree_sha256")
        != source_tree_sha256(workspace)
    ):
        raise ValueError("memory profile requires the active immutable V11 lock")

    matrix = load_model_matrix(
        workspace / str(configuration["model_matrix"]),
        workspace_root=workspace,
        asset_root_override=asset_root,
    )
    job = _job_for(matrix.jobs, model=model, model_seed=int(model_seed))
    runtime = _load_formal_runtime(
        job,
        workspace=workspace,
        configuration=configuration,
        device=device,
    )
    fit_root = workspace / str(configuration["validation_fit_root"])
    validation_root = fit_root / "planning_jobs" / model / f"seed_{int(model_seed)}"
    calibration = _load_json(validation_root / "job_calibration.json")
    if calibration.get("status") != "PASS":
        raise ValueError("validation operator calibration is not PASS")
    plan_root = validation_root / "patient_plans"
    patient_id, patient_plan = _select_profile_patient(
        plan_root,
        calibration.get("patient_ids", []),
    )
    plan_hashes = dict(calibration.get("patient_plan_hashes", {}))
    if plan_hashes.get(patient_id) != sha256_file(plan_root / f"{patient_id}.json"):
        raise ValueError("validation profile plan hash differs from calibration")

    data_configuration = _load_yaml(workspace / "configs/data/brats2023_2d.yaml")
    split = str(configuration["fit_split"])
    split_configuration = data_configuration["splits"][split]
    records = discover_slice_records(
        asset_root / str(split_configuration["image_dir"]),
        asset_root / str(split_configuration["mask_dir"]),
    )
    grouped = _group_records(records)
    if sorted(grouped) != sorted(str(value) for value in calibration["patient_ids"]):
        raise ValueError("active validation patients differ from memory-profile calibration")
    records_by_slice = {record.slice_id: record for record in records}
    norm_calibration = _load_json(fit_root / "norm_and_leakage_calibration.json")
    if norm_calibration.get("status") != "PASS":
        raise ValueError("norm and leakage calibration is not PASS")
    identity = {
        "schema_version": 1,
        "formal_claim_eligible": False,
        "model": model,
        "model_seed": int(model_seed),
        "dose_batch_size": int(dose_batch_size),
        "validation_patient_id": patient_id,
        "validation_patient_plan_sha256": plan_hashes[patient_id],
        "validation_matched_node_count": sum(
            bool(node.get("pairs")) for node in patient_plan["nodes"]
        ),
        "protocol_lock_sha256": sha256_file(protocol_lock_path),
        "configuration_sha256": sha256_file(config_path),
        "source_tree_sha256": source_tree_sha256(workspace),
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "observer_registry_sha256": sha256_named_values(runtime.observer_hashes),
    }
    if resume and output_path.is_file():
        previous = _load_json(output_path)
        if all(previous.get(key) == value for key, value in identity.items()):
            runtime.adapter.to("cpu")
            for group in runtime.observers.values():
                for observer in group.values():
                    observer.to("cpu")
            torch.cuda.empty_cache()
            return previous
        raise ValueError("existing memory profile has a different locked identity")

    processor = FormalPatientProcessor(
        runtime=runtime,
        patient_plan_root=plan_root,
        patient_plan_hashes=plan_hashes,
        records_by_slice=records_by_slice,
        configuration=configuration,
        norm_calibration=norm_calibration,
        device=device,
        dose_batch_size=int(dose_batch_size),
        split=split,
    )
    try:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        processor(patient_id)
        torch.cuda.synchronize(device)
        payload = {
            **identity,
            "status": "COMPLETE_NONFORMAL_VALIDATION_MEMORY_PROFILE",
            "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
            "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 2**20),
            "current_allocated_mib": float(torch.cuda.memory_allocated(device) / 2**20),
            "current_reserved_mib": float(torch.cuda.memory_reserved(device) / 2**20),
            "scientific_outputs_persisted": False,
            "profile_may_not_enter_formal_summary": True,
        }
        _write_json_atomic(output_path, payload)
        return payload
    finally:
        runtime.adapter.to("cpu")
        for group in runtime.observers.values():
            for observer in group.values():
                observer.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/v11_causal_abstraction.yaml"),
    )
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dose-batch-size", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace_root.resolve()
    config_path = args.config if args.config.is_absolute() else workspace / args.config
    lock_path = (
        args.protocol_lock
        if args.protocol_lock.is_absolute()
        else workspace / args.protocol_lock
    )
    output_path = args.output if args.output.is_absolute() else workspace / args.output
    result = profile_v11_memory(
        workspace=workspace,
        config_path=config_path.resolve(),
        protocol_lock_path=lock_path.resolve(),
        asset_root=args.asset_root.resolve(),
        model=str(args.model),
        model_seed=int(args.model_seed),
        device=torch.device(args.device),
        dose_batch_size=int(args.dose_batch_size),
        output_path=output_path.resolve(),
        resume=bool(args.resume),
    )
    print(
        f"{result['model']} peak reserved: {result['peak_reserved_mib']:.1f} MiB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
