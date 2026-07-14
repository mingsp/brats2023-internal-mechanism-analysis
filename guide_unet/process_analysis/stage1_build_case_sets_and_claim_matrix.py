from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def patient_id_from_case_name(case_name: str) -> str:
    parts = str(case_name).split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return str(case_name)


def write_json(payload: Dict, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_run_config(args: argparse.Namespace, out_dir: str) -> None:
    write_json(vars(args).copy(), os.path.join(out_dir, "run_config.json"))


def write_manifest(out_dir: str, rows: Sequence[Dict]) -> None:
    pd.DataFrame(list(rows)).to_csv(os.path.join(out_dir, "run_manifest.csv"), index=False)


CLAIMS = [
    {
        "claim_id": "C1",
        "claim_text": "no-skip 在 Stage 1 早期节点具有更高语义可读性。",
        "primary_modules": "E0,E2,E3",
        "required_evidence": "stage1 curve; response alignment; case-level support",
        "allowed_wording": "no-skip shows higher early-stage semantic readout in this U-Net comparison.",
        "forbidden_wording": "no-skip is universally better in early layers.",
    },
    {
        "claim_id": "C2",
        "claim_text": "baseline 在 down4/up1 附近出现中部低谷。",
        "primary_modules": "E0,E2,E3",
        "required_evidence": "stage1 curve; down4/up1 response quality; case-level support",
        "allowed_wording": "baseline exhibits a mid-network valley around down4/up1.",
        "forbidden_wording": "skip connections always cause a mid-network collapse.",
    },
    {
        "claim_id": "C3",
        "claim_text": "baseline 在 up2->up3->up4 呈现后期加速恢复。",
        "primary_modules": "E0,E2,E3,E4,E5,E6",
        "required_evidence": "semantic acceleration; response approach velocity; delta coupling; perturbation; robustness",
        "allowed_wording": "baseline late acceleration is consistent with rapid decoder-stage recovery of task-relevant responses.",
        "forbidden_wording": "this proves the complete causal mechanism of skip connections.",
    },
    {
        "claim_id": "C4",
        "claim_text": "no-skip 在 up2->up3->up4 呈现后期减速趋稳。",
        "primary_modules": "E0,E2,E3,E5,E6",
        "required_evidence": "semantic deceleration; response saturation; delta coupling; robustness",
        "allowed_wording": "no-skip late deceleration is consistent with limited additional task-relevant response in late decoding.",
        "forbidden_wording": "every no-skip case strictly decelerates.",
    },
    {
        "claim_id": "C5",
        "claim_text": "raw feature response 增量与 task readout 增量的病例级耦合比 CAM 更稳定。",
        "primary_modules": "E3,E5",
        "required_evidence": "case-level delta correlation; aggregation robustness",
        "allowed_wording": "raw feature response provides more stable case-level delta support than CAM in the current evidence set.",
        "forbidden_wording": "CAM is useless or invalid.",
    },
    {
        "claim_id": "C6",
        "claim_text": "高响应区域具有任务相关性，扰动后输出下降大于随机同面积扰动。",
        "primary_modules": "E4",
        "required_evidence": "top-k response perturbation > random same-area and low-response perturbation",
        "allowed_wording": "response-guided perturbation supports the task relevance of identified high-response regions.",
        "forbidden_wording": "perturbation proves the full causal mechanism.",
    },
    {
        "claim_id": "C7",
        "claim_text": "主要结论对 response 聚合方式和 CAM target/variant 具有基本鲁棒性。",
        "primary_modules": "E5,E6",
        "required_evidence": "aggregation stability; CAM target/variant sensitivity",
        "allowed_wording": "the main trends are not artifacts of a single aggregation rule or CAM target.",
        "forbidden_wording": "all explanation methods produce identical maps.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build common case sets and an initial claim evidence matrix for the Stage-1 explanation suite.")
    parser.add_argument("--output-root", type=str, default="stage1_explanation_suite_20260517/results")
    parser.add_argument("--source-case-csv", type=str, default="results/stage1_case_readout_alignment_n512/table_stage1_case_phenomenon_support.csv")
    parser.add_argument("--cam-per-case-csv", type=str, default="results/phenomenon_cause_fullnetwork_gradcam_phase0_3_n512/table_phase1_2_gradcam_per_case.csv")
    parser.add_argument("--top10-summary-csv", type=str, default="results/stage1_selected_case_visuals_n512_top10/stage1_top10_visual_summary.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-size", type=int, default=16)
    parser.add_argument("--dev-size", type=int, default=64)
    parser.add_argument("--formal-size", type=int, default=128)
    parser.add_argument("--formal-large-size", type=int, default=512)
    return parser.parse_args()


def unique_names_from_csv(path: str) -> List[str]:
    if not os.path.isfile(path):
        return []
    df = pd.read_csv(path)
    if "case_name" not in df.columns:
        return []
    names: List[str] = []
    seen = set()
    for name in df["case_name"].astype(str).tolist():
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def selected_nested_cases(names: Sequence[str], size: int, seed: int) -> List[str]:
    if int(size) <= 0 or int(size) >= len(names):
        return list(names)
    rng = np.random.default_rng(int(seed))
    arr = np.asarray(list(names), dtype=object)
    order = rng.permutation(len(arr))
    return [str(arr[idx]) for idx in order[: int(size)]]


def write_case_set(names: Sequence[str], out_path: str, case_set: str, seed: int, source_flags: Dict[str, set]) -> None:
    rows = []
    for rank, case_name in enumerate(names):
        row = {
            "case_set": case_set,
            "selection_rank": int(rank),
            "case_name": str(case_name),
            "patient_id": patient_id_from_case_name(str(case_name)),
            "seed": int(seed),
        }
        for flag_name, flag_values in source_flags.items():
            row[flag_name] = str(case_name) in flag_values
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def build_claim_matrix(out_path: str) -> None:
    rows = []
    for item in CLAIMS:
        rows.append(
            {
                **item,
                "evidence_type": "",
                "source_table": "",
                "source_figure": "",
                "metric": "",
                "model": "",
                "nodes_or_transition": "",
                "n_cases": "",
                "n_slices": "",
                "effect_size": "",
                "ci_low": "",
                "ci_high": "",
                "support_level": "pending",
                "case_level_support": "pending",
                "faithfulness_support": "pending",
                "robustness_support": "pending",
                "boundary_cases": "",
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def build_statistical_protocol(out_path: str) -> None:
    rows = [
        {
            "rule_id": "S1",
            "rule": "summary tables must include n_cases, n_slices, model, node or transition, and seed",
            "status": "required",
        },
        {
            "rule_id": "S2",
            "rule": "curve means should report bootstrap 95% CI when case-level values are available",
            "status": "required",
        },
        {
            "rule_id": "S3",
            "rule": "perturbation reports paired topk-random and topk-low_response differences",
            "status": "required",
        },
        {
            "rule_id": "S4",
            "rule": "random masks must keep the same area and record seed plus repeat index",
            "status": "required",
        },
        {
            "rule_id": "S5",
            "rule": "top10 support/boundary cases are for figures, not a replacement for formal statistics",
            "status": "required",
        },
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    common_dir = os.path.join(args.output_root, "common_case_sets")
    claim_dir = os.path.join(args.output_root, "e7_claim_evidence_matrix")
    ensure_dir(common_dir)
    ensure_dir(claim_dir)

    readout_names = unique_names_from_csv(args.source_case_csv)
    cam_names = unique_names_from_csv(args.cam_per_case_csv)
    top10_names = unique_names_from_csv(args.top10_summary_csv)
    if readout_names and cam_names:
        base_names = sorted(set(readout_names) & set(cam_names))
    else:
        base_names = readout_names or cam_names
    if not base_names:
        raise RuntimeError("No case names found from source CSVs.")

    rng = np.random.default_rng(int(args.seed))
    shuffled = list(base_names)
    rng.shuffle(shuffled)

    case_sets = {
        "smoke_n16": shuffled[: min(int(args.smoke_size), len(shuffled))],
        "dev_n64": shuffled[: min(int(args.dev_size), len(shuffled))],
        "formal_n128": shuffled[: min(int(args.formal_size), len(shuffled))],
        "formal_n512": shuffled[: min(int(args.formal_large_size), len(shuffled))],
    }
    if top10_names:
        case_sets["top10_support_boundary"] = [name for name in top10_names if name in set(base_names)] or top10_names

    source_flags = {
        "in_readout_n512": set(readout_names),
        "in_cam_n512": set(cam_names),
        "in_top10_visuals": set(top10_names),
    }
    manifest_rows = []
    for case_set, names in case_sets.items():
        out_path = os.path.join(common_dir, "{}_cases.csv".format(case_set))
        write_case_set(names, out_path, case_set=case_set, seed=int(args.seed), source_flags=source_flags)
        manifest_rows.append(
            {
                "case_set": case_set,
                "path": out_path,
                "n_cases": int(len(names)),
                "seed": int(args.seed),
                "source_case_csv": args.source_case_csv,
                "cam_per_case_csv": args.cam_per_case_csv,
            }
        )
    pd.DataFrame(manifest_rows).to_csv(os.path.join(common_dir, "table_case_set_manifest.csv"), index=False)
    build_statistical_protocol(os.path.join(common_dir, "table_statistical_protocol.csv"))
    build_claim_matrix(os.path.join(claim_dir, "table_claim_evidence_matrix.csv"))
    pd.DataFrame(CLAIMS).to_csv(os.path.join(claim_dir, "table_paper_claim_wording_guardrails.csv"), index=False)

    write_run_config(args, common_dir)
    write_manifest(common_dir, manifest_rows)
    write_json(
        {
            "purpose": "common case sets and initial claim matrix for Stage-1 explanation suite",
            "num_base_cases": int(len(base_names)),
            "case_sets": {name: int(len(values)) for name, values in case_sets.items()},
            "guardrail": "These case sets support Stage-1 phenomenon explanation only; archived skip/masking assets are not used as current entries.",
        },
        os.path.join(common_dir, "summary_common_case_sets.json"),
    )
    print("Saved common case sets to {}".format(common_dir))
    print("Saved initial claim matrix to {}".format(claim_dir))


if __name__ == "__main__":
    main()
