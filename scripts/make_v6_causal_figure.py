from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pptt.visualization.causal import render_causal_result_figure


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the formal PPTT causal-faithfulness result figure."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    causal_root = workspace / "results" / "v6_pixel_transition_causal"
    artifact_path = causal_root / "causal_representative_case_artifacts.npz"
    artifact_manifest_path = causal_root / "causal_representative_case_artifacts.json"
    conclusion_path = causal_root / "causal_conclusion_gate.json"
    for path in (artifact_path, artifact_manifest_path, conclusion_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    conclusion = json.loads(conclusion_path.read_text(encoding="utf-8"))
    if artifact_manifest.get("status") != "PASS":
        raise ValueError("representative case export is not PASS")
    if not bool(conclusion.get("passed", False)):
        raise ValueError("formal causal conclusion gate is not PASS")

    with np.load(artifact_path, allow_pickle=False) as arrays:
        image = arrays["image"]
        truth = arrays["truth"]
        target = arrays["target_pixels"]
        condition_states = {
            "clean": arrays["clean_states"],
            "corrupt": arrays["corrupt_states"],
            "restore_target": arrays["restore_target_states"],
            "restore_control": arrays["restore_control_states"],
        }
    dose = pd.read_parquet(causal_root / "causal_dose_summary.parquet")
    statistics = pd.read_parquet(causal_root / "causal_patient_statistics.parquet")
    output_root = workspace / "results" / "paper_outputs" / args.language
    output_root.mkdir(parents=True, exist_ok=True)
    output_base = output_root / "fig3_pixel_transition_causal_faithfulness"
    png_path, pdf_path = render_causal_result_figure(
        output_base=output_base,
        language=args.language,
        image=image,
        truth=truth,
        target_mask=target,
        condition_states=condition_states,
        transition_index=int(artifact_manifest["transition_index"]),
        dose=dose,
        statistics=statistics,
        dpi=args.dpi,
    )
    manifest = {
        "status": "PASS",
        "scientific_question": (
            "does_the_registered_path_causally_contribute_to_persistent_pixel_correction"
        ),
        "language": args.language,
        "case_artifact": artifact_path,
        "case_manifest": artifact_manifest_path,
        "dose_table": causal_root / "causal_dose_summary.parquet",
        "statistics_table": causal_root / "causal_patient_statistics.parquet",
        "conclusion_gate": conclusion_path,
        "representative_selection": artifact_manifest["selection"],
        "outputs": {"png": png_path, "pdf": pdf_path},
    }
    manifest_path = output_root / "fig3_pixel_transition_causal_faithfulness.json"
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
