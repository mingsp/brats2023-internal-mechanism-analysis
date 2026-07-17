from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pptt.visualization.method_overview import render_pptt_method_figure


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
        description="Create the publication PPTT method overview figure."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    causal_root = workspace / "results" / "v6_pixel_transition_causal"
    artifact_path = causal_root / "causal_representative_case_artifacts.npz"
    manifest_path = causal_root / "causal_representative_case_artifacts.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise ValueError("method figure source artifact is not PASS")
    with np.load(artifact_path, allow_pickle=False) as arrays:
        image = arrays["image"]
        truth = arrays["truth"]
        states = arrays["clean_states"]
        reliable = arrays["clean_reliable"]
        final_state = arrays["clean_final_state"]
    output_root = workspace / "results" / "paper_outputs" / args.language
    output_root.mkdir(parents=True, exist_ok=True)
    output_base = output_root / "fig1_pptt_method_overview"
    png_path, pdf_path = render_pptt_method_figure(
        output_base=output_base,
        language=args.language,
        image=image,
        truth=truth,
        states=states,
        reliable=reliable,
        final_state=final_state,
        dpi=args.dpi,
    )
    figure_manifest = {
        "status": "PASS",
        "figure_role": "architecture_agnostic_process_xai_method_overview",
        "language": args.language,
        "source_artifact": artifact_path,
        "source_manifest": manifest_path,
        "uses_same_slice_for_all_nodes": True,
        "uses_actual_observer_states": True,
        "contains_stage_numbering": False,
        "contains_cam": False,
        "outputs": {"png": png_path, "pdf": pdf_path},
    }
    output_manifest = output_root / "fig1_pptt_method_overview.json"
    output_manifest.write_text(
        json.dumps(_json_ready(figure_manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(figure_manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
