from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from pptt.visualization.v7_coverage import render_v7_coverage_figure


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the locked V7 effect-and-coverage audit figure."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    config_path = (
        workspace
        / "configs"
        / "experiments"
        / "v7_network_process_intervention_alignment.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result_root = workspace / str(config["output_root"])
    summary_path = result_root / "network_alignment_cell_summary.parquet"
    status_path = result_root / "v7_status.json"
    for path in (summary_path, status_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "INSUFFICIENT_NETWORK_COVERAGE":
        raise ValueError("V7 locked status changed")
    cell_summary = pd.read_parquet(summary_path)
    ordered = cell_summary.sort_values("transition_index", kind="mergesort")
    transitions = tuple(ordered.drop_duplicates("transition_index")["transition"])
    restore_nodes = tuple(str(value) for value in config["intervention"]["restore_nodes"])
    output_root = workspace / "results" / "paper_outputs" / args.language
    output_root.mkdir(parents=True, exist_ok=True)
    output_base = output_root / "figS_v7_network_coverage_audit"
    manifest = render_v7_coverage_figure(
        output_base=output_base,
        language=args.language,
        cell_summary=cell_summary,
        transitions=transitions,
        restore_nodes=restore_nodes,
        status=str(status["status"]),
        dpi=args.dpi,
    )
    manifest["source_sha256"] = {
        "configuration": _sha256(config_path),
        "cell_summary": _sha256(summary_path),
        "status": _sha256(status_path),
    }
    output_base.with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
