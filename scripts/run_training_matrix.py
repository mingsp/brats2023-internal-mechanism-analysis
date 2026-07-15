from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


MODEL_CONFIGS = {
    "unet_baseline": Path("configs/models/unet_baseline.yaml"),
    "unet_noskip": Path("configs/models/unet_noskip.yaml"),
    "transunet_r50_vit_b16": Path("configs/models/transunet_r50_vit_b16.yaml"),
}


def _parse_job(value: str) -> tuple[str, int]:
    model, separator, raw_seed = value.rpartition(":")
    if not separator or model not in MODEL_CONFIGS:
        raise ValueError(f"Invalid training job {value!r}")
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise ValueError(f"Invalid seed in training job {value!r}") from exc
    if seed < 0:
        raise ValueError("Training seeds must be nonnegative")
    return model, seed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the locked segmentation training matrix sequentially."
    )
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/segmentation_training"),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/segmentation_training"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = [_parse_job(value) for value in args.jobs]
    if len(jobs) != len(set(jobs)):
        raise ValueError("Training jobs must be unique")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    overall_passed = True
    for model, seed in jobs:
        output = args.output_root / model / f"seed_{seed}"
        log_path = args.log_dir / f"{model}_seed{seed}.log"
        command = [
            sys.executable,
            "scripts/train_segmentation.py",
            "--config",
            str(MODEL_CONFIGS[model]),
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--output",
            str(output),
        ]
        resumed = args.resume and (output / "latest_training.pt").is_file()
        if resumed:
            command.append("--resume")
        if args.asset_root is not None:
            command.extend(["--asset-root", str(args.asset_root)])
        print(" ".join(command), flush=True)
        if args.dry_run:
            records.append(
                {
                    "model": model,
                    "seed": seed,
                    "status": "DRY_RUN",
                    "command": command,
                    "log": str(log_path),
                    "output": str(output),
                    "resumed": resumed,
                }
            )
            continue
        start = perf_counter()
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write("\nCOMMAND: " + " ".join(command) + "\n")
            stream.flush()
            completed = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = perf_counter() - start
        passed = completed.returncode == 0
        overall_passed &= passed
        records.append(
            {
                "model": model,
                "seed": seed,
                "status": "PASS" if passed else "FAIL",
                "return_code": completed.returncode,
                "elapsed_seconds": elapsed,
                "command": command,
                "log": str(log_path),
                "output": str(output),
                "resumed": resumed,
            }
        )
        if not passed:
            break
    summary = {
        "status": "DRY_RUN" if args.dry_run else ("PASS" if overall_passed else "FAIL"),
        "sequential_execution": True,
        "jobs": records,
    }
    summary_path = args.log_dir / "training_matrix_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if args.dry_run or overall_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
