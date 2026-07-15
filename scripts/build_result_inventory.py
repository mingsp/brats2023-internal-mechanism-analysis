from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch
import yaml

from pptt.io.manifests import write_assets_json
from pptt.reproducibility.inventory import (
    build_result_inventory,
    format_environment_lock,
    verify_result_inventory,
)


def _run_text(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return f"UNAVAILABLE: {(completed.stderr or completed.stdout).strip()}"
    return completed.stdout.strip()


def _git_metadata(workspace: Path) -> tuple[str, bool]:
    commit = _run_text(["git", "-C", os.fspath(workspace), "rev-parse", "HEAD"])
    status = _run_text(["git", "-C", os.fspath(workspace), "status", "--porcelain"])
    return commit, bool(status)


def _runtime_environment() -> dict[str, Any]:
    gpu_names: list[str] = []
    gpu_memory_mib: list[int] = []
    gpu_capabilities: list[str] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu_names.append(properties.name)
            gpu_memory_mib.append(int(properties.total_memory // (1024 * 1024)))
            major, minor = torch.cuda.get_device_capability(index)
            gpu_capabilities.append(f"{major}.{minor}")
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_names": gpu_names,
        "gpu_memory_mib": gpu_memory_mib,
        "gpu_capabilities": gpu_capabilities,
        "nvidia_smi": _run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }


def _packages() -> list[str]:
    output = _run_text([sys.executable, "-m", "pip", "freeze", "--all"])
    if output.startswith("UNAVAILABLE:"):
        return [output]
    return [line for line in output.splitlines() if line.strip()]


def _load_commands(path: Path) -> dict[str, str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    commands = payload.get("commands", payload)
    if not isinstance(commands, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in commands.items()
    ):
        raise ValueError("Run command manifest must map names to command strings")
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or verify the PPTT content-addressed result inventory."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument(
        "--commands",
        type=Path,
        default=Path("manifests/run_commands.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manifests/result_inventory.json"),
    )
    parser.add_argument(
        "--environment-lock",
        type=Path,
        default=Path("manifests/environment_lock.txt"),
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    asset_root = args.asset_root or os.environ.get("PPTT_ASSET_ROOT")
    if asset_root is None:
        raise ValueError("--asset-root or PPTT_ASSET_ROOT is required")
    assets = Path(asset_root).resolve()
    results = (args.results_root or workspace / "results").resolve()
    output = args.output if args.output.is_absolute() else workspace / args.output

    if args.verify:
        inventory = json.loads(output.read_text(encoding="utf-8"))
        problems = verify_result_inventory(
            inventory,
            workspace_root=workspace,
            asset_root=assets,
            results_root=results,
        )
        if problems:
            print(json.dumps({"status": "FAIL", "problems": problems}, indent=2))
            return 1
        print(json.dumps({"status": "PASS", "verified_records": inventory["result_summary"]}, indent=2))
        return 0

    environment = _runtime_environment()
    commit, dirty = _git_metadata(workspace)
    inventory = build_result_inventory(
        workspace_root=workspace,
        asset_root=assets,
        results_root=results,
        run_commands=_load_commands(
            args.commands if args.commands.is_absolute() else workspace / args.commands
        ),
        git_commit=commit,
        git_dirty=dirty,
        environment=environment,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    write_assets_json(output, inventory)
    lock_path = (
        args.environment_lock
        if args.environment_lock.is_absolute()
        else workspace / args.environment_lock
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        format_environment_lock(environment=environment, packages=_packages()),
        encoding="utf-8",
    )
    print(json.dumps(inventory["result_summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
