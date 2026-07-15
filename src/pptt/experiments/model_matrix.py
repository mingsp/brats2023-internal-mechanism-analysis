from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelMatrixJob:
    model: str
    seed: int
    model_config: Path
    checkpoint: Path
    checkpoint_root: str

    @property
    def job_id(self) -> str:
        return f"{self.model}/seed_{self.seed}"


@dataclass(frozen=True)
class ModelMatrix:
    asset_root: Path
    jobs: tuple[ModelMatrixJob, ...]


def _resolve_checkpoint(
    path: str,
    root: str,
    *,
    asset_root: Path,
    workspace_root: Path,
) -> Path:
    source = Path(path)
    if source.is_absolute():
        return source
    if root == "asset":
        return asset_root / source
    if root == "workspace":
        return workspace_root / source
    raise ValueError(f"Unsupported checkpoint root {root!r}")


def load_model_matrix(
    path: str | Path,
    *,
    workspace_root: str | Path,
    asset_root_override: str | Path | None = None,
) -> ModelMatrix:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        payload: Any = yaml.safe_load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
        raise ValueError("Model matrix must contain a jobs list")
    workspace = Path(workspace_root).resolve()
    asset_config = payload["asset_root"]
    if asset_root_override is not None:
        asset_root = Path(asset_root_override)
    else:
        asset_root = Path(
            os.environ.get(
                str(asset_config["environment_variable"]),
                str(asset_config["default"]),
            )
        )
    asset_root = asset_root.resolve()
    jobs: list[ModelMatrixJob] = []
    for raw in payload["jobs"]:
        if not isinstance(raw, dict):
            raise ValueError("Each model matrix job must be a mapping")
        checkpoint = raw["checkpoint"]
        job = ModelMatrixJob(
            model=str(raw["model"]),
            seed=int(raw["seed"]),
            model_config=(workspace / str(raw["model_config"])).resolve(),
            checkpoint=_resolve_checkpoint(
                str(checkpoint["path"]),
                str(checkpoint["root"]),
                asset_root=asset_root,
                workspace_root=workspace,
            ).resolve(),
            checkpoint_root=str(checkpoint["root"]),
        )
        jobs.append(job)
    identifiers = [job.job_id for job in jobs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Model matrix contains duplicate model/seed jobs")
    return ModelMatrix(asset_root=asset_root, jobs=tuple(jobs))
