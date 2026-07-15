from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn


def checkpoint_sha256(path: str | Path) -> str:
    checkpoint_path = Path(path)
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> str:
    """Load a raw tensor state dictionary with strict key checking."""

    checkpoint_path = Path(path)
    payload: Any = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, Mapping) or not payload or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in payload.items()
    ):
        raise ValueError(
            f"Checkpoint must be a non-empty raw tensor state_dict: {checkpoint_path}"
        )
    model.load_state_dict(payload, strict=True)
    return checkpoint_sha256(checkpoint_path)
