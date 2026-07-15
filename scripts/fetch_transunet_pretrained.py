from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from urllib.request import Request, urlopen

import gdown
import numpy as np


TARGET_FILENAME = "R50+ViT-B_16.npz"
STORAGE_URL = (
    "https://storage.googleapis.com/vit_models/imagenet21k/"
    "R50+ViT-B_16.npz"
)
DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd"
)
REQUIRED_KEYS = (
    "embedding/kernel",
    "embedding/bias",
    "Transformer/posembed_input/pos_embedding",
    "Transformer/encoder_norm/scale",
    "Transformer/encoder_norm/bias",
    "conv_root/kernel",
    "gn_root/scale",
    "gn_root/bias",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        keys = tuple(archive.files)
        missing = sorted(set(REQUIRED_KEYS) - set(keys))
        if missing:
            raise ValueError(f"NPZ is missing required keys: {missing}")
        root_shape = tuple(int(value) for value in archive["conv_root/kernel"].shape)
        embedding_shape = tuple(
            int(value) for value in archive["embedding/kernel"].shape
        )
    return {
        "key_count": len(keys),
        "required_keys": list(REQUIRED_KEYS),
        "root_kernel_hwio_shape": list(root_shape),
        "embedding_kernel_hwio_shape": list(embedding_shape),
    }


def _download_storage(destination: Path) -> None:
    request = Request(STORAGE_URL, headers={"User-Agent": "pptt-process-xai/0.1"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)


def _download_drive(destination: Path) -> None:
    entries = gdown.download_folder(
        url=DRIVE_FOLDER_URL,
        quiet=True,
        remaining_ok=True,
        skip_download=True,
    )
    if not entries:
        raise RuntimeError("Official Google Drive folder could not be listed")
    matches = [entry for entry in entries if Path(entry.path).name == TARGET_FILENAME]
    if len(matches) != 1:
        names = sorted(Path(entry.path).name for entry in entries)
        raise RuntimeError(
            f"Expected exactly one {TARGET_FILENAME!r} in official folder, "
            f"found {len(matches)} among {names}"
        )
    result = gdown.download(
        id=matches[0].id,
        output=str(destination),
        quiet=False,
    )
    if result is None or not destination.is_file():
        raise RuntimeError("Official Google Drive download failed")


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def fetch(output: Path, manifest: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, str]] = []

    if output.is_file():
        validation = _validate_npz(output)
        payload = {
            "mode": "pretrained",
            "source": "existing_verified_file",
            "source_url": None,
            "path": str(output.resolve()),
            "sha256": _sha256(output),
            "size_bytes": output.stat().st_size,
            **validation,
        }
        _write_manifest(manifest, payload)
        return payload

    downloaders = (
        ("google_storage", STORAGE_URL, _download_storage),
        ("official_google_drive", DRIVE_FOLDER_URL, _download_drive),
    )
    for source, source_url, downloader in downloaders:
        temporary = output.with_name(f".{output.name}.{source}.part")
        temporary.unlink(missing_ok=True)
        try:
            downloader(temporary)
            validation = _validate_npz(temporary)
            os.replace(temporary, output)
            payload = {
                "mode": "pretrained",
                "source": source,
                "source_url": source_url,
                "path": str(output.resolve()),
                "sha256": _sha256(output),
                "size_bytes": output.stat().st_size,
                **validation,
            }
            _write_manifest(manifest, payload)
            return payload
        except Exception as exc:  # Each official source is attempted in order.
            temporary.unlink(missing_ok=True)
            errors.append({"source": source, "error": f"{type(exc).__name__}: {exc}"})

    payload = {
        "mode": "scratch",
        "source": None,
        "source_url": None,
        "path": None,
        "sha256": None,
        "errors": errors,
        "policy": "All seeds must use the same scratch initialization policy.",
    }
    _write_manifest(manifest, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the pinned official TransUNet initialization asset."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = fetch(args.output.expanduser(), args.manifest.expanduser())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
