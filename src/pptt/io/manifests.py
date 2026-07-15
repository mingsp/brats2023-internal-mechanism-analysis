import csv
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any, TextIO, TypedDict


DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024
CSV_FIELDS = ("relative_path", "size_bytes", "sha256")
_SHA256_LINE = re.compile(
    r"^(?P<sha256>[0-9a-fA-F]{64})(?:  | \*)(?P<relative_path>.+)$"
)
_SHA256_VALUE = re.compile(r"^[0-9a-fA-F]{64}$")


class FileHashRecord(TypedDict):
    relative_path: str
    size_bytes: int
    sha256: str


def _is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or (
        is_junction is not None and bool(is_junction())
    )


def _reject_link_or_junction(path: Path) -> None:
    if _is_link_or_junction(path):
        raise ValueError(
            f"Symbolic links or junctions are not allowed: {path}"
        )


def _require_directory(path: Path, label: str) -> Path:
    candidate = Path(path)
    _reject_link_or_junction(candidate)
    if not candidate.exists():
        raise FileNotFoundError(f"{label} does not exist: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {candidate}")
    return candidate


def _require_file(path: Path, label: str) -> Path:
    candidate = Path(path)
    _reject_link_or_junction(candidate)
    if not candidate.exists():
        raise FileNotFoundError(f"{label} does not exist: {candidate}")
    if not candidate.is_file():
        raise IsADirectoryError(f"{label} is not a file: {candidate}")
    return candidate


def sha256_file(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return a file SHA-256 digest using bounded streaming reads."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    file_path = _require_file(path, "File")
    digest = sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def count_npy_pairs(image_dir: Path, mask_dir: Path) -> int:
    """Count direct .npy pairs after requiring identical filename stems."""
    image_root = _require_directory(image_dir, "image_dir")
    mask_root = _require_directory(mask_dir, "mask_dir")

    def npy_stems(root: Path) -> set[str]:
        stems: set[str] = set()
        for path in root.iterdir():
            _reject_link_or_junction(path)
            if path.is_file() and path.suffix == ".npy":
                stems.add(path.stem)
        return stems

    image_stems = npy_stems(image_root)
    mask_stems = npy_stems(mask_root)
    if image_stems != mask_stems:
        missing_masks = sorted(image_stems - mask_stems)
        missing_images = sorted(mask_stems - image_stems)
        raise ValueError(
            "Unpaired .npy stems: "
            f"missing_masks={missing_masks[:5]} (total={len(missing_masks)}), "
            f"missing_images={missing_images[:5]} (total={len(missing_images)})"
        )
    return len(image_stems)


def regular_files(root: Path) -> list[Path]:
    """Return recursively discovered regular files in stable relative order."""
    directory = _require_directory(root, "Root")
    files: list[Path] = []
    pending_directories = [directory]
    while pending_directories:
        current_directory = pending_directories.pop()
        for path in current_directory.iterdir():
            _reject_link_or_junction(path)
            if path.is_dir():
                pending_directories.append(path)
            elif path.is_file():
                files.append(path)
    return sorted(files, key=lambda path: path.relative_to(directory).as_posix())


def _root_relative_posix(path: Path, root: Path) -> str:
    directory = _require_directory(root, "Root")
    file_path = _require_file(path, "File")
    resolved_root = directory.resolve()
    resolved_file = file_path.resolve()
    try:
        relative_path = resolved_file.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"File is outside root: {file_path}") from error
    return relative_path.as_posix()


def file_hash_record(path: Path, root: Path) -> FileHashRecord:
    """Build a root-relative POSIX hash record for one regular file."""
    file_path = _require_file(path, "File")
    return {
        "relative_path": _root_relative_posix(file_path, root),
        "size_bytes": file_path.stat().st_size,
        "sha256": sha256_file(file_path),
    }


def _normalize_manifest_path(relative_path: str, data_root: Path) -> str:
    candidate = relative_path
    while candidate.startswith("./"):
        candidate = candidate[2:]
    if not candidate or "\\" in candidate or "\x00" in candidate:
        raise ValueError("manifest path must be a relative POSIX path")
    parts = candidate.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("manifest path must be a relative POSIX path")
    if parts[0] == data_root.name:
        parts = parts[1:]
    if not parts:
        raise ValueError("manifest path must name a file")
    return "/".join(parts)


def _read_sha256_entries(
    manifest_path: Path, data_root: Path
) -> dict[str, str]:
    path = _require_file(manifest_path, "SHA-256 manifest")
    entries: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline=None) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            match = _SHA256_LINE.fullmatch(line)
            if match is None:
                raise ValueError(
                    f"Invalid SHA-256 manifest line {line_number}: {path.name}"
                )
            try:
                relative_path = _normalize_manifest_path(
                    match.group("relative_path"), data_root
                )
            except ValueError as error:
                raise ValueError(
                    f"Invalid SHA-256 manifest line {line_number}: {path.name}"
                ) from error
            if relative_path in entries:
                raise ValueError(
                    f"Duplicate SHA-256 manifest path at line {line_number}: "
                    f"{relative_path}"
                )
            entries[relative_path] = match.group("sha256").lower()
    return entries


def verify_sha256_manifest(
    manifest_path: Path,
    data_root: Path,
    *,
    verify_full_hashes: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> int:
    """Validate a sha256sum manifest and optionally rehash every listed file."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    root = _require_directory(data_root, "Data root")
    entries = _read_sha256_entries(manifest_path, root)
    files = regular_files(root)
    files_by_relative_path = {
        path.relative_to(root).as_posix(): path for path in files
    }

    if len(entries) != len(files_by_relative_path):
        raise ValueError(
            f"SHA-256 manifest entry count mismatch for {Path(manifest_path).name}: "
            f"entries={len(entries)}, files={len(files_by_relative_path)}"
        )

    listed_paths = set(entries)
    actual_paths = set(files_by_relative_path)
    if listed_paths != actual_paths:
        missing_entries = sorted(actual_paths - listed_paths)
        unknown_entries = sorted(listed_paths - actual_paths)
        raise ValueError(
            f"SHA-256 manifest paths mismatch for {Path(manifest_path).name}: "
            f"missing_entries={missing_entries[:5]}, "
            f"unknown_entries={unknown_entries[:5]}"
        )

    if verify_full_hashes:
        for relative_path in sorted(entries):
            actual_digest = sha256_file(
                files_by_relative_path[relative_path], chunk_size=chunk_size
            )
            if actual_digest != entries[relative_path]:
                raise ValueError(f"SHA-256 mismatch: {relative_path}")
    return len(entries)


def _canonicalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _canonicalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_canonicalize_json(item) for item in value]
        if items and all(
            isinstance(item, Mapping) and "relative_path" in item for item in items
        ):
            items.sort(key=lambda item: str(item["relative_path"]))
        return items
    return value


def _atomic_text_write(path: Path, writer: Callable[[TextIO], None]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_assets_json(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write a canonical UTF-8 asset manifest by atomic replacement."""
    canonical_manifest = _canonicalize_json(manifest)

    def write(handle: TextIO) -> None:
        json.dump(
            canonical_manifest,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    _atomic_text_write(path, write)


def _normalize_file_hash_record(record: Mapping[str, object]) -> FileHashRecord:
    missing_fields = [field for field in CSV_FIELDS if field not in record]
    if missing_fields:
        raise ValueError(f"Missing SHA-256 CSV fields: {missing_fields}")
    relative_path = record["relative_path"]
    size_bytes = record["size_bytes"]
    digest = record["sha256"]
    if not isinstance(relative_path, str):
        raise ValueError("relative_path must be a string")
    relative_path = _normalize_manifest_path(relative_path, Path("."))
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
        raise ValueError("size_bytes must be a non-negative integer")
    if not isinstance(digest, str) or _SHA256_VALUE.fullmatch(digest) is None:
        raise ValueError("sha256 must contain 64 hexadecimal characters")
    return {
        "relative_path": relative_path,
        "size_bytes": size_bytes,
        "sha256": digest.lower(),
    }


def write_sha256_csv(
    path: Path, records: Iterable[Mapping[str, object]]
) -> None:
    """Write sorted file hash records with fixed columns by atomic replacement."""
    normalized_records = [_normalize_file_hash_record(record) for record in records]
    normalized_records.sort(key=lambda record: record["relative_path"])
    relative_paths = [record["relative_path"] for record in normalized_records]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("Duplicate relative_path in SHA-256 CSV records")

    def write(handle: TextIO) -> None:
        csv_writer = csv.DictWriter(
            handle,
            fieldnames=list(CSV_FIELDS),
            lineterminator="\n",
            extrasaction="ignore",
        )
        csv_writer.writeheader()
        csv_writer.writerows(normalized_records)

    _atomic_text_write(path, write)


def write_manifest_outputs(
    directory: Path,
    manifest: Mapping[str, Any],
    records: Iterable[Mapping[str, object]],
) -> None:
    """Stage and commit assets.json and sha256.csv as one rollback-safe batch."""
    output_directory = _require_directory(directory, "Manifest output directory")
    final_paths = {
        "assets.json": output_directory / "assets.json",
        "sha256.csv": output_directory / "sha256.csv",
    }

    with tempfile.TemporaryDirectory(
        dir=output_directory, prefix=".asset-manifest-stage-"
    ) as staging_name:
        staging_directory = Path(staging_name)
        staged_paths = {
            name: staging_directory / name for name in final_paths
        }
        write_assets_json(staged_paths["assets.json"], manifest)
        write_sha256_csv(staged_paths["sha256.csv"], records)

        backups: dict[str, Path | None] = {}
        for name, final_path in final_paths.items():
            backup_path: Path | None = None
            if final_path.exists() or _is_link_or_junction(final_path):
                existing_path = _require_file(final_path, f"Existing {name}")
                backup_path = staging_directory / f".{name}.backup"
                shutil.copy2(existing_path, backup_path)
            backups[name] = backup_path

        try:
            for name, final_path in final_paths.items():
                os.replace(staged_paths[name], final_path)
        except Exception as commit_error:
            rollback_errors: list[OSError] = []
            for name, final_path in final_paths.items():
                try:
                    backup_path = backups[name]
                    if backup_path is None:
                        final_path.unlink(missing_ok=True)
                    else:
                        os.replace(backup_path, final_path)
                except OSError as rollback_error:
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                raise OSError(
                    "Manifest output commit failed and rollback was incomplete"
                ) from commit_error
            raise
