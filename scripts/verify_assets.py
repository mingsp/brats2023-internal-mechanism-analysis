import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from pptt.io.manifests import (
    FileHashRecord,
    count_npy_pairs,
    file_hash_record,
    regular_files,
    verify_sha256_manifest,
    write_manifest_outputs,
)


EXPECTED_PAIR_COUNTS = {"train": 56937, "val": 8366, "test": 16118}
EXPECTED_RAW_FILE_COUNT = 6255
EXPECTED_CHECKPOINTS = (
    "baseline_seed42_best_val_loss.pth",
    "noskip_unet_seed42_best_val_loss.pth",
)
FULL_HASH_MANIFESTS = (
    ("processed_2d.sha256", "processed_2d"),
    ("raw_3d.sha256", "raw_3d"),
)


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
    _reject_link_or_junction(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    return path


def _require_file(path: Path, label: str) -> Path:
    _reject_link_or_junction(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"{label} is not a file: {path}")
    return path


def _require_asset_subdirectory(
    asset_root: Path, directory_name: str, label: str
) -> Path:
    expected_path = asset_root / directory_name
    directory = _require_directory(expected_path, label)
    resolved_directory = directory.resolve(strict=True)
    if resolved_directory != expected_path:
        raise ValueError(
            f"{label} resolves outside its expected path: {directory}"
        )
    return resolved_directory


def _relative_path(path: Path, asset_root: Path) -> str:
    return path.resolve().relative_to(asset_root.resolve()).as_posix()


def _verify_splits(asset_root: Path) -> dict[str, dict[str, object]]:
    processed_root = _require_directory(
        asset_root / "processed_2d", "processed_2d"
    )
    splits: dict[str, dict[str, object]] = {}
    for split, expected_count in EXPECTED_PAIR_COUNTS.items():
        image_dir = processed_root / f"{split}_Image"
        mask_dir = processed_root / f"{split}_Mask"
        pair_count = count_npy_pairs(image_dir, mask_dir)
        if pair_count != expected_count:
            raise ValueError(
                f"{split} pair count mismatch: "
                f"expected={expected_count}, actual={pair_count}"
            )
        splits[split] = {
            "image_dir": _relative_path(image_dir, asset_root),
            "mask_dir": _relative_path(mask_dir, asset_root),
            "pair_count": pair_count,
        }
    return splits


def _verify_checkpoints(asset_root: Path) -> list[FileHashRecord]:
    checkpoint_root = _require_directory(
        asset_root / "validation_checkpoints", "validation_checkpoints"
    )
    regular_files(checkpoint_root)
    actual_names = {path.name for path in checkpoint_root.iterdir()}
    expected_names = set(EXPECTED_CHECKPOINTS)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ValueError(
            "validation_checkpoints contents mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    records: list[FileHashRecord] = []
    for filename in sorted(EXPECTED_CHECKPOINTS):
        expected_path = checkpoint_root / filename
        path = _require_file(expected_path, "Checkpoint")
        if path.resolve(strict=True) != expected_path:
            raise ValueError(
                f"Checkpoint resolves outside its expected path: {filename}"
            )
        if path.stat().st_size == 0:
            raise ValueError(f"Checkpoint is empty: {filename}")
        records.append(file_hash_record(path, asset_root))
    return records


def _verify_pretrained(asset_root: Path) -> list[FileHashRecord]:
    pretrained_root = _require_directory(asset_root / "pretrained", "pretrained")
    return [file_hash_record(path, asset_root) for path in regular_files(pretrained_root)]


def _verify_full_manifests(
    asset_root: Path, verify_full_hashes: bool
) -> tuple[list[dict[str, object]], list[FileHashRecord]]:
    manifest_root = _require_asset_subdirectory(
        asset_root, "manifests", "manifests"
    )
    json_records: list[dict[str, object]] = []
    hash_records: list[FileHashRecord] = []
    for filename, data_directory in FULL_HASH_MANIFESTS:
        manifest_path = _require_file(
            manifest_root / filename, "Full SHA-256 manifest"
        )
        entry_count = verify_sha256_manifest(
            manifest_path,
            asset_root / data_directory,
            verify_full_hashes=verify_full_hashes,
        )
        hash_record = file_hash_record(manifest_path, asset_root)
        json_records.append({**hash_record, "entry_count": entry_count})
        hash_records.append(hash_record)
    return json_records, hash_records


def _verify_asset_root(
    asset_root: Path, verify_full_hashes: bool
) -> tuple[Path, dict[str, object], list[FileHashRecord]]:
    root = _require_directory(asset_root, "asset_root").resolve(strict=True)
    required_directories = (
        ("manifests", "manifests"),
        ("processed_2d", "processed_2d"),
        ("raw_3d", "raw_3d"),
        ("validation_checkpoints", "validation_checkpoints"),
        ("pretrained", "pretrained"),
    )
    for directory_name, label in required_directories:
        _require_asset_subdirectory(root, directory_name, label)
    manifest_root = root / "manifests"
    for filename, _ in FULL_HASH_MANIFESTS:
        _require_file(
            manifest_root / filename, "Full SHA-256 manifest"
        )
    regular_files(root)

    splits = _verify_splits(root)
    raw_root = _require_directory(root / "raw_3d", "raw_3d")
    raw_file_count = len(regular_files(raw_root))
    if raw_file_count != EXPECTED_RAW_FILE_COUNT:
        raise ValueError(
            "raw_3d file count mismatch: "
            f"expected={EXPECTED_RAW_FILE_COUNT}, actual={raw_file_count}"
        )
    checkpoint_records = _verify_checkpoints(root)
    pretrained_records = _verify_pretrained(root)
    full_manifest_records, full_manifest_hashes = _verify_full_manifests(
        root, verify_full_hashes
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "splits": splits,
        "raw_3d": {
            "relative_path": _relative_path(raw_root, root),
            "file_count": raw_file_count,
        },
        "validation_checkpoints": checkpoint_records,
        "pretrained": pretrained_records,
        "full_hash_manifests": full_manifest_records,
    }
    csv_records = checkpoint_records + pretrained_records + full_manifest_hashes
    return root, manifest, csv_records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify immutable PPTT assets.")
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument(
        "--verify-full-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Recompute every file listed in both full SHA-256 manifests "
            "(default: enabled); use --no-verify-full-hashes for a fast "
            "format, path, and count-only check."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Verify assets and atomically write deterministic manifests."""
    arguments = _build_parser().parse_args(argv)
    try:
        asset_root, manifest, csv_records = _verify_asset_root(
            arguments.asset_root, arguments.verify_full_hashes
        )
        manifests_root = _require_asset_subdirectory(
            asset_root, "manifests", "manifests"
        )
        write_manifest_outputs(manifests_root, manifest, csv_records)
    except (OSError, ValueError) as error:
        print(f"Asset verification failed: {error}", file=sys.stderr)
        return 1

    split_counts = manifest["splits"]
    if not isinstance(split_counts, Mapping):
        raise RuntimeError("Invalid internal split summary")
    print(f"Asset verification passed: {asset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
