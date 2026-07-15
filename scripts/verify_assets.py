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
    write_assets_json,
    write_sha256_csv,
)


EXPECTED_PAIR_COUNTS = {"train": 56937, "val": 8366, "test": 16118}
EXPECTED_CHECKPOINTS = (
    "baseline_seed42_best_val_loss.pth",
    "noskip_unet_seed42_best_val_loss.pth",
)
FULL_HASH_MANIFESTS = (
    ("processed_2d.sha256", "processed_2d"),
    ("raw_3d.sha256", "raw_3d"),
)


def _require_directory(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    return path


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
        path = checkpoint_root / filename
        if not path.is_file():
            raise ValueError(f"Checkpoint is not a regular file: {filename}")
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
    manifest_root = asset_root / "manifests"
    json_records: list[dict[str, object]] = []
    hash_records: list[FileHashRecord] = []
    for filename, data_directory in FULL_HASH_MANIFESTS:
        manifest_path = manifest_root / filename
        if not manifest_path.exists():
            continue
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
    root = _require_directory(asset_root, "asset_root").resolve()
    splits = _verify_splits(root)
    raw_root = _require_directory(root / "raw_3d", "raw_3d")
    raw_file_count = len(regular_files(raw_root))
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
        action="store_true",
        help="Recompute every entry in existing full SHA-256 manifests.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Verify assets and atomically write deterministic manifests."""
    arguments = _build_parser().parse_args(argv)
    try:
        asset_root, manifest, csv_records = _verify_asset_root(
            arguments.asset_root, arguments.verify_full_hashes
        )
        manifests_root = asset_root / "manifests"
        manifests_root.mkdir(parents=True, exist_ok=True)
        write_assets_json(manifests_root / "assets.json", manifest)
        write_sha256_csv(manifests_root / "sha256.csv", csv_records)
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
