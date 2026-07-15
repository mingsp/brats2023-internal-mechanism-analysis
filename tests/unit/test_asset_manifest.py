import csv
import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from pptt.io.manifests import (
    count_npy_pairs,
    file_hash_record,
    regular_files,
    sha256_file,
    verify_sha256_manifest,
    write_assets_json,
    write_sha256_csv,
)
from scripts import verify_assets


def _create_symlink_or_skip(
    test_case: unittest.TestCase,
    link_path: Path,
    target_path: Path,
    *,
    is_directory: bool = False,
) -> None:
    try:
        link_path.symlink_to(target_path, target_is_directory=is_directory)
    except (NotImplementedError, OSError) as error:
        test_case.skipTest(f"symlink creation is unavailable: {error}")


class AssetManifestTests(unittest.TestCase):
    def test_count_npy_pairs_returns_one_for_equal_stems(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "Image"
            mask_dir = root / "Mask"
            image_dir.mkdir()
            mask_dir.mkdir()
            (image_dir / "case_1.npy").write_bytes(b"image")
            (mask_dir / "case_1.npy").write_bytes(b"mask")

            self.assertEqual(count_npy_pairs(image_dir, mask_dir), 1)

    def test_sha256_file_matches_known_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "known.bin"
            path.write_bytes(b"abc")

            digest = sha256_file(path, chunk_size=2)

            self.assertEqual(len(digest), 64)
            self.assertEqual(
                digest,
                "ba7816bf8f01cfea414140de5dae2223"
                "b00361a396177a9cb410ff61f20015ad",
            )

    def test_count_npy_pairs_reports_both_missing_sides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "Image"
            mask_dir = root / "Mask"
            image_dir.mkdir()
            mask_dir.mkdir()
            (image_dir / "image_only.npy").write_bytes(b"image")
            (mask_dir / "mask_only.npy").write_bytes(b"mask")

            with self.assertRaises(ValueError) as context:
                count_npy_pairs(image_dir, mask_dir)

            message = str(context.exception)
            self.assertIn("missing_masks", message)
            self.assertIn("image_only", message)
            self.assertIn("missing_images", message)
            self.assertIn("mask_only", message)

    def test_sha256_file_rejects_invalid_path_and_chunk_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file_path = root / "file.bin"
            file_path.write_bytes(b"content")

            for chunk_size in (0, -1, True, 1.5, "1"):
                with self.subTest(chunk_size=chunk_size):
                    with self.assertRaisesRegex(
                        ValueError, "chunk_size must be positive"
                    ):
                        sha256_file(file_path, chunk_size=chunk_size)
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                sha256_file(root / "missing.bin")
            with self.assertRaisesRegex(IsADirectoryError, "not a file"):
                sha256_file(root)

    def test_count_npy_pairs_rejects_missing_and_non_directory_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask_dir = root / "Mask"
            mask_dir.mkdir()

            with self.assertRaisesRegex(FileNotFoundError, "image_dir does not exist"):
                count_npy_pairs(root / "missing", mask_dir)

            image_file = root / "Image"
            image_file.write_bytes(b"not a directory")
            with self.assertRaisesRegex(
                NotADirectoryError, "image_dir is not a directory"
            ):
                count_npy_pairs(image_file, mask_dir)

            image_file.unlink()
            image_file.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "mask_dir does not exist"):
                count_npy_pairs(image_file, root / "missing-mask")

    def test_file_records_use_sorted_root_relative_posix_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            nested_file = nested / "a.bin"
            root_file = root / "z.bin"
            nested_file.write_bytes(b"a")
            root_file.write_bytes(b"z")

            self.assertEqual(regular_files(root), [nested_file, root_file])
            self.assertEqual(
                file_hash_record(nested_file, root),
                {
                    "relative_path": "nested/a.bin",
                    "size_bytes": 1,
                    "sha256": hashlib.sha256(b"a").hexdigest(),
                },
            )

    def test_verify_sha256_manifest_checks_format_count_and_full_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "raw_3d"
            data_root.mkdir()
            data_file = data_root / "case.bin"
            data_file.write_bytes(b"case")
            manifest_path = root / "raw_3d.sha256"
            manifest_path.write_text(
                f"{hashlib.sha256(b'case').hexdigest()}  case.bin\n",
                encoding="utf-8",
            )

            self.assertEqual(
                verify_sha256_manifest(
                    manifest_path, data_root, verify_full_hashes=True
                ),
                1,
            )

            for chunk_size in (0, -1, True, 1.5, "1"):
                with self.subTest(chunk_size=chunk_size):
                    with self.assertRaisesRegex(
                        ValueError, "chunk_size must be positive"
                    ):
                        verify_sha256_manifest(
                            manifest_path, data_root, chunk_size=chunk_size
                        )

            manifest_path.write_text("not-a-valid-entry\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                verify_sha256_manifest(manifest_path, data_root)

    def test_json_and_csv_outputs_have_deterministic_fields_and_sorting(self):
        digest_a = "a" * 64
        digest_b = "b" * 64
        record_a = {
            "relative_path": "validation_checkpoints/a.pth",
            "size_bytes": 1,
            "sha256": digest_a,
        }
        record_b = {
            "sha256": digest_b,
            "relative_path": "validation_checkpoints/b.pth",
            "size_bytes": 2,
        }
        manifest_a = {
            "validation_checkpoints": [record_b, record_a],
            "schema_version": 1,
            "splits": {
                "val": {"pair_count": 1},
                "train": {"pair_count": 2},
            },
            "raw_3d": {"relative_path": "raw_3d", "file_count": 3},
            "pretrained": [],
            "full_hash_manifests": [],
        }
        manifest_b = {
            "full_hash_manifests": [],
            "pretrained": [],
            "raw_3d": {"file_count": 3, "relative_path": "raw_3d"},
            "splits": {
                "train": {"pair_count": 2},
                "val": {"pair_count": 1},
            },
            "schema_version": 1,
            "validation_checkpoints": [record_a, record_b],
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_a = root / "a" / "assets.json"
            json_b = root / "b" / "assets.json"
            csv_a = root / "a" / "sha256.csv"
            csv_b = root / "b" / "sha256.csv"

            write_assets_json(json_a, manifest_a)
            write_assets_json(json_b, manifest_b)
            write_sha256_csv(csv_a, [record_b, record_a])
            write_sha256_csv(csv_b, [record_a, record_b])

            self.assertEqual(json_a.read_bytes(), json_b.read_bytes())
            self.assertEqual(csv_a.read_bytes(), csv_b.read_bytes())

            payload = json.loads(json_a.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload),
                {
                    "schema_version",
                    "splits",
                    "raw_3d",
                    "validation_checkpoints",
                    "pretrained",
                    "full_hash_manifests",
                },
            )
            self.assertEqual(
                [item["relative_path"] for item in payload["validation_checkpoints"]],
                [
                    "validation_checkpoints/a.pth",
                    "validation_checkpoints/b.pth",
                ],
            )

            with csv_a.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(
                    reader.fieldnames,
                    ["relative_path", "size_bytes", "sha256"],
                )
                self.assertEqual(
                    [row["relative_path"] for row in reader],
                    [
                        "validation_checkpoints/a.pth",
                        "validation_checkpoints/b.pth",
                    ],
                )

    def test_sha256_csv_normalizes_paths_before_duplicate_detection(self):
        canonical_record = {
            "relative_path": "nested/a.bin",
            "size_bytes": 1,
            "sha256": "a" * 64,
        }
        prefixed_record = {**canonical_record, "relative_path": "./nested/a.bin"}

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "sha256.csv"
            write_sha256_csv(output_path, [prefixed_record])

            with output_path.open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["relative_path"], "nested/a.bin")

            with self.assertRaisesRegex(ValueError, "Duplicate relative_path"):
                write_sha256_csv(output_path, [canonical_record, prefixed_record])

    def test_regular_files_and_full_manifest_reject_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "raw_3d"
            data_root.mkdir()
            target_path = root / "target.bin"
            target_path.write_bytes(b"target")
            link_path = data_root / "linked.bin"
            _create_symlink_or_skip(self, link_path, target_path)
            manifest_path = root / "raw_3d.sha256"
            manifest_path.write_text(
                f"{hashlib.sha256(b'target').hexdigest()}  linked.bin\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "links|junctions"):
                regular_files(data_root)
            with self.assertRaisesRegex(ValueError, "links|junctions"):
                verify_sha256_manifest(manifest_path, data_root)


class VerifyAssetsCliTests(unittest.TestCase):
    TEST_PAIR_COUNTS = {"train": 1, "val": 1, "test": 1}

    @staticmethod
    def _write_full_manifest(manifest_path: Path, data_root: Path) -> None:
        lines = []
        for path in regular_files(data_root):
            relative_path = path.relative_to(data_root).as_posix()
            lines.append(f"{sha256_file(path)}  {relative_path}\n")
        manifest_path.write_text("".join(lines), encoding="utf-8")

    def _build_asset_root(self, root: Path) -> Path:
        asset_root = root / "assets"
        processed_root = asset_root / "processed_2d"
        for split in ("train", "val", "test"):
            image_dir = processed_root / f"{split}_Image"
            mask_dir = processed_root / f"{split}_Mask"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            (image_dir / f"{split}_case.npy").write_bytes(b"image")
            (mask_dir / f"{split}_case.npy").write_bytes(b"mask")

        raw_root = asset_root / "raw_3d"
        raw_root.mkdir()
        (raw_root / "case.bin").write_bytes(b"raw")

        checkpoint_root = asset_root / "validation_checkpoints"
        checkpoint_root.mkdir()
        (checkpoint_root / "baseline_seed42_best_val_loss.pth").write_bytes(
            b"baseline"
        )
        (checkpoint_root / "noskip_unet_seed42_best_val_loss.pth").write_bytes(
            b"noskip"
        )

        pretrained_root = asset_root / "pretrained"
        pretrained_root.mkdir()
        (pretrained_root / "encoder.npz").write_bytes(b"pretrained")

        manifests_root = asset_root / "manifests"
        manifests_root.mkdir()
        self._write_full_manifest(
            manifests_root / "processed_2d.sha256", processed_root
        )
        self._write_full_manifest(manifests_root / "raw_3d.sha256", raw_root)
        return asset_root

    def _run_cli(
        self,
        asset_root: Path,
        *extra_arguments: str,
        expected_raw_file_count: int = 1,
    ) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.object(
                verify_assets, "EXPECTED_PAIR_COUNTS", self.TEST_PAIR_COUNTS
            ),
            patch.object(
                verify_assets,
                "EXPECTED_RAW_FILE_COUNT",
                expected_raw_file_count,
                create=True,
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = verify_assets.main(
                ["--asset-root", str(asset_root), *extra_arguments]
            )
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_cli_writes_complete_sorted_outputs_after_full_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            asset_root = self._build_asset_root(Path(directory))
            exit_code, _, _ = self._run_cli(asset_root)

            self.assertEqual(exit_code, 0)
            assets_json = asset_root / "manifests" / "assets.json"
            sha256_csv = asset_root / "manifests" / "sha256.csv"
            payload = json.loads(assets_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["splits"]["train"]["pair_count"], 1)
            self.assertEqual(payload["raw_3d"]["file_count"], 1)
            self.assertEqual(len(payload["validation_checkpoints"]), 2)
            self.assertEqual(len(payload["pretrained"]), 1)
            self.assertEqual(
                [item["entry_count"] for item in payload["full_hash_manifests"]],
                [6, 1],
            )

            with sha256_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["relative_path"] for row in rows],
                sorted(row["relative_path"] for row in rows),
            )
            self.assertEqual(len(rows), 5)

    def test_cli_verification_failure_leaves_outputs_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            asset_root = self._build_asset_root(Path(directory))
            with patch.object(
                verify_assets,
                "EXPECTED_PAIR_COUNTS",
                {"train": 2, "val": 1, "test": 1},
            ):
                stderr = io.StringIO()
                with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                    exit_code = verify_assets.main(
                        ["--asset-root", str(asset_root)]
                    )

            self.assertEqual(exit_code, 1)
            self.assertFalse((asset_root / "manifests" / "assets.json").exists())
            self.assertFalse((asset_root / "manifests" / "sha256.csv").exists())

    def test_cli_defaults_to_full_hash_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            asset_root = self._build_asset_root(Path(directory))
            (asset_root / "raw_3d" / "case.bin").write_bytes(b"tampered")

            exit_code, _, stderr = self._run_cli(asset_root)

            self.assertEqual(exit_code, 1)
            self.assertIn("SHA-256 mismatch", stderr)
            self.assertFalse((asset_root / "manifests" / "assets.json").exists())
            self.assertFalse((asset_root / "manifests" / "sha256.csv").exists())

    def test_cli_allows_explicit_fast_structure_check(self):
        with tempfile.TemporaryDirectory() as directory:
            asset_root = self._build_asset_root(Path(directory))
            (asset_root / "raw_3d" / "case.bin").write_bytes(b"tampered")

            exit_code, _, _ = self._run_cli(
                asset_root, "--no-verify-full-hashes"
            )

            self.assertEqual(exit_code, 0)

    def test_cli_requires_both_full_hash_manifests(self):
        for filename in ("processed_2d.sha256", "raw_3d.sha256"):
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as directory:
                    asset_root = self._build_asset_root(Path(directory))
                    (asset_root / "manifests" / filename).unlink()

                    exit_code, _, stderr = self._run_cli(asset_root)

                    self.assertEqual(exit_code, 1)
                    self.assertIn(filename, stderr)
                    self.assertFalse(
                        (asset_root / "manifests" / "assets.json").exists()
                    )
                    self.assertFalse(
                        (asset_root / "manifests" / "sha256.csv").exists()
                    )

    def test_cli_requires_6255_raw_files(self):
        self.assertEqual(
            getattr(verify_assets, "EXPECTED_RAW_FILE_COUNT", None), 6255
        )
        with tempfile.TemporaryDirectory() as directory:
            asset_root = self._build_asset_root(Path(directory))

            exit_code, _, stderr = self._run_cli(
                asset_root, expected_raw_file_count=6255
            )

            self.assertEqual(exit_code, 1)
            self.assertIn("raw_3d file count mismatch", stderr)

    def test_cli_rejects_checkpoint_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            asset_root = self._build_asset_root(Path(directory))
            checkpoint_path = (
                asset_root
                / "validation_checkpoints"
                / "baseline_seed42_best_val_loss.pth"
            )
            checkpoint_path.unlink()
            _create_symlink_or_skip(
                self, checkpoint_path, asset_root / "raw_3d" / "case.bin"
            )

            exit_code, _, stderr = self._run_cli(asset_root)

            self.assertEqual(exit_code, 1)
            self.assertRegex(stderr, "links|junctions")

    def test_cli_rejects_manifests_directory_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            asset_root = self._build_asset_root(Path(directory))
            manifests_root = asset_root / "manifests"
            real_manifests_root = asset_root / "real_manifests"
            manifests_root.rename(real_manifests_root)
            _create_symlink_or_skip(
                self,
                manifests_root,
                real_manifests_root,
                is_directory=True,
            )

            exit_code, _, stderr = self._run_cli(asset_root)

            self.assertEqual(exit_code, 1)
            self.assertRegex(stderr, "links|junctions")
            self.assertFalse((real_manifests_root / "assets.json").exists())
            self.assertFalse((real_manifests_root / "sha256.csv").exists())

    def test_cli_rolls_back_both_outputs_when_second_replace_fails(self):
        for outputs_exist in (True, False):
            with self.subTest(outputs_exist=outputs_exist):
                with tempfile.TemporaryDirectory() as directory:
                    asset_root = self._build_asset_root(Path(directory))
                    manifests_root = asset_root / "manifests"
                    assets_json = manifests_root / "assets.json"
                    sha256_csv = manifests_root / "sha256.csv"
                    old_assets = b"old assets\n"
                    old_sha256 = b"old sha256\n"
                    if outputs_exist:
                        assets_json.write_bytes(old_assets)
                        sha256_csv.write_bytes(old_sha256)

                    real_replace = os.replace
                    replacement_failed = False

                    def fail_second_final_replace(
                        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
                        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
                    ) -> None:
                        nonlocal replacement_failed
                        if Path(destination) == sha256_csv and not replacement_failed:
                            replacement_failed = True
                            raise OSError("simulated second replacement failure")
                        real_replace(source, destination)

                    with patch(
                        "pptt.io.manifests.os.replace",
                        side_effect=fail_second_final_replace,
                    ):
                        exit_code, _, stderr = self._run_cli(asset_root)

                    self.assertTrue(replacement_failed)
                    self.assertEqual(exit_code, 1)
                    self.assertIn("simulated second replacement failure", stderr)
                    if outputs_exist:
                        self.assertEqual(assets_json.read_bytes(), old_assets)
                        self.assertEqual(sha256_csv.read_bytes(), old_sha256)
                    else:
                        self.assertFalse(assets_json.exists())
                        self.assertFalse(sha256_csv.exists())


if __name__ == "__main__":
    unittest.main()
