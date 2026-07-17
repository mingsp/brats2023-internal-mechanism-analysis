from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from pptt.io.artifacts import load_case_trace, save_case_trace


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _has_margins(path: Path) -> bool:
    with np.load(path, allow_pickle=False) as archive:
        return bool(archive["has_margins"].item()) and "margins" in archive.files


def compact_trace(path: Path) -> tuple[int, int] | None:
    if path.name.startswith(".") or not _has_margins(path):
        return None
    before = path.stat().st_size
    trace = load_case_trace(path)
    save_case_trace(
        path,
        states=trace.states,
        reliable=trace.reliable,
        margins=None,
        truth=trace.truth,
        final_model_state=trace.final_model_state,
        slice_ids=trace.slice_ids,
    )
    return before, path.stat().st_size


def compact_roots(roots: tuple[Path, ...]) -> tuple[int, int, int, int]:
    compacted = 0
    failures = 0
    before_bytes = 0
    after_bytes = 0
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.npz")):
            try:
                result = compact_trace(path)
            except (OSError, ValueError, KeyError) as error:
                failures += 1
                print(f"retry {path}: {error}", flush=True)
                continue
            if result is None:
                continue
            before, after = result
            compacted += 1
            before_bytes += before
            after_bytes += after
            print(
                f"compacted {path} {before / 2**20:.2f} MiB -> "
                f"{after / 2**20:.2f} MiB",
                flush=True,
            )
    return compacted, failures, before_bytes, after_bytes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Atomically remove unused margin tensors from completed case traces."
    )
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--watch-pids", nargs="*", type=int, default=[])
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    roots = tuple(path.resolve() for path in args.roots)
    stable_scans = 0
    while True:
        compacted, failures, before, after = compact_roots(roots)
        if compacted:
            print(
                f"scan compacted={compacted} failures={failures} "
                f"saved={(before - after) / 2**20:.2f} MiB",
                flush=True,
            )
        running = any(_process_exists(pid) for pid in args.watch_pids)
        stable_scans = stable_scans + 1 if compacted == 0 and failures == 0 else 0
        if not running and stable_scans >= 2:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
