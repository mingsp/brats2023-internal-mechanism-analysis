from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pptt.data.brats2d import SliceRecord


_SLICE_INDEX = re.compile(r"[0-9]+")


def patient_id_from_slice(slice_name: str | Path) -> str:
    """Extract a patient identifier using the final underscore separator."""
    filename = Path(slice_name).name
    stem = filename[:-4] if filename.endswith(".npy") else filename
    patient_id, separator, slice_index = stem.rpartition("_")
    if (
        not separator
        or not patient_id
        or _SLICE_INDEX.fullmatch(slice_index) is None
    ):
        raise ValueError(f"Invalid slice name: {slice_name}")
    return patient_id


def patient_ids_from_records(records: Iterable[SliceRecord]) -> set[str]:
    """Return the patient identifiers represented by slice records."""
    return {patient_id_from_slice(record.slice_id) for record in records}


def patient_ids_from_directory(directory: str | Path) -> set[str]:
    """Return patient identifiers from direct lowercase .npy files."""
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"directory is not a directory: {root}")
    return {
        patient_id_from_slice(path.name)
        for path in root.iterdir()
        if path.is_file() and path.suffix == ".npy"
    }


def assert_disjoint_patient_splits(
    splits: Mapping[str, Collection[str]],
) -> None:
    """Raise when any pair of named splits shares a patient identifier."""
    overlap_messages: list[str] = []
    for (left_name, left_ids), (right_name, right_ids) in combinations(
        splits.items(), 2
    ):
        overlap = sorted(set(left_ids) & set(right_ids))
        if overlap:
            overlap_messages.append(
                f"{left_name} vs {right_name}: patients={overlap[:5]} "
                f"(total={len(overlap)})"
            )
    if overlap_messages:
        raise ValueError("Patient split overlap: " + "; ".join(overlap_messages))
