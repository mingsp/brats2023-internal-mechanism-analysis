from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from pptt.interventions.network_alignment import NodeRestoreMasks
from pptt.interventions.network_runtime import run_network_alignment_forwards
from pptt.models.adapters import HookedModelAdapter
from scripts.run_v7_network_alignment import (
    JobAlreadyRunningError,
    JobRunLock,
    build_matrix_cells,
    run_patient_registry,
)


class EightNodeSegmentationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.names = (
            "down1",
            "down2",
            "down3",
            "down4",
            "up1",
            "up2",
            "up3",
            "up4",
        )
        for name in self.names:
            layer = nn.Conv2d(1, 1, kernel_size=1, bias=False)
            layer.weight.data.fill_(1.0)
            setattr(self, name, layer)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        state = image
        for name in self.names:
            state = getattr(self, name)(state)
        return torch.cat([-state, state], dim=1)


def _adapter() -> HookedModelAdapter:
    model = EightNodeSegmentationNet()
    return HookedModelAdapter(
        model,
        OrderedDict((name, getattr(model, name)) for name in model.names),
    ).eval()


def _masks(nodes: tuple[str, ...]) -> dict[str, dict[int, NodeRestoreMasks]]:
    result: dict[str, dict[int, NodeRestoreMasks]] = {}
    for node in nodes[1:]:
        result[node] = {}
        for transition_index in range(7):
            target = torch.zeros((4, 4), dtype=torch.bool)
            control = torch.zeros((4, 4), dtype=torch.bool)
            target.flatten()[transition_index] = True
            control.flatten()[15 - transition_index] = True
            result[node][transition_index] = NodeRestoreMasks(
                target=target,
                control=control,
                target_count=1,
                control_count=1,
            )
    return result


def test_synthetic_eight_node_adapter_produces_full_registered_matrix():
    nodes = (
        "down1",
        "down2",
        "down3",
        "down4",
        "up1",
        "up2",
        "up3",
        "up4",
    )
    image = torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4)
    registered_masks = _masks(nodes)
    forward_set = run_network_alignment_forwards(
        _adapter(),
        image,
        root_node="down1",
        restore_nodes=nodes[1:],
        masks=registered_masks,
        input_equivalent_shift_yx=(1, 1),
    )
    clean_state = forward_set.clean_logits[0].argmax(axis=0).astype(np.uint8)
    transition_masks = {
        index: np.eye(4, dtype=bool).reshape(-1)[:16].reshape(4, 4)
        for index in range(7)
    }
    for index in range(7):
        transition_masks[index] = np.zeros((4, 4), dtype=bool)
        transition_masks[index].reshape(-1)[index] = True
    metadata = {
        (index, node): {
            "status": "EVALUABLE",
            "target_pixel_count": 1,
            "target_feature_count": 1,
            "control_feature_count": 1,
        }
        for index in range(7)
        for node in nodes[1:]
    }
    metadata[(3, "up1")]["status"] = "INELIGIBLE_TARGET_FEATURE_POSITIONS"

    cells = build_matrix_cells(
        forward_set,
        truth=clean_state,
        transition_masks=transition_masks,
        nodes=nodes,
        cell_metadata=metadata,
    )

    assert len(cells) == 49
    assert {cell["transition_index"] for cell in cells} == set(range(7))
    assert {cell["restore_node"] for cell in cells} == set(nodes[1:])
    assert sum(cell["status"] == "EVALUABLE" for cell in cells) == 48
    unavailable = next(
        cell
        for cell in cells
        if cell["transition_index"] == 3 and cell["restore_node"] == "up1"
    )
    assert unavailable["specific_effect"] is None
    assert unavailable["target_retention"] is None


def _patient_payload(patient_id: str) -> dict:
    return {
        "patient_id": patient_id,
        "status": "PASS",
        "matrix_cells": [{"index": index} for index in range(49)],
    }


def test_resume_is_atomic_and_does_not_duplicate_patient_work(tmp_path: Path):
    job_root = tmp_path / "job"
    calls: list[str] = []

    def process(patient_id: str) -> dict:
        calls.append(patient_id)
        return _patient_payload(patient_id)

    identity = {"model": "synthetic", "model_seed": 42, "protocol": "lock-sha"}
    first = run_patient_registry(
        job_root,
        patient_ids=("case-a", "case-b"),
        job_identity=identity,
        process_patient=process,
        resume=True,
        execution_mode="smoke",
    )
    second = run_patient_registry(
        job_root,
        patient_ids=("case-a", "case-b"),
        job_identity=identity,
        process_patient=process,
        resume=True,
        execution_mode="smoke",
    )

    assert first["patient_count"] == second["patient_count"] == 2
    assert calls == ["case-a", "case-b"]
    assert len(list((job_root / "patient_results").glob("*.json"))) == 2


def test_second_same_identity_runner_is_rejected_by_job_lock(tmp_path: Path):
    lock_path = tmp_path / ".run.lock"

    with JobRunLock(lock_path):
        with pytest.raises(JobAlreadyRunningError, match="JOB_ALREADY_RUNNING"):
            with JobRunLock(lock_path):
                pass
