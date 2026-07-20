import pytest
import torch
from torch import nn

from pptt.interventions.transunet_paths import (
    InterventionPath,
    declared_transunet_decoder_paths,
    resolve_module,
)
from pptt.models.adapters import build_adapter


def test_transunet_decoder_paths_match_real_forward_arguments():
    paths = declared_transunet_decoder_paths()

    assert [path.path_id for path in paths] == [
        "bottleneck_to_up1",
        "skip_down3_to_up1",
        "skip_down2_to_up2",
        "skip_down1_to_up3",
    ]
    assert [(path.receiver_node, path.argument_index) for path in paths] == [
        ("up1", 0),
        ("up1", 1),
        ("up2", 1),
        ("up3", 1),
    ]
    assert [path.argument_name for path in paths] == [None, "skip", "skip", "skip"]
    assert [(path.source_index, path.receiver_index) for path in paths] == [
        (3, 4),
        (2, 4),
        (1, 5),
        (0, 6),
    ]
    assert [path.topology_order for path in paths] == list(range(4))


def test_declared_transunet_paths_resolve_to_unique_modules():
    adapter = build_adapter("transunet_r50_vit_b16", img_size=160)
    source_ids = []
    receiver_arguments = []
    for path in declared_transunet_decoder_paths():
        source_ids.append(id(resolve_module(adapter.model, path.source_module_path)))
        receiver_arguments.append(
            (
                id(resolve_module(adapter.model, path.receiver_module_path)),
                path.argument_index,
            )
        )

    assert len(source_ids) == len(set(source_ids))
    assert len(receiver_arguments) == len(set(receiver_arguments))


def test_resolve_module_supports_module_list_indices():
    root = nn.Module()
    root.blocks = nn.ModuleList([nn.Identity(), nn.ReLU()])

    assert resolve_module(root, "blocks.1") is root.blocks[1]


@pytest.mark.parametrize(
    "updates",
    (
        {"source_index": 4, "receiver_index": 4},
        {"argument_index": -1},
        {"topology_order": -1},
        {"argument_index": 1, "argument_name": None},
    ),
)
def test_intervention_path_rejects_invalid_declarations(updates):
    values = {
        "path_id": "path",
        "source_node": "down3",
        "receiver_node": "up1",
        "source_index": 2,
        "receiver_index": 4,
        "source_module_path": "source",
        "receiver_module_path": "receiver",
        "argument_index": 1,
        "argument_name": "skip",
        "topology_order": 0,
    }
    values.update(updates)

    with pytest.raises(ValueError):
        InterventionPath(**values)
