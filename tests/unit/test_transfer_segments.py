import numpy as np

from pptt.pipeline.transfer import (
    contiguous_node_segments,
    mask_discontinuous_reliability,
)


def test_contiguous_segments_preserve_failed_node_breakpoints():
    declared = ("down1", "down2", "down3", "up1", "up2")
    admitted = ("down1", "down2", "up1", "up2")

    assert contiguous_node_segments(declared, admitted) == (
        ("down1", "down2"),
        ("up1", "up2"),
    )


def test_discontinuous_transition_is_never_marked_reliable():
    reliable = np.ones((3, 2, 3, 4), dtype=bool)

    result = mask_discontinuous_reliability(
        reliable,
        declared_nodes=("down1", "down2", "down3", "up1", "up2"),
        traced_nodes=("down1", "down2", "up1", "up2"),
    )

    assert result[0].all()
    assert not result[1].any()
    assert result[2].all()
