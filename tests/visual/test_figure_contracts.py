from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from pptt.transitions.fields import TransitionEvent
from pptt.visualization.common import configure_figure_style
from pptt.visualization.flow_curves import render_flow_and_reconstruction
from pptt.visualization.lineage_maps import render_lineage_depth
from pptt.visualization.transition_maps import render_transition_map


def _assert_publication_pair(base: Path) -> None:
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    assert png.is_file() and png.stat().st_size > 5000
    assert pdf.is_file() and pdf.stat().st_size > 1000
    with Image.open(png) as image:
        dpi = image.info.get("dpi", (0, 0))
        assert min(dpi) >= 299
        pixels = np.asarray(image.convert("RGB"))
        assert pixels.shape[0] >= 900 and pixels.shape[1] >= 900
        assert pixels.std() > 5


def test_transition_map_is_single_transition_and_publication_resolution(tmp_path):
    configure_figure_style("en")
    events = np.full((48, 48), int(TransitionEvent.STABLE_CORRECT), dtype=np.uint8)
    events[8:20, 8:20] = int(TransitionEvent.CORRECTION)
    events[22:32, 12:24] = int(TransitionEvent.DESTRUCTION)
    events[12:24, 30:40] = int(TransitionEvent.WRONG_REENCODING)
    events[34:42, 30:42] = int(TransitionEvent.UNCERTAIN)
    base = tmp_path / "transition"

    render_transition_map(
        events,
        base,
        transition_label="up3 to up4",
        language="en",
    )

    _assert_publication_pair(base)


def test_flow_and_lineage_figures_meet_output_contract(tmp_path):
    configure_figure_style("en")
    nodes = ("down1", "down2", "up1", "up2")
    node_rows = []
    transition_rows = []
    for model, offset in (("unet_baseline", 0.0), ("unet_noskip", 0.05)):
        for index, node in enumerate(nodes):
            node_rows.append(
                {
                    "model": model,
                    "node": node,
                    "class_index": -1,
                    "dice": 0.4 + 0.12 * index + offset,
                }
            )
        for index, (left, right) in enumerate(zip(nodes[:-1], nodes[1:], strict=True)):
            transition_rows.append(
                {
                    "model": model,
                    "transition": f"{left}->{right}",
                    "persistent_net_rate": 0.01 + 0.02 * index - offset / 5,
                }
            )
    flow_base = tmp_path / "flow"
    render_flow_and_reconstruction(
        pd.DataFrame(node_rows),
        pd.DataFrame(transition_rows),
        flow_base,
        nodes=nodes,
        language="en",
    )
    _assert_publication_pair(flow_base)

    lineage = pd.DataFrame(
        [
            {
                "kind": "final_state_depth",
                "class_index": class_index,
                "depth": depth,
                "count": (class_index + 1) * (depth + 2),
            }
            for class_index in range(4)
            for depth in range(-1, len(nodes) + 1)
        ]
    )
    lineage_base = tmp_path / "lineage"
    render_lineage_depth(
        lineage,
        lineage_base,
        kind="final_state_depth",
        nodes=nodes,
        language="en",
    )
    _assert_publication_pair(lineage_base)
