import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from pptt.transitions.fields import TransitionEvent
from pptt.visualization.common import configure_figure_style
from pptt.visualization.flow_curves import render_flow_and_reconstruction
from pptt.visualization.lineage_maps import render_lineage_depth
from pptt.visualization.network_alignment import render_network_alignment_figure
from pptt.visualization.pixel_fate import (
    render_pixel_fate_figure,
    select_median_process_case,
)
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


def _assert_600_dpi_bundle(base: Path) -> dict:
    _assert_publication_pair(base)
    manifest_path = base.with_suffix(".json")
    assert manifest_path.is_file() and manifest_path.stat().st_size > 500
    with Image.open(base.with_suffix(".png")) as image:
        assert min(image.info.get("dpi", (0, 0))) >= 599
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_representative_case_is_nearest_median_not_extreme():
    frame = pd.DataFrame(
        [
            {"patient_id": "extreme-low", "model_seed": 42, "v0": -10.0, "v1": -10.0},
            {"patient_id": "near-low", "model_seed": 42, "v0": -0.1, "v1": -0.1},
            {"patient_id": "median", "model_seed": 42, "v0": 0.0, "v1": 0.0},
            {"patient_id": "near-high", "model_seed": 42, "v0": 0.1, "v1": 0.1},
            {"patient_id": "extreme-high", "model_seed": 42, "v0": 10.0, "v1": 10.0},
        ]
    )

    selected = select_median_process_case(frame, vector_columns=("v0", "v1"))

    assert selected["patient_id"] == "median"


def test_pixel_fate_bundle_uses_one_case_slice_and_all_eight_nodes(tmp_path):
    nodes = ("down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4")
    image = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    truth = np.zeros((32, 32), dtype=np.uint8)
    truth[5:15, 5:15] = 2
    truth[9:13, 9:13] = 1
    truth[11:14, 11:14] = 3
    baseline = np.stack([np.roll(truth, 7 - index, axis=1) for index in range(8)])
    noskip = np.stack([np.roll(truth, max(0, 3 - index), axis=0) for index in range(8)])
    scatter = pd.DataFrame(
        [
            {
                "patient_id": f"case-{index}",
                "model_seed": 42,
                "terminal_dice_difference": (index - 4) / 100,
                "full_path_transition_tv": 0.05 + index / 100,
            }
            for index in range(9)
        ]
    )
    effect_statistics = pd.DataFrame(
        [
            {
                "scope": "fixed_seed_patient_average",
                "model_seed": -1,
                "metric": metric,
                "mean_difference": mean,
                "ci_low": mean - 0.01,
                "ci_high": mean + 0.01,
                "paired_cohens_d": 1.0,
                "holm_p": 0.001,
            }
            for metric, mean in (
                ("middle_damage", 0.04),
                ("late_persistent_net_recovery", 0.18),
                ("terminal_dice", 0.05),
            )
        ]
    )

    for language in ("en", "zh"):
        base = tmp_path / language / "fig2"
        render_pixel_fate_figure(
            output_base=base,
            language=language,
            scatter=scatter,
            patient_id="case-4",
            model_seed=42,
            slice_id="case-4_77",
            image=image,
            truth=truth,
            baseline_states=baseline,
            noskip_states=noskip,
            nodes=nodes,
            process_vector=[0.01] * 7,
            effect_statistics=effect_statistics,
            dpi=600,
        )
        manifest = _assert_600_dpi_bundle(base)
        assert manifest["patient_id"] == "case-4"
        assert manifest["slice_id"] == "case-4_77"
        assert manifest["nodes"] == list(nodes)
        assert manifest["node_count"] == 8
        assert manifest["same_case_and_slice_across_panels"] is True
        assert manifest["scatter"]["point_count"] == 9
        assert manifest["scatter"]["endpoint_close_threshold_pp"] == 2.0
        assert manifest["trajectory"]["later_node_encoding"] == "events_relative_to_previous_node"
        assert set(manifest["trajectory"]["event_totals"]) == {"skip", "no_skip"}
        assert manifest["scatter"]["highlighted"]["terminal_absolute_difference_pp"] == 0.0
        assert manifest["effects"][0]["mean_difference_pp"] == 4.0
        assert manifest["effects"][-1]["metric"] == "late_to_terminal_ratio"


def test_network_alignment_bundle_is_complete_7_by_7_and_preserves_null(tmp_path):
    models = ("unet_baseline", "unet_noskip")
    restore_nodes = ("down2", "down3", "down4", "up1", "up2", "up3", "up4")
    transitions = tuple(f"t{index}" for index in range(7))
    cell_rows = []
    for model in models:
        for seed in (42, 123):
            for transition_index, transition in enumerate(transitions):
                for restore_index, restore_node in enumerate(restore_nodes):
                    unavailable = transition_index == 0 and restore_index == 6
                    cell_rows.append(
                        {
                            "model": model,
                            "model_seed": seed,
                            "transition_index": transition_index,
                            "transition": transition,
                            "restore_node": restore_node,
                            "is_receiving_node_cell": transition_index == restore_index,
                            "evaluable": not unavailable,
                            "mean_specific_effect": (
                                np.nan
                                if unavailable
                                else 0.02 * (transition_index + 1) - 0.01 * restore_index
                            ),
                        }
                    )
    patient_rows = pd.DataFrame(
        [
            {
                "model": model,
                "model_seed": seed,
                "patient_id": f"case-{patient}",
                "endpoint": endpoint,
                "value": 0.04 + 0.005 * patient + (0.01 if endpoint == "micro" else 0),
                "weight": 10,
                "evaluable_transition_count": 6,
            }
            for model in models
            for seed in (42, 123)
            for patient in range(6)
            for endpoint in ("macro", "micro")
        ]
    )
    example_image = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    example_truth = np.zeros((32, 32), dtype=np.uint8)
    example_truth[6:22, 7:24] = 2
    example_truth[10:18, 11:20] = 1
    example_truth[13:17, 14:19] = 3
    example_states = {
        "clean": example_truth,
        "corrupt": np.roll(example_truth, 5, axis=1),
        "restore_target": np.roll(example_truth, 1, axis=1),
        "restore_control": np.roll(example_truth, 4, axis=1),
    }
    example_target_mask = example_truth > 0

    for language in ("en", "zh"):
        base = tmp_path / language / "fig3"
        render_network_alignment_figure(
            output_base=base,
            language=language,
            cell_summary=pd.DataFrame(cell_rows),
            patient_global=patient_rows,
            transitions=transitions,
            restore_nodes=restore_nodes,
            dpi=600,
            bootstrap_iterations=200,
            bootstrap_seed=9,
            example_image=example_image,
            example_truth=example_truth,
            example_states=example_states,
            example_target_mask=example_target_mask,
            example_identity={
                "patient_id": "case-3",
                "model": "unet_baseline",
                "model_seed": 42,
                "slice_id": "case-3_77",
                "transition": "t3",
                "restore_node": "up1",
            },
        )
        manifest = _assert_600_dpi_bundle(base)
        assert manifest["matrix_shape"] == [7, 7]
        assert manifest["dual_axis"] is False
        assert manifest["matrices"]["unet_baseline"][0][6] is None
        assert manifest["matrices"]["unet_baseline"][1][2] == 0.02
        assert len(manifest["matrices"]["unet_noskip"]) == 7
        assert manifest["example"]["patient_id"] == "case-3"
        assert manifest["example"]["slice_id"] == "case-3_77"
        assert manifest["example"]["same_case_and_slice_across_conditions"] is True
