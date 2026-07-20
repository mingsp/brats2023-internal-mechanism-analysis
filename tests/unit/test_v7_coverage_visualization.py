import numpy as np
import pandas as pd

from pptt.visualization.v7_coverage import (
    aggregate_v7_matrices,
    render_v7_coverage_figure,
)


def _cell_summary() -> pd.DataFrame:
    rows = []
    for seed in (42, 123, 3407):
        for transition_index in range(7):
            for node_index in range(7):
                unavailable = transition_index == 0 and node_index == 6
                rows.append(
                    {
                        "model": "unet_baseline",
                        "model_seed": seed,
                        "transition": f"t{transition_index}",
                        "restore_node": f"n{node_index}",
                        "mean_specific_effect": (
                            np.nan
                            if unavailable
                            else 0.01 * (transition_index - node_index)
                        ),
                        "specific_patient_count": 0 if unavailable else 40,
                        "evaluable": not unavailable,
                    }
                )
    return pd.DataFrame(rows)


def test_v7_matrices_preserve_unavailable_cells_instead_of_zero():
    effect, coverage, available = aggregate_v7_matrices(
        _cell_summary(),
        transitions=tuple(f"t{index}" for index in range(7)),
        restore_nodes=tuple(f"n{index}" for index in range(7)),
    )

    assert effect.shape == coverage.shape == available.shape == (7, 7)
    assert np.isnan(effect[0, 6])
    assert coverage[0, 6] == 0
    assert not available[0, 6]
    assert np.isclose(effect[3, 1], 0.02)
    assert coverage[3, 1] == 120


def test_v7_coverage_figure_exposes_locked_boundary(tmp_path):
    manifest = render_v7_coverage_figure(
        output_base=tmp_path / "v7_coverage",
        language="en",
        cell_summary=_cell_summary(),
        transitions=tuple(f"t{index}" for index in range(7)),
        restore_nodes=tuple(f"n{index}" for index in range(7)),
        status="INSUFFICIENT_NETWORK_COVERAGE",
        dpi=300,
    )

    assert manifest["status"] == "INSUFFICIENT_NETWORK_COVERAGE"
    assert manifest["network_wide_claim_authorized"] is False
    assert manifest["matrix_shape"] == [7, 7]
    assert manifest["effect_matrix"][0][6] is None
    assert manifest["unavailable_encoding"] == "gray_NA_not_zero"
