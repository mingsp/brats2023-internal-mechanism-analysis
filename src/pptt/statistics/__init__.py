"""Patient-level statistical analysis."""

from pptt.statistics.network_alignment import (
    compute_global_alignment_statistics,
    evaluate_network_alignment_gate,
    summarize_alignment_cells,
)

__all__ = [
    "compute_global_alignment_statistics",
    "evaluate_network_alignment_gate",
    "summarize_alignment_cells",
]
