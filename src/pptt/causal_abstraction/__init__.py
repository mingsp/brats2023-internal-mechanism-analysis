"""Cross-architecture causal abstraction for pixel decision processes."""

from pptt.causal_abstraction.interventions import (
    StateExchange,
    apply_flat_feature_edit,
    bilinear_resize_matrix,
    bilinear_resize_rows,
    equal_norm_nullspace_control,
    minimum_norm_state_exchange,
    project_feature_edit,
)
from pptt.causal_abstraction.kernels import (
    TransitionProcess,
    estimate_patient_equal_process,
    evaluate_history_dependence,
    pool_architecture_processes,
)
from pptt.causal_abstraction.matching import (
    MatchingRules,
    match_natural_sources,
)
from pptt.causal_abstraction.metrics import (
    CrossArchitectureTransfer,
    DoseDirectionSummary,
    InterventionSpecificitySummary,
    PairedFidelityDifference,
    PathFidelitySummary,
    STATE_DISTRIBUTION_NAMES,
    StructuralProcessContrast,
    counterfactual_gain,
    cross_architecture_transfer,
    paired_fidelity_difference,
    structural_process_contrast,
    summarize_dose_direction,
    summarize_intervention_specificity,
    summarize_path_fidelity,
    total_variation,
)
from pptt.causal_abstraction.runtime import (
    CounterfactualTrace,
    run_state_exchange,
)
from pptt.causal_abstraction.states import (
    ProcessEvent,
    RelationshipState,
    TRUTH_CLASS_NAMES,
    process_events,
    relationship_states,
)

__all__ = [
    "CounterfactualTrace",
    "CrossArchitectureTransfer",
    "DoseDirectionSummary",
    "InterventionSpecificitySummary",
    "MatchingRules",
    "PairedFidelityDifference",
    "PathFidelitySummary",
    "ProcessEvent",
    "RelationshipState",
    "StateExchange",
    "STATE_DISTRIBUTION_NAMES",
    "StructuralProcessContrast",
    "TRUTH_CLASS_NAMES",
    "TransitionProcess",
    "apply_flat_feature_edit",
    "bilinear_resize_matrix",
    "bilinear_resize_rows",
    "counterfactual_gain",
    "cross_architecture_transfer",
    "equal_norm_nullspace_control",
    "estimate_patient_equal_process",
    "evaluate_history_dependence",
    "minimum_norm_state_exchange",
    "match_natural_sources",
    "paired_fidelity_difference",
    "pool_architecture_processes",
    "process_events",
    "project_feature_edit",
    "relationship_states",
    "run_state_exchange",
    "structural_process_contrast",
    "summarize_dose_direction",
    "summarize_intervention_specificity",
    "summarize_path_fidelity",
    "total_variation",
]
