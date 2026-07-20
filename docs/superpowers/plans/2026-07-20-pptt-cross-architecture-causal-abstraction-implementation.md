# PPTT Cross-Architecture Causal Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and formally validate whether one low-capacity pixel-decision state-transition model predicts full downstream intervention trajectories in frozen U-Net and TransUNet models on BraTS2023, while retaining no-skip U-Net only as a whole-path structural control.

**Architecture:** Implement one continuous method path: architecture-specific feature tensors are mapped to a common pixel-decision state trajectory; a closed-form minimum-norm operator translates a high-level state intervention back to the selected node; and a full-path evaluator compares the high-level counterfactual rollout with every downstream network state. Reuse the existing model adapters, frozen linear observers, BraTS patient manifests, atomic result patterns, and nine frozen model checkpoints. Validation fitting and calibration are locked before formal test interventions, and every result is reduced at patient level before architecture-level inference.

**Tech Stack:** Python 3.10, PyTorch 2.1, NumPy 1.26, SciPy 1.11, pandas 2.2, PyArrow 17, PyYAML, pytest, Hypothesis, existing PPTT adapters and observer artifacts.

---

## Authoritative Execution Spine

Only the following three objects constitute the method:

1. `alpha`: the architecture-specific readout that maps each registered feature tensor to one common five-state pixel-decision representation;
2. `omega`: the closed-form minimum-norm translator from `do(S_r=q)` to a reversible feature edit at node `r`;
3. `H`: the low-capacity state-transition process whose complete downstream rollout is compared with the intervened network trajectory.

The implementation is accepted only through three nested validation questions:

1. **Within-architecture faithfulness:** does `H_U` or `H_T` predict the complete downstream intervention trajectory of its source architecture?
2. **Cross-architecture abstraction:** do `H_U -> TransUNet`, `H_T -> U-Net`, and `H_shared` pass the locked full-path conditions across all eight nodes?
3. **Structural diagnostic sensitivity:** does the baseline/no-skip process distance exceed baseline seed variability when the identical method is applied end to end?

Natural matching, observer reliability, intervention OOD checks, dose checks, nullspace edits, randomized observers, and depth permutation are safeguards against alternative explanations. They must remain subordinate audit fields and must not be presented as additional method branches. The no-skip model is a structural control; no per-skip masking or skip-importance ranking is authorized.

### Implementation Status

| Method role | Implementation | Status |
|---|---|---|
| `alpha` state semantics | `states.py`, `kernels.py` | implemented and unit tested |
| `omega` intervention translator | `interventions.py` | implemented and theorem tested |
| full downstream execution | `runtime.py` | implemented and integration tested |
| auxiliary natural-state support | `matching.py` | implemented; audit role only |
| intervention-alignment inference | `metrics.py`, `gates.py` | next implementation block |
| immutable formal execution | V11 lock, runner, summarizer | pending after inference tests |

---

## File Structure

**Create**

- `src/pptt/causal_abstraction/__init__.py`: public causal-abstraction API.
- `src/pptt/causal_abstraction/states.py`: five relationship states, truth-class strata, and process-event mapping.
- `src/pptt/causal_abstraction/kernels.py`: patient-equal first-order kernels, rollout, shared kernels, cross-transfer, and history-dependence admission.
- `src/pptt/causal_abstraction/interventions.py`: bilinear resize matrix, closed-form minimum-norm edit, nullspace control, and operator audit.
- `src/pptt/causal_abstraction/runtime.py`: model-hook execution and complete downstream counterfactual tracing.
- `src/pptt/causal_abstraction/matching.py`: auxiliary natural source/base matching and patient caps; not a method output.
- `src/pptt/causal_abstraction/metrics.py`: one patient-level intervention-alignment criterion, its cross-architecture transfer form, and the secondary structural sensitivity contrast.
- `src/pptt/causal_abstraction/gates.py`: fixed three-state gate system and claim decision table.
- `configs/experiments/v11_causal_abstraction.yaml`: the only V11 numerical protocol.
- `scripts/lock_v11_causal_abstraction_protocol.py`: immutable asset, model, observer, patient, kernel, and configuration lock.
- `scripts/run_v11_causal_abstraction.py`: resumable per-model/per-seed formal runner.
- `scripts/summarize_v11_causal_abstraction.py`: patient-first statistics and final gate.
- `scripts/monitor_v11_causal_abstraction.py`: read-only progress, GPU, disk, failure, and dynamic ETA display.
- `scripts/server/run_v11_causal_abstraction.sh`: safe scheduler for the nine frozen model jobs.
- `tests/unit/test_causal_states.py`
- `tests/unit/test_causal_kernels.py`
- `tests/unit/test_minimum_norm_state_exchange.py`
- `tests/unit/test_causal_source_matching.py`
- `tests/unit/test_causal_abstraction_metrics.py`
- `tests/unit/test_causal_abstraction_gate.py`
- `tests/unit/test_v11_protocol_lock.py`
- `tests/integration/test_causal_abstraction_runtime.py`
- `tests/integration/test_v11_smoke.py`

**Modify**

- `src/pptt/models/protocol.py`: add a reversible checkpoint-transform trace contract without changing existing adapter behavior.
- `src/pptt/models/adapters.py`: expose the existing ordered checkpoint roles through the new protocol.
- `docs/reproducibility_cn.md`: append V11 commands, formal-output contract, and conclusion boundary after implementation passes.
- `docs/final_experiment_audit_cn.md`: append V11 status only after formal server audit.

## Task 1: Baseline and State Semantics

**Files:**
- Create: `src/pptt/causal_abstraction/__init__.py`
- Create: `src/pptt/causal_abstraction/states.py`
- Test: `tests/unit/test_causal_states.py`

- [ ] **Step 1: Record the clean baseline**

Run:

```powershell
python -m pytest -q
```

Expected: the existing suite passes before V11 files are added; platform-specific junction skips may remain skipped.

- [ ] **Step 2: Write failing state tests**

The tests must exercise all valid truth/prediction relations and reject shape or label mismatches:

```python
def test_relationship_states_preserve_pixel_identity_and_error_type():
    truth = np.array([[0, 1, 2, 3, 2]], dtype=np.uint8)
    pred = np.array([[0, 1, 0, 2, 3]], dtype=np.uint8)
    states = relationship_states(truth, pred, num_classes=4)
    np.testing.assert_array_equal(states, [[BC, FC, FN, FW, FW]])


def test_process_events_are_determined_by_adjacent_states():
    before = np.array([FN, FC, FN, FP, BC], dtype=np.uint8)
    after = np.array([FC, FN, FW, BC, BC], dtype=np.uint8)
    np.testing.assert_array_equal(
        process_events(before, after),
        [CORRECTION, DESTRUCTION, ERROR_RECODING, CORRECTION, PERSISTENCE],
    )
```

- [ ] **Step 3: Run the new tests and verify failure**

Run:

```powershell
python -m pytest tests/unit/test_causal_states.py -q
```

Expected: collection fails because `pptt.causal_abstraction.states` does not exist.

- [ ] **Step 4: Implement the state API**

Use integer enums with stable serialized values:

```python
class RelationshipState(IntEnum):
    BC = 0
    FC = 1
    FP = 2
    FN = 3
    FW = 4


class ProcessEvent(IntEnum):
    PERSISTENCE = 0
    CORRECTION = 1
    DESTRUCTION = 2
    ERROR_RECODING = 3


def relationship_states(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    """Return one architecture-independent relation state per pixel."""


def process_events(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Map adjacent relationship states to one exhaustive process event."""
```

Also expose `TRUTH_CLASS_NAMES = {0: "BG", 1: "NCR_NET", 2: "ED", 3: "ET"}` and reject labels outside `[0, num_classes)`.

- [ ] **Step 5: Run focused and package tests**

Run:

```powershell
python -m pytest tests/unit/test_causal_states.py tests/unit/test_package_import.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/pptt/causal_abstraction tests/unit/test_causal_states.py
git commit -m "feat: define pixel decision process states"
```

## Task 2: Patient-Equal High-Level Kernels

**Files:**
- Create: `src/pptt/causal_abstraction/kernels.py`
- Test: `tests/unit/test_causal_kernels.py`

- [ ] **Step 1: Write failing kernel tests**

Cover patient equality, row stochasticity, rollout, architecture-equal pooling, bidirectional transfer, and first-order misspecification:

```python
def test_patient_equal_kernel_is_not_pixel_count_weighted():
    rows = toy_rows(one_large_patient=True)
    kernel = estimate_patient_equal_kernel(rows, state_count=5, alpha=0.5)
    assert kernel.shape == (5, 5)
    np.testing.assert_allclose(kernel.sum(axis=1), 1.0)
    assert kernel[3, 1] == pytest.approx(expected_patient_equal_fn_to_fc)


def test_shared_kernel_weights_architectures_equally():
    shared = pool_architecture_kernels({"unet": unet_kernel, "transunet": trans_kernel})
    np.testing.assert_allclose(shared, 0.5 * (unet_kernel + trans_kernel))


def test_history_admission_rejects_material_second_order_gain():
    result = evaluate_history_dependence(first_order_rows(), second_order_rows(), tolerance=0.03)
    assert result.status == "HIGH_LEVEL_MODEL_MISSPECIFIED"
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
python -m pytest tests/unit/test_causal_kernels.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement immutable model objects**

```python
@dataclass(frozen=True)
class TransitionProcess:
    node_names: tuple[str, ...]
    kernels: np.ndarray  # shape: 8 x 5 x 5; final kernel maps D4 to Y
    alpha: float
    source: str

    def rollout(self, *, intervention_node: int, source_state: int) -> np.ndarray:
        """Return downstream state distributions through Y."""


def estimate_patient_equal_process(
    rows: pd.DataFrame,
    *,
    node_names: tuple[str, ...],
    state_count: int = 5,
    alpha: float = 0.5,
) -> TransitionProcess:
    """Average source-state-normalized counts within patient, seed, then architecture."""


def pool_architecture_processes(
    processes: Mapping[str, TransitionProcess],
) -> TransitionProcess:
    """Return an architecture-equal process without architecture covariates."""
```

Implement five-fold patient splits with a fixed seed. The second-order model exists only inside `evaluate_history_dependence`; it is never serialized as the formal model.

- [ ] **Step 4: Run tests**

Run:

```powershell
python -m pytest tests/unit/test_causal_kernels.py tests/property/test_transition_properties.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/pptt/causal_abstraction/kernels.py tests/unit/test_causal_kernels.py
git commit -m "feat: add patient-equal causal process kernels"
```

## Task 3: Closed-Form Minimum-Norm State Exchange

**Files:**
- Create: `src/pptt/causal_abstraction/interventions.py`
- Test: `tests/unit/test_minimum_norm_state_exchange.py`

- [ ] **Step 1: Write theorem-oriented failing tests**

```python
def test_bilinear_matrix_matches_torch_align_corners_false():
    source = torch.arange(12, dtype=torch.float64).reshape(1, 1, 3, 4)
    matrix = bilinear_resize_matrix((3, 4), (5, 7), dtype=torch.float64)
    expected = F.interpolate(source, (5, 7), mode="bilinear", align_corners=False)
    actual = (matrix @ source.flatten().double()).reshape(1, 1, 5, 7)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


def test_exchange_is_feasible_and_minimum_norm():
    edit = minimum_norm_state_exchange(R, W, delta_logits, rcond=1e-10)
    torch.testing.assert_close(R @ edit.delta_h @ W.T, delta_logits, atol=1e-8, rtol=0)
    assert edit.frobenius_norm <= torch.linalg.vector_norm(edit.delta_h + null_vector) + 1e-9


def test_equal_norm_nullspace_control_preserves_observer_logits():
    control = equal_norm_nullspace_control(W, edit.delta_h, generator=generator)
    torch.testing.assert_close(control @ W.T, torch.zeros_like(delta_logits_native), atol=1e-7, rtol=0)
    assert torch.linalg.vector_norm(control) == pytest.approx(edit.frobenius_norm)
```

- [ ] **Step 2: Verify failure**

Run:

```powershell
python -m pytest tests/unit/test_minimum_norm_state_exchange.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement the closed-form operator**

```python
@dataclass(frozen=True)
class StateExchange:
    delta_h: torch.Tensor
    reconstructed_delta_logits: torch.Tensor
    target_max_abs_error: float
    frobenius_norm: float
    effective_rank_spatial: int
    effective_rank_channel: int


def minimum_norm_state_exchange(
    resize_rows: torch.Tensor,
    observer_weight: torch.Tensor,
    delta_logits: torch.Tensor,
    *,
    rcond: float,
) -> StateExchange:
    left = torch.linalg.pinv(resize_rows, rtol=rcond)
    right = torch.linalg.pinv(observer_weight, rtol=rcond).T
    delta_h = left @ delta_logits @ right
    reconstructed = resize_rows @ delta_h @ observer_weight.T
    residual = reconstructed - delta_logits
    return StateExchange(
        delta_h=delta_h,
        reconstructed_delta_logits=reconstructed,
        target_max_abs_error=float(residual.abs().max().item()),
        frobenius_norm=float(torch.linalg.vector_norm(delta_h).item()),
        effective_rank_spatial=int(torch.linalg.matrix_rank(resize_rows).item()),
        effective_rank_channel=int(torch.linalg.matrix_rank(observer_weight).item()),
    )
```

`bilinear_resize_matrix` must use the exact half-pixel coordinate rule used by PyTorch with `align_corners=False`. Reject non-finite inputs, empty target sets, zero-rank observers, and nullspaces with no available dimension.

- [ ] **Step 4: Run unit and property tests**

Run:

```powershell
python -m pytest tests/unit/test_minimum_norm_state_exchange.py tests/unit/test_linear_observer.py -q
```

Expected: PASS on CPU float64 theorem tests and float32 tolerance tests.

- [ ] **Step 5: Commit**

```powershell
git add src/pptt/causal_abstraction/interventions.py tests/unit/test_minimum_norm_state_exchange.py
git commit -m "feat: add closed-form state exchange operator"
```

## Task 4: Reversible Full-Path Runtime

**Files:**
- Modify: `src/pptt/models/protocol.py`
- Modify: `src/pptt/models/adapters.py`
- Create: `src/pptt/causal_abstraction/runtime.py`
- Test: `tests/integration/test_causal_abstraction_runtime.py`

- [ ] **Step 1: Write failing runtime tests**

Use a tiny deterministic U-Net input and a synthetic observer. Verify all downstream activations are captured after the edit, upstream activations remain clean, dose zero exactly restores clean output, hooks are removed after success and failure, and no model parameters change.

```python
def test_state_exchange_captures_only_registered_downstream_path():
    result = run_state_exchange(
        adapter,
        image,
        node="down3",
        delta_h=delta,
        doses=(0.0, 0.5, 1.0),
        observers=observers,
    )
    assert tuple(result.downstream_states) == ("down3", "down4", "up1", "up2", "up3", "up4", "Y")
    np.testing.assert_allclose(result.final_logits[0.0], clean_logits, atol=1e-6)
    assert parameter_hash(adapter) == before_hash
    assert hook_count(adapter) == 0
```

- [ ] **Step 2: Verify failure**

Run:

```powershell
python -m pytest tests/integration/test_causal_abstraction_runtime.py -q
```

Expected: missing runtime API.

- [ ] **Step 3: Add the reversible protocol contract**

Extend `ModelAdapter` with a concrete context-managed default based on `checkpoint_module`, preserving existing subclasses:

```python
@contextmanager
def transform_checkpoint_output(
    self,
    name: str,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> Iterator[None]:
    def hook(
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(output, torch.Tensor):
            raise TypeError("checkpoint output must be a tensor")
        changed = transform(output)
        if changed.shape != output.shape:
            raise ValueError("checkpoint transform changed tensor shape")
        return changed

    handle = self.checkpoint_module(name).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
```

- [ ] **Step 4: Implement batched dose execution**

```python
@dataclass(frozen=True)
class CounterfactualTrace:
    node: str
    doses: tuple[float, ...]
    downstream_states: Mapping[float, Mapping[str, np.ndarray]]
    final_logits: Mapping[float, np.ndarray]
    audits: Mapping[str, object]


def run_state_exchange(
    adapter: ModelAdapter,
    image: torch.Tensor,
    *,
    node: str,
    delta_h: torch.Tensor,
    doses: Sequence[float],
    observers: Mapping[str, Sequence[LinearObserver]],
    truth: np.ndarray,
    reliability_threshold: float,
) -> CounterfactualTrace:
    """Apply one registered node edit at all doses and read every downstream state."""
```

Batch dose conditions only when their combined peak memory was preflighted; batching changes execution only, not patient membership or statistics.

- [ ] **Step 5: Run adapter and runtime tests**

Run:

```powershell
python -m pytest tests/unit/test_model_protocol.py tests/integration/test_activation_restore.py tests/integration/test_causal_abstraction_runtime.py -q
```

Expected: PASS without changing old adapter outputs.

- [ ] **Step 6: Commit**

```powershell
git add src/pptt/models/protocol.py src/pptt/models/adapters.py src/pptt/causal_abstraction/runtime.py tests/integration/test_causal_abstraction_runtime.py
git commit -m "feat: trace reversible full-path interventions"
```

## Task 5: Auxiliary Validity Guard - Deterministic Natural Source Matching

**Files:**
- Create: `src/pptt/causal_abstraction/matching.py`
- Test: `tests/unit/test_causal_source_matching.py`

- [ ] **Step 1: Write failing matching tests**

```python
def test_matching_preserves_truth_class_and_changes_relation_state():
    matches = match_natural_sources(base, source_bank, rules)
    assert (matches.base_truth_class == matches.source_truth_class).all()
    assert (matches.base_state != matches.source_state).all()
    assert matches.groupby(["patient_id", "node", "source_state"]).size().max() <= 128


def test_matching_is_order_invariant_and_hash_stable():
    left = match_natural_sources(base, source_bank.sample(frac=1, random_state=3), rules)
    right = match_natural_sources(base, source_bank.sample(frac=1, random_state=9), rules)
    pd.testing.assert_frame_equal(left, right)
```

- [ ] **Step 2: Verify failure**

Run:

```powershell
python -m pytest tests/unit/test_causal_source_matching.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement locked matching**

```python
@dataclass(frozen=True)
class MatchingRules:
    same_patient_first: bool
    max_pixels_per_patient_node_state: int
    boundary_edges: tuple[float, ...]
    feature_norm_quantile_bins: int
    seed: int


def match_natural_sources(
    base: pd.DataFrame,
    source_bank: pd.DataFrame,
    rules: MatchingRules,
) -> pd.DataFrame:
    """Return deterministic one-to-one source/base pairs without replacement."""
```

Sort candidates by same-patient indicator, truth class, boundary stratum, norm stratum, deterministic hash, and stable row identity. Emit explicit `NO_MATCH` records rather than dropping conditions.

- [ ] **Step 4: Run matching tests**

Run:

```powershell
python -m pytest tests/unit/test_causal_source_matching.py tests/unit/test_matching.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/pptt/causal_abstraction/matching.py tests/unit/test_causal_source_matching.py
git commit -m "feat: add deterministic natural state matching"
```

## Task 6: Unified Full-Path Intervention Alignment

**Files:**
- Create: `src/pptt/causal_abstraction/metrics.py`
- Test: `tests/unit/test_causal_abstraction_metrics.py`

- [ ] **Step 1: Write failing metric tests**

```python
def test_worst_condition_cannot_be_hidden_by_macro_mean():
    rows = mostly_good_rows_with_one_failed_node_class()
    summary = summarize_path_fidelity(rows)
    assert summary.macro_tv < 0.15
    assert summary.worst_tv > 0.20


def test_cross_transfer_uses_source_architecture_process_only():
    result = cross_architecture_transfer(unet_process, transunet_interventions)
    assert result.process_source == "unet"
    assert result.network_target == "transunet"


def test_structural_distance_is_patient_equal_and_exceeds_seed_null():
    result = structural_process_contrast(baseline_rows, noskip_rows, seed_null_rows)
    assert result.delta_distance == pytest.approx(result.between - result.within)
```

- [ ] **Step 2: Verify failure**

Run:

```powershell
python -m pytest tests/unit/test_causal_abstraction_metrics.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement patient-first metrics**

```python
def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.abs(left - right).sum())


def counterfactual_gain(clean: np.ndarray, edited: np.ndarray, target: np.ndarray) -> float:
    return total_variation(clean, target) - total_variation(edited, target)


def summarize_path_fidelity(
    patient_rows: pd.DataFrame,
    *,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return condition rows and worst-node/class/depth rows after patient aggregation."""
```

Implement class macro TV, worst class/node/state/depth TV, bidirectional transfer, shared-vs-specific TV gap, task-minus-null gain, null-vs-clean TV, dose direction, and `D_proc`/`Delta D_struct`. Use Holm correction only across the locked family of node hypotheses.

- [ ] **Step 4: Run statistics regression tests**

Run:

```powershell
python -m pytest tests/unit/test_causal_abstraction_metrics.py tests/unit/test_patient_statistics.py tests/unit/test_network_alignment_statistics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/pptt/causal_abstraction/metrics.py tests/unit/test_causal_abstraction_metrics.py
git commit -m "feat: quantify full-path causal abstraction"
```

## Task 7: Hard Gates and Claim Levels

**Files:**
- Create: `src/pptt/causal_abstraction/gates.py`
- Test: `tests/unit/test_causal_abstraction_gate.py`

- [ ] **Step 1: Write failing gate tests**

```python
def test_one_failed_node_blocks_full_network_claim():
    report = evaluate_causal_abstraction_gate(rows_with_failed_up3)
    assert report.status == "PARTIAL_NODE_RANGE_ONLY"
    assert report.full_network_claim_authorized is False


def test_shared_claim_requires_both_cross_transfer_directions():
    report = evaluate_causal_abstraction_gate(rows_with_only_unet_to_transunet_pass)
    assert report.status == "ARCHITECTURE_SPECIFIC_PROCESS_ONLY"


def test_randomized_or_null_control_pass_blocks_causal_claim():
    report = evaluate_causal_abstraction_gate(rows_with_nonseparating_control)
    assert report.status == "OBSERVATIONAL_PROCESS_ONLY"
```

- [ ] **Step 2: Verify failure**

Run:

```powershell
python -m pytest tests/unit/test_causal_abstraction_gate.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement one deterministic decision function**

```python
class ConclusionStatus(StrEnum):
    PASS_SHARED_FULL_NETWORK = "PASS_SHARED_FULL_NETWORK"
    ARCHITECTURE_SPECIFIC_PROCESS_ONLY = "ARCHITECTURE_SPECIFIC_PROCESS_ONLY"
    PARTIAL_NODE_RANGE_ONLY = "PARTIAL_NODE_RANGE_ONLY"
    OBSERVATIONAL_PROCESS_ONLY = "OBSERVATIONAL_PROCESS_ONLY"
    INSUFFICIENT_INTERVENTION_SUPPORT = "INSUFFICIENT_INTERVENTION_SUPPORT"
    HIGH_LEVEL_MODEL_MISSPECIFIED = "HIGH_LEVEL_MODEL_MISSPECIFIED"
    FAILED_AUDIT = "FAILED_AUDIT"


def evaluate_causal_abstraction_gate(
    metrics: pd.DataFrame,
    controls: pd.DataFrame,
    audit: Mapping[str, object],
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    """Apply audit, coverage, operator, fidelity, transfer, specificity, and dose gates in that order."""
```

The returned JSON must contain every failed condition and an explicit `allowed_claim` and `forbidden_claims` list.

- [ ] **Step 4: Run gate tests**

Run:

```powershell
python -m pytest tests/unit/test_causal_abstraction_gate.py tests/unit/test_causal_conclusion_gate.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/pptt/causal_abstraction/gates.py tests/unit/test_causal_abstraction_gate.py
git commit -m "feat: gate causal abstraction claims"
```

## Task 8: Configuration and Immutable Protocol Lock

**Files:**
- Create: `configs/experiments/v11_causal_abstraction.yaml`
- Create: `scripts/lock_v11_causal_abstraction_protocol.py`
- Test: `tests/unit/test_v11_protocol_lock.py`

- [ ] **Step 1: Write failing lock tests**

Tests must reject an extra dataset, an extra main architecture, changed nodes, changed doses, altered test patients, dirty source identity, missing observers, changed kernels, and any existing non-identical lock.

```python
def test_v11_scope_is_exactly_two_main_architectures_one_control_and_brats():
    validated = validate_v11_configuration(registered_config())
    assert validated.main_models == ("unet_baseline", "transunet_r50_vit_b16")
    assert validated.control_models == ("unet_noskip",)
    assert validated.dataset == "brats2023_2d"
```

- [ ] **Step 2: Verify failure**

Run:

```powershell
python -m pytest tests/unit/test_v11_protocol_lock.py -q
```

Expected: import failure.

- [ ] **Step 3: Add the exact YAML protocol**

The configuration must contain these immutable identities:

```yaml
name: v11_cross_architecture_causal_abstraction
dataset: brats2023_2d
fit_split: val
formal_split: test
main_models: [unet_baseline, transunet_r50_vit_b16]
control_models: [unet_noskip]
model_seeds: [42, 123, 3407]
observer_seeds: [17, 29, 43]
nodes: [down1, down2, down3, down4, up1, up2, up3, up4]
relationship_states: [BC, FC, FP, FN, FW]
truth_classes: [0, 1, 2, 3]
doses: [0.0, 0.25, 0.5, 0.75, 1.0]
output_root: results/v11_causal_abstraction
```

It must also include the numerical gates from design sections 12.1--12.3, `bootstrap_iterations: 10000`, `holm_alpha: 0.05`, `history_tv_tolerance: 0.03`, and fixed matching caps.

- [ ] **Step 4: Implement lock creation and verification**

The lock records configuration SHA-256, source-tree SHA-256, all nine checkpoint hashes, all 216 individual observer hashes (three model roles by three model seeds by eight nodes by three observer restarts), ordered validation/test patient hashes, fitted process hashes, environment identity, and the exact gate table. `write_protocol_lock` must be atomic and idempotent only for byte-identical content.

- [ ] **Step 5: Run lock tests**

Run:

```powershell
python -m pytest tests/unit/test_v11_protocol_lock.py tests/unit/test_v8_protocol_lock.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add configs/experiments/v11_causal_abstraction.yaml scripts/lock_v11_causal_abstraction_protocol.py tests/unit/test_v11_protocol_lock.py
git commit -m "feat: lock V11 causal abstraction protocol"
```

## Task 9: Resumable Formal Runner

**Files:**
- Create: `scripts/run_v11_causal_abstraction.py`
- Create: `tests/integration/test_v11_smoke.py`

- [ ] **Step 1: Write a failing one-patient smoke test**

The smoke test uses temporary output and synthetic weights. It must verify job isolation, resume behavior, atomic patient completion, all eight nodes, all doses, task/null conditions, complete downstream depths, and explicit non-formal labeling.

```python
def test_v11_smoke_is_complete_resumable_and_nonformal(tmp_path):
    status = run_smoke(tmp_path, patient_limit=1)
    assert status["execution_mode"] == "smoke"
    assert status["node_count"] == 8
    assert status["formal_claim_eligible"] is False
    assert rerun_smoke(tmp_path)["skipped_completed_patients"] == 1
```

- [ ] **Step 2: Verify failure**

Run:

```powershell
python -m pytest tests/integration/test_v11_smoke.py -q
```

Expected: script API is missing.

- [ ] **Step 3: Implement the runner**

Required CLI:

```text
--workspace-root --config --protocol-lock --asset-root
--model --model-seed --execution-mode {smoke,formal}
--patient-limit --resume --device --dose-batch-size
```

Formal mode refuses `--patient-limit`, validates the lock and source tree, opens a per-job PID lock, and writes one patient directory through a temporary directory followed by atomic rename. Each patient artifact contains only states, condition metadata, effect summaries, and audits; full activations are never persisted.

- [ ] **Step 4: Run smoke and legacy integration tests**

Run:

```powershell
python -m pytest tests/integration/test_v11_smoke.py tests/integration/test_causal_trace_smoke.py tests/integration/test_transunet_trace.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/run_v11_causal_abstraction.py tests/integration/test_v11_smoke.py
git commit -m "feat: run resumable V11 interventions"
```

## Task 10: Formal Summary, Monitor, and Scheduler

**Files:**
- Create: `scripts/summarize_v11_causal_abstraction.py`
- Create: `scripts/monitor_v11_causal_abstraction.py`
- Create: `scripts/server/run_v11_causal_abstraction.sh`
- Modify: `docs/reproducibility_cn.md`

- [ ] **Step 1: Add pure-function tests to existing metric and gate suites**

Test duplicate patients, missing jobs, non-finite values, smoke contamination, unequal patient weights, altered lock hashes, and one missing node. Every case must return a failure status rather than silently dropping rows.

- [ ] **Step 2: Implement the summarizer**

It must write atomically:

```text
patient_counterfactuals.parquet
condition_fidelity.parquet
cross_architecture_transfer.parquet
shared_noninferiority.parquet
specificity_and_dose.parquet
noskip_structural_contrast.parquet
v11_gate.json
v11_audit.json
v11_status.json
```

Only `v11_status.json` may authorize the shared full-network claim.

- [ ] **Step 3: Implement the read-only monitor**

Output one refreshed table containing job identity, completed/250 patients, current node, task/null condition counts, failures, GPU memory/utilization, disk free space, recent throughput, and dynamic ETA. It must never spawn, kill, or modify a job.

- [ ] **Step 4: Implement the server scheduler**

The scheduler performs a one-patient memory profile per architecture before formal launch. It may run independent jobs in parallel only when summed peak memory remains below 21.5 GB and output roots are disjoint. It never restarts completed jobs and prints the monitor command on launch.

- [ ] **Step 5: Run the full local suite**

Run:

```powershell
python -m pytest -q
```

Expected: all core tests pass; no V0--V10 regression.

- [ ] **Step 6: Update reproducibility documentation and commit**

```powershell
git add scripts/summarize_v11_causal_abstraction.py scripts/monitor_v11_causal_abstraction.py scripts/server/run_v11_causal_abstraction.sh docs/reproducibility_cn.md tests
git commit -m "feat: complete V11 formal pipeline"
```

## Task 11: Server Preflight and Protocol Lock

**Files:**
- Server workspace: `/root/autodl-tmp/A_scheme_workspace/pptt_process_xai_workspace`
- Formal output: `/root/autodl-tmp/A_scheme_workspace/pptt_process_xai_workspace/results/v11_causal_abstraction`

- [ ] **Step 1: Push or synchronize only committed source**

Verify local branch is clean, then update the server checkout without copying local result directories or credentials.

- [ ] **Step 2: Run the complete server test suite**

```bash
cd /root/autodl-tmp/A_scheme_workspace/pptt_process_xai_workspace
.venv/bin/python -m pytest -q
```

Expected: PASS before any formal lock.

- [ ] **Step 3: Audit frozen assets**

```bash
PPTT_ASSET_ROOT=/root/autodl-tmp/A_scheme_workspace/brats2023_data \
  .venv/bin/python scripts/verify_assets.py
```

Expected: all nine model jobs, all 216 individual observers, ordered validation/test manifests, and eight node adapters pass; otherwise stop with an explicit blocked status.

- [ ] **Step 4: Fit validation processes and run history admission**

Run the lock script in preflight mode to generate `H_U`, `H_T`, and `H_shared` under a temporary validation-only root. Confirm the first-order admission passes before creating the formal directory.

- [ ] **Step 5: Create the immutable lock**

```bash
PPTT_ASSET_ROOT=/root/autodl-tmp/A_scheme_workspace/brats2023_data \
  .venv/bin/python scripts/lock_v11_causal_abstraction_protocol.py \
  --workspace-root . \
  --config configs/experiments/v11_causal_abstraction.yaml \
  --asset-root "$PPTT_ASSET_ROOT"
```

Expected: `v11_protocol_lock.json` reports `LOCKED_BEFORE_FORMAL_INTERVENTION` and a clean source identity.

## Task 12: Formal Execution and Adversarial Result Audit

- [ ] **Step 1: Launch the safe scheduler**

```bash
PPTT_WORKSPACE_ROOT="$PWD" \
PPTT_ASSET_ROOT=/root/autodl-tmp/A_scheme_workspace/brats2023_data \
PPTT_PYTHON_BIN="$PWD/.venv/bin/python" \
bash scripts/server/run_v11_causal_abstraction.sh
```

The launcher prints a persistent monitor command. Do not interrupt, duplicate, or alter running jobs.

- [ ] **Step 2: Verify all nine jobs before summarization**

Each job must contain all 250 registered patients, eight intervention nodes, every registered evaluable state, five doses, task/null controls, and no smoke/debug rows. Any failure is diagnosed read-only.

- [ ] **Step 3: Run the unique formal summarizer**

```bash
.venv/bin/python scripts/summarize_v11_causal_abstraction.py \
  --workspace-root . \
  --config configs/experiments/v11_causal_abstraction.yaml \
  --protocol-lock results/v11_causal_abstraction/v11_protocol_lock.json
```

- [ ] **Step 4: Perform the adversarial audit in fixed order**

Check asset identity, patient completeness, class/state coverage, operator residuals, OOD/leakage, first-order adequacy, absolute full-path TV, both transfer directions, shared-model non-inferiority, task/null separation, dose direction, worst node, and no-skip structural separation. Stop at the first failed prerequisite when assigning the claim level, while still reporting all downstream diagnostics as exploratory.

- [ ] **Step 5: Update the final audit only from formal artifacts**

Append the exact status, strongest authorized conclusion, failed gates, effect sizes, 95% intervals, and forbidden claims to `docs/final_experiment_audit_cn.md`. Do not generate result figures or rewrite the paper in this task.

- [ ] **Step 6: Commit the audited status**

```bash
git add docs/final_experiment_audit_cn.md manifests/result_inventory.json
git commit -m "docs: audit V11 causal abstraction results"
```

## Plan Self-Review

- **Scope:** only BraTS2023, U-Net, TransUNet, and no-skip structural control appear in the implementation matrix.
- **No new training:** all nine segmentation checkpoints and 216 individual observer files are reused; observer recomputation is blocked unless an asset/hash audit explicitly requires it.
- **Full network:** all eight macro nodes and every downstream depth enter the formal worst-condition gate.
- **Single method line:** state process, closed-form exchange, shared model, and controls all test the same causal-abstraction hypothesis; no CAM, region, frequency, or path-specific experiment is added.
- **Causal boundary:** mathematical guarantees cover the operator and conditional error bounds; empirical gates determine whether the approximate abstraction is supported.
- **Outcome independence:** thresholds, patients, states, doses, kernels, and claim decisions are locked before formal test intervention.
- **Value test:** the method must both transfer across U-Net/TransUNet and separate baseline/no-skip beyond seed variability; otherwise the conclusion is downgraded.
