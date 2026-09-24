# CleoLOB v0.4 Roadmap

## External Validity and Real-Market Evidence

CleoLOB v0.4 will focus on moving the project from strong internal research
validity toward stronger external market validity.

v0.3 established reproducible workflows, explicit evidence boundaries,
historical L2 reconstruction, bounded MBO semantics, frozen calibration,
registered PPO/DQN studies, artifact verification, and cross-platform CI.

v0.4 should not primarily be a feature-count release.

The main objective is to answer a harder question:

> Which CleoLOB conclusions survive contact with independent, real-market,
> out-of-sample evidence?

---

## Research Principles

All v0.4 work should preserve the following rules:

1. External holdouts must not influence model selection.
2. Previously inspected holdouts must never be presented as fresh evidence.
3. Aggregate L2 and order-level MBO evidence must remain explicitly separated.
4. Missing source information must not be synthetically reconstructed and
   presented as observed market truth.
5. Actual fills and hypothetical residual valuation must remain separate.
6. Failed calibration, execution, and policy results must be retained.
7. Statistical comparisons must preserve complete paired designs where required.
8. Multiple comparisons must use preregistered family definitions and correction.
9. Benchmark throughput must not be presented as production or HFT latency.
10. Synthetic evidence must not be presented as historical profitability or alpha.

---

# 1. Historical MBO Validation

## Goal

Validate CleoLOB's order-identity and queue-reconstruction layer against genuine
historical market-by-order data.

## Planned work

- Add at least one real order-level market-data adapter.
- Preserve source-native order identities.
- Support source-native ADD, MODIFY, CANCEL, EXECUTE, and snapshot semantics.
- Record source and canonical-data hashes.
- Validate deterministic replay.
- Validate reconstructed aggregate L2 against source snapshots where available.
- Track queue-ahead quantity only when the source actually establishes ordering.
- Distinguish recorded maker executions from trade prints.
- Preserve censoring for orders whose terminal state cannot be observed.
- Explicitly report unsupported exchange semantics.

## Acceptance criteria

- No synthetic order IDs are substituted for missing real identities.
- Replay is deterministic from the same canonical source.
- Source hashes and normalized hashes are recorded.
- Queue metrics are disabled when source semantics do not support them.
- Full-fill semantics remain correct across MODIFY and CANCEL events.
- Priority reset rules are covered by regression tests.
- Real-MBO results are clearly separated from synthetic fixtures.

## Important limitation

If suitable real MBO data cannot legally or practically be distributed with the
repository, CleoLOB may provide adapters and reproducible instructions without
bundling the raw dataset.

This limitation must remain explicit.

---

# 2. Formal Market-Data Capability Contracts

## Goal

Make unsupported inference mechanically difficult.

Each data source should explicitly declare which observations it can establish.

Example capabilities:

- aggregate depth
- order identity
- exact FIFO ordering
- quantity ahead
- observed maker execution
- hidden quantity
- counterfactual passive fill
- exchange sequence continuity
- snapshot ordering guarantees

## Planned work

Introduce a formal capability object shared by historical replay consumers.

Consumers should fail explicitly when they request evidence unavailable from the
underlying dataset.

## Acceptance criteria

- L2 consumers cannot request identity/FIFO evidence.
- MBO consumers cannot assume hidden liquidity visibility.
- Capabilities are immutable at runtime.
- Dataset adapters expose documented capability declarations.
- Capability mismatches are tested.

---

# 3. Multi-Period Historical Generalization

## Goal

Extend calibration evaluation beyond one train/internal/external sequence.

## Planned design

Use multiple chronological periods containing:

- development period
- validation folds
- sealed internal holdout
- untouched external period

Where sufficient data is available, repeat this structure across multiple market
regimes.

## Diagnostics

Track at least:

- spread
- top-level depth
- top-N depth
- imbalance
- volatility
- log returns
- trade intensity
- book-update intensity

## Requirements

- Selection remains based on development data only.
- Internal holdout is unavailable before selection is sealed.
- External data cannot alter:
  - model family
  - parameters
  - thresholds
  - observables
  - acceptance rules

## Acceptance criteria

Mutation of internal/external bytes must not alter the frozen selection artifact.

Fresh external periods must be explicitly distinguished from previously inspected
historical periods.

---

# 4. Calibration Failure Attribution

## Goal

Move beyond a single aggregate calibration score.

v0.4 should explain why a model fails.

## Planned work

Add observable-level diagnostics including:

- normalized distribution distance
- empirical coverage
- quantile error
- mean/variance drift
- tail behavior
- temporal persistence
- cross-observable dependence
- regime transition behavior

Where appropriate, include uncertainty estimates across chronological windows.

## Output

Calibration reports should identify whether failure is primarily associated with:

- spread structure
- depth structure
- imbalance
- volatility
- activity intensity
- dependence structure
- regime persistence

A failed overall model should remain failed even if individual observables pass.

---

# 5. Completion-Constrained Execution Policies

## Goal

Address the low-completion behavior observed in the registered v0.3 DQN study.

## Planned work

Evaluate learned policies with explicit completion constraints.

Candidate approaches may include:

- stronger terminal completion penalties
- action masking where justified
- time-to-horizon features
- remaining-inventory normalization
- forced execution schedules near horizon
- constrained action policies

These mechanisms must not silently turn hypothetical liquidation into actual fills.

## Evaluation

Compare learned policies against the existing control family:

- TWAP
- VWAP
- POV
- Almgren-Chriss
- heuristic controls
- diagnostic controls

Use:

- multiple independent training seeds
- common evaluation market seeds
- unchanged held-out trajectories
- fixed training budgets
- preregistered final checkpoints
- multiplicity correction
- completion metrics reported separately from cost metrics

## Acceptance criteria

A learned policy cannot be described as successful solely because residual
inventory can be economically valued.

Report separately:

- actual fill fraction
- completion rate
- remaining inventory
- realized execution cost
- hypothetical residual liquidation cost
- combined economic metric

---

# 6. Stronger RL Experimental Design

## Goal

Determine whether learned-policy conclusions are stable across training
initialization and market regimes.

## Planned improvements

- larger independent training-seed sets
- explicit learning curves
- training instability diagnostics
- checkpoint provenance
- final-checkpoint-only registered evaluation
- ablation studies
- original / shifted / stress regimes
- complete failed-run retention

## Optional research

Additional algorithms may be added only when their action-space and training
semantics are clearly defined.

Algorithm count is not itself a v0.4 objective.

---

# 7. Performance and Scaling

## Goal

Characterize research-system scaling without implying production trading latency.

## Benchmark families

Measure separately:

- matching-engine operations
- aggregate L2 replay
- MBO replay
- snapshot generation
- experiment orchestration
- policy inference
- artifact serialization

## Measurements

Report:

- p50 / p95 / p99 call latency
- latency-derived operation rate
- wall-clock batch throughput
- workload size
- Python allocations
- process memory where reliable
- source commit
- Python version
- operating system
- CPU identity

## Scaling studies

Benchmark multiple:

- book depths
- event counts
- snapshot frequencies
- active-order counts
- experiment sizes

No result should be described as exchange-colocation or production-HFT latency.

---

# 8. Reproducibility and Artifact Integrity

## Goal

Make externally reproduced CleoLOB experiments easier to verify.

## Planned work

- versioned research artifact schema
- dataset manifests
- explicit source-data identity
- stronger provenance metadata
- deterministic reproduction checks
- compact evidence bundles
- artifact integrity verification
- release-linked evidence manifests

## Optional extension

Investigate cryptographic attestations or signed release manifests where they
provide practical reproducibility value.

---

# 9. Dataset Adapter Interface

## Goal

Make adding market-data sources predictable and auditable.

Each adapter should document:

- source
- instrument identifiers
- timestamps
- sequence semantics
- price units
- quantity units
- snapshot semantics
- order identity availability
- execution semantics
- known gaps
- licensing / redistribution constraints

Adapters should output canonical CleoLOB schemas rather than allowing downstream
research code to depend directly on vendor-specific formats.

---

# 10. Packaging and Research Usability

v0.4 should improve usability without weakening research guarantees.

Planned improvements include:

- PyPI distribution
- installation documentation
- minimal Python API examples
- CLI quickstart
- dataset-adapter documentation
- clearer experiment templates
- public reproduction instructions

Packaging improvements must remain separate from empirical validity claims.

---

# Proposed Work Packages

## M0 ? v0.4 Protocol Freeze

Before new empirical results:

- define objectives
- define acceptance gates
- freeze primary metrics
- freeze external-evaluation rules
- define fresh versus consumed datasets

Deliverable:

`docs/v04-research-protocol.md`

---

## M1 ? Capability Contracts

Implement formal L2/MBO capability declarations and consumer enforcement.

Expected areas:

- `lob/replay/`
- `lob/mbo.py`
- adapter interfaces
- regression tests

---

## M2 ? Real MBO Adapter

Implement one genuine historical MBO adapter and deterministic replay workflow.

Deliverables:

- adapter
- schema validation
- source manifest
- replay tests
- evidence report

If genuine data is unavailable, the milestone remains explicitly incomplete.

---

## M3 ? Multi-Period Generalization

Extend chronological calibration/generalization to multiple independent periods.

Deliverables:

- preregistered design
- fresh holdout registry
- drift reports
- observable-level diagnostics
- sealed selection artifacts

---

## M4 ? Completion-Constrained Policy Study

Implement and register a study specifically targeting incomplete execution.

Deliverables:

- frozen study design
- PPO/DQN or successor policies
- classical controls
- completion metrics
- cost metrics
- multiplicity-aware comparisons
- retained failures

---

## M5 ? Scaling and Performance

Run independent performance families with corrected measurement semantics.

Deliverables:

- workload definitions
- scaling curves
- latency tables
- wall-clock throughput tables
- memory measurements
- hardware/software provenance

---

## M6 ? Independent Reproduction

Run the complete v0.4 workflow from a clean environment.

Validate:

- source checkout
- installation
- artifact verification
- smoke workflow
- historical replay
- registered study reproduction

---

## M7 ? v0.4 Release Review

Before `v0.4.0`:

- full test suite passes
- cross-platform CI passes
- package build passes
- installed wheel smoke passes
- evidence artifacts verify
- claims audit completed
- limitations reconciled with README/docs
- no consumed holdout presented as fresh evidence
- no synthetic result presented as historical evidence

---

# Explicit Non-Goals

v0.4 does not require or imply:

- production high-frequency trading infrastructure
- exchange colocation
- profitable live trading
- historical alpha
- reconstruction of hidden liquidity from L2
- exact counterfactual fills from aggregate L2
- state-of-the-art RL claims
- guaranteed policy superiority
- unrestricted live broker/exchange connectivity

Any future implementation of these areas requires separate evidence.

---

# Success Criteria

v0.4 should be considered successful if it improves the strength and clarity of
CleoLOB's evidence, even if some empirical hypotheses fail.

A scientifically useful v0.4 outcome may therefore include:

- a model that fails new external calibration periods
- an RL policy that fails to beat classical execution controls
- an MBO source that exposes important queue-data limitations
- performance measurements that reveal scaling bottlenecks

Those outcomes are evidence, not release failures.

The release should optimize for trustworthy conclusions rather than favorable
conclusions.
