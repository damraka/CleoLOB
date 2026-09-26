# CleoLOB v0.5 Roadmap

## Real-Market Execution Validation

CleoLOB v0.5 focuses on testing which simulator and execution-policy conclusions
survive when constrained by genuine market-data observability and independent
real-market evidence.

The primary research question is:

> When CleoLOB is constrained by what real market data can actually establish,
> which simulator and execution-policy conclusions remain valid out of sample?

v0.5 is not primarily a feature-count release.

It is an external-validity, market-impact, historical-execution and
microstructure-validation release.

---

# Research Principles

All v0.5 work must preserve the following rules:

1. Real market data and synthetic evidence must remain explicitly separated.
2. Aggregate L2 must never imply exact order identity, FIFO position, hidden
   liquidity or exact passive counterfactual fills.
3. Genuine MBO evidence must preserve source-native identities and semantics.
4. Historical counterfactual execution must report uncertainty or bounds when
   exact fills cannot be established.
5. External holdouts may not influence model family, features, thresholds,
   parameters or acceptance criteria.
6. Previously inspected data may never be relabeled as fresh evidence.
7. Failed calibration and failed execution-policy hypotheses remain published.
8. Actual fills, decision-horizon completion, post-horizon settlement and
   hypothetical residual valuation remain separate.
9. Statistical significance and economic significance must be reported separately.
10. Multiplicity correction must follow a preregistered comparison family.
11. New model complexity must be justified by out-of-sample evidence.
12. No live profitability, production-HFT or alpha claim follows from synthetic
    or historical simulation alone.

---

# M0 — v0.5 Protocol Freeze

## Goal

Freeze the scientific design before new empirical results are inspected.

## Deliverables

- `docs/v05-research-protocol.md`
- versioned preregistration schema
- fresh/consumed dataset registry
- primary and secondary endpoint definitions
- comparison-family definition
- multiplicity method
- resource/compute budget
- dataset-access policy
- evidence-status vocabulary

## Required statuses

- ESTABLISHED
- NOT_ESTABLISHED
- FAILED
- NOT_AVAILABLE
- INVALID
- INCONCLUSIVE

## Acceptance criteria

- Primary hypotheses are frozen before final evaluation.
- Final holdouts are identified before access.
- Repeated access changes freshness status.
- Source/config/metric hashes are recorded.
- Protocol mutation after final evaluation is detectable.

---

# M1 — Genuine Historical MBO Validation

## Goal

Validate CleoLOB's order-identity and queue semantics using genuine historical
market-by-order data.

## Planned work

Implement at least one source-native order-level adapter where suitable data is
legally and practically available.

Preserve:

- source order ID
- venue sequence number
- exchange timestamp
- receive timestamp where available
- ADD
- MODIFY
- CANCEL
- EXECUTE
- snapshot/reset semantics
- side
- price
- quantity
- order lifetime
- partial execution
- full execution
- censoring
- priority-reset semantics

Validate:

- deterministic replay
- lifecycle correctness
- queue-ahead reconstruction
- FIFO ordering only where source semantics establish it
- modify priority rules
- cancel behavior
- partial fills
- full fills
- MBO-to-L2 aggregation
- source snapshot agreement where available

## Required implementation areas

Likely modules:

- `lob/datasets.py`
- `lob/mbo.py`
- `lob/mbo_validation.py`
- `lob/capabilities.py`

Potential new modules:

- `lob/mbo_sources.py`
- `lob/order_lifecycle.py`

## Tests

Add regression tests covering:

- duplicate order IDs
- unknown modify
- unknown cancel
- unknown execution
- non-monotonic sequence
- duplicate sequence
- timestamp inversion
- partial execution
- full execution
- cancel-after-fill
- modify-after-fill
- priority-preserving modify
- priority-resetting modify
- censoring at file boundary
- snapshot resets
- MBO-to-L2 equivalence
- unsupported FIFO evidence

## Evidence boundary

If genuine historical MBO data cannot be obtained:

- do not fabricate data
- do not upgrade synthetic evidence into historical evidence
- keep real-MBO status `NOT_AVAILABLE`
- complete the adapter contract
- document exact required schema
- document acquisition procedure
- preserve synthetic invariant tests separately

---

# M2 — Bounded Historical Counterfactual Execution

## Goal

Evaluate historical passive execution without pretending that aggregate data can
answer unknowable counterfactual questions.

## Core concept

A hypothetical order should not automatically receive a binary historical
fill/no-fill label.

Introduce bounded fill evidence.

Candidate classifications:

- OBSERVED_FILL
- GUARANTEED_FILL
- POSSIBLE_FILL
- GUARANTEED_NON_FILL
- INDETERMINATE
- UNSUPPORTED

## Example semantics

With genuine MBO:

- exact source-observed queue transitions may establish stronger fill evidence
- order identity and queue-ahead may be used only when supported by capabilities

With aggregate L2:

- level depletion may provide bounds
- trades may constrain possible execution
- exact FIFO fill must remain unknown
- hidden liquidity must remain unknown

## Planned modules

- `lob/fill_bounds.py`
- `lob/historical_execution.py`

## Outputs

For each hypothetical child order:

- observable evidence
- lower fill bound
- upper fill bound
- fill classification
- queue evidence used
- trade evidence used
- uncertainty reason
- capability requirements

## Acceptance criteria

- L2 cannot emit exact FIFO fills.
- Missing information produces uncertainty rather than fabricated certainty.
- Optimistic and conservative execution bounds are reproducible.
- All historical-execution results disclose data capability level.

---

# M3 — Calibration v2

## Goal

Test whether a richer simulator family improves real-market generalization rather
than merely improving in-sample fit.

v0.4 showed that the existing model failed fresh external calibration.

v0.5 therefore investigates model-class misspecification.

## Candidate improvements

Evaluate separately and incrementally:

- regime-conditioned event intensities
- spread-conditioned arrivals
- depth-conditioned cancellations
- imbalance-conditioned behavior
- state-dependent market-order intensity
- empirical order-size distributions
- empirical cancellation-size distributions
- depth-decay models
- volatility conditioning
- activity conditioning
- time-varying intensity
- Markov regime transitions
- self-exciting event processes where justified
- event-response kernels

Do not implement complexity solely because it is available.

## Model comparison

Compare:

- v0.4 baseline simulator
- individual extensions
- selected combined model
- deliberately simple controls

## Required observables

At minimum:

- spread
- top-level depth
- top-N depth
- imbalance
- returns
- volatility
- trade intensity
- book-update intensity
- cancellation intensity
- order-size distribution
- trade-size distribution
- interarrival distribution
- depth recovery
- short-horizon autocorrelation
- cross-observable dependence

## Evaluation structure

- development period
- model-selection period
- sealed internal holdout
- fresh external period
- optional cross-instrument external period

## Acceptance criteria

A more complex simulator is preferred only when it improves preregistered
out-of-sample criteria.

Training-fit improvement alone is insufficient.

---

# M4 — Market Impact and Resilience Validation

## Goal

Connect execution-cost models and simulator response dynamics to observed
real-market behavior.

## Historical quantities

Estimate where supported:

- signed trade size
- short-horizon mid-price response
- spread response
- depth depletion
- depth replenishment
- recovery time
- volatility response
- imbalance response
- post-trade persistence

## Conditional analysis

Condition responses on:

- trade-size bucket
- participation proxy
- spread regime
- volatility regime
- depth regime
- imbalance regime
- market direction
- activity regime

## Outputs

Produce empirical response curves for horizons such as:

- immediate
- short
- medium

Exact horizons must be preregistered.

## Simulator validation

Compare historical conditional response distributions with simulator-generated
response distributions.

Report:

- effect sizes
- uncertainty intervals
- distance metrics
- direction agreement
- persistence disagreement

## Almgren-Chriss integration

Where assumptions permit, estimate empirical parameters for:

- volatility
- temporary-impact proxy
- permanent/persistent-impact proxy
- liquidity scale

Do not imply that fitted proxies identify a unique structural impact model.

## Planned modules

- `lob/impact.py`
- `lob/resilience.py`
- `lob/impact_validation.py`

---

# M5 — Strict Completion and Execution Mandates

## Goal

Distinguish actual execution success within the mandate from eventual settlement.

v0.4 observed:

- 1,061 / 1,080 episodes completed inside the decision horizon
- 19 / 1,080 completed during settlement

v0.5 should make this distinction a formal endpoint.

## Required metrics

Report separately:

- within-horizon completion
- completion by final settlement
- residual inventory at horizon
- residual inventory after settlement
- post-horizon filled quantity
- lateness
- time to completion
- fill fraction at horizon
- final fill fraction
- realized fill cost
- hypothetical residual valuation
- fees
- participation
- implementation shortfall
- completion-adjusted economic objective

## Policy-success rules

A policy must never be described as completing its mandate merely because
settlement eventually removed residual inventory.

## Tests

Cover:

- complete exactly at horizon
- one event before horizon
- one event after horizon
- delayed fills
- unresolved cancellations
- latency crossing horizon
- missing liquidity
- kill switch
- order-size limits
- residual valuation without fill
- forced settlement without policy action

---

# M6 — Multi-Regime and Cross-Instrument External Validity

## Goal

Move beyond a small number of periods from one instrument.

## Design

Where data availability permits, construct independent regimes defined before
evaluation using objective market properties such as:

- volatility
- spread
- depth
- trading activity
- imbalance
- trend
- stress

Avoid manually selecting periods based on policy performance.

## Evaluation layers

1. same instrument, unseen time period
2. same venue, different instrument
3. different regime
4. optional different venue where semantics are compatible

## Dataset registry

For each dataset record:

- source
- venue
- instrument
- contract
- date range
- access time
- capability level
- tick size
- lot size
- timezone
- sequence semantics
- source hash
- normalized hash
- licensing constraints
- freshness status

## Acceptance criteria

- no external period is reused as fresh
- selection does not inspect final holdout outcomes
- unsupported cross-market comparisons are rejected
- failures remain visible

---

# M7 — Execution Policy Study on the Stronger Environment

## Goal

Re-evaluate learned and classical execution policies only after simulator and
historical-validation work is frozen.

## Baselines

Retain:

- TWAP
- VWAP
- POV
- Almgren-Chriss
- heuristic controls
- diagnostic/random controls

Retain:

- PPO
- DQN

Potential additional learned policies may include:

- recurrent PPO
- constrained RL
- distributional value methods
- uncertainty-aware policy variants

Additional algorithms are optional.

Algorithm count is not a success criterion.

## State design

Audit:

- aggregate market state
- own-order state
- inventory
- time remaining
- cancellation pending state
- resting quantity
- available quantity
- spread/depth/imbalance
- volatility/activity context
- regime indicators only when observable

Never expose:

- future state
- final-holdout statistics
- unavailable queue information
- hidden liquidity
- simulator truth unavailable to equivalent real-data observations

## Experimental design

Require:

- independent training seeds
- common evaluation market seeds
- fresh evaluation seeds
- frozen checkpoints
- fixed compute budgets
- train-only normalization
- full failed-run retention
- paired comparisons where appropriate
- multiplicity correction
- cost and completion endpoints
- regime-level results
- seed-level results

## New primary/secondary endpoints

Possible preregistered primary gate:

1. economic-cost improvement
2. within-horizon completion noninferiority

Secondary:

- final settlement completion
- tail cost
- seed sensitivity
- residual inventory
- lateness
- regime robustness

## Required claims discipline

One significant contrast does not establish global policy superiority.

A learned policy may be described as superior only if the exact preregistered
success gate passes.

---

# M8 — Historical vs Synthetic Policy Transfer

## Goal

Determine whether policy conclusions from the simulator agree with bounded
historical execution evidence.

## Compare

For comparable policies and scenarios:

- synthetic ranking
- historical bounded-cost ranking
- completion behavior
- regime sensitivity
- direction of pairwise differences

Because historical counterfactual fills may be bounded rather than exact,
comparisons must propagate uncertainty.

## Questions

- Does a policy that looks good in simulation remain plausible historically?
- Are policy rankings stable?
- Do simulation advantages disappear under conservative historical fill bounds?
- Which conclusions are invariant to fill assumptions?

## Acceptance criteria

Do not collapse bounded historical evidence into false point estimates.

---

# M9 — Paper-Style Research Report

## Goal

Make v0.5 readable as a coherent research study rather than a collection of
features.

## Main report

Create:

`docs/v05-paper.md`

Suggested structure:

1. Abstract
2. Research questions
3. Data
4. Market-data observability
5. MBO validation
6. Historical counterfactual methodology
7. Simulator model
8. Calibration v2
9. Market impact and resilience
10. Execution protocol
11. Classical baselines
12. RL policies
13. Statistical design
14. Results
15. Negative/null findings
16. Sensitivity analysis
17. Limitations
18. Reproduction
19. Conclusions

## Supporting documents

- `docs/v05-research-protocol.md`
- `docs/v05-data.md`
- `docs/v05-mbo.md`
- `docs/v05-historical-execution.md`
- `docs/v05-calibration.md`
- `docs/v05-impact.md`
- `docs/v05-execution.md`
- `docs/v05-rl.md`
- `docs/v05-statistics.md`
- `docs/v05-reproduction.md`
- `docs/v05-final-report.md`

---

# M10 — Reproducibility and Evidence

## Goal

Every headline result should trace to immutable research inputs.

Each final study should record:

- Git commit
- package version
- Python version
- dependency versions
- operating system
- CPU/environment metadata where relevant
- source dataset identity
- source hash
- normalized data hash
- adapter version
- config hash
- preregistration hash
- normalization hash
- checkpoint hash
- seeds
- comparison family
- statistical method
- result hash

## Verification

Tampering tests should cover:

- source mutation
- config mutation
- checkpoint mutation
- result mutation
- missing files
- stale artifacts
- mismatched registration
- changed normalization
- wrong dataset
- wrong source commit

## Public evidence bundle

Create a compact public v0.5 evidence bundle that excludes:

- restricted raw market data
- restricted derived data
- oversized training checkpoints where unnecessary
- personally identifying local paths

but retains enough hashes and metadata to verify provenance.

---

# M11 — Performance and Scaling

## Goal

Measure how the new real-data workflows scale.

Benchmark independently:

- MBO parsing
- canonical normalization
- identity replay
- MBO-to-L2 aggregation
- fill-bound computation
- impact analysis
- calibration v2
- simulator generation
- policy evaluation
- artifact serialization

Report:

- workload size
- throughput
- p50
- p95
- p99
- wall-clock duration
- Python allocations
- process memory where reliable
- hardware
- OS
- Python version

No production-HFT claim follows.

---

# M12 — Packaging and Research UX

Maintain:

- `pip install cleolob`
- clean CLI
- Python API
- dataset-adapter examples
- study templates
- reproducibility commands
- clear errors for missing optional dependencies

Potential CLI commands:

- `cleo validate-mbo`
- `cleo historical-execution`
- `cleo fill-bounds`
- `cleo impact-study`
- `cleo resilience-study`
- `cleo calibration-v2`
- `cleo regime-study`
- `cleo policy-study`
- `cleo verify-evidence`

Do not break stable v0.4 commands without a documented migration path.

---

# Testing Requirements

v0.5 should add tests for at least:

- native MBO identity semantics
- queue priority
- modify priority reset
- partial executions
- censoring
- MBO-to-L2 aggregation
- capability refusal
- counterfactual fill bounds
- optimistic/conservative historical fills
- historical uncertainty propagation
- calibration split isolation
- model-selection leakage
- fresh-holdout consumption
- market-impact calculations
- resilience recovery
- regime classification
- completion at horizon
- post-horizon settlement
- residual inventory
- learned-policy observation leakage
- normalization isolation
- paired statistical design
- multiplicity correction
- artifact mutation detection
- installed-wheel CLI behavior

The full existing suite must remain green.

---

# Success Criteria

v0.5 is scientifically successful if it produces stronger conclusions even when
those conclusions are negative.

Acceptable outcomes include:

- real MBO validation exposing unsupported queue assumptions
- richer calibration models still failing fresh real-market periods
- synthetic policy rankings failing to transfer historically
- PPO/DQN failing to outperform classical controls
- within-horizon completion exposing weaknesses hidden by settlement
- impact/resilience behavior differing substantially between simulator and data

Those are research results, not release failures.

---

# Explicit Non-Goals

v0.5 does not claim or require:

- profitable live trading
- historical alpha
- production HFT
- exchange colocation
- unrestricted broker connectivity
- exact passive fills from L2
- reconstruction of hidden liquidity
- guaranteed RL superiority
- universal market calibration
- universal cross-venue transfer

---

# Release Gate

Before `v0.5.0`:

- complete M0 protocol freeze
- complete all software work for M1-M12
- genuine MBO evidence either ESTABLISHED or explicitly NOT_AVAILABLE
- fresh holdouts are consumed correctly
- calibration results retained regardless of outcome
- historical fill uncertainty retained
- policy study registered before final evaluation
- within-horizon completion reported separately
- complete statistical audit
- artifact verification passes
- full test suite passes
- Ruff passes
- compileall passes
- pip check passes
- wheel/sdist build
- isolated wheel smoke
- Linux/Windows CI passes
- public evidence bundle verifies
- README claim audit completed
- final report completed
- release review completed