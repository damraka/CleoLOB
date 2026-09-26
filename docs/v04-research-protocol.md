# v0.4 implementation and evidence protocol

The source of scope is [the unchanged roadmap](v04-roadmap.md). This protocol was
written before v0.4 experiments. Baseline: `c6e9d1a`, after README PR #9 passed
all required checks and merged normally; 713 local tests passed (two legacy
unbounded-observation warnings). Branch: `research/v0.4-external-validity`.

| Roadmap milestone | Implementation and acceptance |
|---|---|
| M0: protocol | Freeze each run's config, hypotheses, source, seeds and input identities before evaluation; retain failed attempts. |
| M1: contracts | Immutable trades/L1/L2/MBO/simulator capabilities; unsupported identity, FIFO, hidden-liquidity and fill requests raise. |
| M2: MBO | Canonical and explicitly mapped ingestion, source/native identities, strict transitions, priority and aggregation diagnostics. No genuine MBO is locally available; real validation remains NOT_AVAILABLE. |
| M3: generalization | Separate physical development/internal/external inputs, chronological folds and embargo, sealed selection, consumed-period registry and observable-level failure attribution. |
| M4: completion | Shared optional urgency constraints, actual completion separate from residual valuation, train-only observation scaling and own-order state, fixed final checkpoints. |
| M5: scaling | Warmed repeated depth/event/snapshot/order/experiment-size sweeps; separate call latency, batch throughput and Python allocation passes. No production latency claim. |
| M6: reproduction | Versioned evidence bindings, tamper tests, installed-wheel CLI and offline pipeline, full tests/lint/build/dependency checks. |
| M7: review | Compact evidence with explicit exclusions, technical report, independent claims audit, protected-branch PR checks. No automatic final release. |

## Historical evidence boundaries

The available April/May 2020 aggregate Deribit inputs and previously registered
July/August/September 2026 evidence are consumed. They may support retrospective
diagnosis, never fresh confirmation. A June 2020 first-day Deribit ETH-PERPETUAL
period is a candidate fresh local evaluation, subject to access and current
provider terms. Its date, model and gates must be declared before download; it
must not affect selection. One new day cannot establish broad regime/venue
generalization. Restricted raw data, trained models and detailed derived data
must remain outside the public evidence bundle. No commercial MBO data is assumed.

Calibration diagnosis reports distribution, tail, coverage, mean/variance,
serial persistence and dependence shifts with chronological-block uncertainty
where supported. These are associations, not identified causes. Aggregate depth
changes cannot identify cancellation or order-arrival intensity. Existing failed
models and thresholds are preserved.

## Policy study design

The new configuration will register PPO and DQN, four independent training seeds,
three arms, 2,048 steps per model and twelve common held-out market seeds across
original/shifted/stress synthetic regimes. This is a bounded exploratory study,
not a convergence or power guarantee. The old study is unchanged. TWAP, VWAP,
POV, Almgren-Chriss, heuristic and random controls share the same execution limits
and completion mechanism. Queue position is excluded from observations.

Primary endpoints are fee-inclusive effective cost and actual completion.
The comparison family and Bonferroni correction include both endpoints, with a
prespecified five-percentage-point completion noninferiority margin. A favorable
cost mean alone is insufficient: superiority requires the adjusted cost interval
strictly below zero and the adjusted completion interval above the margin.
Residual valuation is always hypothetical. Fit normalization only on training
seeds; retain every final checkpoint, failed run and full paired evaluation grid.
Exact resolved hypotheses and metrics are sealed in the run registration.

## Evidence and release rules

Use ESTABLISHED, NOT_ESTABLISHED, FAILED, NOT_AVAILABLE, INVALID or INCONCLUSIVE
where appropriate. Legacy PASS/WARNING/FAIL gates retain their original meaning.
Checksums establish byte integrity, not truth or independent reproduction.
Never regenerate and relabel original artifacts. New attempts use new directories.
Freeze all implementation source before registration and evaluation; record exact
source hashes even when documentation changes later. Publish no v0.4.0 release
until the actual evidence, tests, packaging, CI and remaining blockers are reviewed.
