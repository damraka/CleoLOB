# Implementation status

This is the persistent checklist for the broader 170-section platform brief.
Status values: **DONE**, **PARTIAL**, **NOT STARTED**, **BLOCKED**. A phase is not
DONE merely because its directory exists. See `architecture.md` for the audit and
`research-methodology.md` for assumptions. Update this file after each tested batch.

## Current execution study — 2026-09-22

The active release is the bounded PPO-versus-AC study. The platform-wide phases
below remain partial where their broader features are outside this scope.
Older dated batch sections are historical records, not current test counts or
statements that the new study has never run.

- **DONE:** frozen simulation-based L2 calibration against July 2026 BTC/ETH
  training data, followed by August validation and September holdout. All six
  external cohort/phase gates FAIL. The four external files contain 7,461,728
  snapshots and 345,595 one-second observations; pooled rows reuse those data.
  September is consumed holdout evidence, and the model remains unvalidated.
- **DONE:** positive control under a separately registered execution amendment.
  The original failed grid is preserved. The selected 1,714-lot, 240-second
  setting passed 384 capacity preflight episodes and 64 independent confirmation
  seeds: Random minus TWAP −0.9591 bps, 95% CI [−1.1953, −0.7155].
- **DONE:** frozen `configs/core-study.json` and interval-specific AC fit on
  separate seeds; η=5.51834×10⁻⁵, σ=0.00438465, R²=0.5013, coverage 92.94%.
- **DONE:** all 20 PPO fits in `examples/studies/core/ppo-final-20260922`:
  163,840 steps and 4,957 training episodes, with zero INVALID outcomes. The run repeats the fixed design already
  executed twice in Linux CI; prior runs are not pooled as independent seeds.
  The initial local serial attempt is preserved. A registered computational
  amendment adds isolated processes without changing scientific settings.
- **DONE:** diagnostic power analysis locked 32 final market seeds before final
  evaluation; approximate power is 81.48% for the 0.5 bps primary effect, against
  an 80% target. The resource cap is not binding. All **704/704** planned final
  episodes completed, with **zero INVALID** outcomes. Primary PPO minus
  risk-neutral AC is **−1.6046 bps, 95% CI [−2.0951, −1.0152]**. Actual primary
  fill is **86.90%**, with remaining inventory hypothetically valued; the 100 bps
  penalty arm filled 100%. All 99% family intervals and fixed risk-sensitive AC
  results are retained. This is a synthetic computational replication, not
  historical execution validation.
- **DONE:** four Plotly report types, reward decomposition, primary/family
  intervals and labeled invalid-outcome sensitivity rendering. Final reports
  were generated from the sealed study, which passes integrity verification.
  No test episode was rerun for rendering.
- **DONE:** current full local suite **573 passed**, plus Ruff and compilation.
  Two existing Gymnasium warnings concern the unbounded observation Box. Linux
  CI covers Python 3.11/3.13; the release commit's run is verified separately.
- **DONE:** legacy scripts/100-seed results archived under `legacy/`; non-core
  Portfolio/FX and new interface work frozen. Earlier scientific artifacts and
  failed results remain intact.

See the [current README](../README.md), [protocol](core-research-protocol.md) and
[external calibration assessment](../examples/studies/core/calibration/EXTERNAL_RESULTS.md).

## Baseline audit — 2026-09-13

**DONE**: inspected the Python engine, execution agents, RL environment, runner,
statistics, scenarios, evaluation/training scripts, FastAPI, legacy Streamlit,
Three.js frontend, dependencies, and tests. Existing working-tree changes are
preserved. Baseline: **64 tests passed in 11.30 seconds**, Python 3.14.6 on Windows.
This is a local measurement, not a cross-machine performance claim.

## Phase checklist

| Phase | Status | Scope and next required evidence |
|---|---|---|
| 1. Correctness and reproducibility | PARTIAL | DONE: strict config, lifecycle/TIF, FIFO invariants, clocks/RNG, ledger/reservations, bounded post-horizon settlement and late-fill reconciliation. Remaining: feed/broader latency models, self-trade prevention and full exchange sessions. |
| 2. Historical replay and calibration | PARTIAL | DONE: canonical/L2 reconstruction, public provenance, empirical observable models and simulation-based ZI moment fitting across BTC/ETH dates. All frozen external ZI gates fail; real-market validity, exchange-native/instrument adapters and Parquet remain. |
| 3. Execution baselines and TCA | PARTIAL | DONE: TWAP/VWAP/POV/AC, fees, fill reconciliation, economic/objective separation and unpriced inventory detection. Full spread/impact/timing/adverse-selection attribution remains. |
| 4. Experiments and statistics | PARTIAL | DONE: source/config registry, reproduction, finite designs, stress execution, Holm/Bonferroni families, crossed-seed intervals, power locking, isolated PPO workers and failure retention. Broader scheduling, LHS/Sobol/optimization and result cube remain. |
| 5. Risk | PARTIAL | DONE: execution limits plus linear multi-currency portfolio accounting/reservations, gross/net/leverage/margin/concentration, stale mark/FX rejection, drawdown/daily-loss latch, asset/FX scenarios and empirical VaR/ES. Nonlinear derivatives, operational risk and automatic flattening remain. |
| 6. RL and prediction | PARTIAL | DONE: PPO environment contracts, reward decomposition, train/diagnostic/test seed blocks, 20 trained fixed-budget models and all 704 final episodes with crossed intervals. Prior fixed-design CI runs are documented. SAC, recurrent/continuous policies and supervised prediction remain outside scope. |
| 7. Robustness | PARTIAL | DONE: purged splits, walk-forward fits, multiple BTC/ETH dates, external fidelity gates, six-profile stresses and explicit residual-price sensitivity. Four completion penalties are in the current study. Broader independent venues, historical counterfactual execution and outage/crash dynamics remain. |
| 8. Visualization and reports | PARTIAL | DONE: offline evidence reports plus Plotly comparison intervals, training curves, reward and execution-cost decomposition. Existing web/3D exploration is preserved. Broader replay/risk dashboards remain. |
| 9. Performance | PARTIAL | DONE: bounded matching microbenchmark with separate memory measurement. No Rust/GPU acceleration or full-workload profiling performed. |
| 10. Advanced modeling | NOT STARTED | Hawkes, regimes, market making, multi-instrument markets, auctions, empirical latency and volatility models. |

## Foundation batch verification — historical

- Foundation batch: **DONE for the bounded scope below**. The overall platform
  and several phases remain PARTIAL as explicitly listed above.
- At the original foundation-study date, the PPO/SAC comparative study had **NOT RUN**.
  A subsequent bounded PPO execution study has since been completed; its compact final evidence is
  available under `examples/studies/core/ppo-final-20260922/`.
  This does not constitute evidence of live profitability or historical alpha.
- Historical data mechanics assessment: **DONE** for two complete Deribit ETH
  perpetual sample days. Historical execution strategy study: **NOT RUN**. The
  original canonical fixture remains fabricated and clearly separate.

## Implemented batch and file ownership

- Exchange: `lob/engine.py` retains tick/lot FIFO representation and adds lifecycle, TIF,
  queue and event controls instead of introducing a new package hierarchy.
- Accounting/execution: new `lob/accounting.py` and `lob/risk.py`; updated
  `lob/execution.py`, `lob/rl_env.py`, `lob/runner.py`, public exports and training
  seed handling in `train_rl.py`.
- Replay: new `lob/replay/{schema,book,validation,io,engine}.py` and public API;
  `examples/data/canonical-events.jsonl` is fabricated.
- Research: new `lob/config.py`, `lob/experiments/{registry,runner,report,design}.py`,
  `lob/cli.py`, `lob/benchmarks.py`, typed presets and `pyproject.toml` console entry.
  Updated `lob/stats.py` and the legacy `evaluate.py` to retain failures and apply
  corrected significance rather than silent pair intersections/fallbacks.
- Verification: new accounting/risk, execution-integrity, exchange-foundation,
  replay, config, statistics, registry/CLI and legacy-evaluation tests. Original
  tests are retained; four assertions/fixtures were corrected to use monotonic
  times, isolate cancellation/resilience from random unrelated flow, and permit
  legitimate early completion.
- Documentation/CI: architecture, methodology/limitations, canonical data,
  configuration/CLI, current README, archived original README and this checklist.
  `.github/workflows/research.yml` configures Linux 3.11/3.13 lint, compile, tests,
  config/data validation and a small benchmark. This describes the foundation
  configuration; current CI and retained-study verification are listed above.
- UI: `static/js/app.js` now exposes effective economic cost, fees, status and
  unpriced quantity. Other pre-existing frontend/server working-tree edits are
  preserved. No new screenshot or full visual browser review was performed.

## Tests and executed checks

Foundation local suite: **241 passed**, two warnings for Gymnasium's existing unbounded
observation Box, **23.78 seconds** on the successful final recorded suite run.
Earlier integrated run: 240 passed in 9.82 seconds. Timings depend on concurrent
machine load and are not performance comparisons. Windows temp-directory
restrictions caused fixture setup errors on unprivileged invocations; rerunning
with approved normal temp access passed. No test failure was suppressed.

```sh
python -m pytest -q --tb=line
python -m ruff check lob tests evaluate.py train_rl.py server.py
python -m compileall -q lob evaluate.py train_rl.py server.py
node --check static/js/app.js
python -m pip install -e . --no-deps --no-build-isolation
cleo --help
cleo config validate configs/research.yaml
cleo validate-data examples/data/canonical-events.jsonl
cleo evaluate --config configs/research.yaml --out examples/studies/foundation
cleo reproduce examples/studies/foundation/20260913T191627-0268a8fa8cb3 --out examples/studies/foundation
cleo benchmark --pairs 2000 --repeats 3
```

The 12-event canonical fixture validated and replayed identically, recording two
executions totaling seven units. The FastAPI server was launched locally on a
temporary port: `/` returned 200, an unknown job returned 404, a submitted 500-share
simulation returned both baseline and policy results with status VALID. The
temporary server was stopped. Starlette TestClient was unavailable because the
local installation requires `httpx2`; the HTTP smoke used the running server
instead. Ruff, Python compilation and JavaScript syntax checks passed.

## Bugs addressed

1. Duplicate IDs and invalid numerical orders could corrupt matching.
2. Caller step size changed the sampled market.
3. In-flight orders and pending cancels did not reserve parent quantity.
4. Entire consumed price levels disappeared from resilience deficit calculation.
5. Floating-point release boundaries could omit a due final slice.
6. Partial-depth VWAP was extrapolated to all leftovers.
7. Actual fees, hypothetical fees and noncompletion penalties were conflated or absent.
8. Decision logs needed pre-action rather than subsequent observations.
9. A different RNG seed did not guarantee disjoint training/validation seed domains.
10. Legacy evaluation could call an untrained heuristic PPO and ignore missing pairs.
11. Multiple unadjusted tests could be reported as significance.
12. Saved outcomes lacked the complete dirty-code/configuration provenance needed
    to reproduce or reject a reproduction attempt.

## Assumptions and remaining work

Executed study: `examples/studies/foundation/20260913T191627-0268a8fa8cb3`, **60
episodes**, six agents × ten seeds, 50 VALID and 10 WARNING. Run time 117.38 seconds.
The full rerun `20260913T191842-dca37a631e1c` produced **byte-identical episode
records**, with both manifests verified and the comparison saved under
`examples/studies/foundation/reproductions/`. Every paired difference interval
includes zero; all five Holm-adjusted sign-test p-values are 1.0. No alpha claim.

Recorded local matching benchmark: **64,835 orders/s** median across three 4,000-order
repetitions; 2,000 resting orders used 1,153,972 peak Python-allocated bytes in a
separate memory pass. See `examples/benchmarks/` for exact samples and limits.
This is not end-to-end simulation throughput and not a measured acceleration.

See `research-methodology.md` for exact mechanics and limits. This batch assumes
one continuous synthetic instrument, Poisson order flow, fixed-plus-exponential
message delay, current exchange observations, average-cost accounting and optional
pretrade monetary estimates. No venue calibration, live orders or profitability
claim is made. Reproduction requires the same implementation/runtime; source
snapshots are audit records and are never executed automatically.

The public-data batch below adds real samples and a tested normalized L2 adapter.
Next add instrument-aware units and frozen empirical calibration before describing
any synthetic model as realistic. Then extend registered designs across regimes, track the complete
hypothesis family, and add chronological/purged walk-forward and OOD/stress tests.
Do not interpret an incomplete phase or an existing PPO interface as completion of
the 170-section brief.

## Public-data batch — 2026-09-13

**DONE for bounded ingestion and reconstruction validation.** Added
`lob/data_download.py`, `lob/replay/l2.py`, `lob/replay/assessment.py`,
`download-sample`/`assess-l2` CLI commands, and 144 focused tests. Raw downloads
are ignored under `data/public/`; six provenance sidecars preserve source URLs,
timestamps, provider MD5, local SHA-256 and compressed/expanded sizes.

Downloaded two full days of Deribit ETH-PERPETUAL from the provider's public
first-of-month samples: 2020-04-01 and 2020-05-01. Processed **5,972,671** price-level
updates, exactly matched **2,081,479** published top-five snapshots and validated
**46,195** trade records. Both assessments PASS. No book or exchange timestamp
mismatches, unmatched groups, crossed/empty books or duplicate trade IDs.
Five snapshot blocks exercised initialization and reset handling. Maximum capture
gaps were 7.060 and 5.217 seconds; two April grid samples were older than five
seconds. Sequence completeness cannot be proved from normalized timestamps.

Source: `examples/studies/historical/sources.json`. Saved reports and limitations:
`examples/studies/historical/README.md`. The six compressed files total 84,784,910
bytes; complete gzip validation covered 835,298,573 expanded bytes. Artifact,
input and current implementation hashes were verified for both saved assessments.

Final integrated suite: **385 passed, 2 existing Gymnasium warnings in 12.88s**.
Ruff, compileall, JavaScript syntax and diff-whitespace checks passed. Actual CLI
download reuse and both complete real-data assessments ran successfully. No new
performance claim, remote CI run, model training or execution backtest occurred.

Review caught and tests now cover: mismatched identity even without overlapping
timestamps, Unicode numeric ambiguity, exact exponent values, Decimal aggregation
precision, invalid books between sampling instants, corrupted gzip bodies/trailers,
source changes during assessment, and exchange timestamp disagreement.

Remaining: calibration, instrument contract units, original sequence-ID validation,
historical execution and delayed-order settlement, market-data latency, chronological
held-out periods, multi-regime stress/OOD evaluation, portfolio risk, SAC and the
other phases above. These data validate aggregate mechanics, not FIFO fill models
or profitability. See `docs/public-market-data.md` for commands and assumptions.

## Four priority workflows — 2026-09-19

**DONE: implementation and executed validation for the four requested workflows.**
This does not mark the full 170-section platform complete or certify the fitted
model's realism. The empirical baseline fails later-date diagnostics, as recorded.

- Calibration: `lob/calibration.py` extracts causal one-second L2 features with
  stale/invalid masking, fits immutable joint empirical observable distributions,
  provides deterministic generation and digest-verified save/load. No latent
  FIFO flow or counterfactual execution calibration is claimed.
- Robustness: `lob/robustness.py` supplies purged chronological splits, expanding
  walk-forward fits, immutable train/validation/test plans, registered execution
  stress families, source archives and nested study verification. Checkpoint or
  child provenance changes withhold all family inference.
- Settlement: `lob/settlement.py` plus execution/runner/RL integration stop new
  decisions, retry cancellations, drain late fills/fees, reconcile the ledger,
  and mark unresolved settlement INVALID. Policy series use actual exchange time.
- Portfolio: `lob/portfolio.py` supplies bounded linear multi-currency cash,
  positions, independent reservations, margin/gross/net/concentration/leverage,
  stale mark/FX controls, loss latch, reduce-only admission, asset/FX stress and
  empirical VaR/ES. Precision is isolated at 192 Decimal digits for allowed scales.
- Integration: `calibrate`, `stress`, `portfolio`, generalized `verify/report`,
  settlement settings, `configs/robustness.yaml`, `configs/portfolio_example.json`.
  Earlier sealed config hashes remain verifiable without injecting new defaults.

Final suite: **521 passed, 2 existing Gymnasium warnings in 25.32 seconds**.
Ruff, compilation, JavaScript syntax and old/new artifact integrity checks passed.
No new full visual review, speed claim, remote CI or model-training claim.

Executed evidence: `examples/studies/validation/README.md`.

1. Calibration on April's first 60%, with purged later validation/internal test
   and May external test: 51,778 train / 17,219 validation / 17,279 internal /
   86,399 external samples. Validation WARNING, internal FAIL, external FAIL.
   Four training-only walk-forward folds returned WARNING/FAIL/WARNING/WARNING.
   The model remained frozen; failures were not tuned away using held-out data.
2. Stress: 300 episodes, 204 VALID / 3 WARNING / 93 INVALID. All orders settled;
   827 late shares in 17 episodes, maximum settlement 1.08s. All invalid outcomes
   were insufficient terminal depth; all 24 planned comparisons were withheld.
3. Portfolio: actual demo actions reconciled cash/positions/fees, rejected risk
   increases, preserved cancel reservations and latched losses. Three synthetic
   joint shocks executed. Its deliberately partial reduce-only order is disclosed.

The attempted June public download failed DNS resolution, including escalated
access. Existing checksum-verified April/May files were used. Their prior
reconstruction/descriptive inspection is disclosed; they are holdouts from model
fitting, not never-inspected independent data. No data were fabricated.

Next research decisions must address failed observable generalization and obtain
fresh dates/venues before any robust execution claim. Advanced items remain
explicit in the phase checklist; the four implemented workflows now make these
limitations testable rather than silently assumed.
