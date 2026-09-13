# Implementation status

This is the persistent checklist for the user's 170-section implementation brief.
Status values: **DONE**, **PARTIAL**, **NOT STARTED**, **BLOCKED**. A phase is not
DONE merely because its directory exists. See `architecture.md` for the audit and
`research-methodology.md` for assumptions. Update this file after each tested batch.

## Baseline audit — 2026-09-13

**DONE**: inspected the Python engine, execution agents, RL environment, runner,
statistics, scenarios, evaluation/training scripts, FastAPI, legacy Streamlit,
Three.js frontend, dependencies, and tests. Existing working-tree changes are
preserved. Baseline: **64 tests passed in 11.30 seconds**, Python 3.14.6 on Windows.
This is a local measurement, not a cross-machine performance claim.

## Phase checklist

| Phase | Status | Scope and next required evidence |
|---|---|---|
| 1. Correctness and reproducibility | PARTIAL | DONE: strict config, lifecycle/TIF, FIFO invariants, partition-independent clocks, separate RNG streams, ledger, reservations and regressions. Remaining: broader latency models, self-trade prevention, full exchange sessions and settlement. |
| 2. Historical replay and calibration | PARTIAL | DONE: canonical ID-driven reconstruction and controls; bounded public Tardis downloads, exact aggregate L2 replay, snapshot comparison and descriptive scorecard on two full real days. Exchange-native/instrument adapters, Parquet and empirical calibration remain. |
| 3. Execution baselines and TCA | PARTIAL | DONE: TWAP/VWAP/POV/AC, fees, fill reconciliation, economic/objective separation and unpriced inventory detection. Full spread/impact/timing/adverse-selection attribution remains. |
| 4. Experiments and statistics | PARTIAL | DONE: frozen config/source registry, integrity verification/reproduction, finite-factor design preview, failure retention, paired bootstrap/sign tests and Holm/Bonferroni/BH corrections. Parallel registered sweeps, LHS/Sobol/optimization and result cube remain. |
| 5. Risk | PARTIAL | DONE: reservations through cancel latency; child, position, estimated notional/gross exposure, loss and kill-switch controls. Portfolio/operational risk, automatic flattening and stress engine remain. |
| 6. RL and prediction | PARTIAL | DONE: PPO environment regression checks, explicit reward terms, causal decision logs and disjoint train/validation seed domains. No new trained research models. SAC, recurrent/continuous policies and supervised prediction NOT STARTED. |
| 7. Robustness | NOT STARTED | Chronological/purged splits, walk-forward orchestration, OOD, ablations, sensitivity and independent validation datasets. |
| 8. Visualization and reports | PARTIAL | DONE: offline HTML evidence report with guardrails, complete outcomes and adjusted comparisons. Existing web/3D view preserved and cards now display economic effective cost/invalid depth. New replay/risk/research dashboards remain. |
| 9. Performance | PARTIAL | DONE: bounded matching microbenchmark with separate memory measurement. No Rust/GPU acceleration or full-workload profiling performed. |
| 10. Advanced modeling | NOT STARTED | Hawkes, regimes, market making, multi-instrument markets, auctions, empirical latency and volatility models. |

## Batch verification

- Foundation batch: **DONE for the bounded scope below**. The overall platform
  and several phases remain PARTIAL as explicitly listed above.
- PPO/SAC comparative study: **NOT RUN**. No claim of predictive, execution, or
  trading alpha follows from simulator smoke tests.
- Historical data mechanics assessment: **DONE** for two complete Deribit ETH
  perpetual sample days. Historical execution strategy study: **NOT RUN**. The
  original canonical fixture remains fabricated and clearly separate.

## Implemented batch and file ownership

- Exchange: `lob/engine.py` was completed after the parallel worker stopped at
  a usage limit. Retained tick/lot FIFO representation; added lifecycle, TIF,
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
  config/data validation and a small benchmark; remote CI is **NOT RUN** here.
- UI: `static/js/app.js` now exposes effective economic cost, fees, status and
  unpriced quantity. Other pre-existing frontend/server working-tree edits are
  preserved. No new screenshot or full visual browser review was performed.

## Tests and executed checks

Final local suite: **241 passed**, two warnings for Gymnasium's existing unbounded
observation Box, **23.78 seconds** on the successful final recorded suite run.
Earlier integrated run: 240 passed in 9.82 seconds. Timings depend on concurrent
machine load and are not performance comparisons. Windows sandbox temp-directory
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
