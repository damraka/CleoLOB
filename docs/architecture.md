# Architecture and audit

## Current v0.3 architecture

**IMPLEMENTED / TESTED:** The package retains three distinct market paths:

| Path | Implementation | Evidence boundary |
|---|---|---|
| Synthetic exchange | `engine.py`, `execution.py`, `rl_env.py`, accounting/risk/settlement | Simulated FIFO and actual simulated fills; residual valuation remains hypothetical |
| Aggregate historical L2 | `replay/l2.py`, `replay/assessment.py` | Source price-level reconstruction; identity, exact FIFO and agent counterfactual fills explicitly unavailable |
| Canonical MBO | `replay/book.py` plus `mbo.py` | Recorded source IDs/executions and source-guaranteed snapshot FIFO; no invented matching or fills |

`mbo.py` adds bounded JSONL/gzip streaming, strict sequence/timestamp validation,
exact integer L2 aggregation, queue trajectories, observed maker fill times and
explicit census/prefix censoring. **LIMITATION:** the committed order-level
datasets are synthetic; real historical MBO validation remains pending.

`generalization.py` selects joint IID or spread-state Markov observable models
using expanding/rolling train-validation folds and temporal embargoes. It seals
selection inputs before internal evaluation and loads the external partition only
after selection. It models observables, not executable order flow. The separate
`zi_calibration.py` fits simulator parameters. **FAILED:** preserved historical
observable and simulator studies breached external calibration gates.

`policy_study.py` reuses the execution environment, six classical/diagnostic
controls and existing crossed-seed bootstrap. It adds fixed-budget PPO and DQN
training, separate training/evaluation seeds, original/shifted/stress regimes,
two finite ablations, checkpoint metadata and failure retention. The prior
`core_study.py` PPO–AC experiment and evidence remain independently identified.

**EMPIRICALLY OBSERVED / FAILED:** The v0.3 smoke completed 18 PPO/DQN fits and
576 evaluations. Its 48 corrected cost intervals all include zero, and main
DQN rarely completed its parent order. The selected Markov observable model
returned internal WARNING and external FAIL. These are valid retained outputs
of the architecture, not evidence that market fidelity or execution quality
passed. See the [final measurements](v03-final-report.md).

`performance.py` profiles workloads before timing repeated book, L2, MBO,
snapshot, replay and complete simulation-episode operations. Warmups, size classes,
latency percentiles and separate Python allocation passes are explicit.
`artifacts.py` captures portable runtime/source provenance and seals artifacts;
hash verification establishes integrity, not independent scientific validation.
Large models, source bundles, episode journals and raw data stay outside Git.

**PLANNED:** validated native MBO adapters, fresh real-data generalization and
profile-supported acceleration. No production execution or live-alpha evidence
is established. See [reproduction](reproduction.md), the
[v0.3 report](v03-final-report.md), [MBO](mbo.md),
[calibration](calibration-v03.md) and [RL protocol](rl-v03.md).

The sections below retain the original architecture audit and its development
history. Their baseline omissions are not statements about the current release.
In the current checkout the legacy entry points remain `app.py` and `evaluate.py`,
and the old tables are under `results/baselines-100seeds/`; references to a
`legacy/` location below should not be used as current checkout paths.

## Original baseline architecture — historical audit

`SimConfig` → `ExchangeSimulator` clock/heap/background flow → `OrderBook`
price-time matching → `Trade` → baseline `ExecutionAgent` or `LOBExecutionEnv`
→ `build_report` → `runner` → `evaluate.py`/FastAPI → CSV/HTML or Three.js UI.

The book uses integer price ticks, individual FIFO deques and separate aggregated
level volumes. Sorted ladders are useful and retained. Prices become currency only
at accounting/report boundaries. Strategies share the exchange API. Synthetic
flow is zero-intelligence; no historical calibration is present in the baseline.

`server.py` executes `run_pair` in background threads and the browser polls jobs.
The frontend synchronizes trajectory, price, depth, tape and 3D liquidity views.
`legacy/app.py` is the archived Streamlit dashboard. `train_rl.py` is the
registered SB3 PPO study entry point. The existing test suite exercises matching, agents, Gymnasium,
runner payloads, scenarios and statistics.

## Original audit findings — addressed in subsequent batches

1. Order constructors accepted invalid quantities, prices, timestamps and IDs.
2. The market generated arrivals anew per `step(dt)`, so decision frequency changed
   the sampled market. Background and cancellation/refill randomness were coupled.
3. Strategies subtracted fills but did not reserve in-flight orders. Catch-up slices
   and cancel/replace races could exceed parent quantity.
4. Accounting was a list of fill prices, without explicit cash, position or fees.
5. Terminal valuation used partial-book VWAP for all unfilled quantity, ignoring
   unavailable depth. RL's extra completion penalty was not separately reported.
6. PPO could become a heuristic when loading failed. A label exposed this in the UI,
   but an `agent=ppo` research row still risked misleading downstream analysis.
7. Evaluation wrote seeds and a short commit, not exact resolved settings or a dirty
   code fingerprint. Saved example results are tied to the previous engine.
8. Paired tests silently intersected seeds and lacked multiple-testing correction.
9. A different RNG initial seed is not proof of disjoint training and test markets;
   the training script's original claims of strict seed separation were too strong.
10. Historical reconstruction, source validation and a persistent registry were absent.

## Architecture evolution before v0.3

Retain `lob` as the package and existing public entry points. Add bounded modules
for configuration, accounting, risk, replay, experiments and reporting rather than
renaming every caller or creating empty directories. The `cleo` console command
delegates to these modules. The web UI remains an exploration interface; the
research command is responsible for validity, provenance and failure records.

Canonical historical reconstruction has its own ID-driven book. Venue execution
records must decrement the recorded order; sending them to the synthetic matching
engine would invent different executions. Counterfactual fills in replay require a
separate explicit model and are outside this batch.

Aggregate historical data has a third, separate path: `data_download.py` verifies
bounded public samples; `replay/l2.py` applies Decimal price-level updates in
capture groups; `replay/assessment.py` joins independently published snapshots
and records exact matches, mismatches, quality counters and causal grid summaries.
There are no artificial order IDs or strategy fills in this path. The current
two-day assessment establishes input/reconstruction consistency, not calibrated
synthetic dynamics or historical execution performance.

`calibration.py` fits frozen empirical L2 observable models; `robustness.py` owns
purged chronological/walk-forward evaluation, registered stress families and study
integrity. These observable models do not replace the FIFO arrival mechanism.
`settlement.py` drains post-horizon strategy messages and late fills for baselines
and RL before final valuation. `portfolio.py` independently reconciles linear
multi-currency holdings/reservations and evaluates risk limits and joint shocks.

The v0.2.1 research scope was the PPO–Almgren–Chriss comparison.
`zi_calibration.py` fits the actual FIFO simulator to normalized historical L2
moments. `controls.py` identifies AC parameters, separates exploratory positive
controls from confirmation and reports labeled residual-price sensitivity.
`core_study.py` freezes the environment and seed domains, trains five seeds per
penalty arm, powers the test from diagnostics and evaluates saved policies once.
Portfolio/FX and further product modules remain frozen. Archived entry points and
old 100-seed tables live under `legacy/`.
