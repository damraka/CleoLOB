# Architecture and audit

## Existing architecture

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

## Audit findings

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

## Direction

Retain `lob` as the package and existing public entry points. Add bounded modules
for configuration, accounting, risk, replay, experiments and reporting rather than
renaming every caller or creating empty directories. The `cleo` console command
will delegate to these modules. The web UI remains an exploration interface; the
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

The active research scope is now the PPO–Almgren–Chriss comparison.
`zi_calibration.py` fits the actual FIFO simulator to normalized historical L2
moments. `controls.py` identifies AC parameters, separates exploratory positive
controls from confirmation and reports labeled residual-price sensitivity.
`core_study.py` freezes the environment and seed domains, trains five seeds per
penalty arm, powers the test from diagnostics and evaluates saved policies once.
Portfolio/FX and further product modules are frozen. Archived entry points and
old 100-seed tables live under `legacy/`.
