# Calibration, validation, settlement and portfolio risk

This batch implements four usable research workflows. It does not turn a failed
model diagnostic into a successful model, or describe unobserved fills as real.

## Delayed-order settlement

Every baseline and policy stops sending new children at the decision horizon.
The exchange continues processing until strategy orders are terminal, or the
configured settlement timeout is reached. Cancellation requests retain quantity
reservations. Cancels rejected before an order arrives are retried; late fills
and fees are booked into the same ledger.

`execution.settlement_timeout` defaults to 5 seconds and
`execution.settlement_poll_dt` to 0.01 seconds. The timeout/poll ratio is bounded
at 100,000. The engine event cap remains active. Reports separate decision
duration, settlement duration, total duration, late quantity/fees and unresolved
order IDs. RL terminal rewards include the same late actual fills and fees once.
A timeout truncates the RL episode and makes economic results INVALID/null.
Successful settlement cancels unfilled children; it does not execute a new forced
liquidation. Residual parent quantity is still explicitly valued hypothetically
only where visible depth is sufficient.

## Empirical calibration and later-date evaluation

`lob.calibration` extracts completed aggregate L2 states on a capture-clock grid.
It retains invalid/stale rows and masks returns across invalid observations or
gaps. Extraction requires complete input consumption and stable source hashes.
Prices and native amount units are preserved; an aggregate level is never
converted into an individual FIFO order.

The frozen model fits joint empirical observations of spread, top-five depth,
imbalance and log returns. A bounded bootstrap pool supports deterministic
synthetic observable generation. It is an IID baseline with no temporal dynamics,
latent order-arrival/cancellation identification or hypothetical fill model.
The separate FIFO engine remains an explicitly synthetic stress environment.

```powershell
cleo calibrate --train data/public/deribit_incremental_book_L2_2020-04-01_ETH-PERPETUAL.csv.gz --test data/public/deribit_incremental_book_L2_2020-05-01_ETH-PERPETUAL.csv.gz --out results/calibration
```

Without `--validation`, the development day is split 60/20/20 into ordered
training, validation and internal-test rows; the second file is an external
later-date test. With `--validation PATH`, three distinct chronological files are
used. Boundaries purge a one-second outcome horizon and a 60-second embargo;
return calculations reset at boundaries. Four expanding walk-forward folds run
inside training. No model update or selection uses validation/test results.

Scorecards bind the exact model and holdout hashes. They measure normalized
quantile distance and coverage of the training 95% range. PASS/WARNING/FAIL
thresholds are predeclared diagnostic rules, not p-values or evidence of alpha.
The generated samples, frozen model, full plan, archived source, runtime and
reports are saved together. A model can fail while the study completes correctly.

## Registered execution stress family

```powershell
cleo stress --config configs/robustness.yaml --out results/stress
```

The preset executes 300 episodes: five agents × ten common seeds × six profiles.
Profiles cover reference conditions, reduced depth/refill, aggressive background
flow, slower messages, combined stresses and deliberate liquidity exhaustion.
The complete design is written before execution. Every attempted, failed,
invalid and unstarted outcome is retained with child-run audit logs.

Paired comparisons use a single Holm family across all profiles/candidates.
Any invalid, failed or missing planned outcome withholds the entire family's
inference. Child provenance and any PPO checkpoint must remain identical across
the family. A completed stress suite may contain failed economic outcomes: this
is distinct from an interrupted or corrupt study. Static parameter stresses do
not implement exchange outages, feed dissemination latency or intraday crashes.

## Portfolio accounting and risk

```powershell
cleo portfolio --config configs/portfolio_example.json --out results/portfolio
```

The module exposes funded **linear-instrument** accounting across currencies.
Instrument specifications define quote currency, multiplier, tick and lot size;
FX is base currency per one quote-currency unit. Filled positions, actual cash,
fees, signed exposure and pending reservations reconcile with Decimal arithmetic.
An optional margin-rate overlay limits capacity; it is not a futures settlement
or options pricing engine.

Pretrade checks include gross/net exposure, leverage, instrument notional,
concentration, margin capacity and child notional. Drawdown/daily-loss breaches
latch a kill switch. Explicit reduce-only requests cannot flip positions or
worsen a breached exposure envelope. Pending buys and sells are independently
reserved; a hedge or cancellation request never prematurely releases capacity.
Missing, future or stale marks/FX reject new risk. Actual fills and cancellation
acknowledgements can still be ingested when marks are stale.

Joint scenarios shock instrument prices and FX, including foreign cash. Historical
mode additionally computes inverse-empirical-CDF VaR and expected shortfall with
fractional tail weights; all supplied observations must align. Synthetic shock
examples are labeled stress scenarios, not historical estimates. Scenario PnL
covers filled positions; outstanding order exposures remain separately disclosed.
There is no live routing, automatic liquidation, borrow or nonlinear derivative
settlement.

## Evidence integrity

```powershell
cleo verify PATH_TO_CALIBRATION_STRESS_OR_PORTFOLIO_STUDY
cleo report PATH_TO_CALIBRATION_STRESS_OR_PORTFOLIO_STUDY
```

Study roots contain a frozen plan, source archive, provenance, result and checksum
manifest. Verification includes nested child manifests and every archived file.
Normal synthetic experiment verification remains compatible with earlier sealed
config documents when newer optional defaults are added. Reproduction still
requires the originally recorded implementation/runtime; archived code is never
executed automatically.
