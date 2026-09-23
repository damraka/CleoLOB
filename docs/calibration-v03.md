# v0.3 observable calibration and chronological generalization

**IMPLEMENTED / TESTED:** `lob.generalization` evaluates four registered candidates:
an IID joint empirical observable model and a two-state spread Markov model, each
with expanding and rolling training windows. This extends observable calibration;
it does not calibrate the execution simulator or establish valid historical fills.
The previous simulator calibration (`lob.zi_calibration`) remains separate.

**FAILED — historical evidence preserved:** the April/May 2020 empirical study
retains its WARNING and FAIL outcomes in
[`examples/studies/validation/README.md`](../examples/studies/validation/README.md).
The July/August/September 2026 simulator study retains all six validation/holdout
FAIL outcomes in
[`EXTERNAL_RESULTS.md`](../examples/studies/core/calibration/EXTERNAL_RESULTS.md).
May 2020 and September 2026 are consumed evidence. They cannot become new untouched
holdouts for v0.3. No old artifact, model or threshold is changed by this workflow.

## Registered sequence

Run from the repository root using an installed package or a development environment:

```sh
python -m lob.generalization --config configs/v03-calibration.json --out results/v03/calibration-smoke
python -m lob.generalization --out results/v03/calibration-smoke --verify
```

Each output directory must be new. The program writes the full plan, observable
catalog, fixed gates, candidate design, dataset descriptors and runtime/source
provenance before loading development data. It then:

1. Splits development into a selection period and a strictly later internal test.
2. Fits each family on four chronological train/validation folds. Validation
   blocks do not overlap. Expanding training grows; rolling training retains at
   most 360 rows. Training ends strictly before each validation boundary minus
   the two-second embargo. No future target is fitted; outcome horizon is zero.
3. Chooses the lowest mean validation loss across the four folds, with a lexical
   candidate tie-break. Any fit/evaluation error in a candidate fold disqualifies
   that candidate; the error and completed failed diagnostics stay in the output.
4. Refits the selected family on train plus validation, using the selected window
   rule and an embargo before internal test. It saves the model, selection and a
   SHA-256 seal of all decision inputs **before scoring internal test**.
5. Scores internal test, then loads/generates external observations for the first
   time in that run. It verifies the seal before and after external loading and
   evaluation. Neither internal nor external score can change family, window,
   thresholds, quantity scales, pools or transition rates.

Temporal observables are reset independently at every split. A trailing window
never carries information from the preceding partition. This conservative reset
removes ten initial volatility observations; all missingness is explicit. The
two-second gap is a fixed smoke-study setting, not a universal market-data
embargo. Real applications must predeclare a gap appropriate to data capture and
any downstream outcome horizon before evaluation.

The in-process seal and final manifest detect accidental edits, not dishonesty
by an actor able to replace every hash. The repository commit is the external
record of the registered design. Runtime provenance includes commit, dirty flag,
Python/OS/package versions and hashes of implementation dependencies without
personal absolute paths. `verify_study` verifies artifacts; it does not claim
independent replication or replay an archived interpreter.

## Model families and selection loss

The IID baseline resamples complete contemporaneous spread, side depths,
imbalance and return rows from a bounded train-only pool. It preserves joint
cross-sectional dependence but discards serial dependence. Its role is the same
as the existing empirical observable baseline; it does not simulate order flow.

The Markov family partitions training spread at the training median. It estimates
the first-order transition matrix from consecutive complete rows, with a fixed
one-count Laplace prior per transition, and resamples joint observations within
the current state. At least 16 training rows per state are required. The model
evolves autonomously during evaluation; it never conditions on the current target
spread, return, future state or external regime weights. State-conditioned joint
pools allow spread-associated return volatility and depth to change with regime.
This is a limited, interpretable temporal extension. It is not a Hawkes or
queue-reactive execution model. Fit pools are capped at 128 rows per state in the
smoke configuration, using one registered fit seed.

For each observable, distance is the mean absolute difference between 51 observed
and generated quantiles at probabilities 0.01 through 0.99, divided by the maximum
of training IQR, standard deviation, 1% of absolute mean, and `1e-12`. This is a
quantile-grid approximation to a normalized distribution distance. Coverage is
the fraction of finite target measurements within the generated central 95%
interval. Equal-weight selection loss is the mean of
`distance + max(0, 0.95 - coverage)` across the eight observables. Each simulated
seed has its own path boundaries; temporal estimators do not span seeds.

Training, validation, internal and external simulation seeds are frozen and
disjoint. All candidates use the same validation seed list. They are not
independent Monte Carlo replications of the market and no p-values are inferred
from these short autocorrelated series.

## Observable catalog

All estimators use the declared sample interval (one second in the fixture).
Invalid rows remain in the validity denominator. Measurements are NaN on invalid
rows; temporal values also become NaN across missing samples or time gaps. No
interpolation, forward-fill of observables or silent row deletion is performed.
Every observable has weight 1/8 in the calibration loss.
Distribution distances and generated-interval coverage use finite measurements;
the independent input-validity gate prevents missing books from being silently
treated as good coverage. Training summaries record the valid fraction and number
of complete joint observations used to estimate the pools.

| Observable | Definition / estimator | Units | Required finite history |
|---|---|---|---|
| Spread | `(ask - bid) / mid * 10000` | basis points | current valid book |
| Bid depth | sum of visible top-five bid quantities | native quantity | current valid book |
| Ask depth | sum of visible top-five ask quantities | native quantity | current valid book |
| Imbalance | `(bid depth - ask depth) / total depth` | dimensionless | current valid book |
| Return | `log(mid[t] / mid[t-1])` | log ratio per interval | two consecutive valid samples |
| RMS volatility proxy | square root of trailing mean of 10 squared returns | log ratio | 10 consecutive finite returns |
| Spread change | `spread[t] - spread[t-1]` | basis points | two consecutive valid samples |
| Relative visible depth change | `(depth[t] - depth[t-1]) / depth[t-1]` | dimensionless | two consecutive valid samples |

The volatility measure is an uncentered short-window RMS proxy, not an annualized
volatility estimate. Spread changes assess temporal transitions without asserting
an identified exchange event type. Visible depth changes combine additions,
cancellations, trades and movement of the top-five boundary. They do not identify
individual replenishment or cancellation intensity. Trade intensity, order-level
inter-arrival time, true event OFI and intraday seasonality are deliberately
excluded because the sampled fixture cannot support those estimands.

## Fixed diagnostic gates and attribution

These thresholds are constants registered in `plan.json` before fitting. They are
diagnostic tolerances and cannot be retuned through the study configuration.

| Quantity | WARNING | FAIL |
|---|---|---|
| Normalized quantile distance | greater than 0.25 | greater than 0.50 |
| Generated central-95% coverage | below 0.90 | below 0.75 |
| Valid input fraction | below 0.99 | below 0.90 |
| Finite target samples per observable | — | fewer than 32 |

An observable takes the worse distance/coverage status. A partition takes the
worst observable/data-quality status; the study takes the worst internal/external
status. A candidate with poor diagnostic fit may still be selected as the least
bad candidate; selection never turns FAIL into PASS. Missing finite observations
receive a fixed loss of 1,000,000 and a FAIL diagnostic.

Internal/external scorecards report distance, central coverage, absolute deviation
from nominal 95% coverage, and train-relative quantile drift. Validation reports
retain every fold's status/loss and the variance of fold losses.
Every completed validation fold also retains its per-observable distances,
coverage errors, train-relative drift and finite sample counts, including failed
and warning-level outcomes. Parameter
stability is standard deviation divided by mean absolute value across folds for
median spread, mean side depths, return standard deviation and spread transition
rates. Transition-rate stability is descriptive for the IID family, whose sampler
does not use the rates.

Failure attribution labels a large train-relative shift (`> 0.5`) as
`regime_shift_or_support_failure`; otherwise a failed observable is labeled
`within_regime_model_misspecification_or_sampling_error`. This is a heuristic
diagnostic, **not causal identification**. Missing/stale input is reported
separately. Discrete distributions can exceed nominal interval coverage.

## Evidence scope and real-data use

The default data are a legally redistributable synthetic two-state generator:
1,080 development observations from synthetic time zero, then 360 observations on
synthetic day two with a predeclared spread/return scale shift of 1.8 and inverse
depth shift. It is a smoke study of software behavior under known misspecification.
The test suite also exercises these fixture seeds. Thus an external partition is
untouched by fitting/selection within a run, but the repeated fixture is **not a
previously uninspected real-market holdout or independent generalization study**.
Any FAIL remains in its result. No positive synthetic result overrides prior
historical failures.

For new real data, `kind: "feature_csv"` loads separate development and external
canonical feature CSVs. Each descriptor must include portable repository-relative
`path`, precomputed `sha256`, exact `rows` and `start_us`; the global configuration
declares one exchange, symbol and regular sampling interval. Columns are
`timestamp_us`, boolean `valid`, `mid_price`, `spread_bps`, `bid_depth5`,
`ask_depth5`, `imbalance5`, and `log_return`. Use the existing causal
`lob.calibration.extract_features` to construct a table, retain native quantity
units, and explicitly reset the first return of each input file. Invalid samples
must have NaN measurements. Return and imbalance identities are checked. A source
hash, chronology or schema mismatch stops the run and preserves prior artifacts.

The runner reads and hash-checks the external file only after selection is sealed.
The external descriptor must be later than all development data plus embargo.
Researchers must additionally document raw-source provenance, file-to-instrument
identity, collection clock, permission to use the data, any earlier inspection and
the reason the new holdout is independent. File hashes alone cannot prove that a
researcher has never inspected a dataset. No new defensible historical external
dataset was available for this upgrade, so real-market evaluation of these new
families remains **PLANNED**, and real-market generalization is **NOT ESTABLISHED**.
