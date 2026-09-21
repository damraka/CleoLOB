# Execution research protocol

This document records the scope of the September 2026 execution study. Numeric
settings, source identities and run-specific timestamps are in the JSON plans
under `examples/studies/core/`; those artifacts take precedence over this guide.

## Question and interpretation

The primary question is whether PPO with no additional noncompletion penalty
reduces mean net implementation shortfall relative to an Almgren–Chriss schedule
whose impact and volatility parameters were estimated from the same simulator.
The question is conditional on a fixed training budget and market configuration.
A simulator fidelity failure prevents generalizing a synthetic result to the
historical market, even when its comparison interval excludes zero.

Economic cost includes actual fill shortfall and fees plus explicitly reported
visible-book valuation of any priceable residual. A completion penalty is an
optimization term and is not added to the economic endpoint. Unpriced residuals
remain INVALID in the raw data. Residual-price sensitivity assumes an adverse
100 or 500 bps price relative to arrival; it is neither an observed liquidation
nor a guaranteed worst-case bound. Unaffected paired comparisons remain available.
The PPO study retains each arm's failures, labels incomplete-pair intervals as
descriptive, and uses the prespecified Bonferroni family below. The earlier stress
study separately retains unidentified tests as p=1 in its planned Holm family.

## Real-data split and fidelity

The public Tardis Deribit top-five samples cover BTC-PERPETUAL and ETH-PERPETUAL
on July 1 (fit), August 1 (validation), and September 1, 2026 (holdout).
The six-file identity and hold-out rule were written before downloading.
Integrity checks may read compressed bytes; held-out observations are first
extracted after the simulator parameters and pass thresholds are frozen.

Each instrument's training price and depth scales map observations to normalized
simulator units. These units do not represent funded inverse perpetual positions.
Calibration compares spread, bid/ask depth, book imbalance, quote OFI, return
volatility and the visible-book walk curve. Snapshot OFI and static book walks
do not identify hidden liquidity, queue priorities or causal permanent impact.
All feature families must meet the predeclared gate; failed families are retained.
The source provides first-day samples without an API key; see the
[provider's Deribit coverage documentation](https://docs.tardis.dev/historical-data-details/deribit).

The frozen external assessment is complete: all six cohort/phase assessments
failed, covering ETH, BTC and pooled fits on both August and September. The four
distinct external files contain 7,461,728 published snapshots and 345,595 causal
one-second samples. Pooled rows reuse the individual cohorts. All September
cohorts also failed the 95% valid-observation requirement; the moment-family
error threshold remained 0.60. Models were not retuned after external evaluation.
September is now consumed holdout evidence, not an unopened dataset available
for another confirmation. See the [external results](../examples/studies/core/calibration/EXTERNAL_RESULTS.md).

## Separate fitting, diagnostics and tests

The diagnostic positive-control grid increases parent quantity relative to
top-five depth and execution horizon. The selected cell must have measurable
TWAP–Random cost separation and valid all-wait/all-market capacity controls. Independent
confirmation runs once; a failed confirmation does not trigger another search
on those seeds. AC estimation uses separate no-parent simulator paths and a
spread intercept in the executable-cost-versus-trade-rate regression. Its
temporary-impact estimate is local and conditional on the slice interval.
The primary comparator is risk-neutral AC (risk aversion zero), whose analytical
schedule equals TWAP. Fitted impact and volatility do not make this schedule
front-loaded. The additional risk-sensitive AC comparator fixes κT=1, using
`risk_aversion = temp_impact / (sigma * horizon)^2`, before final evaluation.

PPO uses five optimizer seeds and four separately trained penalty arms:
0, 5, 25 and 100 bps. All use 8,192 training steps per model, identical market,
fee, settlement, observation and timing contracts, and disjoint train, diagnostic
and final market seeds. No final-test checkpoint selection is allowed. Saved
checkpoints and their hashes are retained. This budget does not certify policy
convergence; learning curves and training failures must remain visible. Multiple
training runs and separate evaluation environments follow the
[SB3 evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).
The 20 models therefore use 163,840 optimization steps in total. An INVALID
terminal training episode aborts the run and leaves a failure artifact: missing
residual cost must never become a zero-cost training target. Completed models
may be resumed only after their checkpoint and trace hashes verify.

Diagnostic paired variance determines the final market-seed count for a 0.5 bps
minimum detectable effect at 80% target power, subject to a registered resource
cap. Any power shortfall is reported. Uncertainty resamples both optimizer seeds
and common market seeds; repeated policies on the same market are not counted
as independent observations. The primary 0 bps penalty arm has a 95% interval.
The family of four penalty arms plus the fixed risk-sensitive AC comparison
uses 99% intervals (Bonferroni, 0.05 / 5). The 100/500 bps residual-price
sensitivities use the same family confidence level and remain labeled scenario
assumptions. Diagnostic seeds are 61000–61019; final seeds start at 71000 and
their count is locked before the first final episode. Training markets occupy
the separate [1000000, 2000000) seed range; optimizer seeds are 81001–81005.
The crossed variance estimate includes a training-seed variance floor, so an
80% target cannot always be reached by adding market paths. Report the cap and
estimated power rather than calling a capped design adequately powered.

Reports are derived from the sealed study without rerunning policies. They
include paired intervals, four penalty ablations, the fixed AC sensitivity,
learning curves, signed training reward components and the completion penalty's
absolute reward share. Final cost components and actual fill fractions keep
hypothetical residual valuation separate from executed quantity. Plotly reports
are written outside the sealed study folder so that rendering does not change
the evidence manifest.

## Scope freeze

Portfolio/FX, new asset classes and additional product interfaces are frozen.
The active work is simulation validity and the execution comparison. Prior
Streamlit, flat evaluation and 100-seed artifacts are archived in `legacy/`.
Earlier sealed research evidence remains unchanged and explicitly historical.
