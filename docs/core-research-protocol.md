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

The original three-cell control grid failed and remains archived. A subsequent
execution-only amendment registered its finite search order and additional
capacity checks before running them; it did not refit the physical simulator.
The selected setting sells 1,714 lots over 240 seconds, with a 5-second policy
decision interval and a 30-second warmup. All 384 fixed-action/random capacity
preflight episodes were valid. One independent confirmation on seeds 48000–48063
passed: Random minus TWAP was −0.9591 bps, paired 95% CI [−1.1953, −0.7155],
and both all-wait and all-market controls remained priceable. This establishes
execution discrimination in that synthetic setting, not PPO performance or
historical fidelity. The amendment and rejected settings are retained in
[`execution-amendment/`](../examples/studies/core/execution-controls/execution-amendment/).

The final no-parent AC fit used separate seeds 46000–46015 and the baseline's
12-second child interval. It estimated temporary impact η=5.51834×10⁻⁵ currency
seconds per lot and volatility σ=0.00438465 currency per square-root second;
the spread-intercept fit had R²=0.5013 and 92.94% probe coverage. The policy's
5-second decision interval and AC's 20-slice schedule serve different roles.
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

## Execution chronology and replication

The fixed design was already executed in Linux CI runs
[35649272649](https://github.com/damraka/CleoLOB/actions/runs/35649272649) and
[35649544066](https://github.com/damraka/CleoLOB/actions/runs/35649544066) before
the durable local study was registered on September 22. Those workflow runs
did not upload their study directories. Their printed outcome summaries and
verification logs were recovered separately; these are not substitutes for
checkpoint evidence. Published log exports omit the previous branch label and
retain original/export hashes in
[`export-provenance.json`](../examples/studies/core/ci-prior-runs/export-provenance.json).

The initial local serial attempt, `ppo-final-20260921`, was stopped after one
completed model and an interrupted second fit. That attempt remains preserved.
The retained study, `ppo-final-20260922`, has a new source snapshot and registration
for isolated process scheduling, with eight workers and one Torch thread each.
Its five optimizer seeds, four penalties, 8,192-step budget, simulator and
evaluation rules did not change. The computational amendment was recorded in
[`PARALLEL_REPLICATION.json`](../examples/studies/core/PARALLEL_REPLICATION.json)
before the new models were fitted. Regression checks compare policy tensors and
training traces between serial and spawned-process training. The first actual
8,192-step model also matched the preserved serial fit exactly in every policy
tensor and the full training trace; checkpoint ZIP bytes can differ because of
serialization metadata. The [identity check](../examples/studies/core/parallel-identity.json)
is computational validation, not an additional independent optimizer seed.

The local run is a computational replication of that fixed design, not the
first exposure of those test seeds or a new untouched holdout. Its source,
parameters, training budget and seed blocks remain unchanged after the prior
runs were discovered. Earlier CI runs are reported as provenance and are not
pooled as extra independent optimizer or market seeds. The reported intervals
describe the five optimizer seeds and common market paths in the retained local
run. This chronology is separate from the real-data September holdout, which
was also already consumed in the frozen calibration assessment.

## Scope freeze

Portfolio/FX, new asset classes and additional product interfaces are frozen.
The active work is simulation validity and the execution comparison. Prior
Streamlit, flat evaluation and 100-seed artifacts are archived in `legacy/`.
Earlier sealed research evidence remains unchanged and explicitly historical.
