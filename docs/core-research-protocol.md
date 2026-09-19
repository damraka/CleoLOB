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
nor a guaranteed worst-case bound. Unaffected paired comparisons remain available,
with unidentified tests retained as p=1 in the full planned Holm family.

## Real-data split and fidelity

The public Tardis Deribit top-five samples cover BTC-PERPETUAL and ETH-PERPETUAL
on July 1 (fit), August 1 (validation), and September 1, 2026 (fresh hold-out).
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

## Separate fitting, diagnostics and tests

The diagnostic positive-control grid increases parent quantity relative to
top-five depth and execution horizon. The selected cell must have measurable
TWAP–Random cost separation and priceable all-wait residuals. Independent
confirmation runs once; a failed confirmation does not trigger another search
on those seeds. AC estimation uses separate no-parent simulator paths and a
spread intercept in the executable-cost-versus-trade-rate regression. Its
temporary-impact estimate is local and conditional on the slice interval.

PPO uses five optimizer seeds and four separately trained penalty arms:
0, 5, 25 and 100 bps. All use 8,192 training steps per model, identical market,
fee, settlement, observation and timing contracts, and disjoint train, diagnostic
and final market seeds. No final-test checkpoint selection is allowed. Saved
checkpoints and their hashes are retained. This budget does not certify policy
convergence; learning curves and training failures must remain visible. Multiple
training runs and separate evaluation environments follow the
[SB3 evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).

Diagnostic paired variance determines the final market-seed count for a 0.5 bps
minimum detectable effect at 80% target power, subject to a registered resource
cap. Any power shortfall is reported. Uncertainty resamples both optimizer seeds
and common market seeds; repeated policies on the same market are not counted
as independent observations. Primary and multiplicity-adjusted ablation intervals
are reported separately.

## Scope freeze

Portfolio/FX, new asset classes and additional product interfaces are frozen.
The active work is simulation validity and the execution comparison. Prior
Streamlit, flat evaluation and 100-seed artifacts are archived in `legacy/`.
Earlier sealed research evidence remains unchanged and explicitly historical.
