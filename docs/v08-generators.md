# v0.8 market generators

## Market core

`lob/v08/market` is an aggregate-queue order-book model. It is not a replacement for the frozen v0.5
matching engine, which is unchanged.

**Book representation.**
- Prices are integer ticks.
- Each side keeps 40 contiguous levels from its touch.
- Sizes are totals, as an aggregate-L2 observer would see them.
- Only the strategy's own orders carry identity. Each has an explicit queue position: background volume
  `ahead`, its own size, and volume behind.

**Book rules (each covered by `tests/test_v08_market.py`):**
- FIFO allocation within a price level.
- A depleted touch moves away, and the new deep level is drawn from the refill distribution.
- Additions inside the spread create a new touch.
- Background cancellations remove a contiguous block of background volume. Its position is uniform for the
  simulated ground truth; front or back is used for the bound assumptions.
- An optional hidden-replenishment knob exists. It is a sensitivity device, not a claim about hidden
  liquidity in real data.

**Bins.**
- The generator decides event counts for six marks per 100 ms bin: market buy/sell, add bid/ask,
  cancel bid/ask.
- A shared `BookModel` decides levels, sizes, cancel fractions and refill depth.
- Strategy messages are merged with background events in time order.
- Randomness is drawn per bin (`SeedSequence([seed, stream, bin])`), so runs are seed-addressable. Twin runs,
  with and without a strategy, share background draws until their states diverge.

## Plugin interface (`lob/v08/generators/base.py`)

Every generator exposes:
- `fit`, `initial_state`, `expected`, `draw`, `update`;
- `loglik` (the one-step predictive log-likelihood on recorded frames) and `score`;
- `simulate`/`sample`, `posterior_predictive`;
- `to_dict`/`from_dict` (versioned JSON);
- `capabilities` and `parameter_space`.

Custom generators register with `@register` or through the `cleolob.generators` entry-point group. The CLI
does not need editing.

## Families

| Name | Mechanism | Fit |
|---|---|---|
| `poisson` | homogeneous rates, with tied totals and side shares | bin means (MLE) |
| `hawkes` | discretized linear multivariate Hawkes. Branching is a scalar times a unit-radius shape and is capped at 0.95 | EM per mark; beta by likelihood grid |
| `queue_reactive` | log-rates linear in log touch queue plus a wide-spread indicator | Poisson GLM (IRLS) |
| `regime_switching` | hidden Markov regimes with Poisson emissions | Baum-Welch; forward-filter scoring |
| `state_dependent` | log-linear GLM on standardized book features | ridge IRLS |
| `mixture` | components with fixed-share switching; sequential Bayesian predictive | component MLE plus stacking weights |
| `ecology` | stylized agents: market maker, taker, noise, directional, cancellation-heavy, replenishment | method of moments |
| `endogenous` | wraps a base generator; strategy flow enters through impact kernels: depletion, transient impact, resilience, participation sensitivity | base refit; response parameters are registered settings |
| `scaled` | per-mark rate scaling (used for stress worlds) | not fitted |

**Interpretation limits.**
- The ecology agents are modelling devices. They are not behavioural identities of real participants.
- The endogenous response is a controlled synthetic mechanism. It makes no causal claim about real markets.
- Impact kernels (exponential, power-law, multi-timescale, in `lob/v08/impact.py`) are model components fitted
  to simulated responses. They are not causal real-market impact.

## Real-data anchoring

The registered worlds come from three steps:
1. Development fits on 24 ten-minute windows of the consumed ETH 2020-10-01 day.
2. ABC-SMC posteriors against calibration-day summaries (ETH 2020-11-01). The run is
   `results/v08/inference/posterior-2`.
3. Two disjoint sets of posterior draws: selection worlds and held-out worlds (`results/v08/inference/worlds`).

All of this is RETROSPECTIVE use of consumed data. The worlds are plausible simulator worlds, not validated
replicas of the market.
