# v0.6 realism framework (M1–M4)

**One measurement operator.** History and simulation pass through the same code:
- `lob.v06.tape`: 100 ms causal samples of the top 10 displayed levels per side, plus
  aggressor-grouped trade events;
- `lob.v06.observables`: 85 components in nine families;
- `lob.v06.realism`: distances, family errors, the calibration objective and the bootstrap.

**Capability.** Every component needs only aggregate L2 plus aggressor-signed prints.
Additions and cancellations are *net* changes in displayed levels between 100 ms samples.
They are proxies, not order events.

These are `NOT_AVAILABLE`, with no substitution:
- modify intensity;
- individual order size;
- queue position;
- hidden liquidity.

**Historical validity.** A sample whose last update is older than 5 s, or whose book is
crossed, is invalid. A 600 s block with less than 50% valid samples is skipped and listed.

## Sketches

Each 600 s block becomes additive sufficient statistics. The bins are frozen from development
data (256 bins in a transformed space, overflow kept at its mean). The statistics are:
- histograms;
- moment sums for correlations and autocorrelations;
- event-state transition counts;
- exceedance counts and Hill sums.

Pooling over blocks or seeds is an exact sum, and so is a block bootstrap.

Rank-type dependence uses a Pearson correlation after transforming each variable by its
**development** marginal (mid-rank PIT). It is a Spearman-type statistic with a fixed
reference, not a sample Spearman.

## Distances and errors

| Kind | Primary error | Also reported |
|---|---|---|
| distribution | W1 on the transformed scale ÷ development IQR (floors: 1 tick for spread/returns, ¼ tick for microprice, 0.1 log units otherwise) | KS, energy distance 2∫(F−G)², JSD (base 2, frozen bins), 5/25/50/75/95% and 1/99% quantile errors |
| correlation, autocorrelation, probability | absolute difference | — |
| rate, Fano factor, exceedance probability, Hill index | \|log ratio\| (floors 1e-4 or 1e-3) | — |
| transition structure | total variation | per-row TV |

Every component error is capped at 10. A **family error** is the mean of its evaluable
component errors; the maximum is also reported. A component is `NOT_EVALUABLE` when either
side has fewer samples than its minimum. It is never imputed.

The **calibration objective** is the mean over families of (family error ÷ development
scale). Each development scale is the family error between the first and second 12 hours of
the development day, with a floor of 0.05. The objective is an optimization target, not a
validity score. The scorecard always reports the families separately.

## Sealed development references (`results/v06/m1/design`, ledger `m1-observables-design`)

| Family | Objective scale (dev halves) | Equivalence margin (Apr vs May, real vs real) |
|---|---|---|
| book state | 0.198 | 0.640 |
| event activity | 0.321 | 0.429 |
| sizes | 0.054 | 0.085 |
| returns | 0.274 | 0.320 |
| tail | 1.085 | 0.946 |
| temporal | 0.080 | 0.076 |
| event process | 0.264 | 0.198 |
| dependence | 0.106 | 0.083 |
| resilience | 0.118 | 0.075 |

- The scaled real-versus-real objective (development day vs selection day) is **1.25**. This
  is the yardstick: a simulator that reaches it is as close to the development day as the next
  real month is.
- The lot scale (243.6875 native units per lot) is identical to v0.5's. The design run is
  deterministic: a pilot and the registered run gave byte-identical margins.

## Equivalence

Family errors are nonnegative, so the lower TOST bound holds trivially. Equivalence is a
one-sided test against the sealed margin, at α = 0.05/9 per dataset and model:
- `EQUIVALENT_WITHIN_MARGIN` if the one-sided upper bound is below the margin;
- `FAILED_MARGIN` if the lower bound is above the margin;
- `NOT_EVALUABLE` if the error is missing;
- `NOT_ESTABLISHED` otherwise.

`NOT_ESTABLISHED` is never read as equivalence.

## Observable reference (generated from `lob.v06.observables.registry()`)

Every row also records:
- the required capability: aggregate L2 + trade prints;
- the sampling procedure;
- the invalidity rule: fewer than `min_n` samples on either side, an undefined statistic, or
  a missing development reference makes the component `NOT_EVALUABLE`.

### book state

| Component | Definition | Units | Min n | Error |
|---|---|---|---|---|
| `spread` | (best ask - best bid) / mid on the 1 s grid | bps | 50 | w1 |
| `depth_l1` | best-level quantity per side | log lots | 50 | w1 |
| `depth5` | top-5 side depth | log lots | 50 | w1 |
| `depth10` | top-10 side depth | log lots | 50 | w1 |
| `side_asymmetry` | log(bid top-5 / ask top-5) | log ratio | 50 | w1 |
| `imbalance5` | (bid5 - ask5) / (bid5 + ask5) | fraction | 50 | w1 |
| `concentration` | (L1 bid + L1 ask) / (top-10 bid + top-10 ask) | fraction | 50 | w1 |
| `book_slope` | log(top-5 side depth / L1 side depth) | log ratio | 50 | w1 |
| `level_gap` | price gap between consecutive displayed levels 1-5 per side | ticks | 50 | w1 |
| `microprice_deviation` | (L1-weighted microprice - mid) / mid | bps | 50 | w1 |

### event activity

| Component | Definition | Units | Min n | Error |
|---|---|---|---|---|
| `trade_count_10s` | aggressive events per 10 s window (>= 80% valid samples) | log1p count | 50 | w1 |
| `book_change_count_10s` | 100 ms steps with any top-10 change per 10 s window | log1p count | 50 | w1 |
| `add_count_10s` | net displayed additions per 10 s window | log1p count | 50 | w1 |
| `cancel_count_10s` | net displayed decreases not explained by prints per 10 s window | log1p count | 50 | w1 |
| `trade_interarrival` | positive gaps between aggressive events within a block | log s | 50 | w1 |

### sizes

| Component | Definition | Units | Min n | Error |
|---|---|---|---|---|
| `trade_size` | aggressive event size | log lots | 50 | w1 |
| `add_size` | net displayed addition size | log lots | 50 | w1 |
| `cancel_size` | net displayed decrease not explained by prints | log lots | 50 | w1 |

### returns

| Component | Definition | Units | Min n | Error |
|---|---|---|---|---|
| `return_1s` | log mid return over 1 s | bps | 50 | w1 |
| `return_10s` | non-overlapping 10 s mid returns | bps | 50 | w1 |
| `return_60s` | non-overlapping 60 s mid returns | bps | 50 | w1 |
| `realized_vol_60s` | std of 1 s returns in non-overlapping 60 s windows (>= 50 returns) | log1p bps | 50 | w1 |
| `nonzero_return_fraction` | fraction of nonzero 1 s returns | probability | 300 | abs |

### tail

| Component | Definition | Units | Min n | Error |
|---|---|---|---|---|
| `exceed_spread_p99` | P(spread_bps > development 99.0% quantile) | probability | 300 | logratio |
| `exceed_thin_depth5_p1` | P(depth5 < development 1.0% quantile) | probability | 300 | logratio |
| `exceed_vol_p99` | P(rv60_bps > development 99.0% quantile) | probability | 30 | logratio |
| `exceed_trade_size_p99` | P(trade_size > development 99.0% quantile) | probability | 50 | logratio |
| `exceed_cancel_burst_p99` | P(cancel_count_10s > development 99.0% quantile) | probability | 60 | logratio |
| `exceed_imbalance_p99` | P(abs_imbalance5 > development 99.0% quantile) | probability | 300 | logratio |
| `exceed_return_10s_p999` | large 10 s price moves (jumps) | probability | 300 | logratio |
| `tailq_spread` | 1%/99% quantile error of spread | normalized | 50 | tailq |
| `tailq_depth5` | 1%/99% quantile error of depth5 | normalized | 50 | tailq |
| `tailq_trade_size` | 1%/99% quantile error of trade_size | normalized | 50 | tailq |
| `tailq_return_10s` | 1%/99% quantile error of return_10s | normalized | 50 | tailq |
| `tailq_realized_vol` | 1%/99% quantile error of realized_vol_60s | normalized | 50 | tailq |
| `hill_trade_size` | Hill index above the development 95th percentile | tail index | 50 | logratio |
| `hill_return_10s` | abs_r10_bps | tail index | 50 | logratio |

### temporal

| Component | Definition | Units | Min n | Error |
|---|---|---|---|---|
| `acf_return_l1` | lag-1 autocorrelation of r1 on the 1 s grid | correlation | 300 | abs |
| `acf_return_l2` | lag-2 autocorrelation of r1 on the 1 s grid | correlation | 300 | abs |
| `acf_return_l5` | lag-5 autocorrelation of r1 on the 1 s grid | correlation | 300 | abs |
| `acf_return_l10` | lag-10 autocorrelation of r1 on the 1 s grid | correlation | 300 | abs |
| `acf_abs_return_l1` | lag-1 autocorrelation of abs_r1 on the 1 s grid | correlation | 300 | abs |
| `acf_abs_return_l5` | lag-5 autocorrelation of abs_r1 on the 1 s grid | correlation | 300 | abs |
| `acf_abs_return_l10` | lag-10 autocorrelation of abs_r1 on the 1 s grid | correlation | 300 | abs |
| `acf_abs_return_l30` | lag-30 autocorrelation of abs_r1 on the 1 s grid | correlation | 300 | abs |
| `acf_spread_l1` | lag-1 autocorrelation of spread on the 1 s grid | correlation | 300 | abs |
| `acf_spread_l10` | lag-10 autocorrelation of spread on the 1 s grid | correlation | 300 | abs |
| `acf_depth5_l1` | lag-1 autocorrelation of depth5 on the 1 s grid | correlation | 300 | abs |
| `acf_depth5_l10` | lag-10 autocorrelation of depth5 on the 1 s grid | correlation | 300 | abs |
| `acf_depth5_l60` | lag-60 autocorrelation of depth5 on the 1 s grid | correlation | 300 | abs |
| `acf_imbalance_l1` | lag-1 autocorrelation of imbalance on the 1 s grid | correlation | 300 | abs |
| `acf_imbalance_l10` | lag-10 autocorrelation of imbalance on the 1 s grid | correlation | 300 | abs |
| `acf_imbalance_l60` | lag-60 autocorrelation of imbalance on the 1 s grid | correlation | 300 | abs |
| `acf_signed_volume_10s_l1` | lag-1 autocorrelation of flow10 on the 10 s grid | correlation | 60 | abs |
| `acf_signed_volume_10s_l2` | lag-2 autocorrelation of flow10 on the 10 s grid | correlation | 60 | abs |
| `acf_signed_volume_10s_l5` | lag-5 autocorrelation of flow10 on the 10 s grid | correlation | 60 | abs |
| `acf_event_count_10s_l1` | lag-1 autocorrelation of count10 on the 10 s grid | correlation | 60 | abs |
| `acf_event_count_10s_l6` | lag-6 autocorrelation of count10 on the 10 s grid | correlation | 60 | abs |
| `pacf_return_l2` | lag-2 partial autocorrelation of 1 s returns | partial correlation | 300 | abs |
| `tight_spread_duration` | completed runs of spread <= 1.5 ticks on the 100 ms grid | log s | 50 | w1 |

### event process

| Component | Definition | Units | Min n | Error |
|---|---|---|---|---|
| `event_state_proportions` | total variation between event-state occupancy distributions | TV distance | 300 | tv |
| `event_transition_rows` | occupancy-weighted total variation between transition rows | weighted TV | 300 | rows |
| `p_cancel_after_trade` | P(next 100 ms state = cancel / current state = trade) | probability | 30 | abs |
| `p_trade_after_cancel` | P(next 100 ms state = trade / current state = cancel) | probability | 30 | abs |
| `fano_trade_1s` | Fano factor (variance / mean) of trade_count_1s | var/mean | 300 | logratio |
| `fano_trade_10s` | Fano factor (variance / mean) of trade_count_10s | var/mean | 30 | logratio |
| `burstiness` | trade_interarrival_s | (sd - mean)/(sd + mean) | 50 | abs |
| `depletion_hazard` | best level moved away or L1 fell below 50% within 1 s | probability per s | 300 | logratio |
| `replenishment_probability` | best level back to the pre-depletion price with >= 90% of its quantity 1 s after depletion | probability | 30 | abs |
| `cross_excitation` | corr(trade count in second t, net cancellation count in second t+1); descriptive only | correlation | 300 | abs |

### dependence

| Component | Definition | Units | Min n | Error |
|---|---|---|---|---|
| `dep_volatility_spread` | rank correlation of w_rv and w_spread | rank correlation | 30 | abs |
| `dep_volatility_depth` | rank correlation of w_rv and w_depth | rank correlation | 30 | abs |
| `dep_activity_volatility` | rank correlation of w_count and w_rv | rank correlation | 30 | abs |
| `dep_cancellation_volatility` | rank correlation of w_cancel and w_rv | rank correlation | 30 | abs |
| `dep_imbalance_spread` | rank correlation of w_absimb and w_spread | rank correlation | 30 | abs |
| `dep_imbalance_next_return` | rank correlation of imbalance and next_r1 | rank correlation | 300 | abs |
| `dep_microprice_next_return` | rank correlation of microdev and next_r1 | rank correlation | 300 | abs |
| `dep_flow_return` | rank correlation of flow1 and r1 | rank correlation | 300 | abs |
| `dep_depth_next_abs_return` | rank correlation of depth5_total and next_abs_r1 | rank correlation | 300 | abs |
| `cond_return_by_imbalance` | mean next 1 s return per development imbalance quintile | development return sd | 30 | cond |
| `mi_imbalance_next_sign` | plug-in mutual information of imbalance quintile and next-return sign (5 x 3 table) | nats | 300 | mi_logratio |

### resilience

| Component | Definition | Units | Min n | Error |
|---|---|---|---|---|
| `depth_recovery_time` | time for the hit side's L1 to regain 90% after an event removing >= 50% (30 s censoring) | log s | 50 | w1 |
| `recovered_fraction` | recovered | probability | 30 | abs |
| `spread_recovery_time` | completed runs of spread > 1.5 ticks | log s | 50 | w1 |
| `depth5_recovery_ratio_5s` | log(top-5 hit-side depth 5 s after / before a depleting event) | log ratio | 50 | w1 |
