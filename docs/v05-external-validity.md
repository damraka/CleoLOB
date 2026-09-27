# v0.5 M6 — multi-regime and cross-instrument external validity

Frozen in `configs/v05/m6-design.json`. The development-only thresholds were sealed in ledger
entry 38 before any holdout access.

## Regimes

The development period (April 2020 ETH, 287 five-minute blocks) is split at the block
median for each of six dimensions:
- volatility
- spread
- depth
- trading activity
- mean |imbalance|
- absolute trend

A block is a **stress** block when both volatility and spread exceed their development 90th
percentiles; that labels 15 of 287 development blocks. Labels describe observable conditions
only, and none was chosen from any outcome.

A conclusion is **regime-robust** only if it holds in every label with at least 6 blocks
(30 minutes). Labels with fewer blocks are NOT_AVAILABLE.

## Layers

| Layer | Dataset | Status |
|---|---|---|
| Unseen period, same instrument | ETH 2020-07-01 (fresh) | evaluated |
| Different instrument, same venue | BTC 2020-07-01 (fresh) | evaluated, using the ETH development scale |
| Different regime | every holdout, per label | evaluated |
| Different venue | Bitstamp btcusd vs Deribit perpetuals | **refused** by `lob.dataset_registry.comparable`: different capability levels, instrument types and sequence semantics |
| Retrospective | ETH 2020-06-01 (consumed by v0.4) | evaluated, labeled retrospective |

## Distribution shift

In all 287 blocks of both June and July ETH, depth sits *below* the development median. Both
periods lie in a depth regime that development data covered only half the time.

| Holdout | Spread | Volatility | Stress |
|---|---|---|---|
| June ETH | high in 10 of 287 blocks | high in 155 of 287 | 8 blocks |
| July ETH | low throughout | high in 35 of 287 | none |
| July BTC | low throughout | high in 4 of 287 (NOT_AVAILABLE) | none |

## Results

| Holdout | Regime labels evaluated | Selected passes the M3 gate in every regime | Selected beats the control in every regime | M4 H4 in any regime |
|---|---:|---|---|---|
| June ETH (retrospective) | 13 | no (FAILED in all 13) | **yes** (13/13) | NOT_ESTABLISHED in all |
| July ETH (external) | 11 | no (FAILED in all 11) | **yes** (11/11) | NOT_ESTABLISHED in all |
| July BTC (cross-instrument) | 11 (+1 NOT_AVAILABLE) | no | **no**: loses in `activity=low` and `imbalance=low` | NOT_ESTABLISHED in all |

On ETH, the selected model's E3 ranges from:
- **June:** 0.94 in the low-imbalance label to 1.80 in stress;
- **July:** 1.25 in low volatility to 1.94 in high volatility and high trend.

These are always below the control, and always above the gate.

## Conclusions

The only regime-robust conclusion is the relative one: calibration v2 beats the v0.4 class
in every regime label of the unseen same-instrument period. No absolute calibration and no
impact-agreement conclusion survives any regime, and neither transfers to BTC. The
cross-venue layer is refused, not approximated.
