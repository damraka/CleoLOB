# Executed priority-workflow validation — 2026-09-19

The four requested workflows are implemented and exercised: empirical observable
calibration, chronological/stress testing, delayed-order settlement and linear
portfolio risk. **521 tests passed** (two existing Gymnasium warnings), together
with Ruff, Python compilation and frontend JavaScript syntax checks.

All three saved study roots passed checksum verification, including archived
source and nested child manifests. Their recorded implementation hashes matched
the code used for this batch. This establishes software/evidence integrity, not
successful market-model generalization or profitable execution.

## Empirical calibration: completed; model generalization rejected

The data are the previously verified Tardis Deribit ETH-PERPETUAL samples from
2020-04-01 and 2020-05-01. A third June sample could not be downloaded because
`datasets.tardis.dev` failed DNS resolution, including outside the sandbox. Both
existing days had previously been inspected for reconstruction and descriptive
statistics. May is a chronological holdout from fitting, not a never-inspected
independent dataset. No new prices or outcomes were fabricated.

The model fits April's first 60% only, with a 60-second embargo and one-second
outcome purge at boundaries. It stays frozen throughout validation and testing.
Depth/quantity retain the native feed units. The model is an IID joint empirical
observable baseline, with no queue/fill model or temporal dependence claim.

| Partition | Samples | Diagnostic result |
|---|---:|---|
| April training | 51,778 | Model fitted once |
| April validation | 17,219 | WARNING |
| April internal test | 17,279 | FAIL |
| May external chronological test | 86,399 | FAIL |

Four expanding walk-forward folds within training returned WARNING, FAIL,
WARNING and WARNING. All model and holdout hashes are recorded. A deterministic
1,000-row generated observable sample is also saved, explicitly synthetic.

| May diagnostic | Normalized quantile distance | Coverage of training 95% range | Result |
|---|---:|---:|---|
| Spread, bps | 1.28158 | 20.353% | FAIL |
| Top-five bid depth | 1.24617 | 34.355% | FAIL |
| Top-five ask depth | 1.70298 | 32.603% | FAIL |
| Imbalance | 0.18562 | 83.150% | WARNING |
| One-second log return | 0.34951 | 88.385% | WARNING |

These predeclared distance/coverage rules are diagnostics, not significance tests.
The fitted baseline does not establish a stable market model across these dates.
The test data were not used to retune it after this result.

[Calibration report](calibration/20260919T012858-2ce969ab5e22/report.md) ·
[Full scorecards](calibration/20260919T012858-2ce969ab5e22/result.json) ·
[Frozen plan](calibration/20260919T012858-2ce969ab5e22/plan.json)

## Execution stress: 300/300 episodes recorded

Five agents (TWAP, VWAP, POV, AC and heuristic), ten paired seeds and six frozen
profiles ran with 600-share parents and a five-second decision horizon. These
are synthetic FIFO parameter stresses; they are not historical fill estimates.

| Profile | VALID | WARNING | INVALID |
|---|---:|---:|---:|
| Reference | 50 | 0 | 0 |
| Thin depth/refill | 50 | 0 | 0 |
| Aggressive flow | 50 | 0 | 0 |
| Slow messages | 50 | 0 | 0 |
| Combined stresses | 4 | 3 | 43 |
| Deliberate liquidity exhaustion | 0 | 0 | 50 |
| **Total** | **204** | **3** | **93** |

All 300 episodes completed order settlement with no live strategy orders left.
**827 shares filled late across 17 episodes** and were reconciled with fees and
inventory. Maximum observed settlement duration was **1.08 seconds**.

All 93 invalid economic outcomes had **insufficient terminal depth** for valuing
the remaining parent quantity. No settlement timeout or execution exception
occurred. All 24 planned candidate/profile comparisons were withheld as one
family; invalid scenarios were not excluded to obtain favorable significance.

[Stress report](stress/20260919T012858-55bb892c75e4/report.md) ·
[Complete result](stress/20260919T012858-55bb892c75e4/result.json) ·
[All episodes](stress/20260919T012858-55bb892c75e4/episodes.csv)

## Portfolio risk: reconciled offline assessment

The configured synthetic USD/EUR linear portfolio exercised independent buy/sell
reservations, fills during pending cancellations, explicit terminal acknowledgements,
gross/symbol limit rejection, loss kill switches and a constrained reduce-only fill.

- Final cash: **USD 102,227.50**; marked equity: **USD 98,677.50**; fees: **USD 2.50**.
- Cash/position/fill reconciliation: **true**.
- Daily-loss and drawdown kill switches: latched; further risk-increasing order rejected.
- One deliberately partial reduce-only order remains reserved (15 shares, USD 0.50 fee allowance).
- Three joint price/FX shocks ran; worst filled-position scenario loss: **USD 4,309**.
- These are synthetic shocks. Historical VaR/ES was not mislabeled as measured
  historical performance; its fractional-tail formula has hand-derived tests.

[Portfolio result and audit](portfolio/20260919T012859-6cd624b82611/result.json) ·
[Resolved portfolio plan](portfolio/20260919T012859-6cd624b82611/plan.json)

## Run and verify

```powershell
cleo calibrate --train data/public/deribit_incremental_book_L2_2020-04-01_ETH-PERPETUAL.csv.gz --test data/public/deribit_incremental_book_L2_2020-05-01_ETH-PERPETUAL.csv.gz --out results/calibration
cleo stress --config configs/robustness.yaml --out results/stress
cleo portfolio --config configs/portfolio_example.json --out results/portfolio
cleo verify PATH_TO_STUDY
cleo report PATH_TO_STUDY
```

The [workflow documentation](../../../docs/validation-and-risk.md) defines APIs,
units and assumptions. Remaining platform work includes latent order-flow and
counterfactual execution calibration, fresh independent market datasets,
nonlinear derivatives risk, feed latency and dynamic outage/crash models.
Successful software tests do not remove the observed calibration failures.
