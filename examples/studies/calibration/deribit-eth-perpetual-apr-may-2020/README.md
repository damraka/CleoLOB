# Deribit ETH-PERPETUAL Chronological Calibration Study

This study evaluates the stability and out-of-sample behavior of an observable
model fitted on historical Deribit ETH-PERPETUAL Level-2 market data.

The model is fitted on the earlier portion of the April 1, 2020 session and
frozen before validation, internal testing, and the later-date May 1, 2020
evaluation.

## Methodology

The study uses:

- chronological rather than random splitting,
- disjoint train, validation, and internal-test samples,
- a 60-second embargo between splits,
- a one-second outcome horizon,
- four purged, expanding walk-forward folds,
- model parameters frozen before holdout evaluation,
- a separate later-date external test.

No parameters are updated after validation.

## Dataset

| Role | Exchange | Instrument | Date |
|---|---|---|---|
| Development (train / validation / internal test) | Deribit | ETH-PERPETUAL | 2020-04-01 |
| External test | Deribit | ETH-PERPETUAL | 2020-05-01 |

Raw market-data files are not committed to this repository.

## Split Results

| Split | Samples | Diagnostic Status |
|---|---:|---|
| Training | 51,778 | Fit |
| Validation | 17,219 | WARNING |
| Internal test | 17,279 | FAIL |
| Later-date May test | 86,399 | FAIL |

## Later-Date Distribution Drift

The May holdout differs substantially from the April training distribution,
particularly in quoted depth and spread.

| Observable | Train Mean | May Mean | Coverage | Normalized Distance | Status |
|---|---:|---:|---:|---:|---|
| Ask depth (top 5) | 254,266 | 109,161 | 32.6% | 1.703 | FAIL |
| Bid depth (top 5) | 233,046 | 90,984 | 34.4% | 1.246 | FAIL |
| Imbalance (top 5) | -0.0522 | -0.0823 | 83.2% | 0.186 | WARNING |
| Log return | ≈ 0 | ≈ 0 | 88.4% | 0.350 | WARNING |
| Spread | 4.071 bps | 2.977 bps | 20.4% | 1.282 | FAIL |

The failed later-date diagnostic is intentionally retained rather than
presented as a successful calibration.

## Walk-Forward Diagnostics

| Fold | Training Rows | Holdout Rows | Status |
|---|---:|---:|---|
| 1 | 25,828 | 6,471 | WARNING |
| 2 | 32,300 | 6,471 | FAIL |
| 3 | 38,772 | 6,471 | WARNING |
| 4 | 45,244 | 6,471 | WARNING |

### Fold-Level Observables

| Fold | Ask Depth | Bid Depth | Imbalance | Log Return | Spread |
|---|---|---|---|---|---|
| 1 | WARNING | WARNING | WARNING | PASS | WARNING |
| 2 | FAIL | WARNING | FAIL | PASS | PASS |
| 3 | WARNING | WARNING | WARNING | PASS | PASS |
| 4 | PASS | PASS | PASS | PASS | WARNING |

## Reproduction

Download the April and May incremental L2 samples into `data/public/`, then run:

```bash
cleo calibrate \
  --train data/public/deribit_incremental_book_L2_2020-04-01_ETH-PERPETUAL.csv.gz \
  --test data/public/deribit_incremental_book_L2_2020-05-01_ETH-PERPETUAL.csv.gz \
  --out results/calibration
```

## Evidence

This directory contains:

- `result.json` — complete diagnostic scorecards and study results
- `plan.json` — chronological split and diagnostic configuration
- `manifest.json` — run artifact manifest
- `provenance.json` — sanitized runtime and source provenance
- `report.md` — generated study summary

Absolute local filesystem paths have been removed from the committed artifacts.
Dataset hashes are retained for reproducibility.

## Interpretation

The calibration workflow completed successfully as a research procedure, but
the fitted observable distribution did not generalize to the later May regime
under the configured diagnostics.

This study does **not** establish:

- strategy profitability,
- historical counterfactual fills,
- FIFO queue-position reconstruction,
- hidden-liquidity reconstruction,
- a fully calibrated exchange-level order-flow model.

The diagnostic thresholds are research heuristics, not statistical
significance tests.