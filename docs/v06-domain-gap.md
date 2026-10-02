# v0.6 domain gap, support and transition realism (M5, M7, M13, M18, M20)

## Design

The design was sealed in `configs/v06/holdout-design.json` (ledger `m13-holdout-design`)
before any fresh access.

**Windows and features.**
- Windows are 60 s, with at least 80% valid samples.
- There are 20 scale-free features, computed by the sealed measurement operator:
  - spreads in bps;
  - log depths in development lots;
  - imbalance, concentration and microprice deviation;
  - return dispersion;
  - event counts and sizes;
  - level gaps.
- `lob.v06.domain_gap.feature_matrix` **refuses** any other column. In particular it refuses
  timestamps, clock time, price levels, dataset or file ids, seeds, simulator parameters,
  regime labels and row order.

**Splits.**
- Real windows: chronological, first half for training, 10-window purge, second half for
  testing.
- Simulated windows: 16 seeds × 3 h from the selected v3 model, first 8 seeds for training and
  last 8 for testing.
- Classes are balanced by deterministic subsampling.

**Classifiers.** These are numpy implementations, so no new dependency is needed:
- primary: L2 logistic regression;
- secondary: a depth-3 tree and a 64-tree depth-4 random forest.

**Uncertainty.** The test AUC is block-bootstrapped: real windows in 10-window blocks,
simulated windows in 10-window blocks within each seed. In 200 synthetic null replications
this design had a **5.0%** false-positive rate at nominal 5%. A whole-seed variant had 8% and
was rejected before sealing.

## Results (test-set ROC-AUC; one-sided 95% lower bound)

| Dataset | Logistic | Tree | Forest | Status |
|---|---|---|---|---|
| ETH 2020-09-01 (fresh; **H8**) | **1.000** (1.000) | 1.000 | 1.000 | **ESTABLISHED**: distinguishable |
| BTC 2020-09-01 (fresh, cross-instrument) | 1.000 | 1.000 | 1.000 | distinguishable (secondary, exploratory family) |
| ETH 2020-06-01 (retrospective) | 0.991 (0.978) | 0.965 | 1.000 | distinguishable (secondary) |
| ETH 2020-07-01 (retrospective) | 1.000 | 0.994 | 1.000 | descriptive |
| BTC 2020-07-01 (retrospective) | 1.000 | 1.000 | 1.000 | descriptive |
| ETH 2020-08-01 (retrospective) | 1.000 | 0.989 | 1.000 | descriptive |

**Real and synthetic market windows are almost perfectly separable** by simple, interpretable
classifiers. Each run stores standardized coefficients and test-set permutation importances
for the descriptive features. On June the most important features were log top-5 depth, log
L1 depth and the nonzero-return fraction. Importances are descriptive: correlated features
share credit arbitrarily.

A discriminator near 0.5 would not have shown realism. Here the question does not arise.

## Support (OOD)

**Rule.** Features are standardized by the simulated windows. Each historical window gets its
5-nearest-neighbour distance to the ensemble's simulated windows. The threshold is the 99th
percentile of leave-one-seed-out simulated distances.

| Dataset | Out-of-support fraction | Label |
|---|---|---|
| ETH 2020-09-01 | 1.000 | NO_CLAIM |
| BTC 2020-09-01 | 1.000 | NO_CLAIM |
| ETH 2020-06/07/08-01, BTC 2020-07-01 | 0.999–1.000 | NO_CLAIM |

**Protocol deviation (disclosed).** The protocol text says features are "standardized by
development statistics". The implementation, sealed as code before access, standardizes by
the simulated windows instead. With 99.9–100% of windows out of support under this choice,
the conclusion is unlikely to depend on it. No post-hoc rerun was made.

Essentially every historical window lies outside the calibrated simulators' support.
Absolute realism statements about these datasets are therefore withheld (`NO_CLAIM`). The
relative H1–H3 comparisons remain valid as comparisons of distances.

## Regime-transition realism (M18)

**Setup.**
- Simulated paths: 8 seeds × 6 h from the selected model.
- Labels: v0.5 M6 thresholds on 5-minute blocks.
- Historical sequences are split at gaps, so no transition spans a missing block.

**Mean row total variation of transition probabilities** (selected v3 vs history):

| Dataset | Volatility | Spread | Activity | Stress |
|---|---|---|---|---|
| ETH Jun (retro) | 0.30 | 0.51 | 0.08 | 0.39 |
| ETH Jul (retro) | 0.59 | 0.82 | 0.26 | 0.05 |
| BTC Jul (retro) | 0.70 | 0.82 | 0.14 | 0.05 |
| ETH Aug (retro) | 0.21 | 0.55 | 0.01 | 0.35 |
| ETH Sep (fresh) | 0.12 | 0.57 | 0.00 | 0.05 |
| BTC Sep (fresh) | 0.55 | 0.82 | 0.00 | 0.05 |

- **Activity-regime dynamics** are reproduced closely on most days.
- **Spread-regime dynamics** are not (TV 0.51–0.82). The simulator's spread regime does not
  switch like history's.
- **Volatility-regime dynamics** depend strongly on the day.
