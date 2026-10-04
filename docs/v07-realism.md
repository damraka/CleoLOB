# v0.7 realism, domain gap and fresh-holdout results (M9, M10, M17)

Code: `lob/v07/realism/` (`holdout.py`, `contrast.py`, `metrics.py`, `folds.py`, `scoring.py`).

**Runs.**
- `results/v07/m10/bank`: simulation bank, sealed before any fresh access.
- `results/v07/m17/<dataset>`: holdout evaluations.

**Sealed design.** `configs/v07/holdout-design.json`, ledger entry 101.

**Attempt history.** The first ETH 2020-12-01 attempt was ABORTED by the host for low memory
after the parsed stage, with no statistic computed or inspected (ledger entries 113 and 129).
It was rerun under the same sealed design in `...-2`.

All four realism holdouts passed the registered data-quality rules: 143 of 143 blocks used and
1,439 sixty-second windows per day. Lot scale and tick geometry are the ETH development values,
applied unchanged; the cross-instrument and cross-venue tests are deliberate.

## Hypothesis results (effect first)

Objective differences are model minus G0 point; lower is better, so a negative difference is
an improvement.

| Hypothesis | Dataset | Estimate | Interval (registered alpha) | Status |
|---|---|---|---|---|
| H1 posterior vs point | ETH 2020-11-01 | +0.387 | [0.211, 0.627] (0.025) | FAILED |
| H1 | ETH 2020-12-01 | +0.568 | [0.303, 0.932] (0.025) | FAILED |
| H1 (conjunction) | both | — | — | **FAILED** [C-H1] |
| H2 AUC difference, G1–G4 vs G0 (8 members) | ETH Nov, Dec | about 0 (all AUCs ≈ 1.000) | ceiling | **NOT_ESTABLISHED**, VACUOUS [C-H2] |
| H3 support coverage, G1–G4 vs G0 (8 members) | ETH Nov, Dec | coverage 0.000–0.015 for all | — | **NOT_ESTABLISHED** (coverage under 1%: VACUOUS for most members) [C-H3] |
| H4 G* (G3) vs G0, cross-instrument | BTC 2020-11-01 | +0.717 | [0.643, 0.768] (0.025) | **FAILED** [C-H4] |
| H5 G* (G3) vs G0, cross-venue | BitMEX XBTUSD 2020-11-01 | +0.616 | [0.527, 0.663] (0.025) | **FAILED** [C-H5] |
| H10 execution-sensitive objective, EA vs G0 | ETH 2020-11-01 | +0.189 | [−0.002, 0.311] | NOT_ESTABLISHED |
| H10 | ETH 2020-12-01 | +0.088 | [−0.068, 0.236] | NOT_ESTABLISHED |
| H10 (conjunction) | both | — | — | **NOT_ESTABLISHED** [C-H10] |
| H11 generic noninferiority (δ = 0.246) | ETH 2020-11-01 | +0.436 | one-sided upper 0.486 | FAILED |
| H11 | ETH 2020-12-01 | +0.395 | one-sided upper 0.453 | FAILED |
| H11 (conjunction) | both | — | — | **FAILED** [C-H11] |

**Interpretation.**
- **Posterior calibration (H1).** It made fresh predictive realism *worse* than the v0.6 point
  model on both fresh days. This is consistent with the diffuse ABC posterior (see
  `docs/v07-calibration.md`). It also inherits the ASSUMPTION_DEPENDENT recovery limitation.
- **Execution-aware calibration (H10, H11).** It did not improve execution-sensitive realism
  out of sample, and it lost generic realism beyond the registered noninferiority margin. The
  development-stage ES gain did not survive (metric overfitting, workstream 96).
- **Distinguishability and support (H2, H3).** No generator family reduced classifier
  distinguishability or raised support coverage: every family is perfectly separable from
  history, and almost no historical window lies inside any family's simulated support.

## Generic objective by model and fresh day (bootstrap 95% intervals)

| Model | ETH 2020-11-01 | ETH 2020-12-01 | BTC 2020-11-01 | BitMEX 2020-11-01 |
|---|---|---|---|---|
| G0 point (v0.6) | 2.773 [2.694, 2.852] | **4.350** [4.168, 4.552] | **3.800** [3.738, 3.867] | 4.421 [4.382, 4.476] |
| G0 posterior predictive | 3.160 [3.000, 3.391] | 4.917 [4.604, 5.322] | 3.845 [3.632, 4.078] | 4.419 [4.292, 4.596] |
| Execution-aware (EA) | 3.209 [3.136, 3.294] | 4.745 [4.544, 4.921] | 4.154 [4.092, 4.215] | **4.382** [4.342, 4.439] |
| G1 state-conditioned Hawkes | 4.646 [4.362, 4.995] | 4.972 [4.882, 5.151] | 6.323 [6.024, 6.665] | 6.578 [6.405, 6.805] |
| G2 regime switching | 2.927 [2.799, 3.061] | 5.049 [4.815, 5.274] | 3.889 [3.798, 3.979] | 4.609 [4.545, 4.690] |
| G3 conditional AR (G*) | 2.777 [2.625, 2.880] | 4.391 [4.229, 4.542] | 4.517 [4.434, 4.575] | 5.037 [4.958, 5.091] |
| G4 GRU | **2.660** [2.561, 2.769] | 4.401 [4.234, 4.571] | 4.499 [4.432, 4.569] | 4.835 [4.776, 4.899] |

**Complexity penalty (workstream 95).**
- G3 and G4 were clearly better than G0 on development (1.89 and 1.74 vs 2.19) and on selection
  (2.01 and 2.08 vs 2.46).
- On fresh same-instrument days they are within G0's intervals: November slightly better,
  December slightly worse.
- Cross-instrument and cross-venue, they are clearly worse.
- So the added capacity is **not justified** on untouched data (NOT_ESTABLISHED). The registered
  selection rule picked G3 on selection data, and that choice did not transfer.

## Scorecard and equivalence (workstream 14)

The family-level scorecard uses the v0.6 equivalence margins (500 bootstrap draws, alpha 0.05).

| Model | Families equivalent within margin (of 9) |
|---|---|
| G0 point, EA, G1 | none, on any day |
| G0 posterior, G2, G4 | `returns` on ETH November and BTC November |
| G3 | `event_activity` on ETH November and BTC November |
| any model | none on ETH December |

Per-family errors are in each run's `scorecard`.

## Two-sample statistics and metric agreement (workstream 15)

**Classifier AUC.** 0.998–1.000 for every model, day and timescale (10 s, 60 s, 300 s windows;
workstream 24).

**k-NN precision/recall.** About 0 for every model: generated windows lie outside the real
window manifold, and the converse holds as well.

**Disagreement between metrics.**
- RBF MMD² ranks G1 *closest* to history on three of four days (0.39–0.66).
- The sealed v0.6 objective ranks G1 *farthest* (4.6–6.6).

No single metric decides realism. All metrics agree that no model is close to history.

## Domain-gap attribution (workstreams 16, 17)

**Stored outputs.** Per-window scores, labels, split IDs and fold IDs of the primary classifier
are stored in `discriminator-<model>.json` in each run. The stored rows reproduce the reported
AUC (tested).

**Feature-family ablations.**
- With only one feature family (single-family AUC), the families that most separate real from
  simulated windows are:
  - **spread** and **event rate** on ETH 2020-11-01
  - **depth** on ETH 2020-12-01
  - **spread** and **depth** on BTC and BitMEX
- Dropping any one family does not remove the gap: the remaining features still separate the
  windows perfectly.

This is associational attribution of detectability, not a causal account.

## Temporal hierarchy (workstream 24)

Windows of 10 s, 60 s and 300 s are evaluated for every model; the 100 ms grid is the native
measurement resolution. The gap is complete at every timescale. Event-level comparison is
represented by the v0.6 event-process components. A separate event-by-event realism score is
NOT_AVAILABLE: aggregate L2 does not identify individual events.
