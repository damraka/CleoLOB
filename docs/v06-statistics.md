# v0.6 statistical design (M21–M23)

All inference lives in `lob.v06.inference` and in the analysis modules that call it. Every
function takes an explicit seed, and no global random state is used. The seed table is in
the protocol (`seeds`).

## Resampling units

| Quantity | Unit | Method |
|---|---|---|
| Historical realism | 600 s blocks (5-minute blocks for regime analyses) | block bootstrap. The same historical draw is applied to every model in a contrast, so contrasts are paired on history. |
| Simulated realism | simulator seeds | independent seed bootstrap per model |
| Simulated execution | market seeds (independent episodes) | paired percentile bootstrap within a world. Learned agents are first averaged over training seeds per market seed. |
| Across worlds | worlds, then market seeds | two-stage bootstrap (transfer sources A–D, H11) |
| Historical execution | episodes 600 s apart | paired percentile bootstrap |
| Domain gap | real windows in 10-window blocks; simulated windows in 10-window blocks within each seed | block bootstrap of the test-set AUC |
| Sensitivity (H7) | simulator worlds | world bootstrap of Spearman's ρ |

- **No IID bootstrap over dependent time series.** Dependent samples always stay inside
  their block.
- **Alternatives for sensitivity analyses.** Moving-block and Politis–Romano stationary
  bootstraps are available. They are not the primary method.

## Null calibration of the domain-gap test

Before sealing, the H8 procedure was checked on 200 synthetic null replications with
identical real and simulated distributions, at nominal α = 0.05:

| Simulated resampling unit | False-positive rate |
|---|---|
| whole seeds (4 seeds) | 8% |
| 10-window blocks within seeds, registered layout (8 test seeds × 180 windows) | **5.0%** |

The registered design uses 10-window blocks. This choice used synthetic data only.

## Multiplicity

The protocol stores each family's members, size, correction, α and adjusted α. A family
whose size depends on the ensemble size K is sealed numerically in the design that uses it.

| Family | Kind | Size | Correction |
|---|---|---|---|
| F1 calibration (H1–H3) | confirmatory | 3 | Bonferroni, α = 0.05/3 per contrast |
| F3 equifinality (H5) | confirmatory | 4 × C(K,2) | Bonferroni |
| F4 rank stability (H6) | confirmatory | 15 × K | Bonferroni |
| F5 sensitivity (H7) | confirmatory | 9 | Holm |
| F6 domain gap (H8) | confirmatory | 1 | one-sided α = 0.05 |
| F6b secondary discriminators | exploratory | 4 | Holm |
| F7 regime (H9, H10) | confirmatory | 4 | Bonferroni, α = 0.0125 |
| F8 transfer learning (H11) | confirmatory | 4 | Bonferroni, α = 0.0125 |
| F9 fill semantics (H12) | confirmatory | 30 | Bonferroni, α = 0.05/30 |
| F10 scorecard equivalence | descriptive | 9 per dataset × model | Bonferroni, α = 0.05/9 |
| F11 transfer sources | descriptive | 15 × 4 | Bonferroni |

- FDR is not used in any confirmatory family.
- Exploratory and confirmatory results are never pooled.
- A comparison with a missing or INVALID planned cell is `WITHHELD`. It counts as not
  passing, and the family size does not change.

## Equivalence

Realism errors are nonnegative distances, so the lower TOST bound is satisfied trivially.
Equivalence is therefore one-sided. A family is `EQUIVALENT_WITHIN_MARGIN` when the one-sided
(1 − α) upper bound lies below its sealed real-versus-real margin. A noninferiority margin
(H10) is sealed numerically from selection data before any holdout access.

`NOT_ESTABLISHED` never means equivalence.

## Power and resolution

`minimum_detectable_effect`, `required_units` and `expected_half_width` are normal
approximations for independent units. The power report is computed from
development/selection variability and sealed with the designs.

An inconclusive result is reported as inconclusive. It is not reported as evidence of no
effect.
