# v0.8 statistical decision framework

`lob/v08/statistics.py` is dependency-free and seeded. It provides:

**Intervals and tests.**
- Paired bootstrap.
- Two-stage hierarchical bootstrap (worlds, then seeds), including a paired version that keeps the pairing
  inside every replicate.
- Bayesian bootstrap probabilities.
- Wilson intervals and the exact sign test.

**Decision rules.**
- **Superiority:** the whole interval lies beyond the threshold.
- **Equivalence:** TOST through the (1 − 2α) interval.
- **Non-inferiority:** a one-sided bound against a margin.

"Not significant" is never reported as "equivalent".

**Multiplicity.** Holm, Bonferroni and Benjamini–Hochberg. Families and interval levels are registered in
`configs/v08/statistical-families.json`.

**Power and planning.**
- MDE, power, and required units for a target effect or a target interval width.
- The planner (`lob/v08/planning.py`) turns a pilot standard deviation into seed counts and predicts events,
  memory, disk and wall clock before a run.

**Sequential testing.** Group-sequential boundaries (O'Brien–Fleming, Pocock) are implemented. No interim
look was registered, so they are unused.

**Rankings and tails.** Ranking uncertainty (bootstrap rank distributions, P(best)), regret matrices, CVaR and
Kendall/Spearman correlations.

**How hypothesis statuses are computed.** Each hypothesis status is computed mechanically by
`lob/v08/studies/final.py`, from stored estimates and intervals, using the registered rule text. Intervals
use the family-adjusted level:
- predictive family: 0.0125
- decision family: 0.01
- abstention and historical families: 0.025
- the H5 pairs: 0.0125
