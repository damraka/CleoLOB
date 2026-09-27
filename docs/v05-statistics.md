# v0.5 statistical design

Every family, correction, interval method and gate was fixed in the
[M0 protocol](v05-research-protocol.md) or a design sealed in the ledger before the data it
evaluates were accessed. Statistical and economic significance are reported separately.
No global "best" label is produced from sample means.

| Study | Family | Correction | Interval method | Gate |
|---|---|---|---|---|
| M1 | one gate | — | exact agreement fraction; the 100 ms tolerant diagnostic is secondary only | frozen thresholds 0.99 / 0.90 |
| M2 | descriptive | — | classification shares and bound widths; never point estimates | none |
| M3 | 3 holdouts, selected vs control | Bonferroni, α = 0.05/3 | paired chronological block bootstrap over 600 s historical blocks (2,000 resamples, seed 45901) with fixed simulated summaries | improvement: upper bound < 0; generalization: every family error ≤ 0.60 |
| M4 | 3 size buckets × 3 horizons × 4 datasets = 36 | Bonferroni, α = 0.05/36 | historical: 600 s block bootstrap of event means; simulator: seed bootstrap | H4 per horizon |
| M6 | per regime label | inherits M3/M4 | as M3/M4, within regime blocks | regime-robust only if the conclusion holds in every label with ≥ 6 blocks |
| M7 | 4 regimes × 2 algorithms × 8 references × 2 endpoints = 128 | Bonferroni, α = 0.05/128 | cost: crossed training-seed × market-seed bootstrap (20,000 resamples); completion: market-level Clopper–Pearson | cost upper bound < 0 **and** completion lower bound ≥ −0.05 |
| M8 | pairs × datasets × fill modes | Bonferroni | episode bootstrap within dataset and fill mode (5,000 resamples) | classification only |

## Details that matter

**Paired and crossed designs.**
- M7 pairs every agent with its reference on common market seeds.
- The cost interval resamples training seeds and market seeds independently while
  preserving cell pairing.
- Within each resample, market seeds carry the pairing.

**The M7 completion bound.**
- v0.4 declared every completion contrast INCONCLUSIVE because constant paired differences
  give a degenerate bootstrap. v0.5 preregistered a distribution-free alternative.
- For each market, Z_m = 1 if any training seed of the agent misses within-horizon
  completion while the reference completes.
- The completion difference is at least −P(Z = 1). That probability is bounded by a one-sided
  exact Clopper–Pearson limit over independent markets at α = 0.05/128.
- With 160 markets and no discordance, the limit is 0.0479, inside the 0.05 margin.
- The bound conditions on the trained models. It is valid, but conservative: it ignores
  markets where the agent completes and the reference does not.

**Missing data.**
- INVALID or missing planned cells withhold a comparison (it counts as not passing).
- They never shrink the family.

**Block bootstraps.**
- Historical observations are serially dependent, so M3 and M4 resample 600 s blocks.
- Approximate stationarity between blocks is an assumption.

**Degeneracy.**
- Constant differences are reported, never converted into certainty.
- The M7 CP bound avoids the v0.4 degeneracy by design rather than by redefinition after the
  fact.

**Power.**
- No prospective variance-based power analysis was performed.
- Only the completion bound's attainability (enough markets) was checked, before evaluation.
- Null results do not establish equivalence.
