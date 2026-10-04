# v0.7 transfer, regimes, drift and historical bounded execution (M11, M12, M14, M16)

**Runs.**
- `results/v07/m11/drift`: regimes, drift and half-life
- `results/v07/m12/matrix`: transfer matrices and hierarchy
- `results/v07/m14/policies`, `results/v07/m16/predictions`, `results/v07/m16/transfer`: learned
  policies and the final bounded transfer

**Data.** M11 and M12 use only consumed data (development, validation and retrospective roles).
The selection day is excluded because its role permits only selection. M16 uses the final
transfer holdout, ETH 2021-01-01, after the policy evaluation lock.

## Temporal transfer, drift and calibration half-life (workstreams 22, 23)

**Dated sequence.** ETH, months after the development day: 0 (04-01), 2 (06-01), 3 (07-01),
4 (08-01), 5 (09-01) and 6 (10-01).

**Observable drift.** The real-vs-real distance of each day to the development day:

| Month | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| Objective | 1.25 | 1.33 | 1.98 | 2.22 | 1.73 |

**Support drift.** The fraction of a day's 60 s windows outside development support:

| Month | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| Out of support | 0.60 | 0.12 | 1.00 | 1.00 | 0.79 |

Every day except July is labelled NO_CLAIM for absolute-realism statements.

**Generic objective of each banked model by month** (lower is better):

| Model | 0 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| G0 point | 2.17 | 2.42 | 2.56 | 2.86 | 2.93 | 2.41 |
| G3 | 1.94 | 2.40 | 2.38 | 2.83 | 2.95 | 2.52 |
| G4 | 1.80 | 2.32 | 2.20 | 2.79 | 3.01 | 2.48 |
| G0 posterior | 2.76 | 2.91 | 2.92 | 3.60 | 3.68 | 2.71 |
| EA | 2.47 | 2.86 | 2.95 | 3.53 | 3.51 | 2.82 |
| G2 | 2.62 | 2.85 | 2.83 | 3.51 | 3.56 | 2.75 |
| G1 | 4.00 | 4.00 | 4.50 | 4.23 | 4.18 | 3.91 |

G3 and G4 lose their development advantage over G0 by months 4–6. This matches the fresh-holdout
results in `docs/v07-realism.md`.

**Half-life (EXPLORATORY, not preregistered).** The relative degradation
d(t) = d_∞ (1 − 2^(−t/τ)) is fitted with block-bootstrap intervals over the six dated days:

| Model | Half-life τ, generic (months) | 95% CI | Long-run degradation |
|---|---|---|---|
| G0 point | 1.34 | [0.94, 1.91] | +26% |
| G0 posterior | 1.53 | [1.17, 2.08] | +18% |
| G2 | 1.67 | [1.17, 2.84] | +24% |
| G3 | 1.83 | [1.34, 3.24] | +47% |
| G4 | 2.38 | [1.60, 4.23] | +64% |
| EA | 1.34 | [1.03, 1.99] | +32% |
| G1 | 0.58 | [0.25, 0.90] | +4% |

- Execution-sensitive half-lives range from 0.25 to 0.86 months, with intervals reaching the
  grid floor.
- G1's execution-sensitive degradation is negative, so it is INCONCLUSIVE.
- More flexible models start better and degrade more.

**Change points.** Within each day, binary segmentation of 300 s log realized volatility finds
6–11 mean shifts per day. These are diagnostics, not events with an identified cause.

## Regimes (workstreams 20, 21)

**Latent regimes.** Gaussian HMMs with k = 1, 2 and 3 states were fitted on development 300 s
blocks using three features:
- log volatility
- log spread
- log trade count

On development data, the 3-state model has the best BIC (329, vs. 750 for k = 2 and 850 for
k = 1). However, one of its states is a degenerate zero-volatility state.

**Out-of-sample comparison.** On four of the five later dated days (months 2, 4, 5 and 6),
the frozen discrete volatility-threshold model has a higher log-likelihood per block than every
HMM, for example −13.7 vs. −14.9 in month 2 and −43.8 vs. −48.7 in month 6. In month 3, the
1-state model is best (−17.8 vs. −19.6). The latent models do not show predictive utility
beyond their complexity here.

**Continuous vs. discrete conditioning.** Continuous state conditioning (G1, G3, G4) was compared
with discrete regimes (G2) and no conditioning (G0) on fresh data (`docs/v07-realism.md`).
G3 and G4 are not better than G0 on fresh days. G2 is worse than G0 on all four fresh days.

Latent states are statistical clusters, never participant intent.

**Posterior drift: NOT_AVAILABLE.** Re-running SMC-ABC per day exceeds the registered budget.
Parameter drift is shown instead by the G1 refits below.

## Cross-instrument transfer matrix and hierarchy (workstreams 25, 27) — EXPLORATORY

**Design.** G1 and G3 were refitted on each of eight consumed datasets: ETH months 04, 06, 07,
08, 09 and 10, and BTC months 07 and 09. Each refit was simulated on 4 seeds × 3,600 s and scored
against every dataset.

**Mean objective by transfer type:**

| Family | In-sample | Same-instrument temporal | Cross-instrument (ETH ↔ BTC, same venue) |
|---|---|---|---|
| G3 | 2.43 | 3.05 | 3.99 |
| G1 | 3.61 | 4.35 | 4.69 |

**Activity similarity.** Transfer error correlates with the dissimilarity of event activity
between source and target: Pearson correlation 0.34 for G3, 0.03 for G1.

**Related-instrument cells: NOT_AVAILABLE.** No consumed related-instrument data exist, for
example BTC futures.

**Hierarchy (global → instrument → month).** Across the 96 G1 coefficients, the median share of
variance between instruments, relative to between months within an instrument, is 0.34:
- 26 coefficients are instrument-local (share above 0.5).
- The rest vary mostly over time within an instrument.

The venue level is NOT_AVAILABLE: only one venue has fit-eligible L2 data.

## Cross-venue (workstream 26)

**Same-day comparison.** On 2020-11-01, ETH-calibrated models were scored on Deribit
BTC-PERPETUAL and on BitMEX XBTUSD. These have the same underlying and contract type, so the
comparison separates the venue effect from the instrument effect (`docs/v07-realism.md`).

| Model | Deribit BTC objective | BitMEX XBTUSD objective | Venue effect |
|---|---|---|---|
| G0 point | 3.80 | 4.42 | +0.62 |
| G3 | 4.52 | 5.04 | +0.52 |

H5 is FAILED.

**Venue × venue fitting: NOT_AVAILABLE.** BitMEX data exist only as a fresh holdout, which must
not be fitted.

## Historical bounded transfer, ETH 2021-01-01 (H12, H13; workstreams 5, 32, 92)

**Runs.**
- `results/v07/m14/policies`: PPO and DQN, each trained single-world (G0 point) and
  posterior-world (domain-randomized over the posterior draws), 4 training seeds each, 16,384
  timesteps per model. Offline RL is NOT_AVAILABLE: no logged historical actions exist.
- `results/v07/m16/predictions`: simulated costs of every learned policy in its training
  worlds (64 markets; 8,704 rows; 0 invalid).
- `configs/v07/transfer-design.json`: the policy evaluation lock (ledger entry 182). It fixes the
  policies, predictions, H8 classification (no robust pair), mandate, episode rule and both fill
  modes before any access.
- `results/v07/m16/transfer-3`: the sealed evaluation. Two earlier attempts are retained as
  ABORTED: `m16/transfer` (low memory, entry 186) and `m16/transfer-2` (stopped by the operator
  before parsing, entry 189).

**Episodes.** 144 of 144 planned episodes (first capture + 300 s + 600 s·k, 186 s windows) were
replayed under the conservative and optimistic fill bounds. Every one of the 8 primary
classical policies and 16 learned models ran in each; 0 rows were invalid. Replay has no market
impact of the hypothetical parent, and passive fills are bounds, never exact fills.

### H12 — posterior training and historical transfer

**H12 is NOT_ESTABLISHED.** [C-H12] The estimand is the gap |simulated cost − bounded historical cost|
of posterior-trained minus single-world-trained policies. It is averaged over training seeds
and episodes, with a two-stage bootstrap and Bonferroni alpha 0.0125 per member.

| Member | Gap, posterior (bps) | Gap, single (bps) | Difference | Interval | Status |
|---|---|---|---|---|---|
| PPO, conservative | 0.03 | 0.00 | +0.03 | [−2.49, 1.60] | NOT_ESTABLISHED |
| PPO, optimistic | 0.00 | 0.10 | −0.10 | [−2.76, 1.58] | NOT_ESTABLISHED |
| DQN, conservative | 0.46 | 0.46 | −0.01 | [−2.51, 1.47] | NOT_ESTABLISHED |
| DQN, optimistic | 0.54 | 0.51 | +0.03 | [−2.67, 1.54] | NOT_ESTABLISHED |

Every interval contains zero and is about 4 bps wide, wider than the prospective MDE of
2.585 bps. This is not evidence that the two training regimes transfer equally well.

### H13 — robust conclusions under both fill bounds

**H13 is INCONCLUSIVE.** [C-H13] H8 found no robust policy pair, so there was nothing to carry to
history. The registered rule reports this case (NOT_EVALUABLE) as INCONCLUSIVE. H13 never
upgrades H8.

### Historical costs (descriptive)

Mean completion-adjusted cost over the 144 episodes (bps):

| Policy | Conservative | Optimistic |
|---|---|---|
| POV | 2.08 | 2.08 |
| imbalance-aware | 2.53 | 2.53 |
| urgency | 2.53 | 2.53 |
| liquidity-sensitive | 2.55 | 2.55 |
| VWAP | 2.58 | 2.58 |
| AC | 2.63 | 2.63 |
| TWAP | 2.68 | 2.68 |
| spread-aware | 2.70 | 2.53 |
| DQN single / posterior | 2.80 / 2.89 | 2.75 / 2.81 |
| PPO single / posterior | 3.13 / 3.15 | 3.03 / 3.12 |

- **No pair is determinate.** Of the 28 classical pairs, none is determinate under either fill
  bound at alpha 0.05/56 (`results/v07/m22/final-2`). For example, POV − AC is −0.55
  [−1.83, 0.70] bps (conservative).
- **Fill bounds rarely matter here.** Only spread-aware, which rests passive orders, differs
  between them. The other classical policies take liquidity, so their two bounds coincide.
- **POV is not consistent across simulation and history.** It has the highest mean cost in
  simulation (3.49 bps over 21 worlds) and the lowest point estimate in history. Neither
  ordering is determinate.
- **Learned policies** cost more than every classical policy in point estimate, as in v0.5 and
  v0.6. No learned-vs-classical contrast was registered, so this is descriptive.

### Remaining uncertainty components (workstream 45, descriptive)

These complete the decomposition in `docs/v07-model-risk.md`.

**Training seed (simulation, `m16/predictions`).** SD of the 4 per-seed mean costs in the training
worlds:

| Model | Single-world | Posterior-world |
|---|---|---|
| PPO | 1.29 bps | 0.40 bps |
| DQN | 1.34 bps | 0.16 bps |

Posterior-world training made learned policies much less seed-dependent *in simulation*. In
historical replay the per-seed means of all four variants lie within 2.47–3.28 bps. H12 found no
difference in transfer.

**Historical fill bound (`m16/transfer-3`).** Mean |conservative − optimistic| episode cost:
- 0 for the seven liquidity-taking classical policies;
- 0.21 bps for spread-aware;
- 0.09–0.23 bps for the learned policies, with single episodes up to 7.4 bps.

**Queue assumption (`m3/queue`, development day).** The conservative-to-optimistic width of the
mean passive fill fraction is 0.072 of order size. It matters only for policies that rest passive
orders.

