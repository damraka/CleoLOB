# v0.7 model risk, robustness and decision certification (M15)

Code: `lob/v07/robustness/` (`analysis.py`, `study.py`, `certification.py`, `stress.py`) and
`lob/v07/uncertainty/` (`worlds.py`, `risk.py`).

**Runs.**
- `results/v07/m15/execution-2`: the registered study. The first attempt,
  `results/v07/m15/execution`, was ABORTED by the host for low memory and is kept (ledger).
- `results/v07/m15/stress` (SYNTHETIC_STRESS).

All results are simulation only, with no historical, live or profitability statement. They
inherit the ASSUMPTION_DEPENDENT posterior recovery limitation, because 16 of the 21 worlds are
posterior draws.

## Plausible-world ensemble (workstreams 8, 41)

**Composition.** 21 worlds, each with a manifest and scenario label:
- G0 point (CALIBRATED)
- 16 G0 posterior-predictive draws (POSTERIOR_SAMPLE, systematic seeded draws)
- the point fits of G1, G2, G3 and G4 (CALIBRATED)

**Rules.**
- Synthetic stresses and structural interventions are excluded by the registered scenario rule
  (enforced in code).
- The registered cap is 21 worlds; it binds exactly (16 + 1 + 4).
- No world was dropped: every cell's INVALID fraction was 0.

**Queue model.** Fixed to the engine's FIFO matching in simulated worlds. Queue sensitivity is
reported separately (`docs/v07-execution.md`, M3).

**Design.** 8 primary policies plus exploratory MPC, 256 identical market seeds per world, the
v0.6 mandate (buy 14 lots, 120 s), and fixed policy parameters in every world. VWAP uses one
volume profile estimated on the G0 point world.

**Mean completion-adjusted cost over the 21 worlds** (bps; range of world means; most adverse
world):

| Policy | Mean | Range | Most adverse world |
|---|---|---|---|
| TWAP | 3.17 | 1.03–4.54 | G3 |
| VWAP | 3.26 | 1.91–4.92 | G1 |
| POV | 3.49 | 2.71–6.35 | G3 |
| Almgren–Chriss | 3.22 | 1.51–4.93 | G4 |
| liquidity-sensitive | 3.17 | 2.64–4.32 | G4 |
| urgency | 3.21 | 2.74–4.12 | G4 |
| imbalance-aware | 3.11 | 1.24–4.53 | G1 |
| spread-aware | 3.16 | 2.19–4.89 | G3 |
| MPC (exploratory) | 3.28 | 2.89–4.48 | G4 |

## H7 — model uncertainty material to execution cost

**H7 is ESTABLISHED for one policy, POV** (Holm over 8 policies, between-world SD ≥ 0.5 bps). [C-H7]
For POV, the between-world SD of world means is 0.96 bps and the ratio to the within-world seed
SE is 2.85 (p < 0.001).

For the other 7 policies:
- The ratio R is 1.17–1.50, with p-values 0.016–0.066.
- None passes the Holm thresholds, so each is NOT_ESTABLISHED.
- Their between-world SDs are 0.43–0.78 bps.

## H8 — robust policy comparison

**H8 is NOT_ESTABLISHED.** [C-H8] No pair satisfies the registered robustness rule.

**Edges** (alpha 0.05 / 28):
- 27 pairs: the pooled interval lies inside ±1 bps (EQUIVALENT_WITHIN_MARGIN, descriptive, using
  the registered 1 bps materiality margin).
- 1 pair (TWAP vs POV): INDETERMINATE, pooled −0.32 bps [−1.10, 0.16].
- No pair is MODEL_DEPENDENT or REVERSED among the 21 worlds.

**Worst-plausible search (workstreams 42, 43).**
- 64 further draws from the 90% lowest-distance posterior region, 64 seeds each.
- For every pair, the most adverse single draw differs from the pooled conclusion by 0.75–8.5 bps.

So the pooled equivalence describes the *average* over plausible worlds, not each world. It is
not evidence that the policies perform the same in every plausible market. The adversarial
draws are noisier (64 vs 256 seeds), but deviations of up to 8.5 bps exceed what seed noise
alone explains.

## H9 — single-world rankings under model uncertainty

**H9 is INCONCLUSIVE (descriptive).** [C-H9] No pair is determinate (alpha 0.05 / 28) even in the
single G0 point world, so no apparent single-world ranking exists that model uncertainty could
overturn.

## Ranking topology (workstream 39)

- 21 worlds yield 21 distinct rankings of the 8 policies.
- The lowest-cost policy by world is:

  | Policy | Worlds where it is cheapest |
  |---|---|
  | spread-aware | 5 |
  | liquidity-sensitive | 4 |
  | POV | 3 |
  | TWAP | 3 |
  | AC | 2 |
  | VWAP | 2 |
  | imbalance-aware | 1 |
  | urgency | 1 |

- The mean Kendall distance between world rankings is 13.8 of 28 pairs (maximum 26).
- No robust arcs exist, so the robust-edge graph has no cycles.

## Uncertainty decomposition (workstreams 35, 45)

| Policy | Market-seed SD | Posterior-parameter SD | Model-class SD | World-mean range |
|---|---|---|---|---|
| TWAP | 8.64 | 0.59 | 0.63 | 3.51 |
| VWAP | 9.18 | 0.40 | 0.76 | 3.02 |
| POV | 5.41 | 0.30 | 1.32 | 3.63 |
| AC | 8.29 | 0.48 | 0.86 | 3.41 |
| liquidity-sensitive | 5.18 | 0.28 | 0.53 | 1.68 |
| urgency | 5.71 | 0.28 | 0.44 | 1.37 |
| imbalance-aware | 8.44 | 0.57 | 0.70 | 3.29 |
| spread-aware | 8.71 | 0.34 | 0.74 | 2.70 |

All values are in bps. Market randomness dominates per episode. Among the model components, the
simulator *family* (model class) contributes more than the posterior parameter spread for every
policy except TWAP, where the two are similar.

**Caveat.** There are only 5 model classes and 16 posterior draws, so these SDs are uncertain.
Training-seed, queue and historical-fill components are reported with the transfer results
(`docs/v07-transfer.md`). Regime uncertainty enters only through G2 and the posterior's regime
parameters.

## Stress, certification and abstention (workstreams 36–38, 40)

**Stress.** See `docs/v07-execution.md`. Rankings change under synthetic stresses; this is not
empirical validation.

**Certification.** Decision certification applies the seven registered gates:
1. registration
2. statistics
3. materiality
4. model
5. seeds
6. regime
7. historical bounds

Its truth table is tested exhaustively. The final per-pair certification, which includes both
historical fill bounds, is in `docs/v07-final-report.md`. Abstention ("We cannot support a
conclusion") is returned whenever the evidence does not certify a direction.
