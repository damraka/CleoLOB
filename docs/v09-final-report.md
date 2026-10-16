# v0.9 final report: reality-calibrated market twins, execution science and safe adaptation

Every status below is read mechanically from the sealed final analysis `results/v09/final/final-analysis` (bundled
under `examples/studies/v09/runs/final/final-analysis`). Each status was decided under the frozen protocol
(`configs/v09/protocol.json`) and the frozen confirmatory registry (`configs/v09/hypotheses-confirmatory.json`,
committed in `7cb806e` before any confirmatory day was read). Markers such as [C09-H2] resolve to the sealed
evidence through `cleo claim-audit-v09`.

## Summary

- 36 candidate hypotheses were registered before any exploratory access. The registered promotion gate promoted 12
  of them, and carried the execution conclusions C1–C3 ([C09-PROMOTION]).
- All 12 promoted hypotheses are ESTABLISHED. The other 24 are NOT_ESTABLISHED because they were never promoted and
  never tested on confirmatory data.
- Real-data predictive results (E4) were ESTABLISHED for:
  - neural temporal point processes over classical point processes;
  - dynamic model averaging;
  - execution-aware fill-probability calibration;
  - queue-survival models over Kaplan–Meier;
  - continual calibration against forgetting;
  - adaptive conformal coverage under drift (non-inferiority).
- Three registered execution conclusions were ESTABLISHED in **bounded historical counterfactual replay** on their
  conservative (worst-assumption) bound. They held across time, across instrument and on the untouched 2025 vault:
  - C1: CVaR-MPC vs TWAP, tail cost, medium mandate;
  - C2: TWAP vs sweep, large mandate;
  - C3: passive-with-deadline vs TWAP, small mandate.

  These are HYPOTHETICAL replays under a registered assumption grid. They are not fills, and no live trading,
  profit, HFT-readiness or production claim follows from them.
- The capability-specific realism gate rejected every generator family for execution, realism, strategic and tail
  use (NO_ACCEPTABLE_MODEL); fills reached PARTIAL_MODEL_SET only ([C09-GATE]). So no synthetic-only result is used
  for a real-market statement.
- **The execution conclusions depend on Amendment 3.** Under the original frozen promotion screen (reported as a
  sensitivity analysis), H16, H17, H23, H32, H33 and H36 would not have been promoted, no conclusion would have been
  carried, and H3 would have been tested instead. Amendment 3 was made after the exploratory pilots were inspected
  and is disclosed as incident 2 (see below).

## H1–H36

Promotion outcome and terminal status of every candidate. For promoted hypotheses the estimate is the mean unit
value on the registered confirmatory units, with a paired percentile bootstrap 95% interval, Holm within the
registered family, and the registered practical threshold.

| H | Question | Promotion | Status | Estimate (95% interval) | Units |
|---|---|---|---|---|---|
| H1 [C09-H1] | hybrid replay vs fully generative response realism | measurement (split halves disagree) | NOT_ESTABLISHED | — | — |
| H2 [C09-H2] | neural TPP vs classical point processes (nats/event, threshold 0.02) | PROMOTED | ESTABLISHED | 0.390 (0.317–0.455) | 8 |
| H3 [C09-H3] | sequence models vs classical conditional event models | measurement | NOT_ESTABLISHED | — | — |
| H4 [C09-H4] | latent dynamics vs classical path realism | effect too small | NOT_ESTABLISHED | — | — |
| H5 [C09-H5] | dynamic model averaging vs best single model (nats/event, 0.005) | PROMOTED | ESTABLISHED | 0.076 (0.069–0.085) | 8 |
| H6 [C09-H6] | execution-aware vs generic fill calibration (Brier, 0.005) | PROMOTED | ESTABLISHED | 0.181 (0.047–0.316) | 8 |
| H7 [C09-H7] | queue-survival models vs Kaplan–Meier (integrated Brier, 0.005) | PROMOTED | ESTABLISHED | 0.094 (0.074–0.117) | 8 |
| H8 [C09-H8] | calibrated fill model and execution regret on the hybrid twin (bps, 0.25) | PROMOTED | ESTABLISHED | 0.474 (0.328–0.617) | 10 |
| H9 [C09-H9] | adverse-selection-aware passive execution (C5) | effect too small | NOT_ESTABLISHED | — | — |
| H10 [C09-H10] | state-dependent vs square-root impact | effect too small | NOT_ESTABLISHED | — | — |
| H11 [C09-H11] | self-impact and large-mandate rankings (synthetic) | effect too small | NOT_ESTABLISHED | — | — |
| H12 [C09-H12] | strategic vs static ecology realism | effect too small | NOT_ESTABLISHED | — | — |
| H13 [C09-H13] | active calibration efficiency (synthetic) | effect too small | NOT_ESTABLISHED | — | — |
| H14 [C09-H14] | sequential posterior updating after drift | effect too small | NOT_ESTABLISHED | — | — |
| H15 [C09-H15] | few-shot adaptation vs full refit (non-inferiority) | pilot violates the margin | NOT_ESTABLISHED | — | — |
| H16 [C09-H16] | continual calibration vs naive fine-tuning, forgetting (nats/event, 0.005) | PROMOTED | ESTABLISHED | 0.016 (0.012–0.019) | 8 |
| H17 [C09-H17] | adaptive conformal coverage under drift, non-inferiority (margin 0.02) | PROMOTED | ESTABLISHED | coverage − 0.90 = 0.044; interval of (value + margin) 0.048–0.079 | 8 |
| H18 [C09-H18] | disagreement predicts regret | effect too small | NOT_ESTABLISHED | — | — |
| H19 [C09-H19] | abstention and incorrect selection | effect too small | NOT_ESTABLISHED | — | — |
| H20 [C09-H20] | contextual fallback (synthetic) | effect too small | NOT_ESTABLISHED | — | — |
| H21 [C09-H21] | distributionally robust execution, worst-case regret (synthetic) | effect too small | NOT_ESTABLISHED | — | — |
| H22 [C09-H22] | chance-constrained completion | effect too small | NOT_ESTABLISHED | — | — |
| H23 [C09-H23] | CVaR-MPC vs TWAP tail cost in bounded replay (bps, 1.0) | PROMOTED | ESTABLISHED | 11.4 (8.8–14.2), conservative bound | 12 |
| H24 [C09-H24] | adaptive MPC after drift (synthetic) | effect too small | NOT_ESTABLISHED | — | — |
| H25 [C09-H25] | meta-policy vs best global policy | effect too small | NOT_ESTABLISHED | — | — |
| H26 [C09-H26] | offline RL safe improvement (synthetic) | effect too small | NOT_ESTABLISHED | — | — |
| H27 [C09-H27] | conservative gating and harmful deployments (synthetic) | effect too small | NOT_ESTABLISHED | — | — |
| H28 [C09-H28] | rare-event-aware optimization (synthetic) | effect too small | NOT_ESTABLISHED | — | — |
| H29 [C09-H29] | synthetic ranking predicts hybrid ranking (agreement, level 0.5, threshold 0.15) | PROMOTED | ESTABLISHED | agreement 0.924; interval of (agreement − 0.5) 0.381–0.465 | 8 |
| H30 [C09-H30] | hybrid ranking predicts bounded-historical ranking | measurement | NOT_ESTABLISHED | — | — |
| H31 [C09-H31] | multi-fidelity consensus vs synthetic-only | measurement | NOT_ESTABLISHED | — | — |
| H32 [C09-H32] | transfer across time (2024-06..12 BTCUSDT) | PROMOTED | ESTABLISHED | C1 9.0, C2 3.3, C3 2.1 bps; each meets its rule | 15 |
| H33 [C09-H33] | transfer across instruments (ETHUSDT) | PROMOTED | ESTABLISHED | C1 11.2, C2 4.7, C3 2.3 bps; each meets its rule | 8 |
| H34 [C09-H34] | transfer across venues (Binance L1) | no L1-identifiable conclusion carried | NOT_ESTABLISHED | — | — |
| H35 [C09-H35] | learned representations and transfer | effect too small | NOT_ESTABLISHED | — | — |
| H36 [C09-H36] | the carried conclusions on the untouched 2025 vault | PROMOTED | ESTABLISHED | C1 6.8, C2 7.3, C3 1.3 bps; each meets its rule | 18 |

Notes:
- For the noninferiority (H17) and agreement (H29) tests, the registered decision uses the interval of the shifted
  quantity: value + margin, or agreement − level. That is why it is printed beside the unshifted estimate.
- Transfer and vault estimates are conservative-bound means on the target pool, with Holm over the carried
  conclusions; every Holm-adjusted p is 0.00075. On the vault, the lower 95% bounds are 4.7 (C1), 4.8 (C2) and
  0.48 (C3) bps, against thresholds 1.0, 1.0 and 0.25.
- H36 used 18 vault days, 6 per vault stream. The registered count is 16, and the frozen even-split rule rounds it up
  to equal days per stream (SOLUSDT linear, BTCUSDT linear, BTCUSDT spot).
- H36's status follows its registered decision rule, which is the vault test. It does not mean that every rung of
  the ladder supported each conclusion. The table below shows the rungs.

## The reality ladder of the carried conclusions

[C09-LADDER] reports the status of each carried conclusion on each rung.

| Conclusion | synthetic | multi-model synthetic | hybrid twin | bounded historical | cross-time | cross-instrument | cross-venue | vault |
|---|---|---|---|---|---|---|---|---|
| C1 CVaR-MPC vs TWAP, medium | DOES_NOT_SUPPORT | SUPPORTS | NOT_EVALUATED | ESTABLISHED | ESTABLISHED | ESTABLISHED | NOT_PROMOTED | ESTABLISHED |
| C2 TWAP vs sweep, large | SUPPORTS | SUPPORTS | NOT_EVALUATED | ESTABLISHED | ESTABLISHED | ESTABLISHED | NOT_PROMOTED | ESTABLISHED |
| C3 passive-deadline vs TWAP, small | SUPPORTS | SUPPORTS | NOT_EVALUATED | ESTABLISHED | ESTABLISHED | ESTABLISHED | NOT_PROMOTED | ESTABLISHED |

- **Reality gap** ([C09-GAP]): this is the gap between the synthetic and the bounded-historical effect.
  - C1: the single calibrated generator sees no benefit (−0.7 bps), while bounded replay shows 11.0 bps, a gap of
    −11.7 bps.
  - C2 and C3: the generators overstate the effect, by 4.2 and 1.1 bps.
- **What the gaps mean.** The synthetic rung alone would have mis-sized every conclusion, and it would have missed
  C1 entirely. This is consistent with the gate rejecting the generators.
- **Adversarial stress** is reported descriptively and is not a decision rung.
- **Cross-venue** was not tested: H34 was NOT_ESTABLISHED because no L1-identifiable conclusion was carried.

## Data

[C09-DATA] reports data availability and consumption per pool.

| Pool | Declared | Read | NOT_AVAILABLE (replaced by registered backups) |
|---|---|---|---|
| explore (2023-08..10) | 40 | 34 | 2 |
| confirm (2023-11..2024-03) | 184 | 35 | 6 (quality rejections, BTCUSDT) |
| confirm_late (2024-06..12) | 48 | 17 | 2 (quality rejections) |
| final vault (2025-05..08) | 45 | 36 | 4 (provider HTTP 404, read once after close) |

- **Backups.** Every substitution followed the frozen backup order, which the read-only audit
  `tools/v09_data_audit.py` recomputes independently.
- **Binance data.** No confirmatory Binance day was read, because H34 was not promoted.
- **Vault.** The vault was opened once (`open_namespace`, after `close_confirmatory` at ledger entry 618) and was
  never supplemented.

## Amendments, incidents and retained attempts

[C09-INCIDENTS] holds the full record. Three amendments and five incidents are recorded, and every one is in the
ledger.

| Incident | Kind | Summary |
|---|---|---|
| 1 (entry 435) | engineering_bug | The gap-duration quality rule counted gaps instead of summing them. Two explore days were requalified and replaced before any outcome. |
| 2 (entry 475) | posthoc_change | Amendment 3 replaced the measurement screen after the exploratory pilots were seen. The original-screen decisions are reported as sensitivity: they promote H2, H3, H5, H6, H7, H8 and H29. |
| 3 (entry 483) | other | A superseded batch queue ran concurrently with its replacement on explore descriptive studies. Write-once directories prevented any overwrite, and the duplicate attempts are ledgered. |
| 4 (entry 619) | engineering_bug | Found in a read-only preview after confirmatory close, so after outcomes were visible: the final analysis ordered H16 units by the later day and used 6 of 8. It was fixed to the registered unit definition. H16 is ESTABLISHED with 6 or 8 units, and no other unit count was affected. |
| 5 (entry 690) | engineering_bug | Vault data rounds did not re-apply the registered backup rule, so rounds 2–8 re-requested refused slots without reading data. It was fixed, and the registered backups filled the four slots. |

Retained failed attempts (none erased):

| Attempt | Outcome | Replacement |
|---|---|---|
| explore data preparation | RESOURCE_EXHAUSTED | rerun |
| continual pilot | INVALID: JSON-string keys | `continual-pilot-2` |
| fill pilot | INVALID: NaN at sealing | `fills-pilot-2` |
| descriptive generator-fit and counterexample | INVALID / ABORTED: concurrent queue | `generator-fit-2`, `counterexample-2` |
| three dry runs | INVALID | fixes to a namespace check and a CLI phase, before any confirmatory access |
| first report run | INVALID: claim-graph key | `report-2` |

## What these results do not say

- No result here is a statement about live trading, profitability, HFT readiness or production use. Bounded-replay
  costs are HYPOTHETICAL, under registered queue, latency, hidden-liquidity and impact assumptions.
- No exact fill or queue position is ever observed on public L2. Fill quantities are bounded clearance events.
- Impact estimates from observational order flow are not causal identifications. The causal registry marks them
  NOT_IDENTIFIED unless a design identifies them.
- Verification (`cleo verify-v09`) establishes integrity and binding of the evidence. It is not independent
  replication.
