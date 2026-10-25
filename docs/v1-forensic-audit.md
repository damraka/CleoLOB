# v1.0 forensic audit of v0.9 (milestone M0)

**Scope.** This is a read-only audit of the released v0.9.0 evidence. It was done before any v1 simulator code
changed. Nothing in `configs/v09`, `results/v09` or `examples/studies/v09` was modified.

Every recomputation below is **RETROSPECTIVE**: it is a diagnostic of sealed evidence, not a preregistered test. It
does not change any v0.9 status. The v0.9 statuses stay as published in
[v09-final-report.md](v09-final-report.md).

Reproduce the diagnostics with:

```bash
cleo protocol-v09 verify
cleo verify-v09 --bundle examples/studies/v09
cleo claim-audit-v09 --bundle examples/studies/v09
python tools/v1_retro_dependence.py --out examples/studies/v1/retrospective/dependence.json
python -m pytest tests/test_v1_book_calibration.py tests/test_v1_statistics.py
```

## 1. Release, protocol and bundle verification

| Check | Result |
|---|---|
| Tag `v0.9.0` | Annotated tag object `b1a98602`, peeling to `64b1f917` (merge of PR #24). Local and `origin` agree. |
| GitHub Release | "CleoLOB v0.9 — Reality-Calibrated Market Twins, Causal Execution Science & Safe Autonomous Adaptation". Published 2026-10-16T11:55:13Z, not a draft or prerelease, target `main`. |
| PyPI `cleolob==0.9.0` | `cleolob-0.9.0-py3-none-any.whl` (sha256 `d42062d1…`) and `cleolob-0.9.0.tar.gz` (`2740bae6…`), uploaded 2026-10-16T11:58Z. |
| `cleo protocol-v09 verify` | `valid: true`, 0 issues. Protocol sha256 `eb3d3d22…` (after amendment 3). 700 ledger entries, head `cc64f7fd…`. Confirmatory phase closed. |
| `cleo verify-v09 --bundle` | `valid: true`, 80 runs, 700 ledger entries, 0 issues. |
| `cleo claim-audit-v09 --bundle` | `valid: true`, 42 claims, 0 issues, 0 warnings. |
| Preregistration order | Confirmatory registry commit `7cb806e`, created 05:20:49Z, came before the first confirmatory read (ledger entry 514, 05:40:58Z). |
| Vault order | The vault was opened only after `close_confirmatory` (entry 618), and first read at entry 622. |

**Provenance weakness.** 240 of the 700 ledger entries record `git_dirty: true`.
- This includes the first development access and the first vault read (entry 622).
- On those entries the code version is a commit plus uncommitted changes, so the commit alone does not identify the
  code.

v1 requires a clean tree for every guarded access (see [v1-scientific-protocol.md](v1-scientific-protocol.md)).

## 2. Issue A: spread-improvement calibration defect (confirmed)

**Code path.**
- `lob/v09/generators/classical.py::book_samples` infers additions only where the touch price is unchanged (`same =
  (px[1:] == px[:-1])`), and indexes levels relative to the touch.
- An order that improves the touch changes the touch price, so it is never sampled.
- Every `add_level` is therefore ≥ 0. `lob/v08/market/sim.py::BookModel.fit` then estimates `improve_prob =
  mean(add_level < 0) × 2 = 0`.
- `MarketSim._apply` (`sim.py:138`) places an order inside the spread only with probability `improve_prob`, so it
  never does.

**Evidence.**
- The fitted v0.9 book of the fill study has `"improve_prob": 0.0` (`examples/studies/v09/runs/explore/fills-pilot-2/result.json`).
- On the consumed explore day `bybit-linear-btcusdt-2023-08-19`, used here only as an engineering diagnostic:
  - the v0.9 path yields 0 negative-level samples out of 45,673;
  - the corrected estimator finds improvement probabilities of 0.11, 0.28 and 0.45 for spreads of 2, 3 and ≥ 4 ticks.

**Consequence.** All 8 v0.9 families were rejected identically on spread, depth and tail
(`runs/explore/realism-benchmark`):

| | real (BTCUSDT test days) | every generator family |
|---|---|---|
| P(spread > 1 tick) | 0.0008 | 1.0 |
| mean spread | 1.006 ticks | 39–60,000 ticks |
| P(spread ≥ 3) | 0.0006 | 1.0 |
| classifier AUC | – | 1.0 |

**Simulation check.** These runs use the same Poisson or Hawkes counts, fitted on 2 consumed explore days, over
3,000 bins (engineering diagnostic):

| | real | v0.9 book | corrected book |
|---|---|---|---|
| Poisson mean spread | 1.003 | 35.3 | 1.26 |
| Hawkes mean spread | 1.003 | 38.6 | 1.32 |
| Hawkes P(spread > 1) | 0.0006 | 1.0 | 0.28 |

- The correction removes the runaway spread, but it does not reach real tightness.
- The remaining excess is a v1 research question (H13, H16 and the replenishment model M75). It is not claimed as
  fixed.

**Fix.** `lob/v1/generators/classical/book.py`:
- aligns frames by absolute price;
- fits a state-dependent improvement table with counts and identifiability flags;
- provides `MarketSimV1`.

Regression tests are in `tests/test_v1_book_calibration.py`:
- the frozen v0.9 estimator returns 0 on frames that contain improvements;
- the v1 estimator recovers them;
- a simulate-then-fit round trip identifies a positive probability.

v0.9 code and results are unchanged.

## 3. Issue B: dependence of the confirmatory units (confirmed, wider than H36)

The v0.9 confirmatory analysis resampled instrument-days as independent units with a paired percentile bootstrap.
The units share calendar dates:

| Hypothesis | Units | Distinct dates | Structure |
|---|---|---|---|
| H2, H5, H6, H7, H16, H17, H29 | 8 | 4 | BTCUSDT and ETHUSDT on the same 4 dates |
| H8 | 10 | 5 | BTCUSDT and ETHUSDT on the same 5 dates |
| H23 | 12 | 6 | BTCUSDT and ETHUSDT on the same 6 dates |
| H32 | 15 | 15 | BTCUSDT only |
| H33 | 8 | 8 | ETHUSDT only, but on the same dates as the BTCUSDT units of H23 |
| H36 (vault) | 18 | 7 | SOLUSDT linear, BTCUSDT linear and BTCUSDT spot on the same dates |

- The H36 backups for 2025-08-21 and 2025-08-27 were 2025-07-08 and 2025-06-05, dates already used by another
  stream. This increased the overlap.
- `examples/studies/v1/retrospective/dependence.json` recomputes each registered decision quantity with calendar
  dates as the independent unit.
- The primary method is the CR1 cluster-robust mean with Student t on G − 1 degrees of freedom. An exact cluster
  sign-flip test and a cluster bootstrap are reported alongside.

| H | Registered interval (v0.9) | Date-cluster t 95% interval | Threshold | Exact sign-flip p (floor) |
|---|---|---|---|---|
| H2 | 0.317–0.455 | 0.234–0.546 | 0.02 | 0.0625 (1/16) |
| H5 | 0.069–0.085 | 0.060–0.093 | 0.005 | 0.0625 |
| H6 | 0.047–0.316 | 0.080–0.283 | 0.005 | 0.0625 |
| H7 | 0.074–0.117 | 0.041–0.146 | 0.005 | 0.0625 |
| H8 | 0.328–0.617 | 0.303–0.645 | 0.25 | 0.031 (1/32) |
| H16 | 0.012–0.019 | 0.007–0.024 | 0.005 | 0.0625 |
| H17 (value + margin) | 0.048–0.079 | 0.032–0.096 | 0 | 0.0625 |
| H23 | 8.8–14.2 | 6.3–16.5 | 1.0 | 0.016 |
| H29 (agreement − 0.5) | 0.381–0.465 | 0.407–0.441 | 0.15 | 0.0625 |
| H32 C1 / C2 / C3 | — | 6.4–11.7 / 2.2–4.5 / 1.4–2.8 | 1 / 1 / 0.25 | < 0.001 |
| H33 C1 / C2 / C3 | — | 7.8–14.6 / 2.6–6.9 / 0.49–4.1 | 1 / 1 / 0.25 | 0.004 / 0.004 / 0.020 |
| **H36 C1** | 4.65–9.23 | 3.44–10.13 | 1.0 | 0.0078 |
| **H36 C2** | 4.83–9.91 | 4.92–9.59 | 1.0 | 0.0078 |
| **H36 C3** | 0.48–2.07 | **−0.13–2.75** | 0.25 | **0.0625** |

What changes under date clustering:
- **H36 C3 is not supported.** Its date-cluster interval contains the threshold (one-sided p = 0.06), and C3 is
  close to zero on SOLUSDT (per-stream mean 0.01 bps vs 1.69 on BTCUSDT linear and 2.23 on BTCUSDT spot). The
  intra-date correlation is 0.48, so the 18 units are worth about 10 independent units. The published v0.9 status is
  unchanged. v1 reevaluates it prospectively (H12, H48).
- **Seven E4 hypotheses rest on 4 independent dates.** Their cluster-t intervals still exclude the thresholds, but
  no randomization test on 4 clusters can reach p < 0.0625. v1 requires at least 7 independent date clusters per
  registered test (2^−7 < 0.0125, the strictest family alpha), and a power analysis at the cluster level (M7).
- **Instrument heterogeneity.**
  - H6's effect is ETHUSDT-driven: BTCUSDT unit values are about 0.007 against the 0.005 threshold, ETHUSDT values
    0.20–0.50.
  - C2's vault effect is spot-driven (13.6 bps spot vs 2.3 BTCUSDT linear).
  - Heterogeneity is reported per stream in v1 and tested where registered.

## 4. Issue C: Amendment 3 (two distinct historical analyses)

- **The change.** Amendment 3 (ledger entry 474, incident 2 at entry 475) replaced the frozen measurement screen
  with a split-half mean-consistency screen. This was done after the exploratory pilots had been inspected.
  - The frozen screen was Spearman–Brown reliability of unit-level split halves ≥ 0.5.
  - The replacement passes when the halves' mean difference is within 2 SE.
- **Effect on eligibility.** From `runs/promotion/run/result.json`, `sensitivity_original_gate`:
  - Eligible only because of Amendment 3 (NOT_PROMOTED_MEASUREMENT under the frozen screen):
    - H16 (reliability −5.0)
    - H17 (−0.18)
    - H23 (0.44)
    - H32 (0.21)
    - H33 (−1.5)
    - H36 (0.44)
    - the carried conclusions C1, C2 and C3 (0.21, 0.21 and 0.09)
  - Lost its promotion because of Amendment 3: H3 (reliability 0.91, but the halves differ by 0.0026 > 2 SE 0.0021).
  - Unaffected: H2, H5, H6, H7, H8 and H29 pass both screens.
  - Under the frozen screen H10, H11, H13, H20, H22, H25 and H28 were measurement failures. After Amendment 3 they
    are "effect too small". The terminal status is the same.
- **Assessment.**
  - The stated rationale is substantive: the frozen statistic measures between-day rank stability, not the
    reliability of a mean. Homogeneous effects such as C3 have low between-day variance and fail it.
  - The replacement has an asymmetric property. A noisier pilot widens 2 SE and so passes more easily, while a very
    precise pilot (H3) can fail on a substantively negligible difference.
  - Neither screen measures what promotion needs, which is the probability that the confirmatory decision will be
    correct.
- **v1 treatment.**
  - The original and amended decisions are two distinct historical analyses. Both are cited, and neither is
    authoritative for v1.
  - v1 promotion uses cluster-level power and the sign-flip floor, no split-half screen. It is frozen in
    `configs/v1` before any v1 pilot is read.
  - v1 tests the replications H1–H12 on fresh data irrespective of their v0.9 eligibility path. The eligibility path
    is recorded per candidate.

## 5. Issue D: why synthetic and hybrid rungs disagreed with bounded replay

| Factor | Evidence | Effect on C1–C3 |
|---|---|---|
| Spread defect (Issue A) | Generated spreads of 35–60 ticks vs 1.006 real | Every aggressive child pays tens of ticks. Pairwise differences between aggressive policies (C1: CVaR-MPC vs TWAP; C2: TWAP vs sweep) are dominated by spread crossing, not by the mechanisms the conclusions are about. C1's synthetic effect is −0.7 bps vs +11.0 bps in replay. |
| No price impact in generators | Impact metric: Hawkes 0.022 bps vs real 2.28 bps (BTCUSDT) | The gate ratio passed (0.99) only because the day-to-day reference (2.28) is as large as the real value. Sweep penalties (C2) are mis-sized. |
| Queue and fill clearance too fast | P(back-of-queue clearance within 30 s): Hawkes 0.995 vs real 0.47 | Passive policies (C3) look better in simulation than in replay. The synthetic C3 effect exceeds replay by 1.1 bps. |
| No trade-count overdispersion | Market-count dispersion: Hawkes 1.2 vs real 117 (BTCUSDT) | No flow bursts, so tail costs (C1's CVaR endpoint) are understated. |
| Weak resiliency response | Hit-side depth recovery 5 s after bursts: 1.5 vs 6.3 | Refill after own trades is wrong in both directions across families. |
| Hybrid twin never tested the conclusions | Ladder: "no registered hybrid-twin test of individual execution conclusions" (NOT_EVALUATED) | H36's record lists fidelity `F2, F3`, but only F2 was evaluated. This is a design gap, not an observed disagreement. |
| Bounded replay grid is narrow | Vault bound width: median 0.16 bps, max 1.9 bps, over 54 assumptions | The grid (queue × latency × hidden × impact) barely moves aggressive-vs-aggressive comparisons. "Bounded" mostly means "replay" for C1 and C2. The conclusions are robust to the grid, but the grid does not span mechanisms such as other participants' reaction. |
| Gate references from 4 days | Reference = 95th percentile over 6 pairs of 4 test days | Degenerate references (spread 0.0012) reject any finite error. Wide references (impact) accept a 100× error. v1 replaces this with per-metric distributional tests and an independent holdout reference (M132, M133). |
| Selection | C1–C3 were the conclusions carried because bounded replay on explore days supported them | The synthetic rung was compared on conclusions selected on F2. Agreement was never measured on a design that could detect it (H30 and H31 failed measurement). |
| Observation reconstruction | 100 ms aggregate frames, no order IDs | Inside-bin events merge. Spreads that open and close within a bin are invisible to both real and synthetic frames, so this does not explain the gap. It does bias event inference (documented in `book.py`). |

**Root cause in one sentence.** The synthetic rung failed because a calibration defect (Issue A) and missing
impact, burstiness and queue dynamics made every generator wrong on precisely the capabilities execution needs.
The gate correctly rejected them, but its references were too noisy to rank partial failures. The hybrid rung never
tested the conclusions.

## 6. Per-hypothesis audit of the 12 ESTABLISHED v0.9 hypotheses

The kinds of evidence are:
- **observed:** scored against real events, with no counterfactual or simulation assumption;
- **reconstructed:** scored against events inferred from aggregate L2, such as clearance proxies;
- **model-dependent:** scored inside a simulator or twin;
- **assumption-dependent:** bounded counterfactual replay.

| H | What it compares | Evidence kind | Main limitation found |
|---|---|---|---|
| H2 | Neural TPP vs classical point processes, held-out log-likelihood | observed | 4 dates; the effect is large (0.39 nats/event vs threshold 0.02) |
| H5 | Dynamic model averaging vs best single model | observed | 4 dates; the average includes the H2 neural model |
| H6 | Execution-aware vs generic fill calibration (Brier) | reconstructed (clearance labels) | 4 dates; ETHUSDT-driven; the generic baseline is degenerate on ETHUSDT (p ≈ 1.0) |
| H7 | Queue-survival models vs Kaplan–Meier | reconstructed | 4 dates; back-of-queue clearance proxy |
| H8 | Fill calibration and regret on the hybrid twin | model-dependent (F3) | 5 dates; regret is measured inside the twin, not in history |
| H16 | Continual calibration vs naive fine-tuning (forgetting) | observed | 4 dates; incident 4 (unit ordering) was found after outcomes were visible; ESTABLISHED with 6 and 8 units |
| H17 | Adaptive conformal coverage under drift (non-inferiority) | observed | 4 dates; non-inferiority margin 0.02 |
| H23 | CVaR-MPC vs TWAP tail cost | assumption-dependent (F2) | 6 dates; eligibility via Amendment 3 |
| H29 | Synthetic ranking predicts hybrid ranking | model-dependent (F3 vs F4–F6) | Agreement between two simulators, not with history. H30 (hybrid → history) failed measurement. |
| H32 | C1–C3 across time (2024-06..12, BTCUSDT) | assumption-dependent | Eligibility via Amendment 3; independent dates (15) |
| H33 | C1–C3 across instrument (ETHUSDT) | assumption-dependent | Eligibility via Amendment 3; same dates as the BTCUSDT confirm units |
| H36 | C1–C3 on the 2025 vault | assumption-dependent (F2 only) | Amendment 3; 7 dates; C3 not supported under date clustering; F3 listed but not evaluated |

**Independently observed effects:** H2, H5, H16 and H17. **Reconstructed-observation effects:** H6 and H7.
**Model-dependent:** H8 and H29. **Assumption-dependent:** H23, H32, H33 and H36.

## 7. The 24 non-promoted candidates

All 24 were NOT_ESTABLISHED because they were never promoted (`runs/promotion/run/result.json`, `promotion.decisions`):
- **Measurement (split halves differ):** H1, H3, H30, H31.
- **Effect too small at the planning effect (pilot − 1 SE):** H4, H9–H15, H18–H22, H24–H28, H35.
  - Several pilots had exactly zero effect and zero spread: H20, H22 and H28.
  - Zero spread means the compared policies never differed on the pilot units, which is a design-sensitivity
    failure, not evidence of no effect.
- **Other registered reason:** H34 (no exploratory pilot record, so no L1-identifiable conclusion was carried).

## 8. Dataset consumption and source capability inventory

Every source below is permanently not fresh for v1 confirmatory use. Each may be used for engineering development,
labelled as such in the v1 ledger.

| Source | Consumed by | Days / periods |
|---|---|---|
| Bybit linear BTCUSDT | v0.9 dev, explore, confirm, confirm_late, final | 2023-07-15; 2023-08-19..2023-10-17 (8); 2023-11-04..2024-03-19 (27); 2024-06-01..12-21 (17); 2025-05-06..08-19 (12) |
| Bybit linear ETHUSDT | v0.9 explore, confirm | 2023-08-19..10-17 (8); 2023-11-12..2024-03-08 (8) |
| Bybit linear SOLUSDT | v0.9 final | 2025-05-06..08-19 (12) |
| Bybit spot BTCUSDT | v0.9 dev (format day), final | 2025-09-20; 2025-05-06..08-27 (12) |
| Binance USD-M BTCUSDT and ETHUSDT (L1) | v0.9 explore | 2023-08-19..10-23 (9 each) |
| Deribit BTC- and ETH-PERPETUAL (Tardis first-of-month) | v0.5–v0.8 | 2020 months; 2026-07-01, 2026-08-01, 2026-09-01 |
| BitMEX XBTUSD | v0.5–v0.8 | 2020-11-01 |
| Bitstamp BTCUSD MBO (dev and validation) | v0.5–v0.8 | as in `configs/v08/consumption-ledger.jsonl` |

- **Unread days inside v0.9 pools.** The v0.9 pools declared 318 slots, of which 124 were read. The unread slots lie
  inside periods whose market conditions were seen through neighbouring days.
- v1 treats each v0.9 pool range (2023-07-15..2025-08-31 for the Bybit linear streams) as **period-consumed**. Such
  ranges may serve only development and exploration, never confirmation or the vault.
- Fresh-data availability after the v0.9 cutoff is assessed in [v1-data-sources.md](v1-data-sources.md).

## 9. Carried into v1

| Finding | v1 response |
|---|---|
| Issue A | M70 fix with regression tests. H13 tests whether the fix materially reduces spread mismatch on fresh data. |
| Issue B | M130 cluster inference as the registered default. A minimum-clusters rule. Effective-n feasibility (M7). H12 and H48. |
| Issue C | Promotion frozen before pilots. No split-half screen. Both historical analyses cited. |
| Issue D | Execution-relevant realism metrics and capability gates (M132–M133). The hybrid rung is registered for every carried conclusion (H29–H31, H47). |
| Dirty-tree accesses | Guarded v1 accesses refuse a dirty tree. |
| Narrow assumption grid | The v1 F2 grid adds participant-reaction (refill and cancel response) bounds (M135). |
