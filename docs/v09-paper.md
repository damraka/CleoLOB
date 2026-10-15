# CleoLOB v0.9 research report: which execution conclusions survive contact with reality?

This report covers the v0.9 study cycle: the question, how it was preregistered and gated, what the public data
could and could not say, what was found and what it does not mean. The statuses come from the sealed final analysis
(see the [final report](v09-final-report.md) for the full table).

## 1. Question

> Which execution conclusions remain scientifically defensible when market reconstruction uncertainty, latent
> liquidity, self-impact, strategic response, causal ambiguity, multi-fidelity simulation, distribution shift, tail
> events, model uncertainty, online adaptation and real historical validation are considered simultaneously?

v0.8 could not answer questions like this: its real holdouts were blocked (HTTP 403), and every confirmatory result
was synthetic. v0.9 registered several public sources and backups before freezing. Each was probed by headers only:
- Bybit linear-perpetual full-depth L2 with update ids;
- Bybit spot L2;
- Binance top-of-book quotes;
- trade prints with aggressor side.

## 2. Preregistration and the promotion gate

- **Before any exploratory access**, the protocol was frozen (`configs/v09/protocol.json`). It fixes:
  - the data pools and their randomized slot order with ordered backups;
  - the statuses, evidence levels E0–E10 and practical thresholds;
  - the statistical families (Holm within each) and the seed namespaces;
  - the realism gate;
  - six candidate execution conclusions C1–C6 (policy pair × mandate × endpoint).
- **Exploration.** 36 candidate hypotheses were registered with endpoint, estimator, unit, direction, threshold,
  family, fidelity and capability. Exploration on 2023-08..10 days estimated only effects, variances and feasibility.
- **Promotion.** `hypothesis-promotion-v09` applied the registered gate once, mechanically:
  - a split-half measurement screen;
  - shrunken planning effect ≥ practical threshold;
  - power ≥ 0.80 within the unit ceiling and the compute budget;
  - capabilities present.

  It promoted 12 hypotheses and carried C1–C3 ([C09-PROMOTION]). The registry was committed and pushed before any
  confirmatory day was read.
- **Amendment 3**, made after the pilots were inspected, replaced the frozen reliability screen with a split-half
  consistency screen. It is disclosed as a post-hoc change (incident 2). The original screen's decisions are kept
  as a sensitivity analysis: they promote H2, H3, H5, H6, H7, H8 and H29, and carry no conclusion. So the
  execution-conclusion results below exist only because of the amendment, and must be read with that in mind.
- **Confirmatory run order.** Every confirmatory branch was dry-run on explore days first, and the dry runs caught
  two defects. Then:
  1. all nine designs were sealed;
  2. confirmatory data were read under the sealed data design (8 quality-rejected days were replaced by registered
     backups in frozen order);
  3. the studies ran one at a time;
  4. the phase was closed;
  5. the 2025 vault was opened exactly once.

## 3. What the data could and could not say

- **Observable.** Public L2 shows aggregate depth per price level, never queue positions or order ids. The
  exact-fill condition used throughout is a *clearance* event: a print through a level bounds the fill time of any
  resting order there. So every fill quantity is bounded rather than exact.
- **Bounded counterfactual replay (F2).** A policy is injected into the reconstructed book under a registered grid
  of queue, latency, hidden-liquidity and impact assumptions. Each conclusion is tested on its conservative
  (worst-assumption) bound, and the costs are HYPOTHETICAL.
- **Data coverage** ([C09-DATA]):
  - 35 confirm days, 17 late days and 36 vault days were read;
  - every substitution was a registered backup;
  - Binance confirm data were never read, because the venue hypothesis was not promoted.
- **Realism gate.** It rejected every generator family for execution, realism, strategic and tail capabilities;
  fills were PARTIAL only ([C09-GATE]). Synthetic results therefore never stand in for real-market statements here.

## 4. Results

### 4.1 Event and path modelling on real days (F1, E4)

- **Neural point processes.** On held-out real days, neural temporal point processes predicted event timing and
  marks better than the best classical point process: 0.39 nats/event (0.32–0.46), so H2 is ESTABLISHED [C09-H2].
- **Model averaging.** Dynamic model averaging beat the single model selected on the preceding data by 0.076
  nats/event (0.069–0.085), so H5 is ESTABLISHED [C09-H5].
- **Not promoted:** sequence classifiers vs classical conditional models, because their split halves disagreed; H3
  is NOT_ESTABLISHED [C09-H3].
- **Not promoted (effect too small):**
  - latent path generators vs classical paths: H4 is NOT_ESTABLISHED [C09-H4]. Its pilot relative MMD was
    pathological because the classical MMD was near zero.
  - learned representations: H35 is NOT_ESTABLISHED [C09-H35].
- **Response realism:** hybrid vs fully generative response failed the measurement screen, so H1 is
  NOT_ESTABLISHED [C09-H1]. Strategic vs static ecology was too small to promote, so H12 is NOT_ESTABLISHED
  [C09-H12].

### 4.2 Fills, queues and markouts

- **Execution-aware calibration** of the queue-reactive generator improved the Brier score of back-of-queue
  clearance probabilities over generic calibration by 0.18 (0.05–0.32). H6 is ESTABLISHED [C09-H6]. The interval is
  wide, and the gate rates fills PARTIAL only.
- **Survival models** (best of Cox, Weibull AFT, boosted and neural hazard) beat Kaplan–Meier on integrated Brier
  score of time-to-clearance by 0.094 (0.074–0.117). H7 is ESTABLISHED [C09-H7].
- **Decision regret.** On the hybrid twin (real backbones with a fitted response), a calibrated fill model lowered
  the decision regret of a passive-first policy by 0.47 bps (0.33–0.62). H8 is ESTABLISHED [C09-H8] at E3, a hybrid
  rung, not a real-fill statement.
- **Adverse-selection-aware passive execution** (C5) was too small to promote, so H9 is NOT_ESTABLISHED [C09-H9].

### 4.3 Impact

- State-dependent impact vs the square-root law was too small in the pilot, so H10 is NOT_ESTABLISHED [C09-H10].
  Observational impact estimates are not causal identifications in any case.
- Self-impact ranking changes in synthetic ecologies were too small to promote, so H11 is NOT_ESTABLISHED
  [C09-H11].

### 4.4 Adaptation

- **Promoted:**
  - continual calibration with replay reduced forgetting relative to naive fine-tuning by 0.016 nats/event
    (0.012–0.019), so H16 is ESTABLISHED [C09-H16];
  - adaptive conformal intervals stayed within the registered 0.02 coverage margin of 90% under drift (mean coverage
    0.944), so H17 is ESTABLISHED [C09-H17] on the non-inferiority test.
- **Not promoted:**
  - active calibration, where the planning effect was below the 30% saving threshold: H13 is NOT_ESTABLISHED
    [C09-H13];
  - sequential posterior updating: H14 is NOT_ESTABLISHED [C09-H14];
  - few-shot adaptation, where the pilot already violated the non-inferiority margin: H15 is NOT_ESTABLISHED
    [C09-H15].

### 4.5 Decisions under uncertainty

- **H23 (CVaR MPC vs TWAP).** The registered robust (CVaR) MPC lowered the tail cost (CVaR90 of shortfall) relative
  to TWAP for the medium mandate by 11.4 bps (8.8–14.2) on the conservative bound in bounded replay. H23 is
  ESTABLISHED [C09-H23]. It is a bounded, hypothetical replay result.
- **H29 (synthetic vs hybrid ranking).** Synthetic policy rankings agreed with hybrid-twin rankings on 92% of
  determinate pairs on average per day. H29 is ESTABLISHED [C09-H29]. This is agreement between two model rungs, not with reality.
- **Agreement with bounded-historical rankings** failed the measurement screen:
  - hybrid vs bounded-historical ranking: H30 is NOT_ESTABLISHED [C09-H30];
  - multi-fidelity consensus: H31 is NOT_ESTABLISHED [C09-H31].
- **Pilot effects too small to promote (all NOT_ESTABLISHED):**
  - disagreement-predicts-regret: H18 is NOT_ESTABLISHED [C09-H18];
  - abstention: H19 is NOT_ESTABLISHED [C09-H19];
  - contextual fallback: H20 is NOT_ESTABLISHED [C09-H20];
  - distributionally robust execution: H21 is NOT_ESTABLISHED [C09-H21];
  - chance-constrained completion: H22 is NOT_ESTABLISHED [C09-H22];
  - adaptive MPC: H24 is NOT_ESTABLISHED [C09-H24];
  - meta-policy: H25 is NOT_ESTABLISHED [C09-H25];
  - offline RL safe improvement: H26 is NOT_ESTABLISHED [C09-H26];
  - conservative gating: H27 is NOT_ESTABLISHED [C09-H27];
  - rare-event-aware optimization: H28 [C09-H28].

  The synthetic decision studies among these ran only in simulated worlds with logged propensities.

### 4.6 Carried execution conclusions, transfer and the vault

All three carried conclusions were ESTABLISHED on BTCUSDT confirm days on the conservative bound, with Holm over the
carried set, under bounded hypothetical replay:
- C1, CVaR-MPC vs TWAP tail cost, medium mandate: 11.0 bps (7.9–14.5);
- C2, TWAP vs sweep mean cost, large mandate: 2.2 bps (1.2–3.1);
- C3, passive-with-deadline vs TWAP, small mandate: 2.3 bps (1.1–3.4).

Each then held on every transfer target:

| Target | Hypothesis | C1 | C2 | C3 |
|---|---|---|---|---|
| Later 2024 BTCUSDT days | H32 is ESTABLISHED [C09-H32] | 9.0 | 3.3 | 2.1 |
| ETHUSDT | H33 is ESTABLISHED [C09-H33] | 11.2 | 4.7 | 2.3 |
| Untouched 2025 vault | H36 is ESTABLISHED [C09-H36] | 6.8 | 7.3 | 1.3 |

- The vault covers SOLUSDT linear, BTCUSDT linear and BTCUSDT spot, and was read once after close.
- **Cross-venue:** H34 is NOT_ESTABLISHED [C09-H34], because no L1-identifiable conclusion was carried.
- **The ladder** ([C09-LADDER]) qualifies the vault result:
  - the single calibrated generator did not support C1;
  - the hybrid twin did not evaluate any conclusion;
  - the reality gap ([C09-GAP]) shows the generators mis-sizing every conclusion.

  So H36's ESTABLISHED status is a statement about the registered vault test, not about every rung.

## 5. What this means

The most defensible v0.9 statements are of two kinds.

**(a) Predictive-modelling results measured directly on real days (E4).** They are modest but clear:
- neural point processes;
- model averaging;
- survival models for clearance;
- execution-aware fill calibration;
- continual calibration;
- adaptive conformal coverage.

**(b) Three execution-policy comparisons that hold on a conservative bound of bounded hypothetical replay.** They
hold across time, instrument and an untouched year, and they show that the registered assumption grid does not
erase them.

Neither kind is a measurement of real fills or real P&L. The synthetic and hybrid rungs disagreed with bounded replay
on effect size, and with each other on C1. This is the gap the v0.9 question was built to expose. Simulators
calibrated on generic summaries are not yet trustworthy for execution decisions, and the realism gate says so.

## 6. Limits stated in advance

- All replay costs are HYPOTHETICAL, under registered queue, latency, hidden-liquidity and impact assumptions. No
  fill is exact, and no queue position is observed.
- The execution conclusions depend on Amendment 3, a post-hoc change to the promotion screen. Under the original
  screen they would not have been tested.
- Engineering defects were found and fixed throughout the cycle, and every one is listed in the final report.
  - Two confirm-branch defects were caught by the dry runs on explore days before any confirmatory access.
  - Two were caught after close: the H16 unit order in the final analysis (incident 4) and the vault data rounds
    (incident 5). Both were fixed to the registered definitions, and neither changed any hypothesis status.
- Confirmatory samples are small (8–21 days per test). They were sized by the registered power rule from shrunken
  pilot effects.
- Nothing here supports live trading, profit, HFT or production claims. Verification is integrity, not independent
  replication.
