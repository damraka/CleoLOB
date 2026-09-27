# Real-market validation of a limit-order-book execution simulator: CleoLOB v0.5

*Research report. Branch `research/v0.5-real-market-validation`. Not a release.*

## 1. Abstract

CleoLOB is an open limit-order-book simulator and execution research environment. Version
0.5 asks whether its components and policy conclusions hold up against real market data,
under a protocol frozen before any v0.5 result was produced. Fresh data were consumed once,
through a hash-chained ledger.

**Findings, including the negative ones:**

- **Order-level validation (M1).** The registered order-level validation on a genuine
  Bitstamp order feed is **INVALID**: the frozen adapter met venue message types it had never
  seen. A post-hoc analysis would have **FAILED** the frozen agreement gate (0.867 < 0.90),
  even though the replayed order state matched periodic REST censuses to within one order.
- **Bounded historical fills (M2).** These are genuinely uncertain. A third to two fifths of
  hypothetical join-the-best orders cannot be classified from aggregate L2. Among possible
  fills, the honest conservative-to-optimistic interval spans 17–21% of order size on ETH and
  52% on BTC.
- **Calibration v2 (M3).** A richer simulator family improves out-of-sample fit over the v0.4
  class on a fresh same-instrument period (**ESTABLISHED**). It still **FAILS** every absolute
  per-family gate, and the improvement does not transfer to BTC.
- **Impact and resilience (M4).** Simulator impact responses do **not** agree with history
  (**NOT_ESTABLISHED**) at any horizon or dataset. Historical impact persists; simulated impact
  reverts.
- **Completion (M5).** In stress regimes, up to 8% of mandates complete only after the horizon.
  A single settlement-inclusive completion number would hide this.
- **Regime robustness (M6).** The only regime-robust conclusion is the *relative* M3
  improvement on the unseen ETH period.
- **Policy study (M7).** In a registered 34,560-episode study, 5 of 64 contrasts pass the
  joint cost and completion gate, all DQN and all in two synthetic regimes. PPO passes none.
- **Transfer (M8).** On a fresh historical transfer holdout:
  - the single original-regime gate pass is indeterminate;
  - learned policies are *more* costly than TWAP, VWAP and Almgren–Chriss under both bounded
    fill modes;
  - the historical and synthetic rankings are unrelated.

No profitability, alpha, production-HFT, universal-generalization or learned-policy
superiority claim follows.

## 2. Research questions

| ID | Question | Outcome |
|---|---|---|
| RQ1 | Do identity-preserving MBO semantics reproduce a genuine venue order-level feed? | Registered: **INVALID**; post hoc: would be FAILED |
| RQ2 | How determinate are hypothetical passive fills from aggregate L2 versus order-level data? | Reported as bounds (descriptive) |
| RQ3 | Does a richer simulator family improve out-of-sample fit, and does it pass frozen gates? | Improvement **ESTABLISHED** (ETH); gates **FAILED**; BTC **NOT_ESTABLISHED** |
| RQ4 | Do simulator impact and resilience responses agree with history? | **NOT_ESTABLISHED** |
| RQ5 | How often is completion within the horizon versus only after settlement? | Separated; material in stress |
| RQ6 | Which conclusions survive regimes, an unseen period and a different instrument? | Relative M3 improvement only; cross-venue comparison **refused** |
| RQ7 | Do main-arm PPO or DQN pass the joint cost and within-horizon completion gate? | 5/64 contrasts (DQN only); stress and calibrated costs withheld |
| RQ8 | Do simulator policy conclusions survive bounded historical execution? | Mostly no: 6 reversals, 5 indeterminate, 1 agreement (not a learned advantage); July not evaluable |

## 3. Data

Details: [v05-data.md](v05-data.md).

**Datasets.**

| Dataset | Source | Role |
|---|---|---|
| Deribit ETH-PERPETUAL, 1 April / 1 May / 1 June 2020 | Tardis public L2 and trade samples | development, selection, retrospective holdout (all consumed before v0.5) |
| **ETH 1 July 2020, BTC-PERPETUAL 1 July 2020** | same | **external holdouts, fresh at freeze** |
| **ETH 1 August 2020** | same | **M8 transfer holdout, fresh at freeze** |
| Bitstamp btcusd, 900 s and 1,800 s | live public order-level recordings, replayed offline | development and validation |

**Freshness and access.**
- Freshness is defined only by the append-only, hash-chained consumption ledger
  (`configs/v05/consumption-ledger.jsonl`).
- A holdout cannot be read before a sealed design lists it.
- Consumed periods are never relabelled.

**Licensing.** Raw provider data and detailed derived data stay local. Tardis terms restrict
redistribution, and Bitstamp commercial use needs a licence.

## 4. Market-data observability

Each source declares a capability level, for example aggregate L2 with trades, or order-level
with identities.

- **Refusals.** Questions a source cannot answer are refused rather than approximated: exact
  FIFO position, order identity or hidden size from aggregate L2, and cross-venue comparison
  across different capability levels.
- **Vendor archive.** Genuine *historical vendor* MBO (LOBSTER, Databento) is
  **NOT_AVAILABLE**: access requires terms acceptance or credentials.
- **Bitstamp status.** The Bitstamp recordings are genuine venue messages, not a
  vendor-certified archive. Price-time priority is not established by that feed.

## 5. MBO validation (M1)

Details: [v05-mbo.md](v05-mbo.md).

**Procedure.**
1. The adapter was developed on a 900 s capture.
2. It was sealed (version `bitstamp-capture-4`).
3. A 1,800 s validation capture was then recorded. The first attempt ended in a
   disconnect, which was retained as a GAP; the single permitted retry completed.

**Registered outcome: INVALID.** The retry contained order subtypes at a sentinel price and
zero-price creations never seen in development. The frozen adapter refused them, so the
replay could not proceed.

**Post-hoc analysis (exploratory; one added rule).**
- Every census check matched within one order.
- Priority was concordant in 11,533 of 11,534 adjacent pairs.
- Replay was deterministic.
- The frozen endpoint E1 (exact top-10 agreement with the published `order_book` channel) was
  0.867. The frozen gate needs ≥ 0.90, so the outcome would be FAILED.
- Mismatches typically match a replay state 20–40 ms earlier, so E1 largely measures channel
  lag.

The endpoint stays as frozen.

**Negative finding.** Rules derived from 15 minutes of messages did not cover the venue's
vocabulary over the next 30.

## 6. Historical counterfactual methodology (M2)

Details: [v05-historical-execution.md](v05-historical-execution.md).

A hypothetical passive order receives nested bounds:

```text
conservative_lower ≤ fifo_lower ≤ fifo_upper ≤ optimistic_upper
```

Each bound is defined under explicit assumptions:
- **conservative:** only prints strictly through the price fill the order;
- **observable FIFO:** displayed queue ahead, which assumes price-time priority and no hidden
  size;
- **optimistic:** front of the level.

Every bound set assumes no market impact. Classes are:
- `GUARANTEED_FILL`, `POSSIBLE_FILL`, `GUARANTEED_NON_FILL`;
- `INDETERMINATE`, `UNSUPPORTED`;
- `OBSERVED_FILL`, for real orders only.

**Results (11,496 orders per day).**

| Period | Determinate under conservative bounds | Width, given possible (conservative→optimistic) |
|---|---:|---:|
| ETH development (April) | 67.6% | 0.189 |
| ETH July (external) | 69.6% | 0.214 |
| BTC July (cross-instrument) | 58.9% | 0.525 |

The observable-FIFO interval is much narrower (1–3% on ETH). That narrowing rests entirely on
assumptions aggregate L2 cannot establish.

**Bitstamp order-level check.** On the Bitstamp captures, every real order's actual fill lay
inside [conservative, optimistic] (2,702 / 2,702 post hoc) and equalled the identity-FIFO
value. The bounds were never falsified there.

## 7. Simulator model

The v0.4 engine is unchanged by default. It is an event-driven price-time-priority book with
zero-intelligence limit, market and cancel flow, latency, and a resilience mechanism.

**Calibration v2 extensions** (`lob.sim_v2`) are opt-in and exact under thinning. With none
active, the simulator is bit-identical to v0.4. The extensions are:
- empirical size tables;
- spread-conditioned arrivals;
- depth-conditioned cancellation;
- imbalance-conditioned market flow;
- a Markov activity regime;
- self-exciting market orders.

**Historical replay for M8** (`lob.historical_sim`) sits behind the same simulator interface,
with bounded passive fills.

## 8. Calibration v2 (M3)

Details: [v05-calibration.md](v05-calibration.md).

**Procedure.**
- One observable operator measures 12 families on historical and simulated 100 ms tapes.
- Every family gets the same budget.
- Chronology: fit on April, select on May, seal, then evaluate unchanged.

**Results.**

| Holdout | Selected E3 | v0.4 control E3 | Improvement (H3a, α = 0.05/3) | Gate (every family ≤ 0.60) |
|---|---:|---:|---|---|
| June ETH (retrospective) | 1.594 | 2.355 | ESTABLISHED | FAILED / FAILED |
| **July ETH (fresh)** | **1.390** | **2.567** | **ESTABLISHED** [−1.35, −0.70] | **FAILED / FAILED** |
| July BTC (cross-instrument) | 2.688 | 2.763 | NOT_ESTABLISHED | FAILED / FAILED |

**Disclosed deviation.** The selection seal followed the first July access by the M2 study.
The selection read only April and May.

## 9. Market impact and resilience (M4)

Details: [v05-impact.md](v05-impact.md).

**Design.** Signed mid-price responses to aggressive trades were measured at 1, 10 and 60 s,
by size tercile, with Bonferroni correction over 36 hypotheses.

**H4 is NOT_ESTABLISHED at every horizon and dataset for both simulators.**
- The selected simulator matches large-trade responses on ETH (July, 60 s: 4.38 bps
  historical vs 4.58 bps simulated).
- It misses the middle tercile and sub-lot flow.
- Its impact partially reverts: persistence (60 s / 1 s response) is 0.85, against 1.10–2.72
  historically.
- Resilience recovery share (> 98%) does not discriminate between history and either simulator.

Almgren–Chriss quantities are reported as empirical proxies only.

## 10. Execution protocol (M5)

Details: [v05-execution.md](v05-execution.md).

- **E5 (within-horizon completion)** counts only fills at or before the horizon end.
- **Settlement** allows only cancellations and exchange events. Its fills are post-horizon and
  never count toward the mandate.
- **E6 (completion-adjusted cost)** is realized fill cost plus hypothetical residual
  valuation. Both components are always reported, and residual valuation never creates a fill.

The rules are identical for every control and policy.

## 11. Classical baselines

The baselines are TWAP, VWAP (synthetic original-regime profile), POV, Almgren–Chriss
(parameters identified on dedicated seeds and locked before evaluation), a heuristic and a
random policy. All share the urgency rule and the mandate.

## 12. RL policies (M7)

Details: [v05-rl.md](v05-rl.md).

**Models.**
- PPO and DQN (Stable-Baselines3), each in three arms: `main`, `no_book` and `no_terminal`.
- 8 training seeds × 16,384 steps each, i.e. 48 models.
- Trained only on the original regime and training-domain seeds.
- Normalization was sealed from training-domain exploration before any fit.
- The final fixed-budget checkpoints are evaluated unchanged.

**Evaluation.** 160 unseen market seeds × 4 regimes (original, shifted, stress, calibrated),
for 34,560 episodes.

**Environment freeze.** The environment was frozen (`configs/v05/environment-freeze.json`)
before M7. The M8-only replay module was later revised twice to fix crashes. M7 never
uses it.

## 13. Statistical design

Details: [v05-statistics.md](v05-statistics.md).

- Every family, correction and gate was fixed before the data it evaluates were accessed.
- Bonferroni correction throughout.
- Block bootstraps for serially dependent historical data.
- A crossed training-seed × market-seed bootstrap for M7 cost.
- A market-level one-sided Clopper–Pearson bound for M7 completion. It is preregistered to
  avoid v0.4's degenerate bootstrap.
- INVALID cells withhold comparisons and never shrink the family.
- No prospective power analysis was performed, so null results do not establish equivalence.

## 14. Results

### 14.1 M7: registered policy study

5 of 64 contrasts pass the joint gate (cost upper bound < 0 and completion lower bound
≥ −0.05, α = 0.05/128):

| Regime | Contrast | Cost difference (bps) |
|---|---|---|
| original | DQN − POV | −0.120 [−0.227, −0.013] |
| shifted | DQN − TWAP | −0.37 |
| shifted | DQN − VWAP | −0.35 |
| shifted | DQN − POV | −0.42 |
| shifted | DQN − AC | −0.35 |

- **PPO:** passes none.
- **Stress and calibrated:** cost comparisons are withheld because of INVALID residual
  valuations (109 and 5,477 episodes).
- **`no_terminal` ablation:** vacuous (bit-identical models).

### 14.2 M8: historical versus synthetic transfer

Details: [v05-transfer.md](v05-transfer.md).

**Setup.**
- The registered run (attempt 2) replays 144 episodes per dataset under both bounded fill
  modes.
- Mapping: a clock ratio of 4.12 and a lot scale of 0.0137, both fitted on development data
  only.
- 78 hypotheses, Bonferroni-corrected.

**August 2020 ETH (fresh transfer holdout):**

| Class | Count | Pairs |
|---|---:|---|
| reverses | 6 | PPO and DQN versus TWAP, VWAP and AC. History: learned policies costlier by +0.44 to +0.54 bps, excluding zero in both modes. Synthetic: small negative, non-significant means. |
| indeterminate | 5 | including **DQN − POV**, the only original-regime M7 gate pass: −0.095 [−0.359, 0.155] |
| agrees_survives_conservative | 1 | PPO − heuristic, where both the synthetic and historical directions have PPO *costlier* |

**Other August results.**
- **Ranking:** the historical ranking puts TWAP, AC and VWAP cheapest and DQN, PPO and POV
  most costly. Kendall τ versus the synthetic ranking is −0.14 in both modes.
- **Fill mode:** conservative and optimistic results differ by ≤ 0.05 bps for all but the
  heuristic, so the cost differences are largely invariant to the fill assumption.
- **Completion:** within-horizon completion is 100% for all policies.

**July ETH and July BTC** are **not_evaluable** in the registered run. Three historical
episodes (96 rows) crashed on a transient crossed book inside one replay update.

A post-hoc rerun (attempt 3; EXPLORATORY_POST_HOC) followed the fix. Every previously valid
row is bit-identical in it, and it adds the July comparisons:

- **July ETH:** PPO again reverses against TWAP, VWAP and AC (+0.18 to +0.22 bps). DQN is
  indeterminate throughout.
- **July BTC:** nothing reverses. PPO − POV and DQN − POV favour the learned policy under the
  optimistic fill mode only (`optimistic_assumption_dependent`).

In neither attempt does any learned-policy advantage survive the conservative fill bound on
any dataset.

### 14.3 Completion (M5 endpoint within M7)

- **Original and shifted regimes:** every policy completes 100% within the horizon.
- **Stress:** 91.9–100% within the horizon and 98.1–100% after settlement. Up to 8% of mandates
  complete only during settlement; PPO and DQN have 71 and 69 settlement-only completions
  (of 1,280 each).
- **Calibrated synthetic regime:** 25.0–57.5% for controls, and 27.7% (PPO) / 34.2% (DQN) for the
  learned policies.

## 15. Negative, null and invalid findings (all retained)

1. **M1:** the registered validation is INVALID, and the post-hoc E1 would FAIL.
2. **M2:** aggregate L2 leaves 30–41% of hypothetical passive orders undetermined, and the
   bounds are wide on BTC.
3. **M3:** every absolute calibration gate FAILED on every holdout for both models. The BTC
   improvement is NOT_ESTABLISHED. Seven combined candidates hit the event cap.
4. **M4:** H4 is NOT_ESTABLISHED everywhere, with persistence reversed in the simulator.
5. **M6:** no absolute calibration or impact conclusion survives any regime. Cross-venue
   comparison is refused.
6. **M7:**
   - PPO passes no contrast.
   - DQN fails against the heuristic and random controls.
   - Stress and calibrated costs are withheld (INVALID economics).
   - The `no_terminal` ablation is vacuous.
7. **M8:**
   - Attempt 1 was INVALID.
   - In the registered attempt 2, six reversals show learned policies costlier than
     schedule-based controls historically.
   - The M7 original-regime gate pass is indeterminate.
   - The July comparisons are not_evaluable.
   - Post hoc, no learned advantage survives the conservative bound on any dataset.
   - The design's regime-sensitivity report was omitted from the transfer run and added
     afterwards as a separate descriptive analysis.

## 16. Sensitivity analysis

- **Regimes (M6):** the relative M3 improvement holds in all 11–13 regime labels on ETH, but
  not on BTC.
- **Fill assumptions (M8):** conclusions are nearly invariant between the conservative and
  optimistic modes.
- **Seeds (M7):** results are resampled over 8 training seeds and 160 market seeds.
- **M8 by regime** (descriptive, frozen M6 thresholds):
  - No August conclusion is regime-robust.
  - Reversals recur in 4–7 of 11 labels and are indeterminate elsewhere.
  - No label classifies a reversing pair as agreeing with the synthetic direction.
  - See [v05-transfer.md](v05-transfer.md#regime-sensitivity).

## 17. Limitations

- **No independent historical MBO archive.** The order-level evidence covers one venue and
  short windows.
- **No market impact of hypothetical orders** in historical replay or in the fill bounds.
  Hidden liquidity and queue position remain unobservable in L2.
- **Deribit coverage:** first-of-month single days from mid-2020, on one venue.
- **Consumed July periods:** they were fresh for M2–M6 but already consumed when M8 read them.
- **Calibrated simulator:** it failed its real-market gate, so calibrated-regime policy
  results are synthetic.
- **No power analysis.** Null results do not establish equivalence.
- **Hashes and ledger** provide consistency, not an external timestamp. They cannot prove
  non-inspection outside the workflow.

## 18. Reproduction

Commands, in dependency order, are in [v05-reproduction.md](v05-reproduction.md).
- **Public evidence:** the bundle is `examples/studies/v05/evidence`. It holds summaries,
  bindings and hashes, never raw or detailed derived data.
- **Verification:** `cleo verify-v05 <run>` checks a local run; `cleo verify-artifact` checks
  the bundle.

## 19. Conclusions

v0.5 turns CleoLOB's claims into bounded, registered tests, and most of the answers are
negative or limited:

- the simulator fits real data better than before, but not well enough;
- its impact dynamics differ from history;
- aggregate L2 cannot determine many passive fills;
- the small learned-policy advantages found in simulation do not survive a fresh historical
  holdout under bounded replay.

These are the results the protocol was built to find if they were true. They define what
the simulator can and cannot be used to claim.
