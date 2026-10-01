# When does a calibrated limit-order-book simulator support reliable execution conclusions?

*CleoLOB v0.6 research report (branch `research/v0.6-market-realism`, not a release).*

## Abstract

We preregistered twelve hypotheses on calibration identifiability, calibration uncertainty,
model risk and transfer for a limit-order-book simulator. The simulator was calibrated to
aggregate-L2 data for Deribit ETH-PERPETUAL. All twelve were tested with sealed designs and
three never-inspected holdout days.

**What held:**
- A new multi-objective calibration (v3) reliably improved realism *relative* to the previous
  version: on a retrospective day, a fresh same-instrument day and a fresh cross-instrument
  day.
- Yet no simulator came close to history in *absolute* terms:
  - nearly every realism family missed a margin defined by the difference between two real
    days;
  - simple discriminators separated real from simulated minute-windows with AUC ≈ 1;
  - essentially no historical window lay inside the simulators' support.
- Calibration was not identified: two materially different parameter vectors fit equally
  well, and they differ in self-excitation, inside-spread placement and regime structure.

**What did not hold:**
- Regime-conditioned calibration failed both within and outside its regime.
- In simulation, execution comparisons across plausible worlds, structural interventions and
  regime models were dominated by model risk rather than by policy differences. No ranking
  reversal could be certified at the design's resolution.
- On a fresh historical transfer day, every pairwise conclusion between classical and learned
  agents was indeterminate under both bounded fill semantics. Domain-randomized training did
  not improve transfer.

The evidence supports a negative conclusion. At this data level and scale, calibrated
simulators improve measurably while staying distinguishable from markets, and their execution
rankings do not carry a determinate signal to history.

## 1. Question

v0.5 asked which simulator and execution conclusions survive contact with real data. Its
answers were mostly negative. v0.6 asks *why*:
- Are simulator conclusions unstable because calibration is ambiguous, because the model
  class is misspecified, because regimes shift, or because history cannot resolve them?
- Which realism failures matter for execution decisions?

## 2. Data and chronology

**Data.** Deribit inverse perpetual futures, Tardis public first-of-month samples, aggregate L2
plus aggressor-signed prints. There is no order-level data, so the following are reported
`NOT_AVAILABLE`:
- order size;
- modify intensity;
- queue position;
- hidden liquidity.

**Chronology:**
- **development:** ETH 2020-04-01;
- **selection:** ETH 2020-05-01;
- **retrospective** (consumed by v0.4/v0.5): June, July, August ETH and July BTC;
- **fresh:** ETH and BTC 2020-09-01 (realism) and ETH 2020-10-01 (execution transfer).

Every analysis that read a fresh day was sealed in a hash-chained ledger before that day's
first access. The ledger also enforces role-specific uses, posthoc labelling and source
immutability ([v06-research-protocol.md](v06-research-protocol.md)).

## 3. Methods

- **Realism measurement.** One measurement operator turns history and simulation into
  10-level, 100 ms tapes. It reduces each 600 s block to additive sketches with 85 components
  in nine families ([v06-realism.md](v06-realism.md)).
- **Equivalence margins.** The margin for each family is the real-versus-real distance
  between the development and selection days, sealed before any holdout access.
- **Calibration v3.** 14 active parameters, 8 seeded starts and 2,304 candidates, with
  re-scoring and selection on separate data. Near-optimal and materially-distinct rules were
  frozen in advance ([v06-calibration.md](v06-calibration.md)).
- **Execution.**
  - Mandate: buy 14 lots in 120 s.
  - Agents: TWAP, VWAP, POV and AC, plus PPO and DQN trained either in the selected world or
    with domain randomization over the plausible ensemble.
  - Worlds: 19, with 200 market seeds each ([v06-model-risk.md](v06-model-risk.md)).
- **Domain gap.** Leakage-resistant two-sample discriminators and nearest-neighbour support
  ([v06-domain-gap.md](v06-domain-gap.md)).
- **Transfer.** Bounded historical replay under conservative and optimistic fills
  ([v06-transfer.md](v06-transfer.md)).
- **Statistics.** Block and seed bootstraps, explicit families with Bonferroni or Holm, and
  one-sided equivalence ([v06-statistics.md](v06-statistics.md)).

## 4. Results

**4.1 Relative calibration improvement (H1–H3: ESTABLISHED).** The v3 objective was lower
than the v0.5 control's on every dataset:

| Dataset | v3 − v0.5 |
|---|---|
| June, retrospective | −1.76 [−1.87, −1.63] |
| fresh ETH | −1.02 [−1.18, −0.88] |
| fresh BTC | −1.17 [−1.27, −1.10] |

The improvement is mostly in book state: book-state error 1.37 vs 2.87 on fresh ETH. The
trade-off is visible elsewhere: the v0.5 control matches event activity within margin on both
July days, and v3 does not.

**4.2 Absolute realism fails.**
- On the fresh holdouts:
  - the v3 objective is 2.91 (ETH) and 2.99 (BTC), against a real-versus-real yardstick of
    1.25;
  - no family is equivalent within its margin;
  - returns is the only family not clearly failing (`NOT_ESTABLISHED`).
- A logistic regression on 20 scale-free window features separates real from simulated
  minutes with test AUC 1.00 (H8 ESTABLISHED).
- 100% of fresh windows lie outside the simulators' support.

**4.3 Calibration is not identified (H4: ESTABLISHED).** The two near-optimal vectors are
materially distinct (L∞ = 0.66 in the unit box) and score 2.44 vs 2.59 on the selection day:

| Mechanism | Vector 1 | Vector 2 |
|---|---|---|
| Hawkes branching ratio | 0.04 | 0.66 |
| inside-spread placement | 0.8% | 48% |
| high-activity regime share | 12% | 41% |

Profiles separate strongly constrained rates (cancellation, market order, limit order and
placement depth) from weakly constrained resilience, depth target and excitation decay.

**4.4 Model risk dominates execution comparisons (H5–H7: NOT_ESTABLISHED).** These results
are simulation only.
- The two plausible worlds' point rankings are nearly unrelated (Kendall τ = 0.07), but no
  reversal is certified.
- The largest between-member difference for a classical agent is 1.0 bps, below the design's
  resolution of about 1.5–2.4 bps for differences.
- For learned policies, the spread across regime models (2.0–4.8 bps) and structural
  interventions (about 1–1.4 bps) exceeds both the market-seed standard error (about 0.5 bps)
  and the training-seed spread (0.2–0.6 bps). It is as large as the apparent learned-vs-
  classical cost differences.
- No realism family's error predicts conclusion changes across worlds; the largest is
  temporal, ρ = 0.44.

**4.5 Regime conditioning fails (H9, H10: FAILED).** The high-volatility model improves
realism within its regime (−0.32). The low-volatility model, selected on only 23 blocks,
worsens it (+0.96). Both models degrade outside their regimes, by up to +1.51 against a
noninferiority margin of 0.24.

**4.6 Transfer has nothing determinate to transfer (H11 NOT_ESTABLISHED; H12 vacuously
ESTABLISHED).**
- On fresh ETH 2020-10-01, bounded historical costs lie between 1.48 and 2.40 bps, with 100%
  completion.
- All 15 pairwise contrasts are indeterminate under both fill modes.
- Only the single-world simulator made a determinate prediction (POV costlier than DQN), and
  pooling worlds removed even that.
- Domain-randomized training changed the simulator-to-history gap by −0.6 (PPO) and +0.4
  (DQN) bps, with intervals spanning zero.

## 5. Discussion

- **Realism improved but did not converge.** Calibration v3 moved the simulator closer to
  history on every family where the v0.5 model was worst, and it did so out of sample and
  across instruments. But the measured distance to history is still about 2–3× the distance
  between two real days, and the discriminator result is unambiguous.
- **The fit does not determine the mechanism.** That two different excitation and regime
  structures fit equally well is the most direct evidence for the paper's premise. Observable
  fit does not determine mechanism.
- **Execution conclusions sit inside model risk.** For execution, the uncertainty from
  plausible-world choice and regime modelling is of the same size as the policy differences
  being measured. History, under bounded fills, does not resolve the policies either.
- **The conclusion.** Under these conditions a calibrated simulator does **not** support
  reliable execution-policy conclusions. It supports relative statements about simulators,
  not about markets.

## 6. Threats to validity

- **Data scope:**
  - one venue, one era and two instruments;
  - aggregate L2 only;
  - first-of-month days only.
- **Ensemble size.** The ensemble has two members. Calibration uncertainty is probably
  under-represented, which biases H5 and H6 towards NOT_ESTABLISHED.
- **Power.** Execution contrasts are underpowered relative to the 1 bps margin. NOT_ESTABLISHED
  is not evidence of agreement.
- **Objective dependence.** The objective and margins are particular choices; family-level
  results are reported to limit this. The real-versus-real margin uses one pair of days.
- **Replay assumptions.** Historical replay assumes no market impact of the hypothetical
  parent. Fills are bounded, not exact.
- **Event cap.** The deterministic event cap excludes very active parameter regions by design.
- **Disclosed deviations.** OOD standardization, the domain-gap resampling unit before sealing,
  a ledger rule bug fixed before any holdout byte was read, and two protocol amendments
  before calibration ([v06-final-report.md](v06-final-report.md)).

## 7. Negative results (retained)

- **FAILED:**
  - H9 and H10;
  - every equivalence margin on fresh data;
  - support on every dataset.
- **NOT_ESTABLISHED:**
  - H5, H6 and H7;
  - H11 for both algorithms.
- **Vacuous:** H12. It is reported as stability of indeterminacy.
- **INVALID attempts:** three, all retained ([v06-reproduction.md](v06-reproduction.md)).

## 8. What is not claimed

The following are not claimed:
- live profitability;
- alpha;
- production or HFT performance;
- exact FIFO or exact historical passive fills;
- hidden liquidity;
- causal effects of real-market mechanisms (structural interventions act on the simulator
  only);
- universal cross-market validity.
