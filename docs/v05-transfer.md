# v0.5 M8 — historical versus synthetic policy transfer

**Question (RQ8).** Do the simulator's policy conclusions from M7 (original regime) survive
bounded historical execution?

**Answer.** On the registered transfer holdout (Deribit ETH-PERPETUAL, 1 August 2020) they
mostly do not:
- The single M7 original-regime gate pass (DQN − POV) is **indeterminate** historically.
- Against TWAP, VWAP and Almgren–Chriss, both learned policies are *more* costly by about
  0.5 bps under both fill modes. That **reverses** the synthetic direction, although that
  synthetic direction was itself not significant.
- The historical ranking is unrelated to the synthetic one (Kendall τ = −0.14).
- The July ETH and July BTC comparisons are **not_evaluable** in the registered run because
  three historical episodes crashed (INVALID).

## Frozen design

The design is `configs/v05/m8-design.json`, sealed before any August access.

**Datasets.**
- Transfer holdout: ETH 2020-08-01, fresh at freeze.
- Secondary: ETH and BTC 2020-07-01. These were fresh at freeze but consumed by M2–M6 before M8.
- The mapping uses the development period only.

**Mapping.** It is fitted on April 2020 ETH against the M7 original regime and applied
unchanged everywhere, including BTC:
- clock: 4.12 historical seconds per simulated second, from the ratio of top-of-book change rates;
- lot scale: 0.0137 simulated lots per native unit, from median top-5 side depth.

**Episodes.**
- 144 per dataset, starting 300 s + k·600 s after the first capture.
- Each episode lasts (warm-up + horizon + settlement + 1 s) mapped by the clock, i.e. about
  35 historical seconds.

**Replay** (`lob.historical_sim.HistoricalSimulator`):
- Displayed levels are identity-free liquidity, reset to history at every update. The
  hypothetical parent has no market impact.
- Own marketable orders walk displayed levels only.
- Own resting orders fill only from the M2 fill-bound trackers, in one declared mode:
  - `conservative`: only prints strictly through the price;
  - `optimistic`: front of the level.

  Each mode is a separate self-consistent path, because later decisions depend on earlier
  fills. No exact FIFO fill is claimed.

**Policies.**
- The six M7 controls, including the M7-locked AC parameters.
- Every main-arm PPO and DQN checkpoint (8 training seeds each), with the sealed normalization.
- Checkpoint hashes are re-verified before loading.
- Learned values are averaged over training seeds per episode.

**Statistics.**
- Mean paired episode cost difference (E6), per dataset and fill mode.
- Percentile bootstrap with 5,000 resamples.
- Bonferroni over 13 pairs × 3 datasets × 2 modes = 78, so α = 0.05/78.

**Classification (E7).** The synthetic direction is the sign of the M7 original-regime mean
difference.

| Class | Rule |
|---|---|
| `agrees_survives_conservative` | the historical interval excludes zero in the synthetic direction in both modes |
| `optimistic_assumption_dependent` | it does so in the optimistic mode only |
| `reverses` | it excludes zero in the opposite direction in both modes |
| `indeterminate` | any other evaluable outcome |
| `not_evaluable` | an INVALID or missing episode in either mode, or no synthetic comparison (PPO − DQN) |

## Attempts (all retained)

| Attempt | Directory | Status | What happened |
|---|---|---|---|
| 1 | `results/v05/m8/registered` | **INVALID** execution | 2,185 of 19,008 rows raised a crossed-book assertion when historical liquidity moved onto an own resting order, so all 39 comparisons were not_evaluable. No cost direction was inspected. Fix: such levels are withheld from the engine book (ledger entry 62). |
| — | (none) | interrupted | A session disconnect killed the first attempt-2 launch before any artifact was written. Its ledger accesses (63–64) are kept. |
| **2** | `results/v05/m8/attempt2` | **registered M8 outcome** | 96 of 19,008 rows (3 episodes: ETH July #82, BTC July #90 and #126) raised a crossed-book assertion. The 13 August comparisons are evaluable. The 26 July comparisons are not_evaluable under the frozen rule. |
| 3 | `results/v05/m8/attempt3-posthoc` | **EXPLORATORY_POST_HOC** | See [below](#post-hoc-attempt-3). |

The attempt-2 crash was a transient crossing inside one replay update. `_sync` applied one
side of the book completely before removing stale levels on the other. When one historical
update moved the book by two or more levels, new bids briefly faced stale asks while the
engine checked invariants. The fix removes stale levels on both sides first; the final state
of every update is unchanged. It was recorded in ledger entry 70 before attempt 3 ran.

Attempt 3 followed inspection of the August results, so it cannot replace attempt 2.

## Registered results (attempt 2): August 2020 ETH transfer holdout

Mean paired cost difference (left − right) in bps, with the Bonferroni interval
(α = 0.05/78). Negative means the learned policy is cheaper.

| Pair | M7 synthetic (original regime) | Historical, conservative | Historical, optimistic | Class |
|---|---|---|---|---|
| PPO − TWAP | −0.039 [−0.146, 0.070] | +0.541 [0.148, 0.927] | +0.532 [0.138, 0.896] | reverses |
| PPO − VWAP | −0.041 [−0.151, 0.062] | +0.467 [0.092, 0.838] | +0.458 [0.082, 0.844] | reverses |
| PPO − POV | −0.064 [−0.165, 0.030] | −0.072 [−0.333, 0.162] | −0.080 [−0.337, 0.181] | indeterminate |
| PPO − AC | −0.036 [−0.135, 0.068] | +0.527 [0.159, 0.936] | +0.519 [0.121, 0.941] | reverses |
| PPO − heuristic | +0.017 [−0.077, 0.121] | +0.283 [0.007, 0.597] | +0.333 [0.037, 0.640] | agrees_survives_conservative |
| PPO − random | −0.014 [−0.136, 0.112] | +0.226 [−0.204, 0.632] | +0.217 [−0.176, 0.607] | indeterminate |
| DQN − TWAP | −0.095 [−0.196, 0.018] | +0.517 [0.139, 0.880] | +0.510 [0.155, 0.869] | reverses |
| DQN − VWAP | −0.097 [−0.199, 0.009] | +0.443 [0.089, 0.835] | +0.435 [0.038, 0.843] | reverses |
| **DQN − POV** (M7 gate pass) | **−0.120 [−0.227, −0.013]** | −0.095 [−0.359, 0.155] | −0.103 [−0.413, 0.138] | **indeterminate** |
| DQN − AC | −0.091 [−0.187, 0.014] | +0.504 [0.124, 0.938] | +0.496 [0.098, 0.949] | reverses |
| DQN − heuristic | −0.038 [−0.131, 0.054] | +0.260 [−0.068, 0.551] | +0.310 [0.035, 0.632] | indeterminate |
| DQN − random | −0.070 [−0.182, 0.044] | +0.203 [−0.226, 0.597] | +0.195 [−0.211, 0.609] | indeterminate |
| PPO − DQN | — | +0.023 [−0.085, 0.138] | +0.023 [−0.085, 0.147] | not_evaluable (no synthetic comparison) |

The M7 intervals are the registered M7 crossed-bootstrap intervals at α = 0.05/128.

**Counts over all 39 comparisons:**

| Class | Count |
|---|---:|
| reverses | 6 |
| indeterminate | 5 |
| agrees_survives_conservative | 1 |
| optimistic_assumption_dependent | 0 |
| not_evaluable | 27: 26 July, plus PPO − DQN on August |

**Reading.**
- **The one "agreement" is not a learned-policy advantage.** It is PPO − heuristic, where the
  synthetic mean was slightly *positive* (PPO costlier, not significant). History confirms
  PPO costlier.
- **The reversals are a sign flip.** The synthetic means were small, negative and not
  significant. History excludes zero on the costly side, by about 0.4–0.5 bps, under both
  fill modes.
- **The only M7 gate pass in the original regime, DQN − POV, does not transfer
  determinately.** Its historical mean has the same sign, but the interval includes zero.
- **Fill mode barely matters.** Conservative and optimistic results differ by ≤ 0.05 bps
  except against the heuristic. With horizons of about 8 historical seconds, the bounded
  passive-fill assumption appears to bind rarely. Fill fractions per mode are in the sealed
  rows. The cost differences are therefore largely invariant to the fill assumption.
- **Completion.** Within-horizon completion (E5) is 100% for every policy on August.

**Rankings by mean E6 cost (bps, August).**

| Mode | Ranking, cheapest first | Kendall τ vs synthetic |
|---|---|---:|
| synthetic (M7 original) | DQN, heuristic, PPO, random, AC, TWAP, VWAP, POV | — |
| conservative | TWAP 3.01, AC 3.02, VWAP 3.08, heuristic 3.27, random 3.32, DQN 3.53, PPO 3.55, POV 3.62 | −0.14 |
| optimistic | TWAP, AC, VWAP, heuristic, random, DQN, PPO, POV | −0.14 |

**July periods (descriptive only).** These are means over non-INVALID rows only; no
classification follows. On ETH, TWAP was cheapest in both modes (τ −0.29 / −0.21) and PPO
most costly. On BTC, PPO was cheapest in the conservative mode and the heuristic in the
optimistic mode (τ 0.21 / 0.57). On BTC, every mean lies within 0.12 bps of the others. The
attempt-2 completion rates below 100% on July come only from INVALID rows, which the
descriptive rate counts as non-completions. Every valid row completed within the horizon.

## Post-hoc attempt 3

**Status: EXPLORATORY_POST_HOC.** Attempt 3 reran the unchanged design after the replay fix
(ledger entry 70). It is secondary evidence and does not replace the registered outcome.

**Integrity check.**
- Attempt 3 has 0 INVALID rows out of 19,008.
- Every row that was valid in attempt 2 (18,912) is **bit-identical** in attempt 3, including
  all 6,336 August rows. The fix changed only the three crashing episodes.
- The August classifications are therefore unchanged.

**July results (first observed here).** Mean paired cost difference in bps; Bonferroni
interval at α = 0.05/78.

| Pair | ETH July, conservative | ETH July class | BTC July, conservative (/ optimistic where the class differs) | BTC July class |
|---|---|---|---|---|
| PPO − TWAP | +0.221 [0.090, 0.375] | reverses | −0.026 [−0.108, 0.030] | indeterminate |
| PPO − VWAP | +0.201 [0.076, 0.326] | reverses | −0.025 [−0.106, 0.029] | indeterminate |
| PPO − POV | +0.095 [−0.121, 0.376] | indeterminate | −0.130 [−0.391, 0.009] / −0.141 [−0.414, −0.013] | optimistic_assumption_dependent |
| PPO − AC | +0.185 [0.034, 0.351] | reverses | −0.037 [−0.143, 0.029] | indeterminate |
| PPO − heuristic | +0.154 [0.039, 0.264] | agrees_survives_conservative | −0.029 [−0.104, 0.020] | indeterminate |
| PPO − random | +0.052 [−0.070, 0.183] | indeterminate | −0.048 [−0.163, 0.023] | indeterminate |
| DQN − TWAP | +0.117 [−0.107, 0.334] | indeterminate | −0.011 [−0.092, 0.056] | indeterminate |
| DQN − VWAP | +0.097 [−0.144, 0.306] | indeterminate | −0.010 [−0.091, 0.064] | indeterminate |
| **DQN − POV** | −0.009 [−0.118, 0.108] | indeterminate | −0.115 [−0.372, 0.008] / −0.123 [−0.326, −0.009] | **optimistic_assumption_dependent** |
| DQN − AC | +0.081 [−0.166, 0.322] | indeterminate | −0.022 [−0.113, 0.034] | indeterminate |
| DQN − heuristic | +0.050 [−0.138, 0.247] | indeterminate | −0.014 [−0.076, 0.028] | indeterminate |
| DQN − random | −0.052 [−0.304, 0.167] | indeterminate | −0.032 [−0.104, 0.019] | indeterminate |

**Counts over the 39 comparisons of attempt 3:**

| Class | Count |
|---|---:|
| reverses | 9 |
| indeterminate | 23 |
| agrees_survives_conservative | 2 (PPO − heuristic on August and July ETH) |
| optimistic_assumption_dependent | 2 (PPO − POV and DQN − POV on BTC) |
| not_evaluable | 3 (PPO − DQN) |

**Kendall τ vs synthetic, conservative / optimistic.**

| Period | τ |
|---|---|
| July ETH | −0.29 / −0.29 |
| July BTC | 0.29 / 0.50 |
| August | −0.14 / −0.14 |

**Completion.** Within-horizon completion was 100% for every policy, dataset and mode.

**Reading (post hoc).**
- **ETH reversals.** On July ETH, PPO again reverses against TWAP, VWAP and AC (costlier
  historically), at a smaller magnitude than in August (+0.18–0.22 bps). DQN is
  indeterminate throughout.
- **BTC.** Nothing reverses on BTC. The only direction-consistent results are PPO − POV and
  DQN − POV, and they hold under the optimistic fill mode only. Their conservative intervals
  touch zero, so a learned advantage over POV on BTC depends on the optimistic
  queue-position assumption.
- **Across attempts.** No learned-policy advantage survives the conservative fill bound on
  any dataset in either attempt.

## Regime sensitivity

The design lists regime sensitivity under the frozen M6 thresholds among the reported
quantities, but the transfer run did not compute it. `cleo transfer-regimes`
(`lob/transfer_regimes.py`) was added afterwards as a descriptive analysis.

**Method.**
- It reads the sealed rows unchanged.
- Each episode is labelled by the development-threshold 5-minute block that contains its window
  midpoint.
- The frozen classification is repeated in every label with at least 6 episodes, at the
  registered α. No hypothesis is added.
- A conclusion counts as regime-robust only if it recurs in every evaluated label (the M6 rule).

**Coverage.**
- Per dataset, one of 144 episodes lies in no complete block and is unlabelled.
- Labels with 6–9 episodes cannot produce the frozen interval, which needs ≥ 10 episodes, so
  they are not_evaluable. On August this is the 8-episode `stress` label.

Runs: `results/v05/m8/attempt2-regimes` (registered rows) and
`results/v05/m8/attempt3-posthoc-regimes`.

**August (identical in both runs; 11 labels evaluated; `activity=low` and `imbalance=low` NOT_AVAILABLE).**

| Pair (dataset class) | Labels: reverses / indeterminate / not_evaluable |
|---|---|
| PPO − TWAP, PPO − AC, DQN − TWAP (reverses) | 7 / 3 / 1 |
| PPO − VWAP, DQN − AC (reverses) | 6 / 4 / 1 |
| DQN − VWAP (reverses) | 4 / 6 / 1 |
| DQN − POV (indeterminate) | 1 label `agrees_survives_conservative`, 9 indeterminate, 1 not_evaluable |
| PPO − heuristic (agrees) | 1 agrees, 3 optimistic-dependent, 6 indeterminate, 1 not_evaluable |

**What the August labels show:**
- **Regime robustness:** no August conclusion is regime-robust.
- **Direction:** no evaluable label classifies a reversing pair as agreeing with the
  synthetic direction. The reversal is reproduced in most labels and is indeterminate in the
  rest.

**July, post hoc (attempt-3 rows):**
- **ETH:** PPO's reversals (vs TWAP, VWAP, AC) recur in 6–8 of 11 labels and are
  indeterminate otherwise. Every DQN comparison is indeterminate in all 11 labels.
- **BTC:** the optimistic-only advantage of PPO and DQN over POV appears in 6 of 10 labels;
  everything else is indeterminate in every label.

**How to read "robust" here.** "Indeterminate in every label" is trivially robust, because
label subsets have less power. It is not evidence of equivalence.

**Registered rows, July.** In attempt 2, labels containing a crashed episode are
not_evaluable, so the July results by regime are incomplete.

## Limits

- **No market impact.** Displayed liquidity resets to history, and real orders would change
  others' behaviour.
- **Unobservable liquidity and queues.** Hidden liquidity and queue position are
  unobservable; the two modes bound passive fills, they do not identify them.
- **Mapping.** The clock and lot mappings are development-fitted scalars. The BTC mapping is
  a transfer test.
- **Sample.** 144 short episodes per day on one venue. The July periods had been consumed by
  M2–M6 before M8, so only August is a fresh transfer holdout.
- **No claims** of exact historical fills, profitability, live performance or historical
  superiority of any policy.
