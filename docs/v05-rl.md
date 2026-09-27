# v0.5 M7 — registered execution-policy study

Registered on the frozen environment (`configs/v05/environment-freeze.json`, commit `509d548`,
ledger entry 56). Plan sealed in ledger entry 57 before any training.

- **Design** (frozen at M0 in `configs/v05/policy-study.json`):
  - PPO and DQN, each in three arms (`main`, `no_book`, `no_terminal`);
  - 8 independent training seeds and 16,384 steps per model, i.e. 48 models and 786,432
    training steps;
  - six controls: TWAP, VWAP, POV, Almgren–Chriss, heuristic and random;
  - 160 common unseen market seeds;
  - 4 regimes: original, shifted, stress and calibrated. Calibrated is the M3-selected
    simulator, whose real-market gate FAILED, so it is a synthetic regime.
- **Size:** 34,560 evaluation episodes; all were retained.
- **Training:** original regime and training-domain seeds only. Normalization was sealed from
  training-domain exploration before any fit. Final fixed-budget checkpoints only.
- **Endpoints:**
  - E6: completion-adjusted cost (realized plus hypothetical residual; bps);
  - E5: within-horizon completion. Settlement fills never count.
- **Family:** 64 contrasts × 2 endpoints = 128 hypotheses, Bonferroni-corrected.
  - Cost uses the crossed seed bootstrap.
  - Completion uses the market-level Clopper–Pearson bound.
  - Gate: cost upper bound < 0 **and** completion lower bound ≥ −0.05.

Verification (`cleo policy-study-v05 verify`) recomputes the summary from the episode journal
and checks every checkpoint, normalization and lock hash. It is valid.

## Primary results: joint gate per contrast

**5 of 64 contrasts pass the frozen joint gate. All five are DQN (main arm):**

| Regime | Contrast | Mean cost difference (bps) | Bonferroni interval | Completion lower bound |
|---|---|---:|---|---:|
| original | DQN − POV | −0.120 | [−0.227, −0.013] | −0.048 |
| shifted | DQN − TWAP | −0.366 | [−0.604, −0.130] | −0.048 |
| shifted | DQN − VWAP | −0.353 | [−0.573, −0.127] | −0.048 |
| shifted | DQN − POV | −0.418 | [−0.641, −0.191] | −0.048 |
| shifted | DQN − AC | −0.350 | [−0.570, −0.118] | −0.048 |

**Everything else:**
- **PPO:** no contrast passes; every PPO cost interval includes zero.
- **DQN in the original regime:** its other contrasts (vs TWAP, VWAP, AC, heuristic, random)
  include zero, as do DQN vs heuristic and random in the shifted regime.
- **Stress:** every cost comparison is **WITHHELD**, because 109 stress episodes had INVALID
  economics (a residual that could not be valued at the terminal book).
- **Stress completion:** the lower bounds are −0.37 to −0.50 because of discordant markets.
- **Calibrated:** every cost comparison is **WITHHELD**, from 5,477 INVALID calibrated episodes.
  The thin calibrated book often cannot value residuals. The completion lower bounds are
  −0.28 to −0.79.
- **`no_terminal` ablation:** the models are bit-identical to `main`, with differences exactly
  zero. Under the shared completion rule the terminal penalty never triggers in the training
  regime, so this ablation is vacuous. That is a design finding, not a result about reward
  shaping.

**Reading of the gate.**
- These passes are **per contrast** only. They show that in two simulated regimes, DQN had
  lower completion-adjusted cost than specific classical schedules, without evidence of worse
  within-horizon completion.
- They do **not** establish learned-policy superiority in general. PPO fails, DQN fails
  against the heuristic and random controls, and no regime result transfers to stress or the
  calibrated regime.
- Economic magnitudes are small: 0.12–0.42 bps.
- The M8 transfer study tests whether these directions survive history.

## M5 endpoint: within-horizon versus settlement completion (main arms)

| Regime | Agent | Within horizon | After settlement | Settlement-only completions (of 160 control / 1,280 learned) |
|---|---|---:|---:|---:|
| original, shifted | all | 100% | 100% | 0 |
| stress | TWAP, VWAP, AC | 91.9% | 98.1% | 10 each |
| stress | POV | 100% | 100% | 0 |
| stress | heuristic | 91.9% | 100% | 13 |
| stress | random | 96.3% | 99.4% | 5 |
| stress | PPO | 92.7% | 98.3% | 71 |
| stress | DQN | 93.4% | 98.8% | 69 |
| calibrated | TWAP / VWAP / AC | 55.0% / 57.5% / 53.7% | same | 0 |
| calibrated | POV / heuristic / random | 25.0% / 50.6% / 50.6% | same | 0 |
| calibrated | PPO / DQN | **27.7% / 34.2%** | same | 0 |

- **Stress:** settlement hides part of the non-completion. Up to 8% of mandates complete
  only after the horizon, and v0.4's single "completion" number would have counted them.
- **Calibrated regime:** most mandates are not completed at all, in either window.
- **Learned policies in the calibrated regime:** they complete far less often than most
  controls. They were trained only in the original regime, and this regime is outside their
  training distribution.

## Limits

- The evidence is synthetic, and the calibrated regime failed its real-market gate.
- There was no prospective power analysis.
- The 160-market design makes the completion bound attainable; it does not make it
  sensitive to rare failures.
- Seed-level results (training-seed means and completion rates) are in the sealed local run.
- No historical, live or profitability claim follows.
