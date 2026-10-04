# v0.7 calibration: posterior, recovery, execution-aware calibration, surrogates (M6, M7)

Code:
- `lob/v07/posterior/` (`smc_abc.py`, `study.py`)
- `lob/v07/calibration/` (`execution_aware.py`, `surrogate.py`, `surrogate_study.py`)

## SMC-ABC posterior (workstream 9)

**Algorithm.** Adaptive SMC-ABC (Del Moral, Doucet and Jasra 2012), as registered:
- uniform prior on the v0.6 14-parameter transformed unit box
- distance = the sealed v0.6 objective against the development target (2 seeds × 1,800 s)
- 256 particles, 6 generations, tolerance = median of the current distances
- one Gaussian random-walk move per particle with twice the particle covariance, reflected at
  the box boundary
- three independent runs (seeds 70101–70103), 1,536 evaluations each

The point-optimization baseline (G0, v0.6) is retained.

**Attempts.** All attempts are kept in the ledger.

| Attempt | Outcome |
|---|---|
| `m6-recovery-registered` | **FAILED**: a seeded synthetic truth violated the 500 events/s guard |
| `m6-posterior-registered` | **ABORTED**: the host stopped the process for low memory; nothing was sealed |
| `results/v07/m6/recovery-2` | sealed |
| `results/v07/m6/posterior-2` | sealed; rerun with 8 workers, same seeds and budgets; wall time 35,874 s |

### Synthetic recovery (`results/v07/m6/recovery-2`)

**Rule.** For each of three truths (the v0.6 selected vector plus seeded prior draws), the 90%
marginal intervals must cover the truth for at least 80% of the 14 parameters. One prior draw was
rejected and redrawn because its synthetic target violated the implausibility guard.

| Truth | Coverage | Final tolerance |
|---|---|---|
| v0.6 selected | 0.714 | 2.616 |
| prior draw 1 | 0.714 | 3.726 |
| prior draw 2 | 1.000 | 4.404 |

**Status: ASSUMPTION_DEPENDENT.** Two of three cases fall below the registered 80%. Every
downstream conclusion that uses the G0 posterior inherits this limitation:
- H1, H6
- H7–H9 (posterior worlds)
- H10/H11 (the execution-aware model is a posterior particle)
- H12/H13 (posterior-trained policies)

### Posterior (`results/v07/m6/posterior-2`)

| Run | Final tolerance | Acceptance by generation | Major modes (mass) |
|---|---|---|---|
| 70101 | 3.481 | 0.49, 0.35, 0.21, 0.14, 0.18 | 2 (0.07, 0.05) |
| 70102 | 3.310 | 0.46, 0.26, 0.21, 0.13, 0.09 | 3 (0.07, 0.06, 0.05) |
| 70103 | 3.281 | 0.46, 0.31, 0.20, 0.12, 0.07 | 2 (0.13, 0.05) |

**H6 (descriptive): ESTABLISHED** [C-H6]. The rule is two or more major components in at
least 2 of 3 runs; all three runs are multimodal. The major components hold only 5–13% of
mass each, and the rest of the population is spread out.

**What the posterior looks like.** The pooled final distances have median 3.17 (5%–95%:
2.82–3.43). The G0 point model scores 2.19 on development in M5. So the ABC tolerance after six
generations remains far above the point optimum, and the posterior is diffuse:
- Most parameters' 90% intervals span most of their range. Examples: `hawkes_branching`
  0.05–0.92, `target_level_vol` 0.13–0.92.
- Only `offset_p` (0.01–0.52), `limit_rate` and `market_rate` (about 0.05–0.75) are
  concentrated.

This is an ABC approximation at the reached tolerance, not the exact Bayesian posterior.

**Posterior predictive** (16 draws, `results/v07/m6/select`). The objective is 2.803 on
development and 3.328 on selection, worse than the G0 point model (2.192 / 2.461). Per-draw
selection objectives range from 2.64 to 4.22.

## Execution-aware calibration (workstream 18; H10/H11) — `results/v07/m7/execution-aware`

**Rule.**
1. Take the 64 best distinct posterior particles plus G0, each re-simulated on 3 seeds ×
   1,800 s.
2. Keep candidates whose generic objective is at most 1.10 × G0's on the same seeds.
3. Choose the minimum execution-sensitive (ES) objective among them.

**ES objective.** The mean of 8 registered components, each divided by its sealed v0.6
real-vs-real error:
- spread
- L1 depth
- depletion hazard
- replenishment probability
- aggressive event size
- 10 s signed-volume autocorrelation at lags 1, 2 and 5

**Result.**
- Only 3 of 64 candidates passed the generic guard.
- The selected model (EA, particle c04) has generic 2.479 vs. G0 2.299, and ES 1.789 vs.
  G0 1.838: a small ES improvement bought with a generic loss on development.
- Fresh-data tests are H10 (ES) and H11 (generic noninferiority, δ = 0.10 × the G0 selection
  objective = 0.246).

## Surrogates and active calibration (workstreams 10, 11)

See `results/v07/m7/surrogate` and the section filled from it in `docs/v07-final-report.md`.

**Emulator data.** The emulators are trained on the 2,304 sealed v0.6 development candidates
(grouped split by search start). They are compared on:
- held-out error
- interval coverage (heuristic for the forest)
- ranking
- how many simulations a screen saves

**Active vs. random.** Active calibration (batched GP lower-confidence-bound) is compared with
random search at an equal budget of 48 evaluations, on a synthetic recovery target. It is
EXPLORATORY (3 replicates).
