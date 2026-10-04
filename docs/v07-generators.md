# v0.7 generator families (M4, M5)

Code: `lob/v07/generators/` (`events.py`, `engine.py`, `counts.py`, `regime.py`, `study.py`).
Run: `results/v07/m5/fit` (development fit; development and selection scores).

## Event-driven reference engine (workstream 6)

`BundleSimulator` subclasses the frozen v0.5 `ExchangeSimulator` with every zero-intelligence
clock switched off.

**How a bin works.**
1. Every 100 ms, a count model receives two inputs: the *current* book state (spread in
   ticks, top-5 imbalance, log L1 depth) and the realized counts of the previous bin.
   Realized market counts come from the engine's own trade tape, so a strategy's market
   orders count.
2. The model returns counts for six marks: market buy/sell, add bid/ask, cancel bid/ask.
3. Each event is realized against the live book through the unchanged matching engine.

**How events are drawn.**
- Additions and cancellations draw joint (tick offset, size) pairs from development data,
  conditional on the spread bucket (one tick vs. two or more).
- Cancellations remove a *fraction* of the background volume at the chosen level.

**Mechanics: calibrated vs. assumed (the workstream-6 report).**

| Mechanism | Calibrated from development data | Assumed |
|---|---|---|
| Event counts per 100 ms bin | yes (per family) | the six-mark bundle representation |
| Placement offset and size | yes (joint empirical pairs) | the spread-bucket conditioning |
| Cancellation fraction | yes | which queue position is cancelled (`uniform` by default) |
| State-conditioned intensity | yes (G1, G3, G4) | the three state features |
| Response to a strategy's orders | yes, through the state features and realized counts | that the historical association transfers to a strategy's own flow |
| Matching, priority, latency | no | frozen v0.5 engine semantics |

**Development-stage fixes, made before any scoring on selection data:**
- A first version drew cancellation sizes independently of level size, so levels random-walked
  upward. Cancellation *fractions* fixed this.
- A second version placed inside-spread additions at a one-tick spread, which piled them onto
  the touch. Conditioning on the spread bucket fixed this.

**Tests** (`tests/test_v07_generators.py`, `tests/test_v07_impact.py`):
- valid and deterministic tapes
- every zero-intelligence rate refused
- strategy orders processed
- all cancel-position modes keep the book valid
- the zero-impact limit (a zero-size parent leaves the market path unchanged)
- impact monotone in size on average
- recovery after the parent ends

## Families (workstream 7)

| Family | Representation | Free parameters | Fit (development) |
|---|---|---|---|
| G0 point | v0.6 selected calibration-v3 model (`ExtendedSimulator`) | 14 | sealed in v0.6 |
| G0 posterior | SMC-ABC posterior over the same 14 parameters (see `docs/v07-calibration.md`) | 14 (distribution) | M6 |
| G1 state-conditioned Hawkes | discretized nonlinear marked Hawkes: per mark, log intensity linear in log-decayed past counts of all marks (decays 2/s, 0.2/s) and standardized state | 96 | 14 s |
| G2 latent regime switching | two v0.6 volatility-regime models, switched by a two-state Markov chain (exit rates from development labels) | 30 | from v0.6 M12 |
| G3 conditional autoregressive | nonparametric: resample a historical count vector from the cell (spread × imbalance tercile × previous activity) | nonparametric (stored rows) | 5 s |
| G4 neural temporal | GRU (hidden 32) over 50 bins, Poisson head; torch on CPU, exact numpy forward pass in simulation | about 4,000 | 9 s (4 epochs, 24,000 sequences) |
| Diffusion | **NOT_AVAILABLE**: no GPU and no adequate development data volume; not approximated | — | — |

G1 is the "marked point process" family: its marks are event type × side.

**G1 stability constraint.** The unconstrained G1 fit is superlinear: each mark's total
excitation elasticity is 1.6–2.6. Simulated in closed loop, it exploded into a high-activity
state. The registered fit therefore constrains each mark's elasticity to at most 0.9 and
refits intercepts and state effects with excitation held as an offset. Constraining costs
4–9 percentage points of deviance explained (for example, market buys 0.20 → 0.11).

**Stability guard for G1 and G4.** State features are clipped to the development range, and
per-bin rates are capped at the development 99.9th-percentile count.

## Development and selection scores (run `results/v07/m5/fit`; 5 seeds × 3,600 s)

Lower is better. Scores use the sealed v0.6 objective.

| Family | Development | Selection | Seed SE (selection) |
|---|---|---|---|
| G0 point | 2.192 | 2.461 | 0.026 |
| G0 posterior predictive (M6, 16 draws) | 2.803 | 3.328 | 0.100 (per-draw) |
| G1 state-conditioned Hawkes | 4.074 | 4.101 | 0.679 |
| G2 regime switching | 2.636 | 2.952 | 0.100 |
| G3 conditional autoregressive | **1.888** | **2.005** | 0.017 |
| G4 GRU | 1.741 | 2.082 | 0.030 |

**Interpretation.**
- G3 and G4 fit development and selection better than the v0.6 point model.
- G1, even constrained, fits worst and is unstable across seeds.
- The more complex G4 beats G3 on development but not on selection. Under the
  complexity-penalty rule (workstream 95), its extra capacity is not established as useful.
- These are development/selection statements only. Fresh-data results are in
  `docs/v07-realism.md`.

## G* selection (sealed rule, run `results/v07/m6/select`)

The rule: G* = argmin of the selection objective over G0 posterior and G1–G4, with ties going
to fewer free parameters. **G* = G3_conditional_ar** (selection objective 2.005).
