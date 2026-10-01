# v0.6 execution stability, misspecification and model risk (M9–M11, M14–M16)

Everything on this page is **simulator-only** evidence. No historical, live or
profitability claim follows from it.

## Frozen environment

The environment is frozen in `configs/v06/environment-freeze.json`, ledger seal
`environment-freeze`, before any policy training.

**Mandate:**
- buy 14 lots, the development median L1 depth (14 × 243.7 USD);
- 120 s horizon, 6 s decision interval, 60 s warmup, 5 s settlement;
- maker fee 0 bps, taker fee 1 bps;
- v0.5 completion rule (urgency fraction 0.8) and 25 bps terminal penalty.

**Agents:**
- Classical: TWAP, VWAP, POV (participation 0.3) and AC. AC is identified per world on 4
  seeds. Identification **failed** for the v0.5 control world (R² 0.014), which therefore uses
  default parameters.
- Learned: PPO and DQN, each in two variants:
  - single: trained in the selected world;
  - ensemble: domain-randomized uniformly over the 2 ensemble members.

  Each combination has 4 training seeds and 16,384 steps. The v0.5 M7 hyperparameters are
  used. Observation normalization is sealed per variant, and there were 0 training failures.

**Worlds (19):**
- the selected v3 model;
- 1 distinct ensemble member;
- 14 structural interventions;
- 2 regime models;
- the v0.5 control.

`spread_distortion` is **inert**: the selected `offset_p` already sits at the bound the
intervention moves to. It is retained and flagged.

**Evaluation:** 19 worlds × 20 agent cells × 200 market seeds = **76,000 episodes, 0 INVALID**.
Every agent completed within the horizon in every reported cell of the selected world.

## Mean completion-adjusted cost in the two ensemble worlds (bps)

| Agent | Selected | Member 2 |
|---|---|---|
| TWAP | 2.96 | 3.26 |
| VWAP | 3.11 | 4.04 |
| POV | 4.23 | 3.23 |
| AC | 3.17 | 3.18 |
| PPO single | 3.54 | 2.31 |
| DQN single | 2.50 | 2.28 |
| PPO ensemble | 3.14 | 2.72 |
| DQN ensemble | 2.82 | 1.68 |

Per-episode standard deviations are 5–9 bps.

## Registered hypotheses

| Hypothesis | Status | Evidence |
|---|---|---|
| H5 equifinality | **NOT_ESTABLISHED** | max \|member difference\| over classical agents = 1.00 bps (POV); every Bonferroni interval (family 4) contains 0 |
| H6 rank non-invariance | **NOT_ESTABLISHED** | no certified reversal (family 30); the two members' *point* rankings are nearly unrelated (Kendall τ = 0.07) |
| H7 execution-sensitive realism | **NOT_ESTABLISHED** | 15 worlds; largest ρ = 0.44 (temporal family), interval spans 0; Holm-adjusted p ≥ 0.61 |

**Resolution caveat for H5 and H6.** The power report sealed before any holdout put the
single-mean minimum detectable effect at 1.0–1.7 bps (α 0.05, power 0.8, 200 seeds). A
two-world difference has about 1.4× that. The preregistered 1 bps materiality margin was
therefore **below the design's resolution**. NOT_ESTABLISHED here means the study could not
resolve the differences. It does not mean the worlds agree.

H7 is coarse: across worlds, the disagreement fraction takes only the values 0, 1/15 and
2/15, because most conclusions are indeterminate in every world.

## Model-risk decomposition

The components below are separate standard deviations in bps. They are **not additive** and
are never combined.

| Agent | Market-seed SE | Calibration ensemble | Structural interventions | Regime models | Training seeds |
|---|---|---|---|---|---|
| TWAP | 0.58 | 0.22 | 1.47 | 0.59 | — |
| VWAP | 0.61 | 0.65 | 1.32 | 0.17 | — |
| POV | 0.37 | 0.71 | 0.91 | 1.02 | — |
| AC | 0.50 | 0.01 | 1.34 | 0.45 | — |
| PPO single | 0.50 | 0.87 | 1.01 | 2.00 | 0.62 |
| DQN single | 0.59 | 0.16 | 1.10 | 4.75 | 0.22 |
| PPO ensemble | 0.54 | 0.30 | 1.42 | 3.43 | 0.51 |
| DQN ensemble | 0.49 | 0.81 | 1.18 | 4.14 | 0.16 |

- **Classical schedules:** structural misspecification dominates.
- **Learned policies:** the regime models dominate (2.0–4.8 bps), well above the market-seed
  standard error (about 0.5 bps) and the training-seed spread (0.2–0.6 bps).

So the apparent cost differences between learned and classical agents (0.5–1.7 bps in the
selected world) are of the same order as the model-risk components.

Historical fill-bound effects are in [v06-transfer.md](v06-transfer.md). On ETH 2020-10-01,
the conservative and optimistic paths differ by at most 0.12 bps per agent. The difference is
0 for the classical schedules, which never rest passive orders.

## Misspecification (M9)

Shifts in pairwise conclusions relative to the selected world, from
`results/v06/m10/evaluation`:
- **Two of 15 conclusions change** under `persistence_removed` and `resilience_weak`.
- **One changes** under every other active intervention.
- **None changes** under the inert `spread_distortion`.

The mean absolute shift of the paired cost differences ranges from 0.38 bps (`depth_x0.5`) to
1.20 bps (`limit_flow_x0.5`).

These are interventions on the simulator. They establish nothing about causal mechanisms in
real markets.
