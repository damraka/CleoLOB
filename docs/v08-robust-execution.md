# v0.8 robust execution, MPC and adaptive policies (simulation)

## Policies (`lob/v08/execution/policies.py`, `mpc.py`)

All policies use a typed plugin interface. Each policy declares:
- the observations it reads;
- the market capabilities it needs;
- whether it needs queue identity;
- whether it supports delayed observations;
- whether it needs a model posterior;
- its completion behaviour;
- whether it is stochastic.

`check_capabilities` fails closed: a policy that needs something the run cannot provide is rejected.

**Baselines.** TWAP, VWAP, POV, AC, a spread-aware heuristic and a seeded random control. These are v0.8
implementations; the v0.5 originals are untouched.

**Adaptive variants.** Adaptive TWAP, liquidity-aware TWAP, adaptive VWAP, a volatility-aware schedule,
adaptive POV and dynamic AC.

**MPC.** A receding-horizon controller that searches a grid of (market, passive) child sizes. Its plan cost
adds up five terms:
- the observed-book impact;
- minus the expected passive saving;
- the continuation cost;
- a risk penalty;
- a completion-pressure term.

MPC estimates use observed data only.

**Robust MPC.** Takes the worst case over a registered scenario grid.

**Parametric schedule family.** Urgency × passive fraction × catch-up. This family is the candidate set for
robust selection over simulated worlds.

Policy parameters are fixed across worlds.

## Robust selection across simulated worlds (M24; H8)

`cleo robust-execution-v08` runs in two stages on simulated worlds.

**Stage 1 — selection.** Candidates are evaluated on the selection worlds with development seeds. Each
criterion then picks one candidate:
- nominal;
- posterior-expected;
- worst-case;
- minimax-regret;
- CVaR(0.8);
- KL-DRO with ρ = 0.1.

Each criterion is run with and without the completion constraint.

**Stage 2 — evaluation.** Every candidate and every baseline is evaluated on **held-out** worlds with confirm
seeds.

H8 compares the nominal choice with the CVaR choice on CVaR(0.8) of out-of-sample regret. Its interval comes
from a hierarchical bootstrap that recomputes the regrets in every replicate.

## Controlled benchmark (M25, M26; H9)

`cleo mpc-study-v08` evaluates every registered policy on held-out worlds under both mandates. For each policy
it reports:
- cost, CVaR and the worst world;
- completion;
- regret;
- rank uncertainty.

H9 compares three members with TWAP on CVaR over worlds, with a Holm correction across the three:
- MPC;
- robust MPC (worst case over the registered scenario grid);
- the best adaptive variant on development seeds.

## Stress worlds (M28) and latency (M30)

`cleo adversarial-stress-v08` applies the 13 registered stress scenarios. It also runs a bounded adversarial
search inside a box of rate multipliers, with two limits:
- a fixed evaluation budget;
- a plausibility radius (Mahalanobis, 99th percentile of plausible worlds).

All of this output is labelled SYNTHETIC_STRESS.

`cleo latency-study-v08` runs every registered policy under seven latency profiles. These profiles are
simulator settings, not venue measurements.
