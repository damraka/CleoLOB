# v0.8 completion-constrained and multi-objective execution

## Separated accounting

Every episode row (`lob/v08/execution/simulator.py`) keeps the following quantities apart. None of them is
folded into another.

**Fills and inventory**
- `actual_filled_qty`, `fill_frac` and `actual_completion`, with a within-horizon / post-horizon split.
- `terminal_inventory`, `outstanding_qty` and `overfill_qty`.

**Costs**
- `implementation_shortfall_bps`: per filled notional.
- `gross_cost_bps`: per target notional.
- `fee_bps`, with rebates also reported separately (`rebate_bps`).
- `economic_cost_bps`: actual fills only.

**Residual and objective**
- `hypothetical_residual_cost_bps`: a walk of the final book. This is **not a fill**. When the book has too
  little depth, `residual_unpriceable` is set instead.
- `completion_penalty_bps`.
- `optimization_cost_bps`: economic cost plus hypothetical residual plus penalty. This is an objective, not an
  economic cost.

`tests/test_v08_execution.py` covers these cases:
- full, partial and zero completion;
- maker rebate, taker fee and per-lot fee;
- unpriceable residual;
- feed latency and stale books;
- cancel races;
- hard and urgency completion;
- participation caps.

## Completion modes

| Mode | Behaviour |
|---|---|
| `none` | the policy alone |
| `urgency` | marketable catch-up after 80% of the horizon |
| `hard` | at the horizon, cancel resting orders; once the cancels are acknowledged, buy what the strategy knows is missing, inside the settlement window |

## H7 and M23

`cleo completion-study-v08` runs four members (pov, heuristic, adaptive_twap, mpc) under `none` and `hard`,
paired by world and seed.

H7 applies only to members whose unconstrained completion is below 0.95. For those members it checks two
things:
1. the increase in completion;
2. cost non-inferiority, with a 3 bps margin.

**Pareto fronts.** These are computed over economic cost, completion probability, CVaR, participation, p95
tail, residual inventory, footprint and latency sensitivity.

**Constrained choice.** The completion-constrained choice minimizes expected cost subject to a Wilson lower
bound on completion of at least 0.95. An incomplete low-cost policy is never ranked first on cost alone.
