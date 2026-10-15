# v0.9 policy cards

Registered execution policies. Bounded-replay costs are HYPOTHETICAL and assumption-dependent; no policy is claimed to be profitable or production-ready.

## classical

- `twap` (conclusions: C1, C2, C3, C4, C6; carried)
- `vwap`
- `pov`
- `ac`
- `sweep` (conclusions: C2, C6; carried)

## adaptive

- `adaptive_twap`
- `adaptive_pov`
- `liquidity_pov` (conclusions: C4)
- `imbalance_passive`
- `toxicity_passive` (conclusions: C5)
- `passive_naive` (conclusions: C5)
- `passive_deadline` (conclusions: C3; carried)

## optimization

- `static_opt`
- `mpc_stochastic`
- `mpc_cvar` (conclusions: C1; carried)
- `mpc_minmax`
- `mpc_dro`
- `mpc_chance`
- `mpc_adaptive`

## learning

- `bc`
- `cql`
- `iql`
- `td3bc`
- `decision_transformer`

## meta

- `regime_selector`
- `mixture_of_experts`
- `policy_blender`
- `contextual_fallback`

## Fallback ladder

- low_uncertainty: selected optimized policy
- moderate: liquidity_pov
- high: twap
- completion_danger: completion-safe aggressive controller
