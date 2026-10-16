# v0.9 methods

## Fidelity ladder and evidence levels

| Fidelity | Meaning | Typical evidence level |
|---|---|---|
| F0 | raw historical observations | — |
| F1 | historical replay (prediction or description on real days) | E4 |
| F2 | bounded historical counterfactual replay (assumption grid) | E5 |
| F3 | hybrid historical market twin (real backbone with a fitted response model) | E3 |
| F4 | calibrated classical generative market | E1 |
| F5 | neural generative market (small CPU research models) | E1–E2 |
| F6 | strategic multi-agent market (endogenous response) | E2 |

The evidence levels E6–E9 are reserved for transfer across time (E6), instrument (E7), venue (E8) and the final
multi-axis vault (E9). A hypothesis that is ESTABLISHED inherits its registered target level. Nothing else is
assigned a level.

## Study families

- **Realism (H1–H5, H12, H35).** The methods compared are:
  - leave-one-day-out (pilot) or explore-trained (confirm) comparisons of classical point processes;
  - marked Hawkes and neural TPPs (RMTPP, transformer TPP);
  - GRU/LSTM/transformer sequence classifiers vs a multinomial baseline;
  - latent state-space and neural latent AR path generators vs classical paths (signature MMD);
  - a hybrid twin vs fully generative response to large aggressive prints;
  - a strategic vs static ecology;
  - contrastive representations vs hand features.

  The realism gate (M55) scores each generator family per capability against the real-real day-to-day reference:
  ACCEPTABLE ≤ 1.0×, PARTIAL ≤ 2.0×, otherwise REJECTED. A rejected capability never supports a claim.
- **Fills, queues and markouts (H6–H9).**
  - Execution-aware vs generic calibration of clearance probabilities (Brier score).
  - Survival models (Cox, Weibull AFT, boosted hazard, neural hazard) vs Kaplan–Meier, scored by integrated Brier
    score.
  - The hybrid backbone.
  - Toxicity-aware passive execution (C5).

  All fill quantities are back-of-queue clearance probabilities or bounds, never exact fills.
- **Impact (H10, H11).**
  - A state-dependent impact model vs the square-root law on large-print bursts. The estimates are observational
    and ASSUMPTION_DEPENDENT; the causal registry marks observational flow impact as NOT_IDENTIFIED.
  - Self-impact ranking changes in endogenous vs exogenous synthetic ecologies.
- **Adaptation (H13–H17).**
  - Active (cross-entropy) vs random simulation acquisition.
  - Sequential Dirichlet posterior updating vs a frozen fit.
  - Few-shot adaptation vs a full refit (non-inferiority).
  - Continual learning (replay / EWC) vs naive fine-tuning.
  - Adaptive conformal inference (ACI) coverage under drift (non-inferiority).
- **Decisions (H18–H28).**
  - The hybrid-twin decisions hub: abstention, completion, meta-policy and multi-fidelity agreement.
  - Synthetic world clusters: contextual fallback, DRO, adaptive MPC v2 and rare-event-aware choice.
  - Offline RL (BC, CQL, IQL, TD3+BC, decision policy) with DR off-policy evaluation and a lower-confidence-bound
    deployment gate. This runs only in synthetic worlds with logged behaviour propensities; on historical data OPE
    is NOT_AVAILABLE by construction.
- **Fidelity agreement (H29–H31).** This asks whether directions from hybrid or synthetic fidelities agree with the
  directions that bounded replay identifies on real days.
- **Transfer and the vault (H32–H34, H36).** The conclusions carried from bounded replay on BTCUSDT confirm days are
  re-tested:
  - across time (confirm_late);
  - across instruments (ETHUSDT);
  - across venues (Binance L1, L1-identifiable conclusions only);
  - once on the 2025 vault (SOLUSDT linear, BTCUSDT linear, BTCUSDT spot).

## Bounded counterfactual replay (F2)

Each registered policy and mandate runs on each window under every combination of:
- queue rule (optimistic / neutral / conservative);
- latency (0 / 50 / 250 ms);
- hidden fraction (0 / 0.5);
- impact rule (none / transient with a 10 s half-life / persistent).

For each treatment–control pair, the per-day oriented difference is computed under each assumption separately, so
both policies face the same unknown world. The day's partial-identification interval is [min, max] over the
assumptions.
- The registered decision quantity is the conservative bound (the minimum).
- A day's direction is IDENTIFIED only when the whole interval lies on one side of zero.
- Costs are implementation shortfall in bps of the arrival mid, net of the registered fees (taker 5.5 bps, maker
  2.0 bps), plus a 25 bps penalty on unfinished quantity.

All of these values are HYPOTHETICAL.

## Statistics

- **Unit:** the instrument-day (windows nested within days). Synthetic hypotheses use world clusters.
- **Inference:**
  - a paired percentile bootstrap over units (4000 resamples);
  - a one-sided centred bootstrap p-value for H0: mean ≤ 0;
  - Holm across the promoted members of each registered family (α = 0.05);
  - Holm across carried conclusions inside each transfer test.
- **Superiority:** ESTABLISHED iff Holm rejects and the estimate is at least the practical threshold. FAILED iff the
  95% upper bound is < 0.
- **Non-inferiority:** ESTABLISHED iff the adjusted lower bound is above −margin. FAILED iff the upper bound is
  below −margin.
- **Agreement:** the same rule applied to (agreement − 0.5), with the estimate required to be at least 0.5 plus the
  threshold.
- **Transfer:** ESTABLISHED iff at least one carried conclusion meets its own rule after Holm. FAILED iff every
  carried conclusion is determinate in the opposite direction.
- **Seeds:** the confirm namespace (confirmatory studies) and the final namespace (the final analysis), opened
  through the ledger by sealed designs.

## Descriptive studies (not confirmatory)

These studies run on explore days, synthetic worlds or sealed upstream runs:
- event reconstruction audit;
- bounded replay width;
- queue clearance;
- latent-liquidity proxies;
- flow-risk proxies;
- regimes;
- change points and drift;
- path and tail realism;
- generator fit;
- parameter recovery (identifiability);
- SBC with an overconfident positive control;
- posterior predictive checks;
- cross-impact;
- portfolio timing risk;
- bounded adversarial search and counterexample search;
- OPE quality;
- ranking stability;
- uncertainty decomposition;
- value of information;
- local workload timings.

They inform the cards and the discussion and never decide a hypothesis.
