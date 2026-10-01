# v0.6 roadmap: market realism, calibration uncertainty and model risk

Branch `research/v0.6-market-realism`, package version `0.6.0.dev0`. This is a research
branch, not a release: there is no tag, no PyPI upload and no GitHub release.

## Question

> When is a calibrated limit-order-book simulator realistic enough, and identified well
> enough, to support reliable execution-policy conclusions under parameter uncertainty,
> model misspecification, regime shift and historical transfer?

v0.5 asked which simulator and execution conclusions survive contact with real historical
data. Its answer was largely negative:
- every absolute calibration gate failed;
- impact agreement was not established;
- learned-policy conclusions did not transfer to the August holdout.

v0.6 asks *why* simulator conclusions are unstable. It asks how much of the instability
comes from calibration ambiguity and structural model risk, and which realism deficiencies
actually move execution decisions.

## Contributions

| | Question | Main hypotheses |
|---|---|---|
| C1 | Can materially different parameter vectors fit equally well? (identifiability) | H4 |
| C2 | How much uncertainty remains after calibration? | H1–H3, H5 |
| C3 | How much do execution results vary across plausible simulators? | H5, H6 |
| C4 | Do pairwise execution conclusions survive simulators, regimes and bounded replay? | H6, H11, H12 |
| C5 | Which realism failures matter for execution conclusions? | H7, H8 |

Regime conditioning (H9, H10) is a further strand. Every hypothesis, estimator, threshold
and multiplicity family is frozen in [`configs/v06/protocol.json`](../configs/v06/protocol.json).

## Milestones

| Milestone | Content |
|---|---|
| M0 | Protocol freeze, dataset chronology audit, v0.6 ledger, verifier |
| M1 | Realism observable framework (10-level tapes, additive block sketches) |
| M2 | Distributional and tail realism (W1, KS, energy, JSD, quantile and exceedance errors) |
| M3 | Temporal and event-process realism |
| M4 | Multivariate and conditional (dependence) realism |
| M5 | Leakage-resistant real/synthetic domain-gap study |
| M6 | Calibration v3 (multi-start, multi-objective, Pareto reporting) |
| M7 | Parameter identifiability and sensitivity |
| M8 | Plausible simulator ensemble |
| M9 | Controlled misspecification and model-risk decomposition |
| M10 | Execution-ranking stability |
| M11 | Execution-sensitive realism |
| M12 | Regime-conditioned calibration and transition realism |
| M13 | External validity (fresh ETH, cross-instrument BTC) and OOD support |
| M14 | Historical transfer v2 and domain-randomized policies |
| M15 | Evidence, claim graph and reproducibility |
| M16 | Performance and UX |
| M17 | Paper-style report and final claim audit |

Status is recorded in [v06-final-report.md](v06-final-report.md) once evidence exists. This
roadmap makes no claims about outcomes.

## Rules carried over from v0.5

- The v0.5 protocol, ledger, evidence and results are immutable records. The v0.6 protocol
  pins the v0.5 ledger head, so any rewrite is detected.
- Consumed data never becomes fresh. Holdout analyses are sealed before first access.
- Failed, invalid and post-hoc attempts are kept and labelled.
- No profitability, alpha, HFT, exact-FIFO, exact-fill or causal real-market claim follows.
