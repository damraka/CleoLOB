# CleoLOB v0.8.0 release notes: decision uncertainty, adaptive calibration and adaptive execution

CleoLOB v0.8.0 asks whether execution decisions stay scientifically reliable when:
- market models are imperfect;
- regimes drift through time;
- model, parameter, queue and latency uncertainty is explicit;
- execution completion is mandatory or constrained;
- conclusions must transfer across time, instruments, venues and plausible market worlds.

Every hypothesis (H1–H21), statistical family, threshold, sealed design, seed namespace and holdout role was
preregistered in `configs/v08/`. They were frozen in an append-only, hash-chained ledger (412 entries, two
amendments) before any confirmatory seed was drawn. The registered studies ran on the research branch at development
version `0.8.0.dev0` and are released unchanged. The release changes only version metadata and release wording;
no result, evidence byte, protocol rule, ledger entry, hypothesis, threshold or claim status was altered.

## Scientific limitations

Read these before any result below.

- **SYNTHETIC evidence only.** All H1–H21 inferential outcomes are SYNTHETIC. They come from simulated worlds, and
  the H21 holdout is another simulator, not a market.
- **Generator gate = NO_ACCEPTABLE_MODEL** ([C08-GATE]). Every generator family failed the registered gate
  (discriminator AUC 1.0, support coverage 0.0). The gate was not weakened.
- **Real holdouts NOT_AVAILABLE** ([C08-DATA-BLOCKED]). The registered acquisition probe returned HTTP 403 for both
  the fresh transfer days (2021-02-01) and the final real holdout (2021-03-01). Neither holdout was read, and
  consumed data were never substituted.
- **No live alpha claim**, and no claim of profitability.
- **No exact historical fill claim.** Counterfactual replays on aggregate L2 are bounds, never fills.
- **No production or HFT claim.** The benchmarks are local research-workload timings only.

## Final hypothesis statuses

| Status | Hypotheses |
|---|---|
| ESTABLISHED (SYNTHETIC) | H1, H2, H3, H5, H9, H14, H19, H21 |
| FAILED | H11, H12 |
| NOT_ESTABLISHED | H4, H6, H7, H8, H10, H18 |
| NOT_AVAILABLE | H13, H15, H16, H17, H20 |

Per hypothesis:

- Adaptive vs frozen calibration on drifting worlds [C08-H1]: ESTABLISHED, 0.061 [0.055, 0.069].
- Regime-conditioned vs global calibration [C08-H2]: ESTABLISHED, 0.018 [0.007, 0.030].
- Dynamic model averaging vs the best single generator [C08-H3]: ESTABLISHED, 0.017 [0.013, 0.021].
- Synthetic parameter recovery [C08-H4]: NOT_ESTABLISHED. The Hawkes decay's Wilson lower bound (0.55) is below
  the registered 0.60.
- Material model and parameter uncertainty [C08-H5]: ESTABLISHED, up to 62% of a pairwise cost variance.
- Calibration quality vs decision regret [C08-H6]: NOT_ESTABLISHED, ρ −0.02 [−0.27, 0.23].
- Completion constraints [C08-H7]: NOT_ESTABLISHED.
- CVaR vs nominal selection over simulated worlds [C08-H8]: NOT_ESTABLISHED.
- MPC and robust MPC vs TWAP on simulated CVaR cost [C08-H9]: ESTABLISHED, about 1.5–1.7 bps (Holm over 3).
- Regime-aware policy selection [C08-H10]: NOT_ESTABLISHED.
- OOD-aware abstention [C08-H11]: FAILED. Mean simulated regret rose from 211 to 1,499 bps.
- Safe fallback under OOD shifts [C08-H12]: FAILED. It increased large degradations instead of preventing them.
- Ranking transfer [C08-H13]: NOT_AVAILABLE (data acquisition blocked).
- Bounded counterfactual separation of policy pairs [C08-H14]: ESTABLISHED for 1 of 5 registered pairs.
- Temporal transfer [C08-H15]: NOT_AVAILABLE (data acquisition blocked).
- Instrument transfer [C08-H16]: NOT_AVAILABLE (data acquisition blocked).
- Venue transfer [C08-H17]: NOT_AVAILABLE (data acquisition blocked).
- Calibration half-life and post-drift gain [C08-H18]: NOT_ESTABLISHED.
- Split-conformal cost coverage [C08-H19]: ESTABLISHED at 0.90 under exchangeability; 0.51–0.59 under every
  registered shift.
- Final real holdout [C08-H20]: NOT_AVAILABLE (data acquisition blocked).
- Sealed synthetic holdout [C08-H21]: ESTABLISHED. Both carried MPC conclusions survived unseen simulated worlds,
  with simulated CVaR improvements of +9.4 and +8.7 bps.

## What is included (M0–M35)

Each milestone maps to a module, a CLI command and sealed evidence; see
[v08-requirement-coverage.md](v08-requirement-coverage.md).

- **Protocol and data (M0–M4).** Protocol v0.8 with an append-only ledger, seed namespaces and evidence binding;
  dataset registry v3 with a capability matrix; a market-data quality engine with no silent repairs; clock and
  event-time normalization.
- **Adaptive calibration, regimes and drift (M5–M7, M14).** Regime detection, change-point and drift detection, a
  calibration half-life, and frozen vs adaptive, regime-conditioned and model-switching calibration.
- **Generator and model uncertainty (M8–M11, M17).** Six generator families, a stylized market ecology, controlled
  endogenous response, transient impact kernels and dynamic model averaging.
- **SBI, posteriors and identifiability (M12, M13, M15, M16, M18).** ABC parameter recovery, real-data posteriors,
  hierarchical pooling, plausible world sets, identifiability v3, posterior predictive checks with the generator
  gate, and an aleatoric/epistemic variance decomposition.
- **Completion-constrained execution (M22, M23).** Completion constraints, objectives and Pareto fronts.
- **Execution under model uncertainty and MPC (M24–M26).** CVaR vs nominal selection across simulated worlds, a
  controlled execution benchmark, and MPC with robust and adaptive variants on simulated worlds.
- **OOD, conformal and abstention (M19, M20, M27).** OOD detectors, split-conformal intervals, regime-aware
  selection, OOD-aware abstention and safe fallback.
- **Counterfactual bounds (M29).** Aggregate-L2 historical execution bounds with synthetic coverage checks.
- **Transfer, ranking, latency and stress (M28, M30–M32).** Structured stress with bounded adversarial search;
  feed latency, stale books and cancel races; cross-time, instrument and venue transfer; predicted vs bounded
  historical rankings; portfolio and basket infrastructure.
- **Statistics and runtime (M33, M34).** A statistical decision framework, a study planner and a deterministic
  parallel runner with checkpoint and resume.
- **Evidence and claim audit (M35).** A claim graph, a fail-closed claim audit (`cleo claim-audit-v08`), `cleo
  final-analysis-v08`, `cleo report-v08`, `cleo evidence-export-v08` and `cleo verify-v08`.

## Public evidence bundle

`examples/studies/v08` holds 41 sealed runs, the v0.8 configs and checksums. The final analysis is
`results/v08/final-2`; the earlier `results/v08/final` is a retained INVALID attempt.

```bash
cleo protocol-v08 verify
cleo verify-v08 --bundle examples/studies/v08
cleo claim-audit-v08 --bundle examples/studies/v08
```

Verification checks bytes, run bindings and the ledger chain; it is not independent scientific replication.

## Engineering fixes found during the research cycle

Each fix has a regression test. Each failed attempt is retained in the ledger; none was deleted.

- `3734c6b`: the aggregate-queue book produced fractional strategy queue positions in the simulator, which made
  episodes INVALID. The first confirmatory wave was retained, and the affected studies were rerun as `*-2` on the
  same sealed designs and seeds.
- `0fa4a49`: the fixed-share mixture predictive hit `log(0)`. This invalidated `explore/ood` and
  `confirm/abstention-3`, which were rerun as `ood-2` and `abstention-4`.
- `341cdad`: claim resolution could pick a ledger-retained failed attempt.
- `2279253`: `report-v08` could not receive the exploratory runs its figures need.
- `c48c26c`: Windows pool workers were throttled by EcoQoS (performance only; results are unaffected).
- `9f899c4`: the sealing benchmark broke on the write-once manifest.
- `c4e9ccc`: the H21 study opened the holdout namespace before validating its inputs, and H20 re-probed on reruns.

A host reboot interrupted three studies. They resumed from manifest-guarded checkpoints and sealed with zero INVALID
episodes; replay checks reproduced sampled episodes and each `result.json` byte for byte.

## Documentation

- [v08-paper.md](v08-paper.md): research report
- [v08-final-report.md](v08-final-report.md): hypothesis table, run pointers, attempts and engineering record
- [v08-requirement-coverage.md](v08-requirement-coverage.md): M0–M35 coverage

Install: `python -m pip install cleolob==0.8.0`.
