# v0.8 requirement coverage (M0–M35)

Each milestone has a module, the command that exercises it, and the sealed evidence under `results/v08/` (also in the
public bundle `examples/studies/v08`). Milestones without a study of their own are exercised inside the studies that
use them and by the `tests/test_v08_*.py` suites. Hypothesis statuses are in
[v08-final-report.md](v08-final-report.md).

| Milestone | Module | Command | Sealed evidence | Outcome |
|---|---|---|---|---|
| M0 audit and plan | `docs/v08-implementation-plan.md` | — | the plan | done |
| M1 protocol, ledger, seed namespaces, evidence binding | `protocol.py` | `protocol-v08` | `configs/v08`, 412 ledger entries, 2 amendments | verifies |
| M2 dataset registry v3, capability matrix | `datasets.py` | `dataset-registry-v08` | `configs/v08/dataset-registry.json` | done; exact-queue studies NOT_AVAILABLE by capability |
| M3 market-data quality engine | `quality.py` | `validate-market-data-v08` | quality status in the ledger's `parsed` events and real-data run metadata | done (no silent repairs) |
| M4 clock and event-time normalization | `clocks.py` | used by the real-data studies | `test_v08_market.py` | done |
| M5 regime detection | `regimes.py` | `regime-study-v08` | `explore/regime-study-2` | EXPLORATORY |
| M6 change-point and drift detection | `drift.py` | `change-point-study-v08` | `explore/change-point` | EXPLORATORY |
| M7 calibration half-life | `drift.py` | `change-point-study-v08`, `adaptive-calibration-v08 drift` | `confirm/adaptive-drift`, `explore/change-point` | H18 NOT_ESTABLISHED |
| M8 generator families | `generators/families.py` | all studies | `explore/posterior-study` | gate NO_ACCEPTABLE_MODEL |
| M9 stylized market ecology | `generators/ecology.py` | stress, H21 holdout | `explore/stress`, `final-2` | SYNTHETIC only |
| M10 controlled endogenous response | `generators/endogenous.py` | H21 holdout | `final-2` | SYNTHETIC only |
| M11 transient impact kernels | `impact.py` | used by the simulator | `test_v08_market.py` | done |
| M12 simulation-based inference and recovery | `inference/abc.py` | `inference-v08 recovery` | `confirm/recovery` | H4 NOT_ESTABLISHED |
| M13 real-data posteriors, hierarchical pooling, world sets | `inference/` | `inference-v08 posterior/hierarchical/worlds` | `inference/posterior-2`, `explore/hierarchical`, `inference/worlds` | registered posterior |
| M14 frozen vs adaptive calibration | `inference/adaptive.py` | `adaptive-calibration-v08` | `confirm/adaptive-{drift,regime,switch}`, `explore/adaptive-real` | H1–H3 ESTABLISHED (SYNTHETIC) |
| M15 identifiability v3 | `inference/identifiability.py` | `identifiability-v3` | `explore/identifiability` | EXPLORATORY; weak local identifiability |
| M16 posterior predictive checks, generator gate | `ppc.py` | `posterior-study-v08` | `explore/posterior-study` | NO_ACCEPTABLE_MODEL (gate not weakened) |
| M17 model-class uncertainty, dynamic averaging | `model_averaging.py` | `model-averaging-v08` | `explore/averaging-{synthetic,real}` | EXPLORATORY |
| M18 uncertainty decomposition | `uncertainty.py` | `uncertainty-study-v08` | `confirm/uncertainty-2` | H5 ESTABLISHED (SYNTHETIC) |
| M19 OOD detection | `ood.py` | `ood-study-v08` | `explore/ood-2` | EXPLORATORY |
| M20 conformal coverage | `conformal.py` | `conformal-study-v08` | `confirm/conformal` | H19 ESTABLISHED (SYNTHETIC, exchangeable only) |
| M21 decision-focused calibration, misspecification benchmark | `decision_calibration.py` | `calibration-utility-v08` | `confirm/calibration-utility-2` | H6 NOT_ESTABLISHED |
| M22 completion-constrained execution | `execution/completion.py` | `completion-study-v08` | `confirm/completion-2` | H7 NOT_ESTABLISHED |
| M23 objectives and Pareto fronts | `execution/completion.py` | `completion-study-v08` | `confirm/completion-2` | done |
| M24 selection across simulated plausible worlds | `robust.py` | `robust-execution-v08` | `confirm/robust-2` | H8 NOT_ESTABLISHED |
| M25 controlled execution benchmark | `execution/policies.py` | `mpc-study-v08` | `confirm/mpc-2` | H9 ESTABLISHED (SYNTHETIC) |
| M26 MPC and adaptive variants | `execution/mpc.py` | `mpc-study-v08` | `confirm/mpc-2` | H9 ESTABLISHED (SYNTHETIC) |
| M27 regime-aware selection, OOD abstention, safe fallback | `selection.py` | `policy-selection-v08`, `policy-abstention-v08` | `confirm/selection-2`, `confirm/abstention-4` | H10 NOT_ESTABLISHED; H11, H12 FAILED |
| M28 structured stress, bounded adversarial search | `stress.py` | `adversarial-stress-v08` | `explore/stress` | SYNTHETIC stress, not empirical |
| M29 historical execution bounds | `counterfactual.py` | `counterfactual-bounds-v08` | `confirm/counterfactual`, `explore/counterfactual-real` | H14 ESTABLISHED (SYNTHETIC); real days RETROSPECTIVE |
| M30 feed latency, stale books, cancel races | `execution/simulator.py` | `latency-study-v08` | `explore/latency` | EXPLORATORY |
| M31 cross-time/instrument/venue transfer | `transfer.py` | `transfer-study-v08` | `transfer/fresh`, `explore/transfer-retrospective` | H15–H17 NOT_AVAILABLE |
| M32 predicted vs bounded historical rankings; portfolio infrastructure | `transfer.py`, `portfolio.py` | `ranking-transfer-v08`, `transfer-study-v08 portfolio` | `transfer/ranking-*`, `explore/portfolio` | H13 NOT_AVAILABLE; retrospective EXPLORATORY |
| M33 statistical decision framework | `statistics.py` | all studies | `test_v08_statistics.py` | done |
| M34 planning, deterministic parallel runner, checkpoint/resume | `planning.py`, `runner.py` | all studies, `benchmark-v08` | `explore/benchmark-2` | resume and replay checks pass |
| M35 claim graph, audit, reports, evidence bundle | `claims.py`, `reporting.py` | `final-analysis-v08`, `report-v08`, `evidence-export-v08`, `claim-audit-v08`, `verify-v08` | `final-2`, `report`, `examples/studies/v08` | audit passes |

**Registered real holdouts.** The fresh-transfer days (2021-02-01) and the final multi-axis holdout (2021-03-01)
were never read. The registered availability probe returned HTTP 403 both times, so H13, H15–H17 and H20 are
NOT_AVAILABLE.
