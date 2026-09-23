# v0.3 validity upgrade — implementation plan

## Inspected baseline (2026-09-23)

Branch `research/v0.3-validity-upgrade` starts at current remote main
`9a683106213339670ecceafa3c2fd78efdf426a9`. The working tree was clean;
release tags are unchanged. Inspected engine, replay schemas/books/L2 parser,
calibration and ZI fitting, execution controls, registered PPO studies,
configuration, manifests, packaging, tests and both CI workflows.

Baseline: Python 3.14.6, 573 tests passed in 38.56 seconds. Two Gymnasium
warnings describe the legitimately unbounded observation space. Initial Windows
sandbox runs had temporary-directory setup errors; a fresh workspace basetemp
completed the unchanged suite. This is an environment issue, not suppressed tests.

The README's 521-test count and statement that no completed PPO comparison
exists are stale. Compact core evidence records 20 PPO fits and 704 synthetic
evaluation episodes. Historical L2 reconstruction is strong same-source
consistency evidence; the April/May observable failures and all six later ZI
cohort gates remain failures. May 2020 and September 2026 are consumed holdouts.
No actual MBO feed was found among the available aggregate L2/snapshot/trade files.

## Ordered work and ownership

1. Reuse the existing order-ID historical book in a strict explicit MBO layer;
   add exchange timestamp/reset semantics, bounded parser, observed-order queue
   research, deterministic L2 aggregation and L2 capability refusal. Test FIFO,
   invalid transitions, censored histories, sequence gaps and replay determinism.
2. Add registered chronological development/internal/external evaluation with
   rolling and expanding folds; compare IID observable bootstrap with a
   train-defined spread-state Markov family. Freeze selection and thresholds
   before external access. Run a clearly synthetic shifted-holdout fixture.
3. Reuse the Discrete(5) execution environment for PPO and DQN, all six classical
   controls, independent optimizer seeds, paired market seeds, finite ablations
   and shifted/stress regimes. Preserve failures and low-budget limitations.
4. Profile real workloads before choosing optimizations. Add controlled warmup
   and repeated small/medium/deep benchmarks for mutations, snapshots, L2/MBO
   replay, simulation and separate Python-memory instrumentation.
5. Add portable compact artifact verification, one-command bootstrap/smoke,
   wheel/container/CI checks and contributor/reproduction templates.
6. Reconcile current claims with tested code and retained evidence; leave
   historical artifacts intact. Run all quality/build/install checks and bounded
   experiments only after source is committed. Commit compact summaries and
   hashes, keeping checkpoints, raw data, logs and episode dumps outside git.

## Pre-evaluation commitments

New calibration thresholds, model candidates, seeds, gaps and observables live
in the registered configuration/protocol. New RL comparisons and multiplicity
family live in its frozen plan. Synthetic fixtures validate the workflow, not
historical realism. Real MBO validation and fresh historical confirmation remain
pending until appropriate data exists. Performance timings describe local Python
research workloads only. No historical counterfactual fill or live alpha claim.

## Validation gate

Run focused tests per commit, then Ruff, compileall, full pytest, pip check,
wheel build, clean wheel installation and CLI/evidence smoke. Exercise supported
Python versions where obtainable; distinguish local execution from remote CI.
Do not change mathematical domains merely to remove warnings.
