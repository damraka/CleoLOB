# v0.4 external-validity development report

This branch implements the software paths in the [unchanged roadmap](v04-roadmap.md)
and adds new, bounded evidence. It is **not a v0.4.0 release**. Package version is
`0.4.0.dev0`; release tags and v0.3 evidence remain unchanged.

## Repository and implementation

Work began with a clean local README branch whose PyPI polish was initially unmerged;
[PR #9](https://github.com/damraka/CleoLOB/pull/9) passed every required check and
merged through the protected workflow. Implementation began from updated main
`c6e9d1a` on `research/v0.4-external-validity`. Baseline: 713 tests passed.

| Roadmap | Delivered | Remaining evidence boundary |
|---|---|---|
| M0 protocol | Written protocol; per-run frozen source, metrics, hypotheses, inputs, seeds and gates | Hashes cannot prove prior human non-inspection |
| M1 capabilities | Immutable trades/L1/L2/MBO/simulator contracts; unsupported claims raise | Declarations still require truthful source semantics |
| M2 MBO | Canonical and explicitly mapped CSV adapters; strict identity replay, lifecycle/censoring, FIFO and aggregation diagnostics | Genuine historical MBO/native venue validation NOT_AVAILABLE |
| M3 historical generalization | Physically separate chronological inputs, sealed selection, consumed registry, detailed failure diagnostics; one fresh real period | Broad cross-period/venue validity NOT_ESTABLISHED |
| M4 policy study | Shared urgency, actual completion metrics, train-only normalization, own-order state, registered PPO/DQN study | Superiority, convergence and prospective power NOT_ESTABLISHED |
| M5 scaling | Depth/event/snapshot/episode-size sweeps, parsing, calibration, serialization, timing and allocation measurement | Production latency, process RSS and neural inference latency not measured |
| M6 reproduction | Installed-wheel/new CLI smoke, exact-source/checkpoint/config bindings, compact export and CI | No independent external researcher replication claimed |
| M7 review | Tests, build, limitations audit and review PR | Real MBO milestone and release review remain incomplete |

New package modules: `capabilities`, `datasets`, `mbo_validation`, `multiperiod`,
`calibration_diagnostics`, `historical_generalization`, `completion`, `observations`,
`evidence`, `scaling`. Existing replay/environment/runner/study APIs retain v0.3
defaults; the new policy protocol is opt-in. Archived source copies are retained
as evidence, not treated as redundant application code.

## MBO and calibration evidence

The explicitly synthetic MBO input has 12 events, five identities, three recorded
executions totaling 12 lots, five exact aggregate-reference comparisons and 26
queue checks. Replay is deterministic, with zero mismatches. This establishes
fixture mechanics, not real historical queue truth. Missing order IDs are never
invented; aggregate L2 cannot supply FIFO/quantity-ahead or passive fill evidence.

The new synthetic multi-period study selected a model using development/selection
only. Internal loss was 0.108912 (PASS); shifted external losses were 0.976637 and
0.552638 (both FAIL). No target refit occurred. The separate v0.3 failure diagnosis
reconstructed the exact old frame/model hashes and retained internal WARNING and
external FAIL. Distribution, tail, coverage, persistence and dependence diagnostics
are descriptive associations; causal attribution remains NOT_ESTABLISHED.

The historical path reused the immutable April 2020 model and its gates. Old
validation WARNING, internal FAIL and May FAIL reproduced. The preregistered
June 1, 2020 Deribit ETH-PERPETUAL period **also failed**. June is now consumed;
it cannot be presented as fresh in a subsequent study. The original download
attempt failed before June access because sandbox networking was unavailable.
That sealed NOT_AVAILABLE attempt remains in `results/v04/historical-diagnostics`;
the approved-network retry used the same frozen design in a new directory,
`results/v04/historical-diagnostics-network-retry`, and retained the actual FAIL.

Detailed historical artifacts, raw data and the fitted historical model remain
local under the provider's current redistribution restrictions. The public registry
contains identity/hash/access metadata; see [calibration details](v04-calibration.md)
for official terms and exact local paths. One fresh period from one instrument
cannot establish broad generalization. Historical L2 reconstruction counts from
v0.3 remain same-source consistency evidence and were not rewritten or expanded
by these calibration results.

## Completion-constrained policy results

At clean source commit `9072bb8`, the study trained 24 final models, totaling
49,152 training steps, and evaluated all 1,080 registered cells across three
regimes, four training seeds and twelve evaluation market seeds. All completed
by the end of settlement; zero residuals, INVALID outcomes or WARNING outcomes occurred.
Nineteen completed after the two-second decision horizon, at most 2.056913 seconds.
Post-horizon settlement is explicit; no new orders were submitted after horizon.
Hypothetical terminal valuation never supplied an actual fill.

All agents, including controls, used the same completion rule. The study does not
isolate that rule's causal effect against v0.3: normalization, seeds and training
budget also differ. The old DQN completion failure remains visible.

The family contains 48 paired contrasts with cost and completion endpoints:
96 Bonferroni-adjusted hypotheses. **47/48 cost intervals include zero.** The sole
exclusion is stress DQN minus POV: mean −2.133149 bps, 99.947917% interval
[−4.549744, −0.075294]. All 48 completion contrasts are INCONCLUSIVE under the
preregistered constant-difference rule; identical observed completions do not
exclude unseen future failures. **Zero joint success gates pass.** Learned-policy
superiority, equivalence, adequate power, convergence and live profitability remain
NOT_ESTABLISHED. [The RL report](v04-rl.md) includes selected main-policy means, seed
sensitivity, protocol hashes and the precise inference convention.

## Validation and artifact handling

The separately registered scaling study completed **45 workload configurations**
at clean source `aeed97f`, followed by the warmed per-call harness. Full measurements
include runtime variation, per-call percentiles, throughput units and separate
Python allocation passes. MBO JSONL replay fell from approximately 25,336 events/s
at 20 active orders to 7,422 at 2,000 active orders on the local machine; no speedup
or production latency claim follows. [Performance tables](v04-performance.md)
describe the workload differences and exact environment.

Final local suite: **817 passed**, 104 more than baseline; two unchanged warnings
describe the legacy unbounded Gymnasium observation space. The v0.4 observation
space is bounded and has training-only normalization. Ruff, compileall, dependency
checks, source-distribution/wheel builds and editable installation passed.
A clean wheel environment imports `site-packages` outside the checkout and runs
the offline smoke, new MBO validation and multi-period commands with successful
artifact verification. It reproduces the synthetic calibration FAIL honestly.

Source-stage CI passed all required Linux/Windows Python 3.11–3.14, container,
research correctness and PPO/DQN contract jobs at `9072bb8`. Final review-head
checks are available on [PR #10](https://github.com/damraka/CleoLOB/pull/10).
The optional legacy full-study dispatch remains intentionally skipped.

Audit fixes covered constant completion intervals, missing terminal quotes,
duplicate model identities, all-invalid calibration references, source-hash resource
bounds, fresh-period aliases and historical-data export refusal. Fixes made after
the studies do not rewrite their source snapshots or results. Performance runs use
their separately recorded source revision. Implementation LF checkout rules make
source hashes portable; archived evidence bytes retain their original rules.

The old 77-file v0.3 compact bundle still verifies. A privacy scan found no newly
introduced personal paths; the 23 pre-existing historical files remain untouched.
No secrets, release tags, branch-protection rules or publication credentials were
changed. The public compact export contains permitted synthetic summaries and
performance results; excluded checkpoints/traces/source copies and historical
data are explicitly listed with original artifact hashes.

## Reproduction and release decision

[Exact commands](v04-reproduction.md) cover every workflow and clean-wheel checks.
Full local runs are write-once under `results/v04/`; the public evidence bundle is
`examples/studies/v04/evidence`. Verify it with:

```sh
cleo verify-artifact examples/studies/v04/evidence
```

The implementation is ready for review, not final v0.4.0 publication. Remaining
blockers are genuine historical MBO evidence, permission to redistribute detailed
new historical evidence, and independent/release review. The failed fresh holdout
and absent learned-policy superiority are retained scientific outcomes, not
software failures to tune away.
