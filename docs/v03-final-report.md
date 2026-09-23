# CleoLOB v0.3 validity upgrade — final research report

## Scope, provenance and preserved work

Work resumed on `research/v0.3-validity-upgrade` from the existing partial tree.
No reset, clean, stash drop, release-tag change or historical-evidence rewrite was
performed. The safety backup remains `stash@{0}`, object
`8dab9ee5311a9a44cdd09de993ec3b41b1e9625a`.

Baseline main was `9a683106213339670ecceafa3c2fd78efdf426a9`: 573 tests passed
before major changes; the preserved partial implementation reproduced 687 passing
tests on resume. Inspected all modified/untracked files, existing schemas/books,
L2/Tardis adapters, calibration/ZI modules, RL environment and controls, registered
studies, manifests, packaging and CI. The claims audit found stale README counts
and an incorrect implication that no completed PPO comparison existed.

Calibration and policy studies froze source at
`f6615c11536156f3e13d9a49322896c5194aec4a`; their accurate dirty flag is true
because documentation work continued. Full source hashes identify the exact code.
Later test/documentation changes do not change those scientific source bytes.
Benchmark/smoke provenance independently records its commit and hashes. All new
evidence is synthetic. Dataset definitions, periods, symbols, seeds, metrics and
environments are inside the retained plans and results.

## Implementation and architecture

| Component | Change and verification |
|---|---|
| `lob/mbo.py` | Strict timestamp/exchange-clock/sequence/identity schema; RESET/SNAPSHOT; bounded JSONL/gzip replay; source/canonical hashes; observed-order queue trajectories, age, cancellation ahead, fills, censoring and cohort frequencies. Reuses the existing audited historical book. |
| `lob/replay/l2.py` | Immutable capability declarations and explicit refusal of identity, exact FIFO, hidden quantity and passive-counterfactual requests. |
| `lob/generalization.py` | IID joint and two-state spread Markov observable generators, expanding/rolling chronological folds, elapsed-time embargoes, validation-only selection, sealed model, internal/external scoring, diagnostics and retained failures. |
| `lob/policy_study.py` | PPO and DQN for the existing Discrete(5) action space, separate optimizer/market/identification seeds, deterministic evaluation, fixed checkpoints, all six controls, three regimes, two retrained ablations and crossed-seed uncertainty. |
| `lob/performance.py` | Profiling first; warmed repeated small/medium/deep mutation, snapshot, L2 update/replay and MBO update workloads; complete simulation episodes; separate allocation measurement. |
| `lob/artifacts.py`, `lob/smoke.py` | Portable source/runtime metadata, exact artifact verification and a network-free generated L2/MBO/calibration/execution pipeline that runs from the installed wheel. |
| CLI and contributor tooling | `mbo-replay`, `calibration-study`, `policy-study`, `benchmark-suite`, `smoke`, `verify-artifact`; bootstrap, compact export and privacy scan. Dockerfile, CI matrix, contributor and issue/reproduction templates. |
| Documentation | Claim taxonomy, actual current evidence, retained failed calibrations, RL limitations, L2/MBO distinction and reproduction commands. Historical numerical records remain unchanged. |

The separate synthetic matching engine remains authoritative for simulated fills;
the identity replay engine applies recorded maker IDs without searching for a
different hypothetical maker. `TRADE` is an unlinked print; `EXECUTE` identifies
the maker. No invented order IDs, historical agent fills or queue reconstruction
from aggregate L2 were added.

## MBO validation

**IMPLEMENTED / TESTED:** FIFO/modify priority, partial/full execution,
cancel-before-fill, duplicate identities, invalid transitions, initial census,
sequence continuity/reset semantics, snapshot censoring, deterministic replay,
MBO-to-L2 aggregation, queue movement and parser/resource limits. Source snapshot
priority and venue amendment semantics remain adapter contracts. A censored
trajectory reports unknown remaining quantity rather than inventing zero.

The fixture and generated smoke are synthetic. No genuine historical MBO feed
was available among the local aggregate L2, top-five and trade files. Real MBO
validation is **PENDING**, and exact hidden liquidity/counterfactual passive fills
remain unavailable. Complete-case observed fill frequencies are descriptive and
can suffer selection/censoring bias.

## Calibration: actual registered results

The study uses 1,080 development samples, 360 later shifted samples, four folds,
a two-second embargo and two simulation seeds per evaluation phase. Each seed
generates 1,024 samples. Eight documented observables cover spread, bid/ask depth,
imbalance, log returns, trailing RMS returns, spread changes and relative visible
depth changes. Net depth changes are not labeled cancellations/replenishment.
Inter-arrival/trade/cancellation intensities and seasonality are unavailable from
this short sampled fixture and were deliberately not invented.

Selection minimizes mean validation loss only. Model/window, thresholds and
selection are sealed before internal/external scoring. External observations are
first generated after the seal. Quantile distances, finite-sample coverage,
validity, train-relative drift, parameter stability and fold variance remain
visible. Thresholds are diagnostic heuristics, not significance tests.

| Candidate | Mean validation loss | Four fold statuses |
|---|---:|---|
| IID / expanding | 0.487916 | FAIL, FAIL, FAIL, FAIL |
| IID / rolling | 0.481882 | FAIL, FAIL, FAIL, FAIL |
| **Spread Markov / expanding — selected** | **0.163811** | **FAIL, FAIL, WARNING, PASS** |
| Spread Markov / rolling | 0.269230 | FAIL, FAIL, FAIL, FAIL |

Selected fold-loss variance: 0.0010166562. Internal: **WARNING**, loss 0.290986.
External: **FAIL**, loss 0.937720. Both had 100% valid input rows.

| External observable | Normalized distance | Coverage | Status |
|---|---:|---:|---|
| Spread | 1.113937 | 41.111% | FAIL |
| Bid depth | 1.061183 | 56.667% | FAIL |
| Ask depth | 0.900704 | 58.611% | FAIL |
| Imbalance | 0.056237 | 93.889% | PASS |
| Return | 0.732394 | 77.159% | FAIL |
| RMS return | 1.137397 | 43.429% | FAIL |
| Spread change | 0.392628 | 97.493% | WARNING |
| Relative depth change | 0.115934 | 98.607% | PASS |

The improved validation fit does not establish external generalization. These
are observable generators, not calibrated FIFO execution dynamics. Existing
April/May 2020 WARNING/FAIL results and all six August/September 2026 simulator
FAIL gates remain intact. Those historical holdouts are consumed. The new
synthetic fixture does not replace independent fresh historical confirmation.

## PPO/DQN: actual registered results

18 models × 1,024 steps = **18,432 training steps**, with three independent
training seeds and eight common evaluation market seeds. All **576/576** episodes
completed, with **zero INVALID and zero WARNING** economic reports. Three regimes
and two ablations are fixed in the plan. **All 48 Bonferroni-adjusted comparison
intervals include zero.** This smoke budget does not establish convergence or
superiority.

Mean fee-inclusive effective cost, bps (lower is cheaper):

| Agent | Original | Shifted | Stress |
|---|---:|---:|---:|
| TWAP | 2.0399 | 2.5361 | 2.9858 |
| Synthetic-profile VWAP | 1.9961 | 2.4761 | 2.8629 |
| POV | 1.9645 | 2.5715 | 3.7129 |
| AC | 2.0345 | 2.5223 | 3.4108 |
| Heuristic | 1.9332 | 2.2282 | 2.2008 |
| Random | 1.9878 | 2.3356 | 3.2495 |
| PPO main | 2.0585 | 2.4495 | 3.7378 |
| DQN main | 1.9959 | 2.4841 | 2.0401 |

**DQN failed to complete most orders.** Completion rates were 0%, 4.17%, 4.17%,
with mean terminal inventory 272.125, 205.375 and 108.0833 of 300 units. Its cost
includes hypothetical residual liquidation; that is not an actual fill. PPO and
TWAP/VWAP/POV/AC completed 100% in each regime. PPO's stress mean was worse than
TWAP by 0.7519 bps and AC by 0.3270 bps; the adjusted intervals cross zero.

| Policy/regime | Ordinary 95% crossed mean interval, bps | Training-seed SD, bps |
|---|---|---:|
| PPO original | [1.7423, 2.3849] | 0.0301 |
| PPO shifted | [1.9616, 3.0257] | 0.0017 |
| PPO stress | [2.5202, 5.0286] | 0.1857 |
| DQN original | [1.4774, 2.5261] | 0.0486 |
| DQN shifted | [1.7273, 3.1722] | 0.0497 |
| DQN stress | [0.1187, 4.2402] | 0.3980 |

PPO-minus-AC 99.8958% family intervals: original −0.4348 to +0.4267 bps;
shifted −0.5153 to +0.8127; stress −1.2619 to +2.0706. Median, standard deviation,
p95/worst/tail mean, residual inventory, fill fraction, impact-related proxy,
training curves and seed sensitivity are retained in the compact result/metadata.
The proxy includes spread/timing/stochastic price movement and is not causal
permanent impact.

AC identification passed on separate seeds in all regimes: R²
0.9042/0.8206/0.4269 and probe coverage 1.0000/0.9944/0.9583. No fallback was used.
Both retrained ablations ran: stress no-terminal-minus-main cost was PPO +1.3168
and DQN +0.3526 bps; DQN no-book-minus-main was +3.1801. Corrected ablation
intervals also include zero. No failed historical calibration is labeled a valid
calibrated RL regime.

## Performance: actual local measurements

CPU: **Intel Core i7-13620H**, 16 logical processors; Windows 11 build 26200;
CPython 3.14.6; NumPy 2.5.1, pandas 3.0.5, Gymnasium 1.3.0, SB3 2.9.0,
Torch 2.13.0. Fixed simulation seed 27. Profiling preceded timing; local test and
training jobs had finished before the benchmark. No affinity/frequency controls
were imposed. No acceleration or production-HFT claim is made.

Each mutation/snapshot workload uses 20 warmup calls and 3 × 200 timed calls.
Throughput below is median batch operations/second; latency entries are p50 /
p95 / p99 microseconds over 600 calls.

| Workload | Small: 10 levels/side | Medium: 100 | Deep: 1,000 |
|---|---|---|---|
| Synthetic book modify ops/s | 207,684 | 222,370 | 217,368 |
| Modify latency μs | 6.20 / 9.90 / 14.91 | 5.20 / 6.905 / 8.507 | 6.10 / 7.60 / 8.801 |
| Top-five snapshot ops/s | 413,907 | 591,017 | 625,586 |
| Snapshot latency μs | 2.30 / 2.70 / 4.10 | 1.70 / 2.20 / 2.40 | 1.60 / 1.90 / 2.10 |
| L2 in-memory update ops/s | 561,325 | 580,215 | 707,214 |
| L2 update latency μs | 1.70 / 1.90 / 3.001 | 1.70 / 1.805 / 2.00 | 1.40 / 1.60 / 2.40 |
| MBO update ops/s | 68,074 | 48,281 | 11,297 |
| MBO update latency μs | 13.80 / 16.70 / 35.115 | 19.90 / 24.815 / 49.802 | 86.10 / 106.735 / 153.313 |
| CSV L2 replay rows/s | 74,011 | 84,287 | 88,396 |
| Peak Python bytes: synthetic/L2/MBO | 26,872 / 7,984 / 15,851 | 273,368 / 61,128 / 156,708 | 2,981,664 / 573,528 / 1,656,581 |

Whole two-second TWAP episode throughput: **166.13 episodes/s** median batch,
p50/p95/p99 **5.9672/6.35434/6.355108 ms** across nine measured episodes.
Replay has only three measured batches; its percentiles are coarse descriptive
estimates. Memory is separately measured Python allocation, not process RSS.
L2 in-memory updates exclude parsing; replay includes parsing and snapshots.
MBO includes schema/book checks but no watched queues in this benchmark.

Profile hotspots were `step`, background limit flow, order admission/process and
numeric validation. Correctness checks were retained; no unjustified rewrite or
speedup claim was introduced. Deep MBO cost shows room for a later measured
optimization, not a reason to mislabel research throughput as exchange latency.

## Validation and failures encountered

- Python 3.14.6 with RL dependencies: **700 passed, 2 warnings**.
- Isolated Python 3.11.9, 3.12.10 and 3.13.15 core environments: each **691 passed,
  9 optional-RL tests skipped, 2 warnings**. The separate RL CI job installs Torch.
- Ruff, compileall and dependency check passed; wheel build succeeded.
- A clean wheel-only environment outside the checkout ran the CLI, 12-episode
  offline smoke and artifact verification successfully. Bootstrap also passed.
- MBO/canonical/L2 review: 191 passed; calibration/generalization/ZI review: 70
  passed; policy review: 25 passed; benchmark/artifact/CLI smoke: 10 passed.
- Added 127 test cases relative to the 573-test original main. Tests cover
  invalid transitions, deterministic replay, censorship, holdout ordering,
  future/internal perturbations, checkpoint behavior, missing-grid inference,
  failure retention, artifact tampering and wheel-style execution.
- The two remaining warnings describe Gymnasium's unbounded observation Box;
  valid mathematical domains were not narrowed to silence them.
- Initial sandbox runs had Windows temporary-directory ACL setup failures.
  Approved runs with normal temporary access passed. Initial 3.11/3.12 runs
  exposed an old test's sub-clock-resolution timing assumption; its clock is now
  explicitly controlled, with no change to experiment budget behavior.
- An initial new smoke integration used a live-queue lookup after full fill;
  this was corrected to the retained observed-order record before the final run.
- No failed research outcome was removed: shifted calibration FAIL, DQN
  completion failure, PPO's unfavorable stress mean and old historical FAILs
  remain visible.

Remote CI and container verification status is recorded in `v03-ci.json` when
available. Local Docker/WSL were unavailable; the configured Linux CI container
job builds and executes the smoke rather than claiming a local Docker run.

## Commands executed

The following are the exact relative command forms used from the checkout
(stdout logs live only under ignored `results/v03/`). Review/diagnostic calls
also inspected the files and Git diffs described above.

```powershell
git branch --show-current
git status --short
git stash list
git log --oneline --decorate -10
.\.venv\Scripts\python.exe -m ruff check lob tests tools train_rl.py server.py
.\.venv\Scripts\python.exe -m compileall -q lob tools train_rl.py server.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip wheel . --no-deps -w results/v03/wheels
.\.venv\Scripts\python.exe -m pytest -q tests/test_artifacts_performance.py tests/test_v03_cli.py --tb=short
.\.venv\Scripts\python.exe -m pytest tests/test_mbo.py tests/test_replay.py tests/test_l2_replay.py -q
.\.venv\Scripts\python.exe -m pytest -q tests/test_generalization.py tests/test_calibration.py tests/test_zi_calibration.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_policy_study.py
.\results\v03\runtimes\py311\python.exe -m pytest -q
.\results\v03\runtimes\py312\python.exe -m pytest -q
.\results\v03\runtimes\py313\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools/bootstrap.py --environment results/v03/bootstrap-env
.\.venv\Scripts\python.exe -m venv results/v03/wheel-env
.\results\v03\wheel-env\Scripts\python.exe -m pip install results/v03/wheels/cleolob-0.3.0.dev0-py3-none-any.whl
.\.venv\Scripts\python.exe -m lob.cli calibration-study --config configs/v03-calibration.json --out results/v03/calibration-final
.\.venv\Scripts\python.exe -m lob.generalization --out results/v03/calibration-final --verify
.\.venv\Scripts\python.exe -m lob.policy_study register --config configs/v03-policy-study.json --out results/v03/policy-final
.\.venv\Scripts\python.exe -m lob.policy_study train --out results/v03/policy-final
.\.venv\Scripts\python.exe -m lob.policy_study evaluate --out results/v03/policy-final
.\.venv\Scripts\python.exe -m lob.policy_study verify --out results/v03/policy-final
.\.venv\Scripts\python.exe -m lob.cli benchmark-suite --operations 200 --repeats 3 --warmup 20 --out results/v03/benchmark-final
.\.venv\Scripts\python.exe -m lob.cli verify-artifact results/v03/benchmark-final
.\.venv\Scripts\python.exe -m lob.cli smoke --out results/v03/smoke-final
.\.venv\Scripts\python.exe -m lob.cli verify-artifact results/v03/smoke-final
.\.venv\Scripts\python.exe tools/compact_v03.py --calibration results/v03/calibration-final --policy results/v03/policy-final --benchmark results/v03/benchmark-final --smoke results/v03/smoke-final --out examples/studies/v03
.\.venv\Scripts\python.exe -m lob.cli verify-artifact examples/studies/v03
.\.venv\Scripts\python.exe tools/privacy_scan.py --base main --out results/v03/privacy-scan.json
git diff --check
```

From `results/v03/wheel-run`, the isolated wheel commands were:

```powershell
..\wheel-env\Scripts\python.exe -m pip check
..\wheel-env\Scripts\python.exe -m lob.cli --help
..\wheel-env\Scripts\python.exe -m lob.cli smoke --out smoke
..\wheel-env\Scripts\python.exe -m lob.cli verify-artifact smoke
```

## Artifact policy and privacy

The committed compact export contains 24 payload files plus its top-level seal,
**311,582 bytes** total: plans, frozen calibration, diagnostics, policy summaries,
model/trace hashes and summarized learning curves, benchmark/profile and smoke
results. Nested full-run manifests deliberately refer to files excluded from
this compact export; use the top-level `verify-artifact` command to verify it.

Intentionally excluded: 18 model ZIPs, episode/training journals, source snapshots,
raw exchange files, full smoke inputs, local virtual environments/runtimes,
wheel binaries, build output, temporary tests, console/CI logs and profiler binary.
The full RL run remains locally in ignored results: 122 files, 4,398,379 bytes.
Nothing was discarded or represented as a committed model distribution.

Privacy scan: **zero new personal path values**, but **23 pre-existing historical
files** on main contain absolute personal paths. The report lists only relative
file names, line numbers and hashes in `v03-privacy.json`. Those old sealed
records were not altered, honoring the explicit preservation instruction.
Consequently a repository-wide claim of zero personal paths would be false.
New v0.3 provenance omits the local executable path.

## Remaining limitations and acceptance boundaries

Real historical MBO validation and a fresh historical external calibration
holdout could not be completed because suitable unconsumed/order-identity data
was unavailable. No fake substitute was reported. No historically validated
calibrated RL regime could be claimed because the preserved fidelity gates fail.
The additional RL baseline is DQN, appropriate for five discrete actions; SAC
would require a separately justified continuous-action environment.

RL smoke establishes reproducible execution of the finite design, not adequate
training, precise tail estimates or SOTA. Only observation and terminal-penalty
ablations were run; no combinatorial reward/risk/frequency search was performed.
Performance remains local Python research measurement; process RSS, production
latency and compiled acceleration are not established. Community adoption and
independent review have not been fabricated. Legacy privacy exposure remains
disclosed rather than silently rewriting historical evidence.

## Review and version recommendation

Recommend **v0.3.0** after review; package metadata is deliberately
`0.3.0.dev0`, with v0.2.0/v0.2.1 tags preserved. Suggested review groups mirror
the logical commits: MBO; calibration; policy studies; benchmarks/artifacts;
reproduction/CI; portability/privacy; claims and compact evidence.

Recommended PR title: **Strengthen research validity with MBO, frozen calibration,
PPO/DQN studies and reproducible evidence**.

Recommended description: Add explicit source-identity MBO replay, leakage-resistant
rolling/expanding observable comparison, a registered multi-seed PPO/DQN framework
with six controls and bounded generalization/ablations, and profiled workload
benchmarks. Add wheel-safe smoke, portable verification, bootstrap, container and
Linux/Windows quality gates. Preserve historical failures: the new synthetic
calibration still fails its shifted holdout, DQN rarely completes, and all 48
corrected policy intervals include zero. Validation and remaining limitations are
recorded in this report. No historical fills, alpha or production-latency claims.

Safe README claims: these capabilities are implemented and tested; recorded
aggregate L2 matches are same-source consistency evidence; MBO is fixture-tested;
new calibration is synthetic and externally failed; the PPO/DQN finite smoke ran
with disclosed incomplete execution; benchmarks measure local research workloads;
portable installation/smoke/seal verification were actually exercised.
