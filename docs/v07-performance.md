# v0.7 performance, parallelism and test hardening (M18, M19)

**Scope.** All timings are local research-workload measurements on one Windows machine (16
logical CPUs, no GPU, CPU-only torch). They are never exchange, network or HFT latency.

## Reference and accelerated paths (workstreams 53, 77)

The reference engines stay the auditable source of truth:
- the frozen v0.5 matching engine
- the v0.6 tape builder
- the v0.6 support rule
- the torch GRU

Accelerated or alternative paths are used scientifically only after a differential test passes:

| Path | Reference | Differential test |
|---|---|---|
| v0.7 tape builder (configurable reader limits) | `lob.v06.tape.tape_from_tardis` | bit-identical on synthetic files (`tests/test_v07_data.py`) and on the full development day (ledger entries 41–43) |
| numpy GRU forward pass in simulation | torch `nn.GRU` | `tests/test_v07_performance.py`, tolerance 1e-5 |
| precomputed-distance support coverage | `lob.v06.domain_gap.support` | exact equality, `tests/test_v07_performance.py` |
| deterministic replay hash chain | `lob.replay.l2.L2Replay` final book | `tests/test_v07_exchange.py` |

**Compiled backends** (Numba, C++, Rust) are **NOT_AVAILABLE**: they are not installed, and none
is used.

## Parallel execution and resumability (workstream 54)

`lob.v07.benchmark.parallel.run`:
- returns results in task order regardless of completion order, so single-process and parallel
  runs are identical
- writes atomic per-task checkpoints
- resumes without recomputation
- refuses to mix checkpoints from different tasks

The study runners use process pools with per-task seeds. The worker count therefore changes wall
time, not results.

**Memory incident.** The host stopped the first M6 posterior run (14 workers, plus a concurrent
dry run) when memory ran critically low. It is recorded as ABORTED. The rerun used 8 workers
with identical seeds and budgets.

## GPU (workstream 55)

**NOT_AVAILABLE.** There is no CUDA device. The GPU stack is optional and absent, and the
package installs and runs CPU-only (verified in the isolated install of the final engineering
audit).

## Compute accounting (workstream 56)

`lob.v07.benchmark.compute.Meter` records per study:
- wall time
- main-process CPU
- peak memory (Windows API / `resource`; NOT_AVAILABLE elsewhere)
- workers, with an upper-bound CPU of wall time × workers
- OS, Python, CPU count and GPU status
- study counters

The compute table of the registered studies is in `docs/v07-final-report.md`.

## Test hardening (workstreams 73–77)

- **Property-based tests** (Hypothesis, `tests/test_v07_properties.py`):
  - matching-engine quantity conservation (resting + 2 × traded + cancelled = submitted)
  - book ordering
  - nonnegative sizes
  - queue-model ordering and bounds
  - record serialization round trip
  - ledger hash-chain integrity under arbitrary appends
- **Fuzzing** — sequences with these events must never crash or corrupt the book:
  - duplicate IDs
  - invalid and repeated cancels
  - timestamp reversal
  - empty book sides
  - negative sizes
  - crossing orders
  - partial-fill and replace races

  Fuzzed L2 record streams are always classified, never crashing.
- **Metamorphic relations** — each with a documented assumption:
  - price-scale invariance of dimensionless metrics
  - deterministic reseeding
  - equivalent event transformations (an intermediate size overwritten within one local
    timestamp group leaves the tape unchanged)
  - aggregation consistency of additive sketches and of 10 s vs. 60 s counts
- **Cross-platform determinism** — **NOT_AVAILABLE** locally. Only Windows is available; Linux
  CI (`.github/workflows`) runs only when the branch is pushed, and it was not pushed. The
  invariant tests are platform-independent, and exact-equality tests use integer or
  canonical-JSON digests.

## Benchmarks

The frozen benchmark tasks (`configs/v07/benchmark-tasks.json`, `cleo benchmark-v07 all`) cover:
- reconstruction
- queue
- calibration
- identifiability
- certification
- execution
- transfer
- performance

Measured timings of the registered studies (posterior, bank, holdout evaluation, execution
study, policy training) appear in the compute table of the final report.
