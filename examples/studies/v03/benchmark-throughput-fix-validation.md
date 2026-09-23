# Post-fix benchmark methodology evidence

This benchmark was regenerated after separating instrumented per-call latency
measurement from wall-clock batch throughput measurement.

## Code under validation

- Git commit: `51d62d5a42ffe4dc412feea1c9e068716114a7cd`
- `lob/performance.py` SHA-256:
  `d0149014c4d581461723b15ab7672225d53c8004a414ca6bd812a36f18470f41`

## Methodology

Per-call p50/p95/p99 values are measured with individual
`perf_counter_ns()` instrumentation.

`operations_per_second` and
`latency_derived_operations_per_second` are derived from those sampled
per-call durations.

`batch_operations_per_second` is measured separately on a fresh equivalent
state with one outer wall-clock timer and without per-operation timing
instrumentation.

This prevents instrumented call latency from being mislabeled as literal
batch wall-clock throughput.

## MBO results

Small:
- p50: 13,300 ns
- p95: 14,805 ns
- p99: 24,209 ns
- latency-derived throughput: ~71,051 ops/s
- median wall-clock batch throughput: ~71,390 ops/s

Medium:
- p50: 19,400 ns
- p95: 21,300 ns
- p99: 26,704 ns
- latency-derived throughput: ~50,531 ops/s
- median wall-clock batch throughput: ~50,081 ops/s

Deep:
- p50: 82,100 ns
- p95: 91,040 ns
- p99: 129,786 ns
- latency-derived throughput: ~11,944 ops/s
- median wall-clock batch throughput: ~11,617 ops/s

## Interpretation

These are warm, repeated, in-process local Python research workload
measurements. They include Python/runtime and operating-system effects and do
not establish production, exchange-colocation, tail-latency, or HFT
performance.

The benchmark uses synthetic workloads. Earlier benchmark evidence is
intentionally retained unchanged.
