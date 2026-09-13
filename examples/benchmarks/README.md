# Matching microbenchmark

[Recorded output](matching-windows-python314.json), measured locally on Windows 11
with Python 3.14.6 after the study processes finished:

- Three repetitions of 4,000 orders, producing 2,000 fills per repetition.
- Median **64,835 orders/second**; samples **62,958 / 64,835 / 67,681**.
- Separate untimed memory pass with 2,000 resting orders: **1,153,972 bytes** peak
  Python allocation reported by `tracemalloc` (about 1.10 MiB).

```sh
cleo benchmark --pairs 2000 --repeats 3
```

The workload alternates a new resting sell limit and a fully matching buy market
order. It reconciles fill counts, an empty final book and book invariants. The
memory pass measures a different, explicitly disclosed resting-book workload.
`tracemalloc` is not process RSS. Timing excludes replay, market generation,
latency scheduling, policy inference and the experiment registry. The first
exploratory run during an active study measured about 75,869 orders/second;
ordinary machine variability makes those samples unsuitable for a speedup claim.

No performance threshold, production throughput, improvement over the old
engine, Rust acceleration or GPU benefit is claimed. Use repeated measurements
under controlled conditions before deciding what to optimize.
