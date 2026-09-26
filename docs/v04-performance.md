# v0.4 local scaling measurements

`cleo scaling-study --out results/v04/scaling-new` freezes the workload dimensions,
source hashes, generator identity and seeds before profiling and measurement.
`cleo verify-evidence results/v04/scaling-new` checks bytes and semantic bindings.
An optional `--config` accepts exactly the fields in `lob.scaling.DEFAULT_CONFIG`.

The default sweep uses 10/100/1,000 levels per side, 100/1,000 mutations,
snapshots every 1/10 mutations, 128/512/2,048 calibration rows and 2/8 episodes.
Each workload has one complete warmup, three timed repetitions and one separate
Python-allocation pass. The factories create fresh equivalent state each time.

| Family | Included work | Unit |
|---|---|---|
| Engine / MBO mutations | Initial book, amendments, periodic snapshots, invariants | mutation events |
| L2 CSV replay | Initial census, parsing, reconstruction, top-five snapshots | source rows |
| MBO JSONL replay | Parsing, initial census, identity transitions | canonical events |
| Calibration | Fit and diagnostic evaluation on separate chronological synthetic samples | feature rows |
| Heuristic policy | Complete simulation episodes | episodes |
| Experiment runner | Episodes, statistics, reports, source snapshots and artifact writing | episodes |
| JSON roundtrip | Encode and decode canonical records | records |

The output separates mean/median/stdev wall time, batch throughput, amortized
per-unit cost and peak traced Python bytes. Batch percentiles are whole-workload
durations, not per-event tail latency; three repeats cannot reliably estimate
extreme tails. Process RSS is explicitly NOT_AVAILABLE. Timed runs do not include
the allocation tracer. No threshold is applied in CI because local timing is noisy.

For individually instrumented mutation/snapshot calls, also run
`cleo benchmark-suite --operations 200 --repeats 3 --warmup 20 --out results/v04/calls-new`.
That existing harness separates per-call p50/p95/p99 and latency-derived rates
from a fresh-state outer wall-clock throughput pass. Its MBO workload excludes
queue watches; the new sweep also does not establish queue-watch scaling.

These are fixed synthetic Python research workloads. They do not establish
exchange/colocation latency, historical strategy performance or neural inference
latency. Hardware, OS, Python, dependencies, source and exact workload sizes are
recorded with each run. Run heavy benchmarks separately from training and tests.

## Recorded local run

The 45-workload sweep and separate call study ran after training, historical
extraction and tests finished, at clean source `aeed97f`. Hardware: Intel
Core i7-13620H, 16 logical CPUs; Windows 11 build 26200; Python 3.14.6.
NumPy 2.5.1, pandas 3.0.5, Gymnasium 1.3.0 and the remaining dependency versions
are recorded in provenance. No CPU affinity/frequency control was imposed.

For 1,000 mutations plus the declared initial census:

| Levels per side | L2 CSV rows/s, median | MBO JSONL events/s, median | MBO peak Python bytes | MBO update call p50/p95/p99, microseconds |
|---:|---:|---:|---:|---:|
| 10 | 72,247 | 25,336 | 156,101 | 12.8 / 13.6 / 16.009 |
| 100 | 83,157 | 21,293 | 359,599 | 19.4 / 22.805 / 35.9 |
| 1,000 | 94,205 | 7,422 | 2,408,713 | 81.8 / 97.04 / 240.76 |

L2 rates count 1,020/1,200/3,000 rows respectively; MBO rates count 1,001 events
including one multi-order census. These different units must not be compared as
equivalent work. Call percentiles come from 600 instrumented amendments per depth
in the separate harness, not the JSONL parser. Engine snapshots are top-five;
MBO mutation/snapshot sweeps materialize full depth, as recorded per workload.

Calibration fit/diagnostic median runtime was 0.0193/0.0258/0.0511 seconds for
128/512/2,048 feature rows. Full experiment orchestration was 0.8405/1.1087 seconds
for 2/8 short episodes including reports and provenance. Source/raw samples,
repetition variability and allocations remain in the machine-readable results.
These describe local scaling costs; they are neither optimization claims nor
evidence of production trading performance.
