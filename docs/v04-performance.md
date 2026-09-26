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
