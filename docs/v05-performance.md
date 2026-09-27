# v0.5 M11 — performance and scaling of the real-data workflows

These are local, single-process Python measurements on deterministic synthetic generators
(`lob/benchmarks_v05.py`), so they are redistributable and repeatable. They are **not**
exchange, colocation or HFT latencies, and throughputs in different units are not
comparable.

- **Run:** `cleo v05-benchmark --out results/v05/m11/benchmarks`, sealed and bound like every
  v0.5 run. Its summary is in the public bundle.
- **Hardware:** 13th Gen Intel Core i7-13620H, 16 logical CPUs.
- **Software:** Windows 11 (10.0.26200); CPython 3.14.6; numpy 2.5.1; torch 2.13.0;
  stable-baselines3 2.9.0.
- **Protocol:** three sizes (×1, ×4, ×16). One warm-up, then 3 timed repetitions, reporting the
  median wall time. Per-call p50/p95/p99 are given where the operation is separable. Peak
  traced Python allocation is measured in a separate pass.
- **Process memory:** RSS is **NOT_AVAILABLE**; it is not reliable across platforms here.
- **Total wall time:** 734 s.

| Workload (roadmap item) | Unit | ×1 | ×4 | ×16 | p50 / p95 / p99 per call at ×16 | Peak traced alloc at ×16 |
|---|---|---:|---:|---:|---|---:|
| MBO parsing and canonical normalization | order messages/s | 86,000 | 22,100 | 21,600 | — | 95 MB |
| Identity (lifecycle) replay | events/s | 236,000 | 53,900 | 53,200 | 17.6 / 19.8 / 31.6 µs | 3.9 MB |
| MBO-to-L2 aggregation (top 10) | aggregations/s | 102,000 | 36,300 | 34,400 | 24.7 / 25.4 / 44.1 µs | 1.3 MB |
| Fill-bound computation | tracker updates/s | 1,790,000 | 560,000 | 548,000 | 789 / 1,101 / 1,307 µs (400 updates per call) | 0.2 MB |
| Streaming aggregate-L2 historical execution | L2 rows/s | 49,300 | 11,600 | 11,400 | — | 30 MB |
| Impact and resilience extraction | aggressive events/s | 203,000 | 214,000 | 209,000 | — | 19 MB |
| Calibration v2 summary (simulate + measure) | simulated s per s | 72 | 92 | 101 | — | 114 MB |
| Extended simulator generation | simulated s per s | 29 | 32 | 32 | — | 87 MB |
| Policy evaluation | episodes/s | 15.9 | 15.6 | 15.6 | — | 1.0 MB |
| Evidence serialization | records/s | 269,000 | 271,000 | 262,000 | — | 11 MB |

## Reading

**Linear scaling.** Several workloads have constant throughput across sizes:
- impact extraction;
- policy evaluation;
- serialization;
- simulation (it improves slightly as fixed costs amortize).

**Step then plateau (×1 → ×4).** Order-level workloads lose throughput from ×1 to ×4, then
hold constant:
- MBO parsing and identity replay;
- aggregation;
- historical execution.

The per-event replay cost rises from 3.9 to 17.6 µs as the synthetic book holds more
resting orders.

**Memory.** Parsing holds the whole normalized stream in memory: about 1.5 KB of traced
allocation per order message. A multi-hour capture therefore needs chunked parsing. v0.5's
30-minute captures (271,290 messages) fit comfortably.

**Fill bounds.** At about 0.5–1.8 million tracker updates per second, fill bounds are cheap
relative to the L2 replay that feeds them.

**Simulation dominates registered-study cost.**
- The extended simulator produces about 30 simulated seconds per wall second.
- The calibration v2 search (48 candidates × 3 seeds × 900 s per family) is the most
  expensive v0.5 path.
- The M8 transfer study (19,008 replay episodes on 12 worker processes) took about
  36–37 minutes.

**No production-HFT claim follows.**
