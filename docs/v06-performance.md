# v0.6 performance (M16)

These are registered local benchmarks of v0.6 research workloads (`cleo benchmark-v06`; sealed
run `results/v06/m16/benchmarks-2`).

- The measurements are single-process Python, on the selected v3 world and on synthetic
  inputs. No restricted data is read.
- **Units differ by workload and must not be compared across rows.**
- None of these numbers is an HFT, latency, colocation or exchange benchmark.

| Workload | Throughput | Median s / repeat |
|---|---|---|
| simulate a 10-level tape (selected v3 world) | 644 simulated seconds / s | 11.19 for 7,200 s |
| measure 600 s blocks | 54.6 blocks / s | 0.22 |
| sketch blocks | 384 blocks / s | 0.031 |
| realism comparison (85 components, 9 families) | 171 comparisons / s | 0.117 for 20 |
| block-bootstrap draws (one model) | 164 draws / s | 0.30 for 50 |
| calibration candidate (1 seed × 600 s) | 1.04 evaluations / s | 0.96 |
| execution episode (TWAP, mandate of record) | 4.06 episodes / s | 0.74 for 3 |
| logistic discriminator fit (20 features) | 2.06 × 10⁶ windows / s | 0.001 for 2,000 |

The first benchmark attempt failed before a run directory was created: its synthetic world
produced too few depleting events to freeze bins. It is recorded as INVALID in the v0.6
consumption ledger (entry 79) and in the research chronology. The ledger note names the
intended path `results/v06/m16/benchmarks`, but no such directory exists.

## Wall-clock cost of the registered study

Local machine: 16 logical CPUs, 14 worker processes.

| Run | Wall clock |
|---|---|
| observable design (two full days, 10-level tapes) | ≈ 6 min |
| calibration v3 search (2,304 candidates + 64 re-scores) | 114 min |
| calibration selection (33 candidates × 5 × 3,600 s) | 6 min |
| regime-conditioned calibration (2 × 1,152 candidates) | 95 min |
| identifiability diagnostics (≈ 1,890 evaluations) | 37 min |
| policy training (16 models × 16,384 steps) | 13 min |
| world evaluation (76,000 episodes plus world realism) | 68 min |
| holdout evaluation (per dataset) | 2–5 min |
| transfer evaluation (per dataset, 5,760 rows) | 4–5 min |

The deterministic event-rate cap (protocol amendment 2) bounds the cost of implausible
candidates. 468 of the 2,304 search candidates hit it.
