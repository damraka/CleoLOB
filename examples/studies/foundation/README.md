# Foundation execution study — 2026-09-13

**Completed mechanics study; insufficient evidence for alpha.** This is a small
study of execution costs in an uncalibrated synthetic market, included to exercise
the research pipeline with actual recorded outcomes.

## Design and assumptions

The frozen [research preset](../../../configs/research.yaml) runs TWAP, VWAP, POV,
Almgren–Chriss, a passive/catch-up heuristic and a random policy on seeds 101–110.
Each episode sells 2,000 endowed shares over at most 10 seconds, with decisions
every 0.5 seconds. The tick size is 0.01 currency units; the initial midpoint is
10,000 ticks. Taker fees are 1 bps, maker fees zero, and the maximum child quantity
is 1,000 shares. Latency is 5 ms plus exponential jitter with mean 5 ms.

Arrival price is measured after a 5-second warmup. The synthetic market uses
Poisson arrivals, exponential sizes/cancellation lifetimes and state-dependent
resilience. VWAP uses volume forecasts from five synthetic shifted-seed paths;
it is not calibrated to historical volume. Observations have no feed delay.

This evaluates **one** market configuration × six agents × ten seeds = **60
episodes**. All combinations in that declared set ran. The wider parameter space
was not covered. The primary run took **117.38 seconds** locally; its reproduction
is a separate run. These wall times include configuration-specific invariant
checking and do not measure engine throughput.

## Recorded outcomes

Economic effective implementation shortfall in bps; lower is cheaper. Confidence
intervals are marginal 95% percentile bootstrap intervals over the ten seeds.
The metric includes actual fees and hypothetical terminal liquidation cost/fees
when depth is sufficient. It excludes the separate RL noncompletion penalty.

| Agent | Mean bps | 95% CI | Median bps | P95 bps | Mean actual fill |
|---|---:|---:|---:|---:|---:|
| TWAP | 2.237 | [1.964, 2.508] | 2.110 | 2.824 | 100.00% |
| VWAP | 2.266 | [1.910, 2.662] | 2.107 | 3.283 | 100.00% |
| POV | 2.240 | [1.783, 2.694] | 2.236 | 3.199 | 100.00% |
| Almgren–Chriss | 2.295 | [2.075, 2.536] | 2.150 | 2.916 | 100.00% |
| Heuristic | 2.213 | [1.873, 2.529] | 2.147 | 2.837 | 90.87% |
| Random | 2.153 | [1.709, 2.623] | 2.092 | 3.169 | 90.83% |

All five candidate-minus-TWAP bootstrap intervals include zero; all five
Holm-adjusted sign-test p-values are **1.0**. These observations provide no clear
evidence that a candidate beats TWAP in this small design. They also do not prove
equivalence. The random policy's lower sample mean does not establish an edge.

There were **50 VALID and 10 WARNING** episode records. No failed/invalid episode
was removed. Warning status discloses residual working orders or risk rejections;
the detailed order/risk logs identify the event for each episode. Remaining
working orders receive delayed cancels at the horizon. Terminal valuations are
hypothetical and are not completed liquidation trades. All leftover quantities
in this run had sufficient visible depth for economic valuation.

## Artifacts and reproduction

- [Primary offline HTML report](20260913T191627-0268a8fa8cb3/report.html)
- [Every episode](20260913T191627-0268a8fa8cb3/episodes.csv)
- [Descriptive statistics](20260913T191627-0268a8fa8cb3/summary.csv)
- [Paired tests and corrections](20260913T191627-0268a8fa8cb3/paired.csv)
- [Resolved configuration](20260913T191627-0268a8fa8cb3/resolved_config.json)
- [Provenance and code hashes](20260913T191627-0268a8fa8cb3/metadata.json)
- [Independent rerun](20260913T191842-dca37a631e1c/report.html)
- [Saved reproduction comparison](reproductions/20260913T191842-dca37a631e1c.json)

Both complete runs passed artifact integrity checks. Their 60-row episode JSONL
files are **byte-for-byte identical**. Each run also contains strategy order/fill/
risk logs, causal policy observations, seed manifests and a source snapshot.
Reproduction requires the recorded implementation/runtime and never executes
archived source automatically.

```sh
cleo evaluate --config configs/research.yaml --out examples/studies/foundation
cleo reproduce examples/studies/foundation/20260913T191627-0268a8fa8cb3
cleo verify examples/studies/foundation/20260913T191627-0268a8fa8cb3
```

## Tests not performed

PPO training/comparison: **NOT RUN**. SAC: **NOT RUN**. Historical-market
validation/calibration: **NOT RUN**. Stress, OOD, parameter perturbations,
walk-forward, capacity and ablations: **NOT RUN**. The canonical replay fixture
is fabricated and is not the data source for this study.

The current implementation is sufficient to reproduce this mechanics comparison,
not to claim live profitability or robust execution alpha. The next study should
use a licensed historical dataset and explicitly predeclare a wider hypothesis
family and untouched future windows.
