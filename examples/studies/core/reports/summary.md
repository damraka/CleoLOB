# Fixed-budget PPO versus simulator-fitted AC

Primary, 0 bps completion penalty: **-1.605 bps (95% CI [-2.095, -1.015])**.
Positive differences mean PPO costs more. Economic cost includes actual fills, fees and
hypothetical visible-book residual liquidation; it excludes the completion penalty.

Inference status: `PRIMARY_IDENTIFIED`. These are conditional synthetic results;
the separately frozen real-data calibration failed its external fidelity gates.

The study fitted **20 PPO models**: 5 optimizer seeds × 4 penalties, 8,192 steps each. This fixed budget is not convergence evidence.

Final evaluation retained **704/704 planned episodes**, including **0 INVALID**, across **32 common market seeds**.

## Penalty ablation

Intervals independently resample optimizer and market seeds. The simultaneous family contains
four penalty arms and the prespecified risk-sensitive AC comparison.

| Training penalty | PPO minus risk-neutral AC | Complete markets | INVALID PPO episodes | Mean actual fill |
|---|---|---:|---:|---:|
| 0 bps | -1.605 bps (99% CI [-2.213, -0.810]) | 32/32 | 0 | 86.90% |
| 5 bps | -1.839 bps (99% CI [-2.256, -1.343]) | 32/32 | 0 | 98.42% |
| 25 bps | -1.745 bps (99% CI [-2.164, -1.237]) | 32/32 | 0 | 98.67% |
| 100 bps | -1.675 bps (99% CI [-2.160, -1.126]) | 32/32 | 0 | 100.00% |

Risk-neutral AC equals TWAP analytically. The secondary comparator fixes κT=1 using
simulator-fitted impact and volatility; it is not selected from final results.

Primary PPO minus fixed risk-sensitive AC: **-1.658 bps (99% CI [-2.229, -0.877])**
(`IDENTIFIED`).

## Power and execution

Diagnostic power analysis locked 32 markets for a 0.5 bps effect and 80% target power.
Approximate achieved power: 0.815; resource/training-variance cap active: False.
Five optimizer seeds impose a variance floor that extra market episodes cannot remove.

| Policy | Priced / total episodes | Mean net cost (bps) | Actual fill | Completion penalty (bps) |
|---|---:|---:|---:|---:|
| Risk-neutral AC | 32/32 | 2.188 | 100.00% | 0.000 |
| Fixed risk-sensitive AC | 32/32 | 2.241 | 100.00% | 0.000 |
| PPO / 0 bps | 160/160 | 0.583 | 86.90% | 0.000 |
| PPO / 5 bps | 160/160 | 0.349 | 98.42% | 0.079 |
| PPO / 25 bps | 160/160 | 0.443 | 98.67% | 0.332 |
| PPO / 100 bps | 160/160 | 0.513 | 100.00% | 0.000 |

Priceable residuals are hypothetical valuations, not actual filled quantities. Cost means
use priced episodes only; missing outcomes and their denominators remain visible.

The 100/500 bps residual-price sensitivities in `report-data.json` are assumptions, not
observed liquidation prices or guaranteed worst cases. If every outcome is valid, they
reproduce the raw comparisons. No policy or test episode was rerun for this report.

## Training reward decomposition

Each table entry averages the five optimizer-level summaries equally. The share uses
absolute reward components before averaging, so offsetting gains/losses do not hide
the completion penalty. These are training returns, not final economic costs.

| Training penalty | Mean return (bps) | Mean penalty contribution (bps) | Mean absolute penalty share |
|---|---:|---:|---:|
| 0 bps | -0.923 | 0.000 | 0.00% |
| 5 bps | -0.942 | -0.017 | 1.11% |
| 25 bps | -0.918 | -0.019 | 1.18% |
| 100 bps | -0.976 | -0.075 | 4.60% |

## Replication scope

Intervals use this run's five optimizer seeds and common market paths. Repeating a fixed
design on the same seeds is a computational replication, not a fresh independent holdout.

This retained Windows run repeats the design already executed in Linux CI runs
35649272649 and 35649544066. Those prior computations are not pooled as extra seeds.
The original local serial attempt was preserved after one completed model and an
interrupted second fit. A new source snapshot and registration introduced isolated
process scheduling without changing the 20-model design, seeds, budget or settings.
The registered computational amendment and prior-run export provenance are included
with verified hashes in `report-data.json`.
