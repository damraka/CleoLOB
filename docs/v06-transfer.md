# v0.6 historical execution transfer v2 (M14/M24; H11, H12)

## Design

The design was sealed in `configs/v06/transfer-design.json` (ledger `m14-transfer-design`).
That was after the execution-evaluation lock and before the transfer holdout was accessed.

**Replay.** Episodes use `lob.historical_sim.HistoricalSimulator`, unchanged:
- displayed historical liquidity is identity-free and is reset to history at every update;
- the hypothetical parent has no market impact;
- passive fills come only from bounded trackers, under **conservative** and **optimistic**
  modes, which are separate self-consistent paths;
- no exact FIFO is assumed.

**Mapping.** The clock ratio is 1, because v0.6 worlds are calibrated in real seconds.
Lots = native ÷ 243.6875, the development scale.

**Episodes.** There are 144 episodes per dataset, starting at first capture + 300 s + 600 s·k.
Each window covers warmup + horizon + settlement + 1 s, with 20 book levels.

**Agents.** TWAP, VWAP, POV, AC (with the selected world's AC parameters) and the 16
registered PPO/DQN models. Policies and mandate are frozen exactly as in simulation.

**Datasets.**
- registered: fresh ETH 2020-10-01;
- retrospective: ETH 2020-08-01, consumed by v0.5 M8.

## Results: ETH 2020-10-01 (registered; 5,760 rows, 0 INVALID)

| Agent | Conservative cost (bps) | Optimistic cost (bps) | Within-horizon completion |
|---|---|---|---|
| TWAP | 1.80 | 1.80 | 100% |
| VWAP | 1.77 | 1.77 | 100% |
| POV | 2.40 | 2.40 | 100% |
| AC | 1.78 | 1.78 | 100% |
| PPO single | 1.68 | 1.63 | 100% |
| DQN single | 1.59 | 1.50 | 100% |
| PPO ensemble | 1.76 | 1.67 | 100% |
| DQN ensemble | 1.60 | 1.48 | 100% |

- The classical schedules never rest passive orders, so fill semantics do not affect them.
- **H12 (fill semantics): ESTABLISHED (registered status), vacuous under the registered
  rule.** All 15 pairwise conclusions are *indeterminate* under **both** modes (Bonferroni,
  0.05/30). No conclusion depends on the fill assumption, because no historical conclusion is
  determinate at all. It supports no substantive stability or equivalence conclusion.
- **H11 (ensemble vs single-world training): NOT_ESTABLISHED** for both algorithms:
  - PPO: gap(ensemble) − gap(single) = −0.58 [−1.78, 1.16] conservative and −0.71
    [−1.85, 1.11] optimistic;
  - DQN: +0.41 [−1.36, 2.11] conservative and +0.38 [−1.37, 1.99] optimistic.

  The point estimates point in opposite directions for the two algorithms, and every interval
  spans zero.
- **Prediction sources A–D (descriptive, family 60).** Only the single selected world (A)
  makes a determinate prediction: POV costlier than single-world DQN. The pooled sources —
  ensemble (B), regime models (C) and ensemble plus interventions (D) — make no determinate
  prediction. That is the expected effect of adding model uncertainty. No source's prediction
  could be scored as agreeing or reversing, because history is indeterminate for every pair.

## Retrospective ETH 2020-08-01 (consumed; descriptive)

Costs ranged from 3.9 to 4.5 bps, and all agents completed within the horizon. The H12-type
rule is vacuous in the same way: all pairs are indeterminate in both modes. The H11-type gaps are
indeterminate: PPO −0.74 [−2.18, 1.25] and DQN +0.27 [−1.63, 2.10] (conservative). Source
predictions are all indeterminate.

## Interpretation

Under a 14-lot, 120 s mandate on Deribit ETH in 2020, bounded historical execution showed no
determinate pairwise difference between the classical and learned agents at the registered α
(0.05/30).
- **This is not equivalence:** no equivalence margin was preregistered for this analysis.
- **Interval widths vary:** some pairs are tightly bounded (TWAP vs VWAP within ±0.2 bps),
  others wide (TWAP vs POV about ±1.1 bps).

Simulator-trained conclusions therefore have nothing determinate to transfer to. No evidence
was established that domain-randomized training over the 2-member ensemble made transfer more
consistent (H11 NOT_ESTABLISHED).

None of this is an exact historical fill, a profitability result or a live-trading result.
