# v0.8 transfer and simulator decision utility

## Retrospective transfer (EXPLORATORY)

`cleo transfer-study-v08 retrospective` replays the historical-replay policies under bounds on four consumed
days:
- the selection day: ETH 2020-12-01;
- the next temporal day: ETH 2021-01-01;
- a cross-instrument day: BTC 2020-11-01;
- a cross-venue day: BitMEX XBTUSD 2020-11-01.

For each day it reports pair verdicts and neutral-assumption costs. Across days it reports dominance stability
per axis and a Kendall-tau matrix.

These days were consumed by earlier versions, so this phase cannot establish transfer.

## Fresh transfer (H13, H15–H17) and the final holdout (H20)

Both the `fresh` phase and the final analysis follow the registered acquisition rule:
1. A sealed reader design comes first.
2. Next comes a headers-only probe of an already-consumed control file.

If the probe fails:
- NOT_AVAILABLE attempts are ledgered;
- no holdout byte is requested;
- the holdouts stay fresh for a future version.

## Simulator decision utility (H13)

`cleo ranking-transfer-v08` compares two rankings:
- the plausible-world mean ranking (confirm seeds);
- bounded historical rankings.

The comparison uses:
- Kendall and Spearman correlation;
- pairwise accuracy on determinate pairs, with an exact binomial test;
- top-k retention;
- regret transfer.

## Cross-venue, basket and routing infrastructure (M31)

`cleo transfer-study-v08 portfolio` covers three things:
- **Synthetic multi-venue routing.** The routers are single-venue, equal split, liquidity-weighted, and SOR.
  SOR scores each venue by marginal observed cost plus fee plus a latency penalty.
- **A basket.** It includes a cross-impact sensitivity sweep.
- **Hypothetical market-order routing** on the time-aligned consumed Deribit BTC and BitMEX XBTUSD days. The
  fees used here are illustrative assumptions.

This is research infrastructure, not a production router.
