# Real-data reconstruction assessment

**Both days passed the mechanics checks.** This is a validation of the public-data
pipeline and aggregate L2 reconstruction, not a strategy backtest.

Source: Tardis's public Deribit **ETH-PERPETUAL** samples for **2020-04-01** and
**2020-05-01**, downloaded on 2026-09-13. The provider documents
[sample access](https://docs.tardis.dev/historical-data-details/deribit),
[CSV fields](https://docs.tardis.dev/downloadable-csv-files/data-types) and
[reconstruction semantics](https://docs.tardis.dev/faq/order-books).

| Check | 2020-04-01 | 2020-05-01 | Total |
|---|---:|---:|---:|
| Incremental price-level rows | 2,321,160 | 3,651,511 | 5,972,671 |
| Completed capture groups | 1,531,713 | 2,265,297 | 3,797,010 |
| Published top-five snapshots checked | 988,235 | 1,093,244 | 2,081,479 |
| Exact snapshot matches | 988,235 | 1,093,244 | 2,081,479 |
| Price/size mismatches | 0 | 0 | 0 |
| Exchange timestamp mismatches | 0 | 0 | 0 |
| Missing or unmatched reference groups | 0 | 0 | 0 |
| Crossed/empty/one-sided completed books | 0 | 0 | 0 |
| Trade records checked | 17,288 | 28,907 | 46,195 |
| Duplicate trade IDs | 0 | 0 | 0 |
| Snapshot initialization/reset blocks | 3 | 2 | 5 |
| Maximum capture gap, seconds | 7.059940 | 5.217385 | — |

Each file spans approximately 24 hours. Six compressed files total **84,784,910
bytes**; full gzip verification read **835,298,573 expanded bytes**. Provider MD5,
gzip integrity, source SHA-256, assessment artifacts and implementation hashes
were verified. Input and implementation changes during assessment are rejected.
Raw files remain in ignored `data/public/`; [source provenance](sources.json)
contains URLs and checksums for obtaining and verifying them independently.

## Descriptive market observations

Each day contributes 86,399 samples on a one-second UTC grid. The most recent
completed capture group is used, without future interpolation or extrapolation
past EOF. These are descriptive observations, with no significance tests or fit.

| Metric | 2020-04-01 | 2020-05-01 |
|---|---:|---:|
| Mean spread, bps | 4.1724 | 2.9774 |
| Median spread, bps | 3.7900 | 2.3756 |
| 95th percentile spread, bps | 7.5786 | 4.8031 |
| Mean top-five bid depth, native units | 239,901.70 | 90,983.75 |
| Mean top-five ask depth, native units | 241,854.57 | 109,161.17 |
| Grid samples with book age over 5 seconds | 2 | 0 |

The two stale April samples and observed capture gaps are disclosed. A quiet
period and lost updates cannot be distinguished from timestamps alone; normalized
CSV does not contain the original exchange sequence IDs. The provider's reference
snapshots share the underlying feed, so exact agreement is not independent
confirmation of exchange completeness.

## Reports and reproduction

- [April assessment](20260913T202441-ae4e5d848775/report.md), [full JSON](20260913T202441-ae4e5d848775/assessment.json)
- [May assessment](20260913T202528-c5d5d426358c/report.md), [full JSON](20260913T202528-c5d5d426358c/assessment.json)
- [Download and assessment commands](../../../docs/public-market-data.md)

The full project suite passed **385 tests**. Ruff, Python compilation and frontend
JavaScript syntax checks passed. Two existing Gymnasium unbounded-Box warnings
remain. No matching-speed improvement was attempted or claimed in this batch.

## What this establishes

Millions of real updates exercise exact decimal ingestion, capture ordering,
snapshot resets, aggregate reconstruction, trade parsing, causal sampling and
provenance. All published reference rows matched, with complete input consumption.

This data has no individual order queues, hypothetical agent fills or simulated
market impact. Contract units have not been converted into equity shares or ETH.
No execution strategy was run against these recordings, no model was calibrated,
and no profitability or out-of-sample performance claim follows. Two historical
days from one instrument/provider are sufficient for this bounded mechanics test;
broader dates, regimes and venues remain necessary for quantitative research.
