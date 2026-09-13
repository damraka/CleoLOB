# Historical L2 mechanics assessment

Status: **PASS**
Instrument: deribit / ETH-PERPETUAL
Observed span: 23.9999 hours

Exact top-five matches: 1,093,244 / 1,093,244 reference rows.
Different books: 0; missing reference groups: 0; unmatched reference rows: 0.

All distributions use a one-second UTC grid and the latest completed captured book.

| Metric | Mean | Median | 95th percentile |
|---|---:|---:|---:|
| mid_price | 211.504 | 211.525 | 214.775 |
| spread_bps | 2.97742 | 2.37558 | 4.80307 |
| bid_depth_5_native | 90983.8 | 89407 | 167667 |
| ask_depth_5_native | 109161 | 104878 | 203690 |
| imbalance_5 | -0.0823024 | -0.0888921 | 0.454841 |

Trade records checked: 28,907.

## Scope and limitations

- L2 amounts are native feed units; they are not automatically ETH units or stock shares.
- L2 does not expose order IDs, FIFO queue positions, hidden liquidity or hypothetical strategy fills.
- Both inputs come from one provider/feed; matching them is a reconstruction check, not independent market truth.
- The normalized CSV lacks original exchange sequence IDs; capture gaps cannot be proven absent.
- Provider snapshots may remove crossed levels; this replay never silently cleans crossed books.
- No model was fitted and no strategy profitability, calibration or out-of-sample claim is made.

Full counters, exact input hashes, timestamps and mismatch locations are in `assessment.json`.
Source implementation hashes and runtime are in `provenance.json`.
