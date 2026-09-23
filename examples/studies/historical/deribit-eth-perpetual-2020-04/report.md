# Historical L2 mechanics assessment

Status: **PASS**
Instrument: deribit / ETH-PERPETUAL
Observed span: 23.9999 hours

Exact top-five matches: 988,235 / 988,235 reference rows.
Different books: 0; missing reference groups: 0; unmatched reference rows: 0.

All distributions use a one-second UTC grid and the latest completed captured book.

| Metric | Mean | Median | 95th percentile |
|---|---:|---:|---:|
| mid_price | 131.814 | 131.975 | 133.925 |
| spread_bps | 4.17242 | 3.79003 | 7.57863 |
| bid_depth_5_native | 239902 | 242820 | 354902 |
| ask_depth_5_native | 241855 | 244314 | 342629 |
| imbalance_5 | -0.00944028 | -0.000919076 | 0.386657 |

Trade records checked: 17,288.

## Scope and limitations

- L2 amounts are native feed units; they are not automatically ETH units or stock shares.
- L2 does not expose order IDs, FIFO queue positions, hidden liquidity or hypothetical strategy fills.
- Both inputs come from one provider/feed; matching them is a reconstruction check, not independent market truth.
- The normalized CSV lacks original exchange sequence IDs; capture gaps cannot be proven absent.
- Provider snapshots may remove crossed levels; this replay never silently cleans crossed books.
- No model was fitted and no strategy profitability, calibration or out-of-sample claim is made.
- Some grid samples carry a book older than five seconds; inspect feed gaps.

Full counters, exact input hashes, timestamps and mismatch locations are in `assessment.json`.
Source implementation hashes and runtime are in `provenance.json`.
