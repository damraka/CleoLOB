# Deribit ETH-PERPETUAL Historical L2 Validation

This study evaluates CleoLOB's aggregate Level-2 reconstruction pipeline
against published Deribit ETH-PERPETUAL top-five snapshots.

## Dataset

- Exchange: Deribit
- Instrument: ETH-PERPETUAL
- Period: April 1, 2020
- Incremental L2 feed
- Published top-five snapshots
- Public trade records

Raw market data is not committed to this repository.

## Reconstruction Results

| Metric | Result |
|---|---:|
| Observed span | 23.9999 hours |
| Incremental L2 rows | 2,321,160 |
| Reconstructed states | 1,531,713 |
| Reference snapshots | 988,235 |
| Exact top-five matches | 988,235 |
| Match rate | 100.000% |
| Different books | 0 |
| Timestamp mismatches | 0 |
| Crossed groups | 0 |
| Missing level deletes | 0 |
| Trade records checked | 17,288 |

## End-to-End Performance

Measured locally for the complete `assess-l2` workflow:

| Metric | Result |
|---|---:|
| Runtime | 111.06 s |
| Effective L2 throughput | ~20,900 rows/s |
| Effective state throughput | ~13,792 states/s |

These values are not isolated matching-engine microbenchmarks.

## Reproduction

Download the required public samples:

```bash
cleo download-sample \
  --exchange deribit \
  --symbol ETH-PERPETUAL \
  --date 2020-04-01 \
  --type incremental_book_L2 \
  --out data/public

cleo download-sample \
  --exchange deribit \
  --symbol ETH-PERPETUAL \
  --date 2020-04-01 \
  --type book_snapshot_5 \
  --out data/public

cleo download-sample \
  --exchange deribit \
  --symbol ETH-PERPETUAL \
  --date 2020-04-01 \
  --type trades \
  --out data/public