# Public market-data validation

The platform now has a separate **aggregate L2** path for Tardis CSV files. It does
not feed aggregate levels into the exact-order-ID replay engine or invent FIFO
orders. Historical reconstruction and a simulated execution backtest answer
different questions.

## Download and provenance

`cleo download-sample` fetches first-of-month public files without credentials.
The default cap is 64 MiB per compressed file, with a separate 2 GiB expanded cap.
Transport length, provider MD5 (when supplied), complete gzip CRC, and SHA-256 are
checked. The adjacent `.provenance.json` records the URL, download time, sizes,
checksums and response metadata. Existing files are reused only after verification;
different or incomplete files are rejected rather than overwritten.

The raw data lives in ignored `data/public/`. Downloading public samples does not
make them project-owned or freely redistributable. This project saves aggregate
validation results and source links; users obtain the raw files from the provider
under its [terms](https://docs.tardis.dev/legal/terms-of-service).

The [provider's Deribit page](https://docs.tardis.dev/historical-data-details/deribit)
documents first-of-month access and the captured feed. Use
[the CSV schema](https://docs.tardis.dev/downloadable-csv-files/data-types) and
[reconstruction rules](https://docs.tardis.dev/faq/order-books) when adding formats.

```powershell
foreach ($sampleDate in @('2020-04-01', '2020-05-01')) {
    foreach ($sampleKind in @('incremental_book_L2', 'book_snapshot_5', 'trades')) {
        cleo download-sample --exchange deribit --symbol ETH-PERPETUAL --date $sampleDate --type $sampleKind --out data/public
    }
    cleo assess-l2 --updates "data/public/deribit_incremental_book_L2_${sampleDate}_ETH-PERPETUAL.csv.gz" --snapshots "data/public/deribit_book_snapshot_5_${sampleDate}_ETH-PERPETUAL.csv.gz" --trades "data/public/deribit_trades_${sampleDate}_ETH-PERPETUAL.csv.gz" --out results/historical
}
```

## Reconstruction and checks

`L2Replay` reads one instrument in capture order using exact Decimal prices and
amounts. It applies each local-timestamp group before exposing its book. Quantities
replace levels; zero removes a level. A new snapshot block clears old state, and
rows preceding the first snapshot are counted and skipped. Exchange-clock
regressions are counted; capture-clock regressions are rejected. Crossed,
one-sided and empty books remain visible in quality counters.

Limits cover file bytes, expanded bytes, rows, line length, group size and total
observed levels. Stopping iteration early leaves `complete=false`; unread gzip
trailers cannot be treated as verified. No sorting or crossing repair is applied.

`assess-l2` compares exact top-five prices and sizes at corresponding capture
timestamps with a separately published snapshot file. It checks both missing
reference rows and changed replay states without references, plus exchange
timestamp discrepancies. It validates all trade records, side enums, identity,
date and duplicate IDs when a trade file is supplied. This is a same-provider
consistency check. Both files ultimately rely on the same captured exchange feed.

For descriptive statistics, books are sampled on a one-second UTC grid using the
latest completed capture group. There is no interpolation from future states or
extension beyond the final record. The report discloses stale and invalid samples.
Depth and trade amount remain in the feed's native contract units; they must not
be interpreted as ETH quantities or equity shares without an instrument adapter.

Outputs are appended to unique directories with `assessment.json`, `report.md`,
`provenance.json` and an artifact checksum manifest. Input and implementation
hashes are checked for changes during assessment. `PASS` means the described
mechanics checks passed, not that a calibrated model or trading policy is valid.
Canonical experiment `cleo verify/reproduce` commands are specific to synthetic
experiment directories; re-run `assess-l2` to independently repeat this assessment.

## Remaining evidence

Two complete sample days can exercise millions of updates and many book states.
They do not cover independent venues, current regimes, exchange sequence gaps,
individual queues, strategy impact, fills, PnL, margin or funding. The normalized
CSV omits exchange sequence IDs, and the provider can clean crossings in its
published snapshots. Discrepancies are reported rather than hidden.

The later [calibration and validation workflow](validation-and-risk.md) now fits
frozen observable parameters and evaluates purged chronological holdouts, alongside
registered synthetic execution stresses. Instrument-aware historical execution,
several independent venues/regimes and counterfactual fill/impact modeling remain.
The reconstruction assessment itself fits no parameters and makes no historical
strategy-performance claim.
