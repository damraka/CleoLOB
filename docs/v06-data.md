# v0.6 data, governance and chronology audit

## Sources

All v0.6 data are Deribit inverse perpetuals from Tardis.dev public first-of-month samples:
- `incremental_book_L2`: aggregate price-level updates;
- `trades`: aggressor-signed prints.

The capability is `aggregate_l2`, with trade prints. The data **cannot** express:
- order identity, exact FIFO position or quantity ahead;
- hidden liquidity;
- observed fills of a hypothetical order;
- modify events.

Every v0.6 observable that would need these is reported `NOT_AVAILABLE`. Static facts are in
[`configs/v06/datasets.json`](../configs/v06/datasets.json):
- contract;
- tick and lot size;
- timezone;
- sequence semantics;
- redistribution terms;
- acquisition notes.

## Chronology audit (2026-10-01, base commit `50ed08b`)

**Sources inspected:**
- `configs/v04-consumed-data.json`;
- the v0.5 protocol and its 80-entry consumption ledger;
- every repository file and git history (`git log --all -S`);
- the local `data/` directories.

| Period | Prior use | v0.6 classification | v0.6 role |
|---|---|---|---|
| ETH 2020-04-01 | v0.3/v0.4 calibration; v0.5 development | DEVELOPMENT (consumed) | `development` |
| ETH 2020-05-01 | v0.4 validation; v0.5 selection | DEVELOPMENT (consumed) | `selection` |
| ETH 2020-06-01 | v0.4 holdout; v0.5 internal holdout | CONSUMED | `retrospective` |
| ETH 2020-07-01 | v0.5 external holdout (M2–M8) | CONSUMED | `retrospective` |
| BTC 2020-07-01 | v0.5 cross-instrument holdout | CONSUMED | `retrospective` |
| ETH 2020-08-01 | v0.5 M8 transfer holdout | CONSUMED | `retrospective` |
| ETH 2020-09-01 | none found | FRESH_CANDIDATE | `fresh_external` |
| BTC 2020-09-01 | none found | FRESH_CANDIDATE | `cross_instrument_external` |
| ETH 2020-10-01 | none found | FRESH_CANDIDATE | `transfer_holdout` |
| ETH/BTC 2026-07/08/09 | v0.3/v0.4 | CONSUMED | not used |
| Bitstamp live captures | v0.5 M1/M2 (M1 INVALID) | CONSUMED | not used |
| Vendor historical MBO | — | NOT_AVAILABLE | — |

Fresh availability is **unverified** until the first ledger-recorded access. Tardis rejects
HEAD requests, so availability could not be checked without starting a download. A
download that fails after one retry makes the dependent hypotheses `NOT_AVAILABLE`. No
other date or instrument is substituted.

No consumed v0.5 holdout is used as fresh v0.6 evidence. The retrospective role permits
only the `retrospective` use, and its results are labelled retrospective.

## Governance

- **Raw files** are never committed. They live in ignored directories: `data/public`,
  `data/v05/tardis`, and `data/v06/tardis` for new files. Tape caches are in `data/v06/cache`.
- **Detailed derived data** stays in ignored `results/v06/` run directories. This covers
  windows, episodes and per-block statistics.
- **The public bundle** `examples/studies/v06/evidence` carries the protocol, ledger,
  configurations, summaries, hashes and registration artifacts. Each excluded file is listed
  with its SHA-256.
- **No credentials** are used. The Tardis public endpoint needs none.
- **Synthetic MBO** is never presented as genuine. v0.6 core uses no MBO.
