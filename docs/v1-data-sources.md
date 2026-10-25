# v1 data sources, fresh-data feasibility and capability labels

This report covers milestones M10–M19 and Part IV.1 of the v1 mandate.

- **Status:** written before any fresh v1 day was read.
- **Probe date:** 2026-10-25.
- **What a probe shows:** a probe is an HTTP HEAD request. A successful one proves only that a file can be reached.
  It says nothing about the file's quality. Quality is decided per day by the registered checks in
  `lob/v1/data/daycache.py`.

## What earlier versions consumed (never fresh again)

`configs/v1/consumed-inventory.json` is built read-only from the v0.9 ledger by `lob.v1.data.firewall.inventory`. It
lists:
- 169 datasets read by v0.1–v0.9;
- the 82 calendar dates on which those datasets were read.

All of them are imported into the v1 ledger as `import_consumed`: 169 datasets, one period rule and 4 excluded dates.
The **period rule** (`lob/v1/data/firewall.py`) is:

- **Up to 2025-08-31:** every date is period-consumed for every venue and instrument.
  - v0.9 observed market conditions across its pools through 2025-08-28, and probed archives to 2025-08-31.
  - Streams on the same date are dependent (audit Issue B). So a date seen through one stream is not fresh through
    another.
- **Excluded later dates:**
  - 2025-09-20, the v0.9 Bybit spot BTCUSDT format day;
  - 2026-07-01, 2026-08-01 and 2026-09-01, the v0.8 Deribit first-of-month samples.
- **Allowed use of consumed data:** development and exploration only (namespaces `dev` and `explore`).

## Fresh sources (HEAD probes, 2026-10-25)

| Source | Content | Fresh coverage | Capability label |
|---|---|---|---|
| Bybit linear `ob500` (`quote-saver.bycsi.com`) | 500-level snapshot+delta MBP | Ends 2025-08-20 (404 from 2025-08-21) | — (all of it is period-consumed) |
| Bybit linear `ob200` | 200-level snapshot+delta MBP | 2025-08-21 → 2026-10-24 (BTCUSDT probed at 2025-09-10 and 2026-06-10; ETH and SOL are verified at acquisition) | `l2`, 200 levels, aggregate queue only |
| Bybit spot `ob200` | 200-level MBP | 2025-09 → 2026-10-24 for BTC, ETH, SOL, XRP and DOGE (2026-10-25 not yet published) | `l2`, 200 levels |
| Bybit trades (`public.bybit.com`) | Every print, with aggressor side and trade id | Linear and spot through 2026-10 | trades with side |
| Binance USD-M `trades`, `aggTrades` | Prints with side and id, with published SHA-256 | Through 2026-10 | trades with side |
| Binance USD-M `bookDepth` | Cumulative depth within ±1..5 % bands, about 1 per minute | Through 2026-10 | `depth_bands`. **Not an order book**: no level, spread or queue quantity may be derived from it |
| Binance USD-M `bookTicker` (L1) | Best bid/ask updates | Ended 2024-03-30 | unavailable for fresh periods |
| Tardis free first-of-month samples | Incremental L2 for many venues | HTTP 404 for every exchange and month probed, 2025-08 → 2026-10 | unavailable |
| Bitstamp public websocket (`live_orders`, `live_trades`, `order_book`) plus REST `order_book?group=2` | **Genuine order-level (MBO/L3) data with order ids**, recorded prospectively, read-only | From 2026-10-25 15:51 UTC, in continuous hourly segments | `l3`, exact queue observability (order ids), explicit cancels, receive timestamps |

**Format checks on period-consumed development days.** Bybit linear BTCUSDT and spot ETHUSDT 2025-08-22, and
Binance USD-M BTCUSDT 2025-08-22, were used. They are development role and ledgered.
- **Bybit `ob200` files** have the same JSON schema as `ob500`. Grids: BTCUSDT linear tick 0.1 / lot 0.001; ETHUSDT
  spot tick 0.01 / lot 1e-5.
- **Full-day check:** over the BTCUSDT day, 863,420 deltas, 0 sequence gaps and 0 crossed states. The book
  reconstructed through the day matches the venue's next snapshot on **400 of 400** levels
  (`lob.v1.data.reconstruction.validate`).
- **Timestamps:** messages carry both `ts` (push time) and `cts` (matching-engine time). `ts − cts` has a median of
  3 ms (99th percentile 9 ms). v1 uses `ts`, as v0.9 did; switching to `cts` changes the at-touch print fraction from
  35.3 % to 35.5 %.
- **Book lag:** prints are reflected in the book after a median of 56 ms (99th percentile 546 ms), consistent with
  the 100 ms `ob200` push interval. In the pre-print book state, 34 % of prints are at the touch, 62 % deeper (fills
  of sweeping orders within one push interval) and 2 % beyond the visible book.
- **Trade timestamps** are parsed as exact decimals. The v0.9 float parsing put 928 of 2,020,781 prints (0.05 %) in
  the neighbouring 100 ms bin. This is a minor v0.9 defect, recorded in the audit.
- **Bitstamp:** a 30 s development capture held 3,783 messages, of which 1,813 were `order_created` and 1,830
  `order_deleted`. The reconstructed order set matched the next census on 99.98 % of 9,081 orders (2 missing,
  3 extra).
- **Bitstamp transient crossings:** marketable orders appear as `order_created` before their executions. The
  resulting transient crossed states are counted and never repaired.

## Prospective MBO capture

`lob/v1/data/connectors/bitstamp_capture.py` records `btcusd` and `ethusd`:
- one write-once directory per hour, each with SHA-256 manifests;
- explicit `GAP` segments on disconnects;
- a census every 10 min;
- no outbound message other than subscriptions and public GETs.

A 23-minute DNS outage on 2026-10-25 (16:37–17:00 UTC) was recorded as GAP segments, and capture resumed by itself.

The data rate is about 17 MB/h compressed for BTCUSD and about 10 MB/h for ETHUSD. Capture days become MBO pool days
under the registered rule in `configs/v1/dataset-registry.json` (`mbo_pools`):
- only complete UTC days count: at least 95 % of seconds in COMPLETE segments and at least two censuses;
- counting starts at 2026-10-26;
- days 1–5 are explore days;
- days 6 onward are confirm days, at most 16.

## Registered pools (dates are the independent clusters)

All streams of a pool use the same registered date order: SHA-256-tagged `SeedSequence` permutation, primaries then
backups. A date belongs to exactly one pool.

| Pool | Role | Dates (range) | Primaries + backups | Streams |
|---|---|---|---|---|
| explore | explore | 2025-09-01 → 2025-11-28 | 16 + 6 | Bybit linear BTC/ETH/SOL, spot BTC/ETH |
| calibrate | calibrate | 2025-12-02 → 2025-12-25 | 8 + 4 | same |
| confirm | confirm | 2026-01-02 → 2026-05-31 | 40 + 12 | same |
| replicate | replicate | 2026-06-15 → 2026-08-31 | 24 + 8 | same |
| vault | vault | 2026-09-04 → 2026-10-23 | 20 + 8 | same |

**Effective sample sizes.** The unit of every test is the calendar date.
- A test needs at least **8 date clusters**, the minimum at which an exact sign-flip test can reach the strictest
  family-wise per-test alpha of 0.00625.
- Direct replications use 12 dates.
- Every other promoted test uses the number of dates its cluster-level power analysis requires, up to 40.
- With 5 L2 streams per date, a stream-date costs about 5.3 min to reconstruct (measured on the development day).
  12 dates × 2 streams take about 2 h.

## Licences and access policy

`lob.v1.data.sources.catalogue()` lists every stream with its licence and access method:
- the provider, the terms URL and the date the terms were read (2026-10-25);
- permitted use: non-commercial research;
- no redistribution: raw data and caches stay under the git-ignored `data/v1`.

No paid source was used. Genuine MBO data from licensed vendors (LOBSTER, Databento, Tardis paid) would need
separate authorisation and is not part of v1.
