# v0.5 data, registry and access

All v0.5 datasets are declared in the frozen [protocol](v05-research-protocol.md).
Their static facts are in [`configs/v05/datasets.json`](../configs/v05/datasets.json).
Freshness, consumption, access time and source hashes come only from the append-only
[consumption ledger](../configs/v05/consumption-ledger.jsonl).

Print the current registry:

```sh
python -c "from lob.dataset_registry import build; import json; print(json.dumps(build(), indent=2))"
```

## Registered datasets

| Dataset | Provider / capability | Role | Status |
|---|---|---|---|
| Deribit ETH-PERPETUAL 2020-04-01 | Tardis L2 + trades (+top-5) | development | consumed before v0.5 |
| Deribit ETH-PERPETUAL 2020-05-01 | Tardis L2 + trades | selection | consumed before v0.5 |
| Deribit ETH-PERPETUAL 2020-06-01 | Tardis L2 + trades | internal holdout | consumed by v0.4 (retrospective) |
| Deribit ETH-PERPETUAL 2020-07-01 | Tardis L2 + trades | external holdout | fresh at freeze |
| Deribit BTC-PERPETUAL 2020-07-01 | Tardis L2 + trades | cross-instrument holdout | fresh at freeze |
| Deribit ETH-PERPETUAL 2020-08-01 | Tardis L2 + trades | M8 transfer holdout | fresh at freeze |
| Bitstamp btcusd, 900 s | order-level live capture | MBO adapter development | recorded after freeze |
| Bitstamp btcusd, 1,800 s | order-level live capture | MBO validation | recorded after adapter freeze |

Each registry entry carries every roadmap field:
- source, venue, instrument and contract;
- date range, access time and capability level;
- tick size, lot size and timezone;
- sequence semantics;
- source and normalized hashes;
- license, freshness and consumption status.

Normalized hashes are computed from the tape arrays (`lob.dataset_registry.tape_sha256`)
or from the normalized order-level event stream.

## Access rules in practice

- `lob.v05_data.acquire` writes a ledger `access` entry **before** downloading a fresh
  dataset, then a second entry with exact SHA-256 hashes after integrity checks. It
  also refuses to access a holdout unless a sealed design lists it.
- Later accesses re-record the hashes. A different hash for the same file is rejected.
  Changed bytes would invalidate every earlier binding.
- `declare_fresh` is refused for consumed datasets and for venue/instrument/date aliases.
  Deleting a ledger line breaks the hash chain and the anchors held by evidence.
- Raw provider files, derived tapes (`data/v05/cache`) and captures (`data/v05/bitstamp`)
  sit under ignored directories and are never committed.

## Deribit aggregate L2 (Tardis)

Tardis first-of-month samples are downloaded without credentials, following Tardis's
[Deribit details](https://docs.tardis.dev/historical-data-details/deribit). The
[terms](https://docs.tardis.dev/legal/terms-of-service), checked 2026-09-26, restrict
redistribution of raw and detailed derived data. For that reason:
- public evidence holds only summaries and hashes;
- per-order bounds, tapes and fitted details stay local.

The CSVs carry no exchange sequence numbers, so losses upstream of Tardis cannot be
detected beyond timestamp gaps.

## Genuine order-level data (Bitstamp)

Bitstamp's public websocket v2 channels `live_orders_<pair>`, `live_trades_<pair>` and
`order_book_<pair>`, plus REST `order_book/<pair>/?group=2`, together provide:
- individual resting orders with IDs (a complete census);
- order creation, change and deletion events;
- trades carrying both order IDs;
- an independently published top-100 aggregate book.

`tools/capture_bitstamp.py` records every message's exact text with local receive
clocks. It takes REST censuses at the start, every 300 s and at the end. A
disconnect or venue reconnect request ends the capture as a GAP, which is never
bridged.

Semantics were established from the development capture and are declared in the
hashed `SourceContract` (`lob.mbo_sources.bitstamp_contract`). The findings:
- **Sequence chain:** the order channel's `event_id` and `pre_event_id` form a continuous chain.
- **Changes are executions:** `order_changed` always reduces the remaining amount through an execution.
- **`amount_traded` is unreliable:** it is sometimes cumulative and sometimes per event.
- **Price field on changes:** it reports execution prices while an aggressor sweeps.
- **Deletions:** zero remaining means the remainder executed; a positive remaining amount means it was cancelled.
- **Crossing orders:** orders created through the opposite best are incoming aggressors, and most are cancelled in the same millisecond.
- **Market orders:** zero-price orders are market orders and never rest.

Price-time priority is **not** established by the feed. The census listing order
agreed with tracked priority in every development check, which is evidence
consistent with FIFO, not proof of it.

These data are genuine venue messages recorded live and replayed offline. They are not a
vendor-certified historical MBO archive. The captures cover one instrument, one venue and
short windows. Commercial use needs a Bitstamp data licence, and the raw captures are not
redistributed.

## Adding another real MBO source

The canonical target is `lob.order_lifecycle.LifecycleEvent`, in exact integer ticks and
lots:

| Field | Meaning |
|---|---|
| `sequence` | contiguous normalized order assigned by the adapter |
| `source_sequence` | venue sequence number when available (gaps become anomalies) |
| `exchange_ts_ns`, `receive_ts_ns` | exchange and capture clocks, kept separate |
| `kind` | ADD, MODIFY, CANCEL, EXECUTE, TRADE, SNAPSHOT (`reset` or `check`), RESET, GAP |
| `order_id`, `side`, `price`, `quantity` | source identity and state; never generated |
| `maker_order_id`, `taker_order_id`, `aggressor_side` | trade linkage when published |

A new adapter needs:
1. A hashed `SourceContract` declaring identity, sequence, timestamp, snapshot, priority,
   execution and hidden-liquidity semantics.
2. Tests on synthetic fixtures for each rule.
3. A development sample for deriving rules, kept separate from a validation sample.
4. Registration of the validation sample as fresh in the protocol and ledger before
   access.

Candidates:
- **LOBSTER** (NASDAQ): the adapter `lob.mbo_sources.LobsterAdapter` is complete. Sample
  access now requires registration and terms acceptance, so it was not used.
- **Databento MBO:** needs credentials. Its fill notifications do not change the book, so
  they must not be mapped to EXECUTE.
- **Coinbase or Kraken L3:** these channels require authentication.
