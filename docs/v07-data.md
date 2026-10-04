# v0.7 data layer

Code: `lob/v07/data/` (schema, quality, tape, access, capability) and `lob/v07/adapters/`
(adapter kit, Tardis, Bitstamp).

## Canonical schema (workstream 1)

`lob.v07.data.schema.Record` is the single record type every adapter emits.

**Record kinds.**
- `book_snapshot`, `book_delta`, `trade`, `gap`
- order-level only: `order_add`, `order_cancel`, `order_modify`, `order_execute`

**Values.** Prices and amounts are exact decimals in native units. Timestamps are integer
microseconds, recorded separately for the exchange and the local capture.

**Validation rules.**
- Aggregate-L2 records cannot carry an order id.
- Order events require one.
- Prices must be positive.
- Book amounts must be nonnegative and trade amounts positive.

Each record belongs to a registered `VenueSpec`: tick, amount unit, contract type,
underlying and capability level. The spec hash is stored with every tape's metadata.

| Venue / instrument | Tick | Amount unit | Capability |
|---|---|---|---|
| Deribit ETH-PERPETUAL | 0.05 | USD face value | aggregate L2 + trade prints |
| Deribit BTC-PERPETUAL | 0.5 | USD face value | aggregate L2 + trade prints |
| BitMEX XBTUSD | 0.5 | USD contracts | aggregate L2 + trade prints |
| Bitstamp btcusd (v0.5 captures) | 0.01 | BTC | order identity; FIFO **not** established by the feed |

## Adapters and adapter kit (workstreams 2, 85)

**Adapter kit.** `Adapter` is the base class. `conformance(adapter)` checks the
obligations every new adapter must meet:
- schema validity
- nondecreasing capture time
- no fabricated identities
- deterministic output (same digest twice)
- an honest capability declaration

**Adapters.**
- `TardisAdapter` reads Tardis normalized `incremental_book_L2` and `trades` CSV for Deribit
  and BitMEX. It parses exact decimals and refuses rows of another venue or instrument.
  BitMEX `orderBookL2` level ids are not exposed and are never treated as order identities.
- `BitstampAdapter` maps the frozen v0.5 capture adapter (`bitstamp-capture-4`) onto the
  schema:
  - The initial REST census becomes `order_add` records.
  - Later censuses are validation references placed at exchange time, so they are skipped
    (otherwise they would duplicate orders).
  - Both consumed captures pass conformance. The first conformance run found exactly this
    duplication problem, and it was fixed before any use.

## Quality validation (workstream 1)

`lob.v07.data.quality.assess` runs while a tape is built. Failures make dependent results
`INVALID` and are never repaired.

**Registered thresholds.**
- FAIL if:
  - valid fraction < 0.90
  - crossed-sample fraction > 1%
  - duration < 6 h
  - the replay is incomplete
- WARN if:
  - valid fraction < 0.98
  - capture gap > 300 s
  - pre-snapshot rows > 1%
  - unknown aggressor side > 1%
  - duplicate trade ids
  - exchange clock regressions
  - spreads off the declared tick grid

## Tapes

`lob.v07.data.tape.build_tape` uses the v0.6 measurement grid unchanged. It differs from
the frozen `lob.v06.tape.tape_from_tardis` in two ways only:
- explicit reader limits (2 GiB compressed, 32 GiB expanded, 1.5 × 10⁹ rows), because BitMEX
  files are far larger than Deribit's
- the quality report

**Differential check.**
- A test checks both builders on synthetic files.
- On the full development day (ledger entries 41–43), all nine tape arrays are bit-identical
  to the v0.6 tape and quality is PASS: valid fraction 0.99997, no crossed samples, 86,399.6 s.

## Ledgered access (stages)

`lob.v07.data.access.acquire` / `load_tape` record these stages:

| Stage | When it is ledgered | What it records |
|---|---|---|
| `downloaded` | before any network request for missing files; this consumes a holdout | — |
| `opened` | after the files are read | source SHA-256 (it may never change) |
| `parsed` | after the tape is built | quality status |

The study then records `evaluated`, and `inspect` when an outcome is first looked at.

**Download failure.** A failed download is retried once. A second failure is an `attempt`
with outcome `NOT_AVAILABLE`, and no other date, instrument or venue is substituted.

**Size limits.** These come from `configs/v07/dataset-registry.json`. A file above the limit
is `NOT_AVAILABLE`, never truncated.

## Capability-aware validation (workstreams 81, 82)

`lob.v07.data.capability.check(protocol, dataset, analysis)` returns `AVAILABLE` or
`NOT_AVAILABLE` with the missing capabilities.

| Analysis | Deribit / BitMEX (L2) | Bitstamp (order-level) |
|---|---|---|
| realism, domain gap, bounded replay | available | available |
| order survival | NOT_AVAILABLE | available |
| queue position, exact passive fill | NOT_AVAILABLE | NOT_AVAILABLE (FIFO not established) |
| hidden liquidity | NOT_AVAILABLE | NOT_AVAILABLE |
