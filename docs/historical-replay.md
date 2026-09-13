# Canonical historical reconstruction

`lob.replay` validates a complete bounded event stream before replay starts. It
reconstructs recorded order IDs independently of the synthetic matching engine.
It does not estimate fills for hypothetical strategy orders. The included
`examples/data/canonical-events.jsonl` is a deliberately fabricated mechanics
fixture, not a market dataset or evidence of trading performance.

## Python API

```python
from lob.replay import HistoricalReplay, load_events, replay_file, validate_file

path = "examples/data/canonical-events.jsonl"
quality = validate_file(path)
assert quality.valid, quality.to_dict()

replay = HistoricalReplay(load_events(path))
replay.step()                      # one recorded event
replay.advance_to(3_000_000)        # all events through this feed timestamp
replay.pause()
replay.step()                      # manual stepping is allowed while paused
replay.resume()
replay.run()                       # finish; no wall-clock sleeping
state = replay.book.snapshot()     # JSON-safe depth and exact resting orders
report = replay.summary()
replay.reset()                     # rewind to an empty book

report = replay_file(path)         # load, validate, run, summarize
```

`validate_file()` returns a quality report even when data are invalid.
`load_events()`, `replay_file()` and `HistoricalReplay()` raise
`DataValidationError` on critical failures; the exception's `.report.to_dict()`
is suitable for JSON output. Invalid input is neither sorted nor repaired.
`validate_events()` accepts canonical `MarketEvent` objects. `HistoricalBook`
also checks each directly applied event and leaves state unchanged on rejection.

## File schema

CSV, JSONL and NDJSON use the same fields. Prices and quantities are integers;
no implicit currency-to-tick or timestamp conversion occurs. JSON integer
numbers and decimal digit strings are accepted, allowing exact CSV round trips.
Booleans, floating point numbers, exponential notation, negatives and values
outside signed 64-bit range are rejected. Blank CSV optional cells and JSON
null optional fields mean absent.

| Field | Meaning |
| --- | --- |
| `timestamp_ns` | Required nonnegative integer nanoseconds in one consistent time basis |
| `sequence` | Required nonnegative integer; each subsequent event must increment by one |
| `event_type` | Required uppercase canonical type below |
| `symbol`, `venue` | Required nonempty string identifiers; one symbol/venue per stream |
| `order_id` | Required string for order lifecycle events; IDs cannot be reused by ADD |
| `side` | `BUY` or `SELL`; maker/resting side for EXECUTE assertions |
| `price_ticks` | Positive integer ticks; tick size belongs in instrument metadata |
| `quantity` | Positive integer quantity in one consistent lot/share/contract unit |
| `orders` | SNAPSHOT only: array of order objects in queue-priority order |

Every snapshot order has exactly `order_id`, `side`, `price_ticks`, `quantity`.
CSV encodes `orders` as an appropriately quoted JSON array in one cell.
JSON objects cannot contain duplicate keys, and CSV headers cannot contain
duplicate columns. Unknown fields, malformed rows and invalid UTF-8 are errors.

| Event | Required payload | State change |
| --- | --- | --- |
| ADD | ID, side, price, quantity | Append a new resting order; crossing is invalid |
| CANCEL | ID, quantity | Subtract quantity; remove order if exhausted |
| DELETE | ID | Remove entire order; quantity must be omitted |
| EXECUTE | ID, quantity | Subtract from that exact recorded maker ID and record execution |
| MODIFY | ID, new price and/or new quantity | Quantity replaces remaining size; it is not a delta |
| TRADE | Price, quantity | Store a separate print; do not modify the book |
| SNAPSHOT | Explicit `orders` list, including `[]` | Replace the complete resting book |
| HALT | No payload | Enter halted state |
| RESUME | No payload | Leave halted state |

CANCEL, DELETE and EXECUTE may include side and price as assertions; they must
match the resting order. MODIFY may assert side but cannot change it. A
same-price size reduction retains queue position. Size increases and price
changes move the amended order to the back of its destination queue.

EXECUTE never calls the synthetic book's matching logic. An execution against
an ID away from the best quote still reduces that ID; it never silently consumes
a different maker. Reported executions and trade prints remain separate because
feeds can report the same transaction through both channels. Summing these two
volumes would double count. Auction/off-book prints may appear during a halt;
ADD, MODIFY and EXECUTE are rejected while halted, but CANCEL, DELETE and
SNAPSHOT are allowed. Snapshot application preserves halt state.

## Validation and bounds

Equal timestamps are valid and preserve sequence order. The first sequence may
start at any nonnegative value. Gaps, duplicate/reversed sequences, timestamp
reversals, mixed streams, missing IDs, over-execution, invalid halt transitions,
and locked/crossed books are critical. A snapshot is authoritative: it can
introduce IDs not previously seen and replace existing orders. The event list
must still have contiguous sequences; snapshots do not silently repair gaps.
If a real feed is filtered by instrument, its adapter must create a contiguous
canonical sequence while separately verifying completeness of the source feed.

State validation stops at the first invalid transition because later errors
would depend on an unknown book. `event_count` gives the parsed dataset size;
`checked_events` gives the valid prefix. Parse failures report the number of
successfully parsed rows and a physical source row. Lifecycle issue rows refer
to the one-based event index (CSV excludes its header). Reports contain the
first/last feed timestamps, event type counts, canonical SHA-256 and full raw
source SHA-256 when the entire source was read. Partial source hashes are never
presented as full dataset fingerprints. Canonical hashes are independent of
CSV versus JSONL representation.

Defaults bound one file to 128 MiB, one physical row to 2 MiB, and one replay to
1,000,000 events. An individual snapshot permits at most 100,000 orders. Pass
`max_source_bytes`, `max_row_bytes`, or `max_events` to file APIs to set tighter
limits. Exceeding a limit rejects the whole dataset; it never silently truncates.
Files are streamed during parsing but validated events, the active book, seen
IDs, and execution/print history are held in memory. Total memory depends on
record and identifier sizes as well as event count. This is bounded research
replay, not a claim of production-scale feed ingestion.

## Limits and adapter responsibilities

This implementation supports a continuous, order-level book. Venue-specific
ITCH/LOBSTER formats, decimal prices, L2-only snapshots, auctions/crossed books,
hidden liquidity, special execution prices, session rollover, corporate
actions, instrument calendars, ID reuse, and alternate priority rules require
explicit adapters and tests. Do not reinterpret L2 levels as true order IDs or
claim queue accuracy from them. Parquet support, wall-clock speed controls,
scrubbing checkpoints, artificial latency, agent interaction, and calibration
are not provided by this module. No proprietary data or external feed access
is bundled or required.

The separate [public L2 adapter](public-market-data.md) now supports exact decimal
aggregate Tardis updates and comparisons with published top-five snapshots. It
does not convert levels into canonical individual orders. See the two-day
[real-data assessment](../examples/studies/historical/README.md) for observed
results, source provenance and the limits of this mechanics check.
