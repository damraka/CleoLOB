# MBO identity and queue research

**IMPLEMENTED / TESTED:** `lob.mbo` adds bounded streaming MBO replay and observed
order research to the existing `lob.replay.HistoricalBook`. It reuses that
book's validated identity, modification, execution and FIFO semantics. The
separate aggregate L2 implementation continues to reconstruct price levels.

**LIMITATION:** No real MBO dataset was found in the inspected environment.
Available Deribit files contain aggregate L2 updates, top-five snapshots and
trade prints. Those cannot establish individual order identity or FIFO. Real
MBO adapter validation remains pending. Both bundled canonical order datasets
are synthetic fixtures; they are not historical market evidence.

## Schema and source contract

`MBOEvent` retains the existing exact integer conventions: `price_ticks` uses
positive integer ticks, `quantity` uses positive integer lots/shares/contracts,
and clocks use nonnegative signed-64-bit integer nanoseconds. An adapter must
record tick and lot sizes in its instrument metadata. Binary floats, negative
values, Boolean numbers and out-of-range integers are rejected. Decimal digit
strings are accepted for compatibility with existing canonical CSV adapters.

Required fields are `timestamp_ns`, `sequence`, `event_type`, `symbol`, and
`venue`. `exchange_timestamp_ns` is optional; its absence stays absent. The
capture/replay timestamp must not decrease. Exchange-clock regressions are
counted and retained without reordering. Canonical sequences are contiguous;
the first sequence may be any nonnegative value. No gap is silently repaired,
including at RESET or SNAPSHOT. Source adapters must verify original exchange
sequence coverage separately when instrument filtering changes numbering.

| Event | Payload and meaning |
| --- | --- |
| ADD | Actual ID, side (`BUY`/`SELL`), price and quantity; append to price queue |
| MODIFY | ID and new remaining quantity and/or price; quantity is absolute |
| CANCEL | ID and quantity removed; partial cancellation is supported |
| DELETE | ID; remove all remaining quantity |
| EXECUTE | Recorded maker ID and executed quantity; partial/full fills supported |
| TRADE | Price and quantity of an unlinked print; does not decrement the book |
| SNAPSHOT | Explicit list of `SnapshotOrder` objects in source-guaranteed FIFO order |
| RESET | Explicit empty-book census; omit all order payloads |
| HALT / RESUME | Existing canonical trading-state transition rules |

Every stream must start with a SNAPSHOT or RESET. An empty RESET is an explicit
assertion that the captured book starts empty; do not use it to disguise a
missing initial census. Snapshot FIFO is a source/adapter obligation: unordered
identity snapshots are insufficient for exact position. L2 levels must never
be converted into invented order IDs. Unknown/duplicate fields, duplicate IDs,
over-cancellation/execution, crossed continuous books, mixed instruments and
invalid sequences are rejected. All rejected events leave book and research
state unchanged. ADD cannot reuse an ID, even after removal or RESET. A later
authoritative snapshot may contain an already known identity.

Same-price size reductions retain priority. Size increases and price changes
move the order to the back of its destination queue. Venues with different
rules require an explicit translating adapter. EXECUTE decrements its recorded
ID even if it is behind another visible order or away from the best quote; the
reconstructor does not substitute a simulated matching decision. TRADE and
EXECUTE channels remain separate to avoid counting the same transaction twice.

## Run the synthetic fixture

```python
from lob.mbo import MBOReplay

replay = MBOReplay("examples/data/mbo-events.jsonl")
for event in replay:
    if event.event_type == "ADD":
        replay.book.watch(event.order_id)

assert replay.stats["complete"]
assert replay.stats["events"] == 12
assert replay.book.aggregate_l2() == {"bids": [], "asks": [[102, 9]]}
assert replay.book.order_research("b2")["time_to_full_fill_ns"] == 60
print(replay.book.cohort_summary(horizon_ns=60))
print(replay.summary())
```

The fixture uses `SYNTHETIC-FIXTURE` as its venue. It exercises partial fills,
cancellation ahead, queue advancement, modification, separate prints and RESET.
It is project-authored and redistributable under the repository license.

`MBOBook.apply(MBOEvent(...))` supports direct adapters. `aggregate_l2(depth=None)`
returns deterministic JSON-safe `bids` and `asks`, sorted best-first, with exact
integer quantities summed at each price. `depth` limits output, not retained
book state. The engine does not implicitly convert those units into the Decimal
price/quantity conventions of the independent Tardis L2 parser.

## Queue metrics and estimands

`queue_metrics(order_id)` gives the current zero-based position at its price,
order count ahead, quantity ahead/behind and age since its observed ADD.
Snapshot-introduced orders have `age_ns=None` and `age_is_left_censored=True`.
This age is identity age, not time since its most recent priority-losing amend.
Orders at better prices are not included in same-price quantity ahead.

`watch(order_id)` registers prospective observation of an actual resting ID.
Register immediately after ADD to measure position at submission. A watch
registered later remains labelled as such, and is excluded from submission
cohort estimates even when its original ADD timestamp is known. Register a
cohort rule before observing outcomes; selective watching can bias frequencies.
The book retains one observation origin per ID for a bounded run.

`order_research(order_id)` returns a copied trajectory and timing information:

| Metric | Definition and units |
| --- | --- |
| Position at submission | Number of same-price orders ahead at the observed ADD |
| Queue movement | Initial position minus current position; positive means advancement |
| Quantity ahead/behind | Current displayed quantity in integer source units |
| Cancellation ahead | Sum of CANCEL, DELETE and same-price size reductions of orders ahead immediately before each event |
| First-fill time | Nanoseconds from observed submission to first recorded maker EXECUTE |
| Full-fill time | Time until cumulative recorded executions reach the initial submitted quantity |
| Terminal reason | Final remaining size executed, cancelled/deleted, or censored by a census |

Cancellation ahead excludes executions and repricing. Full-fill time uses the
initial quantity even if the order is later amended: cancelling four units of
an initial ten and executing the remaining six does **not** count as a full
initial-size fill. After a size increase, the initial-size threshold may be
reached while additional quantity remains resting. `terminal_reason="FILLED"`
means the final remainder was executed; it does not override this threshold.

`cohort_summary(horizon_ns)` computes descriptive frequencies at a fixed
nanosecond horizon, using submission watches only:

- Any passive fill: at least one recorded maker execution by the deadline.
- Full fill: cumulative executions reach the initial quantity by the deadline.
- Queue survival: identity remains resting at the deadline, even after partial
  executions or modifications.

The denominator includes orders observed for the entire horizon and orders
that terminate through cancellation or execution before it. Orders censored
by the observed prefix ending or by a RESET/SNAPSHOT at or before the horizon
are excluded and counted separately. A later census does not invalidate an
already completed shorter horizon. Snapshot replacements are censoring events,
not invented cancellations or fills. Their terminal trajectory observation has
`quantity=None`, because a census does not establish the intervening fate of
the prior remainder, even when the same ID is present in the new census.
`terminal_timestamp_ns` records when observation ended. Frequencies are `None` when no eligible
denominator exists. No independence-of-censoring assumption or survival-model
extrapolation is made. These are complete-case empirical cohort frequencies,
not an unbiased population model or a fill-probability forecast.

**LIMITATION:** Every execution above is an observed source maker execution.
The API explicitly returns no counterfactual fill probability and no hidden
liquidity. Identity alone cannot establish execution outcomes for hypothetical
agent orders, source completeness, venue matching exceptions, or live alpha.

## Bounds, integrity and L2 refusal

The JSONL/NDJSON parser optionally accepts gzip. Defaults bound compressed/source
bytes to 128 MiB, expanded bytes to 256 MiB, physical lines to 2 MiB, events to
one million, active orders to 100,000, cumulative ADD/snapshot order records to
one million, watched orders to 1,000 and stored observations to 100,000. Limits
reject rather than truncate. Cumulative snapshot-record limits also bound the
reference book's seen-ID history. Each active watch records every applied event,
so dense tracking can reach its observation limit before the event limit.

Streaming consumers see only a validated prefix until exhaustion. Only a
successful EOF and gzip trailer check set `stats["complete"]` and publish raw
source/canonical SHA-256 digests. A stopped iterator or corrupt suffix remains
incomplete; no partial hash is described as a full source hash. A replay is
one-pass. Full preflight-before-observation remains available through the older
canonical `HistoricalReplay` interface when its schema is sufficient.

`L2State.capabilities` and `L2Replay.capabilities` explicitly mark individual
identity, exact FIFO, quantity ahead, observed order fill, counterfactual
passive fill and hidden quantity as unavailable. Consumers can call
`require_capability(name)` to get an explicit refusal. Aggregate depth remains
available. These flags do not change L2 reconstruction or its historical
validation results.

## Validation status

**TESTED:** New deterministic and seeded randomized tests compare FIFO and
aggregation against an independent insertion-ordered oracle. They cover fills,
cancels, modification priority, observed cohort denominators/censoring, strict
schema and sequence handling, atomic rejection, bounded parsing, gzip CRC,
hashes, deterministic replay, and the L2 evidence boundary. Existing canonical
replay and L2 regression tests also remain applicable.

```bash
python -m pytest tests/test_mbo.py tests/test_replay.py tests/test_l2_replay.py -q
```

**PLANNED:** Validate a genuine, licensed order-level source adapter against
vendor/exchange semantics and independently published aggregation checks. Do
not claim historical MBO validation until that work has actual source evidence.
