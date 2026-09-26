# v0.4 market-data contracts and MBO ingestion

`REAL_HISTORICAL_MBO_VALIDATION = NOT_AVAILABLE`. Inspection found the registered
Deribit aggregate L2/snapshot/trade files and project-authored canonical MBO
fixtures, with no genuine historical identity feed. Roadmap M2 remains incomplete.
The available software path validates documented canonical semantics; synthetic
ingestion checks do not change the empirical status.

## Observable capabilities

`lob.capabilities.CapabilityContract` is frozen, defensively copies its capability
set, rejects incompatible declarations, and raises `CapabilityError` for requests
outside the declaration. `restrict()` can remove evidence but cannot grant it.
The enum distinguishes trades, L1, L2, MBO and simulator internal observations.

| Source | Supported observations | Not established by that source alone |
| --- | --- | --- |
| Trades | Recorded trade prints | Book depth, order identities, queue position |
| L1 | Best quotes, spread, top-level imbalance | Full depth, identities, FIFO |
| Aggregate L2 | Displayed depth, quotes, spread, depth imbalance, aggregate changes | Individual arrivals/cancels, identities, queues, passive fills |
| MBO | Declared identities and depth; additional FIFO/maker-fill claims require their own capability | Hidden orders or hypothetical agent fills |
| Simulator | Internal book and simulated fills | Historical executions, exchange truth or profitability |

No default contract grants hidden-liquidity or real passive-counterfactual-fill
evidence. Exchange sequence continuity is independent from normalized row
numbering. MBO queue access requires identity, snapshot ordering and FIFO;
quantity ahead also requires aggregate depth. Execution events require observed
maker-fill capability. Snapshot output omits priority when ordering is unknown.

```python
from lob.capabilities import Capability, L2_CONTRACT

L2_CONTRACT.require(Capability.AGGREGATE_DEPTH)
# Raises CapabilityError:
L2_CONTRACT.require(Capability.QUANTITY_AHEAD)
```

`L2Replay` and `L2State` expose `capability_contract`; their v0.3 flags and refusal
methods remain compatible. `MBOBook` and `MBOReplay` also expose the formal
contract. Direct legacy MBO construction retains its documented canonical FIFO
obligations. New dataset adapters require an explicit declaration instead of
granting FIFO merely because a file is labelled MBO.

## Adapter interface and source manifest

`lob.datasets.DatasetAdapter` exposes `metadata`, `capability_contract`, adapter
name/version, one-pass `events()`, and copied `stats`. Completion and canonical
hashes are published only after exhausting valid input and checking source bytes.
Stopping early is an unvalidated prefix. The source hash is checked before and
after consumption; a different same-size file cannot masquerade as the original.

`DatasetMetadata` requires source identity, venue, instrument, source kind
(`synthetic` or `historical`), data level, SHA-256, timestamp/sequence meanings,
UTC timezone, tick and lot sizes, field mapping, provenance, licensing,
snapshot/execution/priority semantics and missing-data behavior. Unknown L2 units
may be `null`; integer tick/lot MBO must have known positive decimal unit sizes.
The supported missing-data policy is rejection. Metadata and mappings are
immutable. A manifest declaration is a source obligation, not an authenticity test.

The [complete synthetic manifest](../examples/studies/v04/mbo-input/adapter-manifest.json)
is a reproducible example. Never change its source-kind label to historical.

| Adapter | Canonical output | Input boundary |
| --- | --- | --- |
| `TardisL2Adapter` | Completed `L2State` capture groups with exact Decimal levels | Existing eight-column Tardis CSV/gzip, unchanged replay semantics |
| `CanonicalMBOAdapter` | Applied `MBOEvent` objects | Strict canonical JSONL/NDJSON, optionally gzip |
| `MappedMBOCSVAdapter` | Applied `MBOEvent` objects | Explicit canonical-field to source-column mapping and explicit action/side maps |

For mapped CSV, header columns must exactly match the declared mapping; unknown
actions, missing order IDs and unmapped sides are errors. CSV uses UTC integer
nanoseconds, integer ticks and integer lots. A source with decimal prices must
first convert them exactly using instrument tick size and retain transformation
provenance. Snapshot `orders` is a JSON array of canonical order objects. Source
row order is retained, no IDs are generated, and no gaps or invalid records are
skipped. The adapter does not infer a vendor's amendment/execution rules.

The supported MBO semantics are explicit manifest values:

| Field | Accepted values |
| --- | --- |
| `sequence_semantics` | `normalized_contiguous_per_instrument` or `source_contiguous_per_instrument` |
| `snapshot_semantics` | `complete_census_source_fifo` or `complete_census_ordering_unknown` (FIFO disabled) |
| `priority_semantics` | `same_price_reduction_retains_increase_or_reprice_resets` |
| `execution_semantics` | `maker_execute_decrements_once_trade_is_unlinked` |

The [MBO schema](mbo.md) specifies ADD, MODIFY, CANCEL, DELETE, EXECUTE, TRADE,
SNAPSHOT, RESET, HALT and RESUME payloads, ordering, censoring and resource limits.
The first observation must be a complete census or a genuine empty-book RESET.
An incomplete initial book cannot be repaired by inventing RESET.

## Validation workflow

```powershell
python -m lob.mbo_validation examples/studies/v04/mbo-input/synthetic-events.jsonl --manifest examples/studies/v04/mbo-input/adapter-manifest.json --references examples/studies/v04/mbo-input/synthetic-aggregate.jsonl --out results/v04-mbo-reproduction
cleo verify-artifact results/v04-mbo-reproduction
```

The output directory must not exist. It receives the frozen adapter manifest,
`result.json`, `report.md`, environment/source provenance and an artifact seal.
Failed reconstruction retains the valid prefix and first error with `INVALID`;
counts beyond that error remain unknown. A complete stream with mismatched
references receives `FAILED`. Failed results are not repaired or overwritten.

The pipeline reports event/action counts, observed identities, partial/terminal
executions, cancellations, priority resets, left/right/census censoring,
observed terminal lifetimes, sequence gaps, duplicate sequences/IDs, invalid
transitions, queue checks and independent aggregation consistency. It replays
twice and compares complete canonical digests, counters and final states. A
separate insertion-ordered oracle checks surviving queues after all event kinds,
including snapshots and terminal removals; no observed queues means the queue
result is `NOT_ESTABLISHED`.

References use one JSON object per explicitly observed boundary:

```json
{"sequence":104,"timestamp_ns":40,"depth":null,"bids":[[99,10]],"asks":[[101,12]]}
```

`depth=null` means full depth; an integer means top-N. Levels use the same exact
tick/lot units as MBO and are distinct, best-first. Reference sequences must be
increasing and unique. Missing boundaries, timestamp mismatches, changed price
levels, changed quantities and absolute lot differences are counted separately.
No independent reference means source aggregate agreement is `NOT_AVAILABLE`.
Passing a synthetic fixture remains real-MBO `NOT_AVAILABLE`, even with matching
synthetic reference levels. For historical metadata without reference checks,
real-MBO validation remains `NOT_ESTABLISHED`.

## Adding a real feed

Obtain a licensed sample independently; retain its native bytes, instrument
definition, acquisition date, usage restrictions, checksum and transformation
log. Audit source-native identities, sequence coverage, snapshots, modifications
and executions before declaring capabilities. A source using different priority
or execution rules needs an explicit, tested normalization adapter.

Databento's official [MBO schema](https://databento.com/docs/schemas-and-data-formats/mbo)
is one candidate, but it is not a drop-in canonical source. Its
[field/action/flag definitions](https://databento.com/docs/standards-and-conventions/common-fields-enums-types)
state that `F` fill notifications do not change the resting book, whereas canonical
EXECUTE does. Mapping both that notification and the subsequent book reduction
would double-decrement. Flags also distinguish aggregated MBP/L1 records and
unrecoverable channel gaps. Its [snapshot conventions](https://databento.com/docs/standards-and-conventions/mbo-snapshot)
preserve per-price priority but use historical sequence values; these are not
canonical contiguous event indices. Documentation inspected 2026-09-26. No
Databento data was acquired, and no Databento-native validation is claimed.

M2 requires actual licensed historical input, audited venue semantics and source
reference agreement. This implementation supplies the bounded ingestion and
validation path and records that unresolved empirical dependency.
