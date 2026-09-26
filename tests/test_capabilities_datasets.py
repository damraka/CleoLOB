"""Evidence boundaries and adapter ingestion, using only synthetic inputs."""
from __future__ import annotations

import csv
import io
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from lob.capabilities import (
    CANONICAL_MBO_CONTRACT, L1_CONTRACT, L2_CONTRACT, SIMULATOR_CONTRACT, TRADES_CONTRACT,
    Capability, CapabilityContract, CapabilityError, DataLevel,
)
from lob.datasets import (
    CanonicalMBOAdapter, DatasetAdapter, DatasetMetadata, MappedMBOCSVAdapter,
    TardisL2Adapter, mbo_adapter, source_sha256,
)
from lob.mbo import MBOBook, MBOEvent
from lob.mbo_validation import AggregateReference, load_references, run_mbo_validation, validate_mbo
from lob.artifacts import verify_artifacts


def event(sequence=0, kind="RESET", **values):
    result = {"timestamp_ns": sequence * 10, "sequence": sequence, "event_type": kind,
              "venue": "SYNTHETIC", "symbol": "TEST"}
    if kind == "ADD":
        result.update(order_id=f"o{sequence}", side="BUY", price_ticks=100, quantity=10)
    result.update(values)
    return result


def source(tmp_path, events=None):
    path = tmp_path / "mbo.jsonl"
    events = [event(), event(1, "ADD"), event(2, "EXECUTE", order_id="o1", quantity=4),
              event(3, "MODIFY", order_id="o1", quantity=8),
              event(4, "EXECUTE", order_id="o1", quantity=8)] if events is None else events
    path.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    return path


def metadata(path, **changes):
    result = dict(source_id="authored-test-fixture", venue="SYNTHETIC", instrument="TEST",
                  data_level=DataLevel.MBO, source_kind="synthetic", source_sha256=source_sha256(path),
                  timestamp_semantics="capture UTC nanoseconds; exchange clock optional",
                  sequence_semantics="normalized_contiguous_per_instrument", tick_size="0.01", lot_size="1",
                  field_mapping={key: key for key in event()}, provenance={"author": "CleoLOB tests"},
                  licensing="project synthetic fixture", snapshot_semantics="complete_census_source_fifo",
                  execution_semantics="maker_execute_decrements_once_trade_is_unlinked",
                  priority_semantics="same_price_reduction_retains_increase_or_reprice_resets")
    return DatasetMetadata(**(result | changes))


def adapter(path, **changes):
    return CanonicalMBOAdapter(path, metadata(path, **changes), CANONICAL_MBO_CONTRACT)


@pytest.mark.parametrize("contract,allowed,denied", [
    (TRADES_CONTRACT, Capability.TRADE_PRINTS, Capability.TOP_OF_BOOK),
    (L1_CONTRACT, Capability.TOP_OF_BOOK, Capability.AGGREGATE_DEPTH),
    (L2_CONTRACT, Capability.AGGREGATE_DEPTH, Capability.QUANTITY_AHEAD),
    (CANONICAL_MBO_CONTRACT, Capability.ORDER_IDENTITY, Capability.HIDDEN_ORDER_QUANTITY),
    (SIMULATOR_CONTRACT, Capability.SIMULATED_FILLS, Capability.OBSERVED_ORDER_FILL),
])
def test_distinct_market_data_levels_enforce_observability(contract, allowed, denied):
    contract.require(allowed)
    with pytest.raises(CapabilityError, match="unavailable"):
        contract.require(denied)
    with pytest.raises(CapabilityError):
        contract.require(Capability.COUNTERFACTUAL_PASSIVE_FILL)
    with pytest.raises(FrozenInstanceError):
        contract.available = frozenset(Capability)


def test_contract_defensively_copies_and_rejects_invalid_evidence_and_dependencies():
    allowed = {Capability.AGGREGATE_DEPTH}
    contract = CapabilityContract(DataLevel.L2, allowed, "test")
    allowed.add(Capability.ORDER_IDENTITY)
    assert not contract.supports(Capability.ORDER_IDENTITY)
    with pytest.raises(CapabilityError, match="cannot establish"):
        CapabilityContract(DataLevel.L2, allowed, "test")
    with pytest.raises(CapabilityError, match="requires"):
        CapabilityContract(DataLevel.MBO, {Capability.QUANTITY_AHEAD}, "test")
    with pytest.raises(CapabilityError, match="cannot grant"):
        contract.restrict({Capability.ORDER_IDENTITY})
    with pytest.raises(CapabilityError, match="Unknown"):
        contract.require("imaginary")


def test_identity_only_mbo_does_not_grant_fifo_or_fill_evidence():
    contract = CANONICAL_MBO_CONTRACT.restrict({Capability.ORDER_IDENTITY, Capability.AGGREGATE_DEPTH})
    book = MBOBook(capability_contract=contract)
    book.apply(MBOEvent.from_mapping(event()))
    book.apply(MBOEvent.from_mapping(event(1, "ADD")))
    assert book.aggregate_l2() == {"bids": [[100, 10]], "asks": []}
    assert "priority" not in book.snapshot()["orders"][0]
    assert book.orders["o1"].quantity == 10
    with pytest.raises(CapabilityError):
        _ = book.orders["o1"].priority
    for call in (lambda: book.queue_metrics("o1"), lambda: book.queue("BUY", 100), lambda: book.watch("o1")):
        with pytest.raises(CapabilityError):
            call()
    before = book.summary()
    with pytest.raises(CapabilityError, match="observed_order_fill"):
        book.apply(MBOEvent.from_mapping(event(2, "EXECUTE", order_id="o1", quantity=1)))
    assert book.summary() == before


@pytest.mark.parametrize("changes,message", [
    ({"tick_size": None}, "requires known"), ({"lot_size": "nan"}, "decimal"),
    ({"timezone": "local"}, "UTC"), ({"missing_data_behavior": "interpolate"}, "reject"),
    ({"source_kind": "historical"}, "synthetic fixture"), ({"source_sha256": "bad"}, "SHA-256"),
    ({"source_kind": "unspecified"}, "explicitly"),
])
def test_malformed_dataset_metadata_is_rejected(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        metadata(source(tmp_path), **changes)


def test_metadata_is_immutable_and_roundtrips(tmp_path):
    declared = metadata(source(tmp_path))
    assert DatasetMetadata.from_mapping(declared.to_dict()) == declared
    exported = declared.to_dict()
    exported["field_mapping"]["order_id"] = "invented"
    assert "order_id" not in dict(declared.field_mapping)
    with pytest.raises(FrozenInstanceError):
        declared.source_kind = "historical"


def test_canonical_adapter_validates_hashes_identity_and_one_pass(tmp_path):
    path = source(tmp_path)
    item = adapter(path)
    assert isinstance(item, DatasetAdapter)
    assert len(list(item.events())) == 5
    assert item.stats["complete"]
    assert item.stats["source_sha256"] == source_sha256(path)
    twin = item.reopen()
    list(twin.events())
    assert twin.stats == item.stats
    with pytest.raises(ValueError, match="one-pass"):
        list(item.events())
    with pytest.raises(ValueError, match="identity"):
        list(adapter(path, instrument="OTHER").events())
    bad = CanonicalMBOAdapter(path, replace(metadata(path), source_sha256="0" * 64), CANONICAL_MBO_CONTRACT)
    with pytest.raises(ValueError, match="SHA-256"):
        list(bad.events())
    assert not bad.stats["complete"]


def test_prefix_and_midstream_source_mutation_cannot_publish_complete_hash(tmp_path):
    path = source(tmp_path)
    item = adapter(path)
    iterator = item.events()
    next(iterator)
    iterator.close()
    assert not item.stats["complete"] and item.stats["canonical_sha256"] is None
    item = adapter(path)
    iterator = item.events()
    next(iterator)
    # Same-length mutation can evade size-only verification; digest must detect it.
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b"TEST", b"BEST"))
    with pytest.raises(ValueError):
        list(iterator)
    assert not item.stats["complete"]


def test_mbo_native_semantics_must_be_declared_and_compatible(tmp_path):
    path = source(tmp_path)
    for changes in ({"priority_semantics": "pro_rata"}, {"execution_semantics": "fill_notification"},
                    {"snapshot_semantics": "unordered_identity_census"}):
        with pytest.raises(ValueError):
            adapter(path, **changes)
    contract = CapabilityContract(DataLevel.MBO,
                                  CANONICAL_MBO_CONTRACT.available | {Capability.EXCHANGE_SEQUENCE_CONTINUITY}, "test")
    with pytest.raises(ValueError, match="normalized sequence"):
        CanonicalMBOAdapter(path, metadata(path), contract)


def test_mapped_csv_preserves_ids_and_exact_units_and_matches_jsonl(tmp_path):
    path = source(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    mapping = {key: "native_" + key for key in fields}
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(mapping.values()), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({mapping[key]: value for key, value in row.items()})
    csv_path = tmp_path / "native.csv"
    csv_path.write_text(output.getvalue(), encoding="utf-8")
    item = MappedMBOCSVAdapter(csv_path, metadata(csv_path, field_mapping=mapping), CANONICAL_MBO_CONTRACT,
                               action_mapping={kind: kind for kind in {r["event_type"] for r in rows}},
                               side_mapping={"BUY": "BUY", "SELL": "SELL"})
    original = adapter(path)
    assert list(item.events()) == list(original.events())
    assert item.stats["canonical_sha256"] == original.stats["canonical_sha256"]
    assert validate_mbo(item.reopen())["deterministic_replay"]
    csv_path.write_text(output.getvalue().replace("o1", ""), encoding="utf-8")
    broken = MappedMBOCSVAdapter(csv_path, metadata(csv_path, field_mapping=mapping), CANONICAL_MBO_CONTRACT,
                                 action_mapping={kind: kind for kind in {r["event_type"] for r in rows}},
                                 side_mapping={"BUY": "BUY"})
    report = validate_mbo(broken)
    assert report["status"] == "INVALID"
    assert report["errors"][0]["code"] == "MISSING_ID"


def test_l2_adapter_keeps_existing_exact_reconstruction(tmp_path):
    path = tmp_path / "depth.csv"
    path.write_text("exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount\n"
                    "SYNTHETIC,TEST,1,1,true,bid,99.1,0.2\n"
                    "SYNTHETIC,TEST,1,1,true,ask,99.2,0.4\n", encoding="utf-8")
    item = TardisL2Adapter(path, metadata(path, data_level=DataLevel.L2, tick_size=None, lot_size=None))
    states = list(item.events())
    assert str(states[0].bids[0][1]) == "0.2"
    assert item.stats["complete"] and item.stats["replay"]["rows"] == 2
    with pytest.raises(CapabilityError):
        item.capability_contract.require(Capability.QUANTITY_AHEAD)


def test_validation_lifetimes_censoring_aggregation_and_synthetic_status(tmp_path):
    result = validate_mbo(adapter(source(tmp_path)), [AggregateReference(2, 20, ((100, 6),), ())])
    assert result["status"] == "ESTABLISHED"
    assert result["real_historical_mbo_validation"] == "NOT_AVAILABLE"
    assert result["source_aggregate_agreement"] == "ESTABLISHED"
    assert result["deterministic_replay"]
    assert result["counts"]["executed_quantity"] == 12
    assert result["counts"]["priority_resets"] == 1
    assert result["counts"]["partial_executions"] == 1
    assert result["counts"]["terminal_executions"] == 1
    assert result["counts"]["queue_inconsistencies"] == 0
    assert result["lifetime_ns"]["mean"] == 30
    censor = validate_mbo(adapter(source(tmp_path, [event(), event(1, "ADD"), event(2), event(3, "ADD")])))
    assert censor["counts"]["census_censored_trajectories"] == 1
    assert censor["counts"]["right_censored_at_eof"] == 1
    assert censor["source_aggregate_agreement"] == "NOT_AVAILABLE"


def test_reference_mismatches_and_missing_boundaries_are_quantitative(tmp_path):
    references = [AggregateReference(2, 19, ((101, 7),), ()), AggregateReference(100, 1000, (), ())]
    result = validate_mbo(adapter(source(tmp_path)), references)
    assert result["status"] == "FAILED"
    assert result["counts"]["aggregate_book_mismatches"] == 1
    assert result["counts"]["price_level_mismatches"] == 2
    assert result["counts"]["volume_mismatches"] == 2
    assert result["counts"]["absolute_volume_difference_lots"] == 13
    assert result["counts"]["reference_timestamp_mismatches"] == 1
    assert result["counts"]["unmatched_references"] == 1
    with pytest.raises(ValueError, match="increasing"):
        validate_mbo(adapter(source(tmp_path)), [references[0], references[0]])


@pytest.mark.parametrize("bad,code,count", [
    (event(3, "ADD"), "SEQUENCE_GAP", "sequence_gaps"),
    (event(1, "ADD"), "DUPLICATE_SEQUENCE", "duplicate_sequences"),
    (event(2, "ADD", order_id="o1"), "DUPLICATE_ORDER_ID", "duplicate_order_ids"),
    (event(2, "ADD", timestamp_ns=0), "TIMESTAMP_REVERSAL", "timestamp_reversals"),
    (event(2, "EXECUTE", order_id="o1", quantity=11), "EXCESS_QUANTITY", "invalid_transitions"),
])
def test_invalid_prefix_never_reports_complete_or_historical_evidence(tmp_path, bad, code, count):
    result = validate_mbo(adapter(source(tmp_path, [event(), event(1, "ADD"), bad, event(4)])))
    assert result["status"] == "INVALID"
    assert result["counts"][count] == 1
    assert result["counts"]["events"] == 2
    assert result["errors"][0]["code"] == code
    assert result["source"]["source_sha256"] is None
    assert result["deterministic_replay"] is None


def test_sealed_output_rejects_overwrite_and_detects_mutation(tmp_path):
    path = source(tmp_path)
    manifest = {"adapter": "canonical-mbo-jsonl", "metadata": metadata(path).to_dict(),
                "capabilities": CANONICAL_MBO_CONTRACT.to_dict()["available"]}
    assert mbo_adapter(path, manifest).metadata == metadata(path)
    root = tmp_path / "evidence"
    run_mbo_validation(path, manifest, root)
    assert verify_artifacts(root)["valid"]
    with pytest.raises(FileExistsError):
        run_mbo_validation(path, manifest, root)
    (root / "result.json").write_text("{}", encoding="utf-8")
    assert not verify_artifacts(root)["valid"]


def test_reference_schema_and_reader_reject_ambiguous_levels(tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        AggregateReference(1, 1, ((100, 2), (100, 3)), ())
    path = tmp_path / "refs.jsonl"
    path.write_text(json.dumps(AggregateReference(1, 10, ((100, 10),), ()).to_dict()) + "\n", encoding="utf-8")
    assert load_references(path)[0].bids == ((100, 10),)
    path.write_text('{"sequence":1,"sequence":2}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_references(path)


def test_snapshot_queue_checks_and_no_observation_status(tmp_path):
    orders = [{"order_id": oid, "side": "BUY", "price_ticks": 100, "quantity": 2} for oid in ("z", "a")]
    result = validate_mbo(adapter(source(tmp_path, [event(kind="SNAPSHOT", orders=orders)])))
    assert result["counts"]["queue_checks"] == 1
    assert result["queue_consistency_status"] == "ESTABLISHED"
    empty = validate_mbo(adapter(source(tmp_path, [event()])))
    assert empty["queue_consistency_status"] == "NOT_ESTABLISHED"


def test_terminal_event_audits_remaining_queue_order(tmp_path, monkeypatch):
    item = adapter(source(tmp_path, [event(), event(1, "ADD"), event(2, "ADD"), event(3, "ADD"),
                                     event(4, "CANCEL", order_id="o1", quantity=10)]))
    original = item.book.queue

    def corrupted(side, price):
        queue = original(side, price)
        return tuple(reversed(queue)) if queue == ("o2", "o3") else queue

    monkeypatch.setattr(item.book, "queue", corrupted)
    result = validate_mbo(item)
    assert result["status"] == "FAILED"
    assert result["counts"]["queue_inconsistencies"] == 1
