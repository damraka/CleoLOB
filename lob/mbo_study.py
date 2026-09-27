"""M1/M2 study on a genuine Bitstamp order-level capture, under the frozen design.

Order of operations: verify the capture manifest hash, record the ledger access,
run the frozen adapter/semantics, evaluate the frozen M1 gate, compute secondary
diagnostics, run the frozen M2 order-level bounds and observed-order coverage,
then bind and seal. The detailed outputs are derived from venue data and stay local.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .experiments.registry import PROJECT_ROOT
from .historical_execution import run_mbo
from .mbo_sources import BitstampCaptureAdapter, m1_gate, tolerant_reference_agreement, validate_source
from .order_lifecycle import LifecycleSemantics
from . import preregistration as pr
from .v05_evidence import LEDGER_PATH, PROTOCOL_PATH, finalize, new_run, write_json

DESIGN_PATH = "configs/v05/m1-m2-design.json"


def load_design(root: Path = PROJECT_ROOT) -> dict:
    return json.loads((root / DESIGN_PATH).read_text(encoding="utf-8"))


def frozen_semantics(design: dict) -> LifecycleSemantics:
    return LifecycleSemantics(**design["m1"]["semantics"])


def run(capture_dir: str | Path, out: str | Path, *, dataset_id: str, root: Path = PROJECT_ROOT,
        record_access: bool = True) -> dict:
    capture_dir = Path(capture_dir)
    manifest = json.loads((capture_dir / "capture-manifest.json").read_text(encoding="utf-8"))
    source = capture_dir / manifest["file"]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("capture bytes differ from their manifest")
    design = load_design(root)
    protocol = pr.load_protocol(root / PROTOCOL_PATH)
    pr.dataset_declaration(protocol, dataset_id)
    if record_access:
        pr.append_event(root / LEDGER_PATH, "access", {"dataset_id": dataset_id,
                        "source_sha256": {manifest["file"]: digest}, "purpose": "frozen M1/M2 order-level study"},
                        protocol=protocol)
    out = new_run(out)
    semantics = frozen_semantics(design)
    adapter = BitstampCaptureAdapter(source, alignment=design["m1"]["adapter"]["reference_alignment"])
    if adapter.contract.adapter_version != design["m1"]["adapter"]["adapter_version"]:
        raise ValueError("adapter version differs from the frozen design")
    validation = validate_source(adapter, semantics, depth=design["m1"]["reference_depth"])
    events, references = adapter.build()
    tolerant = tolerant_reference_agreement(events, references, semantics, depth=design["m1"]["reference_depth"])
    gate = m1_gate(validation)
    m2 = run_mbo(events, design["m2"]["mbo_design"], semantics=semantics, dataset_id=dataset_id,
                 observed_lifetime_s=design["m2"]["mbo_observed_order_window_s"])
    write_json(out, "m2-orders.json", m2["orders"])
    result = {
        "dataset_id": dataset_id, "capture_status": manifest["status"], "capture_sha256": digest,
        "m1": {"gate": gate, "validation": {k: v for k, v in validation.items() if k != "report"},
               "lifecycle_report": validation["report"], "secondary_tolerant_reference_agreement": tolerant},
        "m2": {"summary": m2["summary"], "observed_order_validation": m2["observed_order_validation"],
               "capability": m2["capability"]},
        "status_scope": ("Genuine venue order-level messages recorded live and replayed offline for one "
                         "instrument, one venue and one capture window. Not a vendor-certified archive; "
                         "FIFO is not established by the source."),
    }
    config = {"design": design, "design_sha256": pr.document_sha256(design), "contract": adapter.contract.to_dict(),
              "semantics": semantics.to_dict(), "capture_manifest": manifest}
    finalize(out, analysis=f"m1-m2-{dataset_id}", dataset_ids=[dataset_id], config=config, result=result, root=root)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    args = parser.parse_args(argv)
    result = run(args.capture_dir, args.out, dataset_id=args.dataset_id)
    print(json.dumps({"m1_gate": result["m1"]["gate"], "m2_summary": result["m2"]["summary"],
                      "observed": result["m2"]["observed_order_validation"]}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
