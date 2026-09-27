"""M2 study: frozen hypothetical child orders on one aggregate-L2 Deribit period.

Uses the design frozen in ``configs/v05/m1-m2-design.json``; the dataset must be
registered in the protocol. Per-order bounds derived from provider data stay in
the local run directory; summaries are bound to the protocol and ledger.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiments.registry import PROJECT_ROOT
from .historical_execution import run_l2
from .mbo_study import load_design
from . import preregistration as pr
from .v05_data import acquire
from .v05_evidence import finalize, new_run, write_json


def run(dataset_id: str, out: str | Path, *, root: Path = PROJECT_ROOT) -> dict:
    design = load_design(root)
    if dataset_id not in design["m2"]["l2_datasets"]:
        raise ValueError("dataset is not registered for the frozen M2 aggregate-L2 design")
    files = acquire(dataset_id, root=root, purpose="frozen M2 bounded historical execution")
    out = new_run(out)
    result = run_l2(files["l2"], files["trades"], design["m2"]["l2_design"], dataset_id=dataset_id)
    write_json(out, "orders.json", result["orders"])
    summary = {"dataset_id": dataset_id, "capability": result["capability"], "summary": result["summary"],
               "by_size_and_lifetime": _cells(result["orders"]), "source_stats": result["source_stats"],
               "scope": "Aggregate L2 plus trade prints: exact FIFO, identities and hidden size are never inferred."}
    config = {"design": design["m2"], "design_sha256": pr.document_sha256(design),
              "files": {kind: path.name for kind, path in files.items()}}
    finalize(out, analysis=f"m2-l2-{dataset_id}", dataset_ids=[dataset_id], config=config, result=summary, root=root)
    return summary


def _cells(orders: list[dict]) -> dict:
    cells: dict[str, dict] = {}
    for order in orders:
        key = f"{order['side']}|{order['quantity']}|{(order['expire_ts'] - order['submit_ts']) // 1_000_000}s"
        cell = cells.setdefault(key, {"orders": 0})
        cell["orders"] += 1
        cell[order["classification"]] = cell.get(order["classification"], 0) + 1
    return dict(sorted(cells.items()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run(args.dataset_id, args.out)
    print(json.dumps(summary["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
