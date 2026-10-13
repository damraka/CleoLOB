"""validate-market-data-v09 (M13-M19): prepare day caches for registered slots and report quality and drift.

Inputs: ``--dataset`` ids, or ``--input``-free pool selection through ``--design``-less exploratory use
(``--namespace explore``/``dev``) with the registered stream slots ``pool:stream:count`` given as ``--dataset``
entries of the form ``slots:<pool>:<venue>:<instrument>:<n>``. Every dataset goes through ``store.prepare``
(ledgered download, normalization, reconstruction, quality decision). Failed slots are replaced only by the
registered backups. The sealed run lists per-dataset quality decisions, profiles, grid anomalies and a descriptive
drift report; it never inspects any study outcome.
"""
from __future__ import annotations

from ...preregistration import ProtocolError
from .. import protocol as pr, quality
from ..runner import Task, run_tasks
from . import common

ANALYSIS = "validate-market-data-v09"


def prepare_task(dataset_id: str, use: str, design: str, root: str) -> dict:
    from pathlib import Path
    from ..data import store
    from ..data.sources import DataNotAvailable
    try:
        cache, record = store.prepare(dataset_id, use=use, design=design, root=Path(root),
                                      purpose="day cache for registered studies")
    except DataNotAvailable as exc:
        return {"dataset": dataset_id, "status": "NOT_AVAILABLE", "reason": str(exc)}
    q = cache["quality"]
    return {"dataset": dataset_id, "status": q["decision"]["status"], "cache_sha256": record["cache_sha256"],
            "checks": q["decision"]["checks"], "profile": q["profile_v09"],
            "grid_anomalies": q.get("grid", {}).get("anomalies", []),
            "reconstruction": {k: q["reconstruction"][k] for k in ("messages", "sequence_gaps", "duplicates",
                                                                   "out_of_order", "crossed", "max_gap_ms")},
            "trades": q["trades"], "valid_bin_fraction": q["valid_bin_fraction"]}


def _expand(args, protocol) -> list[str]:
    out: list[str] = []
    for item in args.dataset:
        if item.startswith("slots:"):
            _, pool, venue, instrument, n = item.split(":")
            resolved = common.units(args, pool, [f"{venue}:{instrument}"], int(n))[f"{venue}:{instrument}"]
            out.extend(resolved["datasets"])
        else:
            pr.dataset_declaration(protocol, item)
            out.append(item)
    if not out:
        raise ValueError("no datasets selected (use --dataset ID or --dataset slots:POOL:VENUE:INSTRUMENT:N)")
    return list(dict.fromkeys(out))


def run(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    use = common.use_for(args)
    datasets = _expand(args, protocol)
    roles = {pr.dataset_declaration(protocol, d)["role"] for d in datasets}
    if roles & pr.GUARDED_ROLES and not args.design:
        raise ProtocolError("guarded datasets are prepared only by their sealed design")
    design = args.design or f"{args.namespace}-data-prep"
    out = common.start(args)
    tasks = [Task(d, "lob.v09.studies.data:prepare_task", {"dataset_id": d, "use": use, "design": design,
                                                           "root": str(root)}, namespace=args.namespace)
             for d in datasets]
    rows, info = run_tasks(tasks, workers=args.workers or 1, contract=common.contract(args, workers_default=1))
    by_id = {r["dataset"]: r for r in rows}
    ok = [d for d in datasets if by_id[d]["status"] == "AVAILABLE"]
    profiles = {d: by_id[d]["profile"] for d in ok}
    drift = quality.drift(profiles, ok) if len(ok) >= 3 else None
    result = {"datasets": by_id, "available": ok,
              "not_available": sorted(d for d in datasets if by_id[d]["status"] != "AVAILABLE"),
              "drift": drift, "thresholds": protocol["data_quality"]["thresholds"], "runner": info}
    return common.finish(args, out, analysis=ANALYSIS, config={"datasets": datasets, "use": use, "design": design},
                         result=result, datasets=ok)
