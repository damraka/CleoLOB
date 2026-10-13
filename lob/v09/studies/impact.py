"""impact-study-v09 (M71-M73, M75-M77; H10): observational impact models on aggressive-flow bursts.

Pilot: leave-one-explore-day-out. Both models are fitted on the bursts of the other days and scored on the held-out
day; the unit endpoint is (MSE_sqrt - MSE_state) / MSE_sqrt of the 10 s response (positive = the state-dependent
model fits out of sample better). Split-half reliability uses odd vs even bursts of each held-out day. Confirm: both
models are fitted once on all explore days and scored on every confirm day. Also reported (descriptive): the
temporary/transient/persistent decomposition, the propagator kernel, BTC->ETH cross impact on matched dates and the
causal identification status of the flow-impact estimand (NOT_IDENTIFIED: unobserved information).
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .. import causal, protocol as pr
from ..capabilities import provided
from ..impact import models as M
from ..runner import Task, run_tasks
from . import common

ANALYSIS = "impact-study-v09"
STREAMS = ["bybit-linear:BTCUSDT", "bybit-linear:ETHUSDT"]


def day_task(dataset_id: str, use: str, design: str, root: str) -> dict:
    from ..data import store
    cache, _ = store.prepare(dataset_id, use=use, design=design, root=Path(root), purpose="impact bursts")
    b = M.bursts(cache)
    return {"dataset": dataset_id, "X": b["X"].tolist(), "y": {str(h): b["y"][h].tolist() for h in M.HORIZONS},
            "decomposition": M.decomposition(b), "kernel": M.kernel(cache)}


def _xy(days, horizon="10"):
    X = np.concatenate([np.asarray(d["X"], float).reshape(-1, 7) for d in days])
    y = np.concatenate([np.asarray(d["y"][horizon], float) for d in days])
    return X, y


def _score(train, held):
    X, y = _xy(train)
    sq, st = M.SqrtImpact().fit(X, y), M.StateImpact().fit(X, y)
    Xh, yh = _xy([held])
    if len(yh) < 10:
        return None
    out = {}
    for name, sel in (("all", slice(None)), ("odd", slice(1, None, 2)), ("even", slice(0, None, 2))):
        a = float(np.mean((yh[sel] - sq.predict(Xh[sel])) ** 2))
        b = float(np.mean((yh[sel] - st.predict(Xh[sel])) ** 2))
        out[name] = {"mse_sqrt": a, "mse_state": b, "effect": (a - b) / a if a > 0 else 0.0}
    out["bursts"] = int(len(yh))
    out["Y_sqrt"] = sq.Y
    return out


def run(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    t0 = time.time()
    if args.phase == "pilot":
        res = common.units(args, "explore", STREAMS, protocol["pools"]["explore"]["count"])
        datasets, train_ids, design_name = [d for s in STREAMS for d in res[s]["datasets"]], None, "explore-impact"
    else:
        design = common.design_document(args)
        datasets, train_ids, design_name = list(design["datasets"]), set(design["train_datasets"]), args.design
    out = common.start(args)
    tasks = [Task(d, "lob.v09.studies.impact:day_task", {"dataset_id": d, "use": common.use_of(protocol, d),
                                                         "design": design_name, "root": str(root)},
                  namespace=args.namespace) for d in datasets]
    days, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args))
    if args.phase == "pilot":
        rows = {}
        for i, held in enumerate(days):
            r = _score([d for j, d in enumerate(days) if j != i], held)
            if r is not None:
                rows[held["dataset"]] = r
        from .bounds import reliability
        eff = np.asarray([r["all"]["effect"] for r in rows.values()])
        caps = {c: provided(protocol, "explore", c) for c in ("incremental_l2", "trades_with_side")}
        pilot = {"H10": {"data_available": len(eff) > 0, "capabilities": caps,
                         "reliability": reliability([r["odd"]["effect"] for r in rows.values()],
                                                    [r["even"]["effect"] for r in rows.values()]),
                         "units": int(len(eff)), "effect": float(eff.mean()) if len(eff) else None,
                         "sd": float(eff.std(ddof=1)) if len(eff) > 1 else None,
                         "compute_hours_per_unit": (time.time() - t0) / 3600 / max(len(days), 1),
                         "unit_values": eff.tolist()}}
    else:
        train = [d for d in days if d["dataset"] in train_ids]
        rows = {d["dataset"]: _score(train, d) for d in days if d["dataset"] not in train_ids}
        rows = {k: v for k, v in rows.items() if v is not None}
        pilot = None
    cross = {}
    by_date = {}
    for d in days:
        by_date.setdefault(d["dataset"][-10:], {})[d["dataset"].split("-")[2]] = d["dataset"]
    result = {"phase": args.phase, "units": rows, "pilot": pilot,
              "decomposition": {d["dataset"]: d["decomposition"] for d in days},
              "kernel": {d["dataset"]: d["kernel"] for d in days},
              "causal": causal.registry_report(), "cross_impact_note": "computed in cross-impact-v09",
              "label": "observational, ASSUMPTION_DEPENDENT; the flow-impact effect is NOT_IDENTIFIED causally",
              "runner": info}
    del cross, by_date
    return common.finish(args, out, analysis=ANALYSIS, config={"phase": args.phase, "datasets": datasets,
                                                               "quantile": M.QUANTILE, "horizons": M.HORIZONS},
                         result=result, datasets=datasets)
