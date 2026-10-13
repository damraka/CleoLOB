"""survival-study-v09 (M67, M68; H7): queue-survival models for the back-of-queue clearance time.

Pilot (explore days only): leave-one-day-out. For each held-out day the registered model set is fitted on the other
days; the model with the lowest integrated Brier score on an inner validation split (the last quarter of the training
days in registered order) is refitted on all training days and evaluated on the held-out day. The unit endpoint is
IBS(Kaplan-Meier) - IBS(selected model), positive = the survival model predicts time to clearance better. Split-half
reliability uses odd vs even windows of each held-out day.

Confirm: the model family selected most often in the pilot (ties -> lower mean pilot IBS) is fitted once on all
explore days (sealed in the confirm design) and evaluated on every confirm day.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .. import protocol as pr
from ..capabilities import provided
from ..queue import clearance as C, survival as S
from ..runner import Task, run_tasks
from . import common

ANALYSIS = "survival-study-v09"
MAX_TRAIN = 60_000
STREAMS = ["bybit-linear:BTCUSDT", "bybit-linear:ETHUSDT"]


def events_task(dataset_id: str, use: str, design: str, root: str) -> dict:
    from ..data import store
    cache, _ = store.prepare(dataset_id, use=use, design=design, root=Path(root), purpose="queue-survival events")
    ev = C.events(cache)
    win = (np.searchsorted(np.asarray([w for w in C.WINDOW_STARTS_S], float), ev["t"], side="right") - 1)
    return {"dataset": dataset_id, "X": ev["X"].tolist(), "T": ev["T"].tolist(), "E": ev["E"].tolist(),
            "window": win.tolist(), "confidence": C.queue_confidence(cache)}


def _stack(days: list[dict], seed: int, cap: int = MAX_TRAIN):
    X = np.concatenate([np.asarray(d["X"]) for d in days])
    T = np.concatenate([np.asarray(d["T"]) for d in days])
    E = np.concatenate([np.asarray(d["E"], bool) for d in days])
    if len(T) > cap:
        keep = np.sort(np.random.default_rng(seed).choice(len(T), cap, replace=False))
        X, T, E = X[keep], T[keep], E[keep]
    return X, T, E


def _models(seed: int) -> dict:
    out = {"cox": S.CoxPH(seed=seed), "weibull_aft": S.WeibullAFT(), "boosted_hazard": S.BoostedHazard(seed=seed)}
    if provided({}, None, "neural_backend"):
        out["neural_hazard"] = S.NeuralHazard(seed=seed)
    return out


def _eval_day(day: dict, model, km, windows=None) -> tuple[float, float]:
    X, T, E = np.asarray(day["X"]), np.asarray(day["T"]), np.asarray(day["E"], bool)
    if windows is not None:
        sel = np.isin(np.asarray(day["window"]), windows)
        X, T, E = X[sel], T[sel], E[sel]
    return S.integrated_brier(km.predict(X), T, E), S.integrated_brier(model.predict(X), T, E)


def lodo(days: list[dict], seed: int) -> dict:
    rows = {}
    for i, held in enumerate(days):
        train = [d for j, d in enumerate(days) if j != i]
        cut = max(1, int(round(0.75 * len(train))))
        inner_tr, inner_va = train[:cut], train[cut:]
        Xa, Ta, Ea = _stack(inner_tr, seed)
        Xv, Tv, Ev = _stack(inner_va, seed + 1)
        scores = {}
        for name, m in _models(seed).items():
            scores[name] = S.integrated_brier(m.fit(Xa, Ta, Ea).predict(Xv), Tv, Ev)
        best = min(scores, key=scores.get)
        X, T, E = _stack(train, seed)
        model = _models(seed)[best].fit(X, T, E)
        km = S.KaplanMeier().fit(X, T, E)
        ibs_km, ibs_m = _eval_day(held, model, km)
        odd = _eval_day(held, model, km, windows=list(range(1, 24, 2)))
        even = _eval_day(held, model, km, windows=list(range(0, 24, 2)))
        Xh, Th, Eh = (np.asarray(held[k]) for k in ("X", "T", "E"))
        full = S.evaluate(model.predict(Xh), Th, Eh.astype(bool))
        rows[held["dataset"]] = {"selected": best, "inner_ibs": scores, "ibs_km": ibs_km, "ibs_model": ibs_m,
                                 "effect": ibs_km - ibs_m, "odd": odd[0] - odd[1], "even": even[0] - even[1],
                                 "c_index": full["c_index"], "ece_5s": full["ece_5s"], "brier": full["brier"]}
    return rows


def run(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    t0 = time.time()
    if args.phase == "pilot":
        res = common.units(args, "explore", STREAMS, protocol["pools"]["explore"]["count"])
        datasets = [d for s in STREAMS for d in res[s]["datasets"]]
        design_name, train_ids = "explore-survival", None
    else:
        design = common.design_document(args)
        datasets, design_name, train_ids = list(design["datasets"]), args.design, list(design["train_datasets"])
    out = common.start(args)
    tasks = [Task(d, "lob.v09.studies.survival:events_task", {"dataset_id": d, "use": common.use_of(protocol, d), "design": design_name,
                                                              "root": str(root)}, namespace=args.namespace)
             for d in datasets]
    days, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args))
    if args.phase == "pilot":
        rows = lodo(days, args.seed)
        from .bounds import reliability
        eff = np.asarray([r["effect"] for r in rows.values()])
        counts = {}
        for r in rows.values():
            counts[r["selected"]] = counts.get(r["selected"], 0) + 1
        chosen = sorted(counts, key=lambda k: (-counts[k], np.mean([r["ibs_model"] for r in rows.values()
                                                                    if r["selected"] == k])))[0]
        hours = (time.time() - t0) / 3600 / max(len(days), 1)
        caps = {c: provided(protocol, "explore", c) for c in ("incremental_l2", "trades_with_side",
                                                              "queue_clearance_observable")}
        pilot = {"H7": {"data_available": len(rows) > 0, "capabilities": caps,
                        "reliability": reliability([r["odd"] for r in rows.values()],
                                                   [r["even"] for r in rows.values()]),
                        "units": len(eff), "effect": float(eff.mean()), "sd": float(eff.std(ddof=1)),
                        "compute_hours_per_unit": hours, "unit_values": eff.tolist(), "selected_model": chosen}}
        result = {"phase": "pilot", "units": rows, "selected_model": chosen, "pilot": pilot,
                  "confidence": {d["dataset"]: d["confidence"] for d in days},
                  "label": "BOUNDED survival target (back-of-queue clearance), not an actual fill"}
    else:
        train = [d for d in days if d["dataset"] in set(train_ids)]
        test = [d for d in days if d["dataset"] not in set(train_ids)]
        X, T, E = _stack(train, args.seed)
        model = _models(args.seed)[design["model"]].fit(X, T, E)
        km = S.KaplanMeier().fit(X, T, E)
        rows = {}
        for d in test:
            a, b = _eval_day(d, model, km)
            rows[d["dataset"]] = {"ibs_km": a, "ibs_model": b, "effect": a - b}
        result = {"phase": "confirm", "model": design["model"], "units": rows,
                  "label": "BOUNDED survival target (back-of-queue clearance), not an actual fill"}
    result["runner"] = info
    return common.finish(args, out, analysis=ANALYSIS, config={"phase": args.phase, "datasets": datasets,
                                                               "seed": args.seed, "max_train": MAX_TRAIN},
                         result=result, datasets=datasets)
