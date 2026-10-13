"""hybrid-twin-v09 (M49, F3; H1 and H12): response realism of the hybrid twin, generators and ecologies.

Pilot: leave-one-explore-day-out. Per fold:

* the hybrid response model is fitted on the bursts of all training days;
* every classical family, the static ecology and the strategic ecology are fitted on the training frame of at most
  the six most recent inner-training days (registered compute cap; the same days for every family; never the
  inner-validation days) and produce probe tables (seeded);
* the best fully generative model (excluding the hybrid) is chosen by response error on an inner validation split
  (bursts of the last quarter of training days);
* on the held-out day: H1 unit = (err_generative_best - err_hybrid) / err_generative_best;
  H12 unit = (err_static_ecology - err_strategic_ecology) / err_static_ecology.

Split-half: odd vs even bursts. Errors are standardized squared errors over the six registered targets.
"""
from __future__ import annotations

import time

import numpy as np

from .. import protocol as pr
from ..generators import classical as G, hybrid as Hy
from ..generators.strategic import StrategicEcology
from ..runner import Task, run_tasks
from . import common

ANALYSIS = "hybrid-twin-v09"
GEN_DAYS = 6


def _generators(frame, seed):
    book = G.book_model(frame)
    touch = G.median_touch(frame)
    gens = {name: G.fit_family(name, frame, book) for name in list(G.FAMILIES) + list(G.STATIC_ECOLOGY)}
    strat = StrategicEcology(None, book).fit(frame)
    strat.book = book
    gens["strategic_ecology"] = strat
    return {n: G.probe_responses(g, touch=touch, seed=seed) for n, g in gens.items()}, touch


def fold_task(held: str, train: list[str], root: str, design: str, seed: int) -> dict:
    caches = {d: common.load_cache(d, root, design, "response realism") for d in sorted(train, key=lambda d: d[-10:])}
    bursts = {d: Hy.burst_targets(c) for d, c in caches.items()}
    order = list(caches)
    n_inner = max(1, min(len(order) - 1, int(round(0.75 * len(order)))))
    inner, val = order[:n_inner], order[n_inner:]
    Xtr = np.vstack([bursts[d]["X"] for d in order])
    Ytr = np.vstack([bursts[d]["Y"] for d in order])
    sd = Ytr.std(0)
    gen_days = inner[-GEN_DAYS:]                 # generators never see the inner-validation days
    frame = G.training_frame([caches[d] for d in gen_days])
    tables, touch = _generators(frame, seed)
    del caches
    Xv = np.vstack([bursts[d]["X"] for d in val])
    Yv = np.vstack([bursts[d]["Y"] for d in val])
    relv = np.concatenate([bursts[d]["rel"] for d in val])
    generative = [n for n in tables]
    val_err = {n: float(Hy.response_error(Hy.table_prediction(tables[n], relv), Yv, sd).mean()) for n in generative}
    best = min(val_err, key=val_err.get)
    hybrid = Hy.HybridResponse().fit(Xtr, Ytr)
    del Xv
    h = Hy.burst_targets(common.load_cache(held, root, design, "response realism"))
    if len(h["Y"]) < 10:
        return {"held": held, "status": "NOT_EVALUABLE", "bursts": int(len(h["Y"]))}
    err = {n: Hy.response_error(Hy.table_prediction(tables[n], h["rel"]), h["Y"], sd) for n in generative}
    err["hybrid"] = Hy.response_error(hybrid.predict(h["X"]), h["Y"], sd)
    odd = np.arange(len(h["Y"])) % 2 == 1

    def rel(a, b, sel=slice(None)):
        x, y = float(a[sel].mean()), float(b[sel].mean())
        return (x - y) / max(x, 1e-12)

    return {"held": held, "status": "OK", "bursts": int(len(h["Y"])), "selected_generative": best, "val_err": val_err,
            "err": {n: float(v.mean()) for n, v in err.items()},
            "h1": rel(err[best], err["hybrid"]), "h1_odd": rel(err[best], err["hybrid"], odd),
            "h1_even": rel(err[best], err["hybrid"], ~odd),
            "h12": rel(err["ecology_static"], err["strategic_ecology"]),
            "h12_odd": rel(err["ecology_static"], err["strategic_ecology"], odd),
            "h12_even": rel(err["ecology_static"], err["strategic_ecology"], ~odd),
            "tables": tables, "touch": touch, "generator_days": gen_days}


def run(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    t0 = time.time()
    if args.phase == "pilot":
        units = common.explore_units(args, protocol)
        folds, design_name = [(d, [x for x in units if x != d]) for d in units], "explore-hybrid"
    else:
        design = common.design_document(args)
        units, design_name = list(design["test_datasets"]), args.design
        folds = [(d, list(design["train_datasets"])) for d in units]
    out = common.start(args)
    tasks = [Task(h, "lob.v09.studies.hybrid:fold_task", {"held": h, "train": tr, "root": str(root),
                                                          "design": design_name, "seed": args.seed},
                  namespace=args.namespace) for h, tr in folds]
    rows, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args),
                           checkpoint=root / (args.out + ".checkpoint"), design={"folds": folds}, inputs=units)
    ok = [r for r in rows if r["status"] == "OK"]
    hours = (time.time() - t0) / 3600 / max(len(rows), 1)
    caps = common.caps_for(protocol, ("incremental_l2", "trades_with_side"))
    pilot = {k.upper(): common.pilot_record([r[k] for r in ok], [r[f"{k}_odd"] for r in ok],
                                            [r[f"{k}_even"] for r in ok], caps=caps, hours=hours)
             for k in ("h1", "h12")}
    result = {"phase": args.phase, "units": {r["held"]: r for r in rows},
              "pilot": pilot if args.phase == "pilot" else None, "runner": info}
    return common.finish(args, out, analysis=ANALYSIS, config={"phase": args.phase, "units": units,
                                                               "seed": args.seed, "generator_days": GEN_DAYS},
                         result=result, datasets=units)
