"""conformal-study-v09 (H17): sequential conformal coverage of 60 s-ahead realized volatility under drift.

Pilot: leave-one-explore-day-out (base predictor and training scores from the other days). Confirm: base predictor
and scores from all explore days, evaluated on every confirm day. Unit endpoint: ACI coverage - 0.90 (non-inferiority,
registered margin 0.02). All five methods and their widths, conditional and post-shock coverage are reported.
"""
from __future__ import annotations

import time

import numpy as np

from .. import protocol as pr
from ..runner import Task, run_tasks
from ..uncertainty import conformal as C
from . import common

ANALYSIS = "conformal-study-v09"


def fold_task(held: str, train: list[str], root: str, design: str) -> dict:
    tr = [C.samples(common.load_cache(d, root, design, "conformal samples")) for d in train]
    X = np.vstack([s["X"] for s in tr])
    y = np.concatenate([s["y"] for s in tr])
    coef = C.fit_base(X, y)
    scores = np.abs(y - C.predict(coef, X))
    test = C.samples(common.load_cache(held, root, design, "conformal samples"))
    return {"held": held, "methods": C.run_day(coef, scores, X[:, 1], test), "coef": coef.tolist()}


def run(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    t0 = time.time()
    if args.phase == "pilot":
        units = common.explore_units(args, protocol)
        folds, design_name = [(d, [x for x in units if x != d]) for d in units], "explore-conformal"
    else:
        design = common.design_document(args)
        units, design_name = list(design["test_datasets"]), args.design
        folds = [(d, list(design["train_datasets"])) for d in units]
    out = common.start(args)
    tasks = [Task(h, "lob.v09.studies.conformal:fold_task", {"held": h, "train": tr, "root": str(root),
                                                             "design": design_name}, namespace=args.namespace)
             for h, tr in folds]
    rows, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args))
    hours = (time.time() - t0) / 3600 / max(len(rows), 1)
    caps = common.caps_for(protocol, ("incremental_l2", "trades_with_side"))
    aci = [r["methods"]["aci"] for r in rows]
    pilot = {"H17": common.pilot_record([a["coverage"] - (1 - C.ALPHA) for a in aci],
                                        [a["odd"] - (1 - C.ALPHA) for a in aci],
                                        [a["even"] - (1 - C.ALPHA) for a in aci], caps=caps, hours=hours)}
    result = {"phase": args.phase, "units": {r["held"]: r for r in rows},
              "pilot": pilot if args.phase == "pilot" else None, "alpha": C.ALPHA, "gamma": C.GAMMA, "runner": info}
    return common.finish(args, out, analysis=ANALYSIS, config={"phase": args.phase, "units": units}, result=result,
                         datasets=units)
