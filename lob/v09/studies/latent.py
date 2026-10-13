"""latent-dynamics-v09 (M47, M52; H4): path realism of latent-dynamics vs classical path generators.

Pilot: leave-one-explore-day-out. Per fold, the best classical generator (iid bootstrap, regime switching) and the
best latent generator (latent activity SSM, neural latent AR) are chosen by signature MMD on an inner validation split
(last quarter of training days), refitted on all training paths, and each samples as many 60 s paths as the held-out
day has (registered seeds). MMD = sqrt(max(MMD^2, 0)) of level-3 signatures against the held-out real paths (scale =
training sd of the 60 s mid move). Unit endpoint: (MMD_classical - MMD_latent) / max(MMD_classical, 1e-6).
Split-half: paths from odd vs even windows. The real-vs-real reference (held-out day vs training paths) is reported.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .. import protocol as pr
from ..generators import latent as L
from ..realism import path as P
from ..runner import Task, run_tasks
from . import common

ANALYSIS = "latent-dynamics-v09"


def _paths(ids, root, design):
    return [P.real_paths(common.load_cache(d, root, design, "path realism")) for d in ids]


def _mmd(real_sig, gen_paths, scale):
    return math.sqrt(max(P.mmd2(real_sig, P.signature(gen_paths, scale)), 0.0))


def fold_task(held: str, train: list[str], root: str, design: str, seed: int) -> dict:
    parts = _paths(train, root, design)
    allp = np.concatenate(parts)
    scale = float(np.std(allp[:, -1, 0])) or 1.0
    n_inner = max(1, min(len(parts) - 1, int(round(0.75 * len(parts)))))
    inner, val = np.concatenate(parts[:n_inner]), np.concatenate(parts[n_inner:])
    val_sig = P.signature(val, scale)
    val_mmd = {}
    for name, M in {**L.CLASSICAL, **L.LATENT}.items():
        val_mmd[name] = _mmd(val_sig, M().fit(inner).sample(len(val), seed), scale)
    best_c = min(L.CLASSICAL, key=val_mmd.get)
    best_l = min(L.LATENT, key=val_mmd.get)
    held_p = _paths([held], root, design)[0]
    sig = P.signature(held_p, scale)
    gc = {"cls": L.CLASSICAL[best_c]().fit(allp), "lat": L.LATENT[best_l]().fit(allp)}
    samples = {k: g.sample(len(held_p), seed + 101) for k, g in gc.items()}
    win = np.arange(len(held_p))
    odd = (win // 9) % 2 == 1

    def eff(sel):
        mc = _mmd(P.signature(held_p[sel], scale), samples["cls"][sel], scale)
        ml = _mmd(P.signature(held_p[sel], scale), samples["lat"][sel], scale)
        return (mc - ml) / max(mc, 1e-6), mc, ml

    e, mc, ml = eff(np.ones(len(held_p), bool))
    return {"held": held, "selected_classical": best_c, "selected_latent": best_l, "val_mmd": val_mmd,
            "mmd_classical": mc, "mmd_latent": ml, "effect": e, "odd": eff(odd)[0], "even": eff(~odd)[0],
            "real_vs_real": math.sqrt(max(P.mmd2(sig, P.signature(allp[:: max(1, len(allp) // 400)], scale)), 0.0)),
            "paths": int(len(held_p))}


def run(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    t0 = time.time()
    if args.phase == "pilot":
        units = common.explore_units(args, protocol)
        folds, design_name = [(d, [x for x in units if x != d]) for d in units], "explore-latent"
    else:
        design = common.design_document(args)
        units, design_name = list(design["test_datasets"]), args.design
        folds = [(d, list(design["train_datasets"])) for d in units]
    out = common.start(args)
    tasks = [Task(h, "lob.v09.studies.latent:fold_task", {"held": h, "train": tr, "root": str(root),
                                                          "design": design_name, "seed": args.seed},
                  namespace=args.namespace) for h, tr in folds]
    rows, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args),
                           checkpoint=root / (args.out + ".checkpoint"), design={"folds": folds}, inputs=units)
    hours = (time.time() - t0) / 3600 / max(len(rows), 1)
    caps = common.caps_for(protocol, ("incremental_l2", "trades_with_side", "neural_backend"))
    pilot = {"H4": common.pilot_record([r["effect"] for r in rows], [r["odd"] for r in rows],
                                       [r["even"] for r in rows], caps=caps, hours=hours)}
    result = {"phase": args.phase, "units": {r["held"]: r for r in rows}, "neural_sde": L.NEURAL_SDE_STATUS,
              "pilot": pilot if args.phase == "pilot" else None, "runner": info}
    return common.finish(args, out, analysis=ANALYSIS, config={"phase": args.phase, "units": units,
                                                               "seed": args.seed}, result=result, datasets=units)
