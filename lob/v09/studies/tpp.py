"""neural-point-process-v09 (M45, M17/M37; H2 and H5): marked TPPs of aggressive orders on real days.

Pilot (explore days): leave-one-day-out over the Bybit BTCUSDT and ETHUSDT explore days. In each fold the size-class
cuts are computed on the training days only; the best classical model (marked Poisson, marked Hawkes) and the best
neural model (RMTPP, Transformer TPP) are chosen by NLL on an inner validation split (the last quarter of the training
days), refitted on all training days and scored on every aggressive order of the held-out day.

* H2 unit endpoint: NLL(selected classical) - NLL(selected neural), nats per event (positive favours neural);
* H5 unit endpoint: NLL(single model selected by inner validation) - NLL(fixed-share dynamic model average over all
  four models, share 0.01, uniform start), nats per event (positive favours averaging).

Split-half reliability: odd vs even hourly segments of the held-out day. Confirm: models are fitted once on all
explore days and scored on each confirm day with the same rules.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np

from .. import protocol as pr
from ..capabilities import provided
from ..generators import point_process as PP
from ..runner import Task, run_tasks
from . import common

ANALYSIS = "neural-point-process-v09"
STREAMS = ["bybit-linear:BTCUSDT", "bybit-linear:ETHUSDT"]
SHARE = 0.01


def _events(dataset_id: str, root: str, protocol: dict, design: str) -> dict:
    from ..data import store
    cache, _ = store.prepare(dataset_id, use=common.use_of(protocol, dataset_id), design=design, root=Path(root),
                             purpose="aggressive-order events")
    return PP.aggressive_events(cache)


def dma(nlls: np.ndarray, share: float = SHARE) -> np.ndarray:
    """Fixed-share dynamic model averaging: per-event NLL of the online mixture (rows = models)."""
    k, n = nlls.shape
    logw = np.full(k, -math.log(k))
    out = np.empty(n)
    for i in range(n):
        ll = -nlls[:, i]
        mix = np.logaddexp.reduce(logw + ll)
        out[i] = -mix
        post = logw + ll - mix
        w = (1 - share) * np.exp(post) + share / k
        logw = np.log(w)
    return out


def _fit_select(train_ev: list[dict], seed: int):
    cuts = PP.size_cuts(train_ev)
    n_inner = max(1, int(round(0.75 * len(train_ev))))
    inner = [s for e in train_ev[:n_inner] for s in PP.segments(e, cuts)]
    val = [s for e in train_ev[n_inner:] for s in PP.segments(e, cuts)]
    full = [s for e in train_ev for s in PP.segments(e, cuts)]
    models = {**PP.CLASSICAL, **(PP.NEURAL if provided({}, None, "neural_backend") else {})}
    val_nll = {name: float(cls(**({"seed": seed} if name in PP.NEURAL else {})).fit(inner).nll(val).mean())
               for name, cls in models.items()}
    fitted = {name: cls(**({"seed": seed} if name in PP.NEURAL else {})).fit(full) for name, cls in models.items()}
    return cuts, val_nll, fitted


def fold_task(held: str, train: list[str], root: str, design: str, seed: int) -> dict:
    protocol = pr.load_protocol(Path(root) / pr.PROTOCOL_PATH)
    train_ev = [_events(d, root, protocol, design) for d in train]
    held_ev = _events(held, root, protocol, design)
    cuts, val_nll, fitted = _fit_select(train_ev, seed)
    segs = PP.segments(held_ev, cuts)
    nll = {name: m.nll(segs) for name, m in fitted.items()}
    classical = min((n for n in fitted if n in PP.CLASSICAL), key=val_nll.get)
    neural = [n for n in fitted if n in PP.NEURAL]
    best_neural = min(neural, key=val_nll.get) if neural else None
    single = min(val_nll, key=val_nll.get)
    names = sorted(nll)
    mix = dma(np.vstack([nll[n] for n in names]))
    seg_id = np.concatenate([np.full(len(t) - 1, k) for k, (t, _) in enumerate(segs)])
    odd, even = seg_id % 2 == 1, seg_id % 2 == 0

    def eff(a, b, sel=slice(None)):
        return float(np.mean(a[sel]) - np.mean(b[sel]))

    out = {"held": held, "events": int(len(mix)), "val_nll": val_nll, "nll": {n: float(v.mean()) for n, v in nll.items()},
           "selected_classical": classical, "selected_neural": best_neural, "selected_single": single,
           "dma_nll": float(mix.mean()),
           "h5": eff(nll[single], mix), "h5_odd": eff(nll[single], mix, odd), "h5_even": eff(nll[single], mix, even)}
    if best_neural:
        out.update({"h2": eff(nll[classical], nll[best_neural]), "h2_odd": eff(nll[classical], nll[best_neural], odd),
                    "h2_even": eff(nll[classical], nll[best_neural], even)})
    out["fit_info"] = {n: getattr(m, "fit_info", {}) for n, m in fitted.items()}
    return out


def run(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    t0 = time.time()
    if args.phase == "pilot":
        res = common.units(args, "explore", STREAMS, protocol["pools"]["explore"]["count"])
        units = [d for s in STREAMS for d in res[s]["datasets"]]
        folds = [(d, [x for x in units if x != d]) for d in units]
        design_name = "explore-tpp"
    else:
        design = common.design_document(args)
        units, design_name = list(design["test_datasets"]), args.design
        folds = [(d, list(design["train_datasets"])) for d in units]
    out = common.start(args)
    tasks = [Task(h, "lob.v09.studies.tpp:fold_task", {"held": h, "train": tr, "root": str(root),
                                                       "design": design_name, "seed": args.seed},
                  namespace=args.namespace) for h, tr in folds]
    rows, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args),
                           checkpoint=root / (args.out + ".checkpoint"), design={"folds": folds}, inputs=units)
    hours = (time.time() - t0) / 3600 / max(len(rows), 1)
    caps = {c: provided(protocol, "explore", c) for c in ("incremental_l2", "trades_with_side", "neural_backend")}
    from .bounds import reliability

    def record(key):
        vals = np.asarray([r[key] for r in rows if key in r], float)
        return {"data_available": len(vals) > 0, "capabilities": caps,
                "reliability": reliability([r[f"{key}_odd"] for r in rows if key in r],
                                           [r[f"{key}_even"] for r in rows if key in r]),
                "units": int(len(vals)), "effect": float(vals.mean()) if len(vals) else None,
                "sd": float(vals.std(ddof=1)) if len(vals) > 1 else None, "compute_hours_per_unit": hours,
                "unit_values": vals.tolist()}

    result = {"phase": args.phase, "units": {r["held"]: r for r in rows},
              "pilot": {"H2": record("h2"), "H5": record("h5")} if args.phase == "pilot" else None,
              "runner": info}
    return common.finish(args, out, analysis=ANALYSIS, config={"phase": args.phase, "units": units, "seed": args.seed,
                                                               "share": SHARE}, result=result, datasets=units)
