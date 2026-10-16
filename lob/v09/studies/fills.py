"""fill-study-v09 (M66, M68; H6): execution-aware vs generic calibration of fill (queue-clearance) probabilities.

Design (pilot): per instrument the explore days in chronological order; the first half are calibration days and the
second half evaluation units. The queue-reactive family is calibrated two ways on the calibration days:

* generic - its own maximum-likelihood fit with the ``BookModel`` moment estimates (summary/rate realism);
* execution-aware - the same family with the book refill depth, cancel fraction and market-size scale chosen from a
  registered 24-point seeded random search to minimize the Brier score of P(back-of-queue clearance within 5 s |
  queue-size tercile) against the calibration days' observed clearance events.

Each calibrated generator simulates 8 windows; its predicted clearance probability per queue tercile is compared with
the evaluation day's observed clearance events (same terciles, from the calibration days). Unit endpoint:
Brier(generic) - Brier(execution-aware) (positive favours execution-aware calibration). Split-half: odd vs even
windows of the evaluation day. A tercile a generator never visits is not estimable (None); only probes whose
tercile both calibrations predict are scored, and a unit (or half) with no such probe has effect None and is excluded
by the pilot record. Confirm: calibration on all explore days, evaluation on the design's confirm days.
"""
from __future__ import annotations

import time

import numpy as np

from .. import protocol as pr
from ..generators import classical as G
from ..queue import clearance as C
from ..realism.benchmark import _clearance
from ..runner import Task, run_tasks
from . import common

ANALYSIS = "fill-study-v09"
SEARCH = 24


def _observed(cache, cuts):
    ev = C.events(cache)
    q = ev["X"][:, 0]
    y = (ev["T"] <= 5.0) & ev["E"]
    tercile = np.digitize(q, cuts)
    window = (np.searchsorted(np.asarray(C.WINDOW_STARTS_S, float), ev["t"], side="right") - 1)
    return tercile, y.astype(float), window


def _predicted(g, cuts, seed):
    sims = [g.simulate(3300, seed + k, warmup_bins=600) for k in range(8)]
    out = np.zeros(3)
    for t in range(3):
        vals = []
        for fr in sims:
            q = np.log1p(fr.depth[:, 0, 0].astype(float))
            keep = np.digitize(q, cuts) == t
            if keep.sum() < 50:
                continue
            sub = fr.slice(0, len(fr))
            sub.valid = sub.valid & keep
            vals.append(_clearance(sub, horizons=(5.0,))[0])
        out[t] = float(np.mean(vals)) if vals else float("nan")
    return out


def _probs(p) -> list:
    """Tercile probabilities for the sealed result: a tercile the generator never visits is None (not estimable)."""
    return [float(v) if np.isfinite(v) else None for v in np.asarray(p, float)]


def _mean(x):
    return float(x.mean()) if x.size else None


def score(pg, pe, tercile, y, window) -> dict:
    """Brier comparison on the probes whose tercile both calibrations predict; empty subsets are None."""
    pg, pe = np.asarray(pg, float), np.asarray(pe, float)
    ok = np.isfinite(pg[tercile]) & np.isfinite(pe[tercile])
    a, b = (pg[tercile] - y) ** 2, (pe[tercile] - y) ** 2
    odd = window % 2 == 1
    ga, eb = _mean(a[ok]), _mean(b[ok])
    oa, ob, ea, ebv = _mean(a[ok & odd]), _mean(b[ok & odd]), _mean(a[ok & ~odd]), _mean(b[ok & ~odd])
    return {"brier_generic": ga, "brier_exec": eb, "effect": None if ga is None else ga - eb,
            "odd": None if oa is None else oa - ob, "even": None if ea is None else ea - ebv, "probes": int(ok.sum())}


def calibrate_task(stream: str, train: list[str], root: str, design: str, seed: int) -> dict:
    caches = [common.load_cache(d, root, design, "fill calibration") for d in train]
    allq = np.concatenate([C.events(c)["X"][:, 0] for c in caches])
    cuts = np.quantile(allq, [1 / 3, 2 / 3])
    obs = [_observed(c, cuts) for c in caches]
    frame = G.training_frame(caches)
    del caches
    book = G.book_model(frame)
    generic = G.fit_family("queue_reactive", frame, book)
    p_generic = _predicted(generic, cuts, seed)
    rng = np.random.default_rng(np.random.SeedSequence([9, 66, seed]))
    best = None
    T = np.concatenate([o[0] for o in obs])
    Y = np.concatenate([o[1] for o in obs])
    for k in range(SEARCH):
        mult = np.exp(rng.uniform(np.log(0.25), np.log(4.0), 3))
        b = book.replace(refill_mean=max(1.0, book.refill_mean * mult[0]),
                         cancel_frac_mean=float(np.clip(book.cancel_frac_mean * mult[1], 0.01, 1.0)),
                         market_size_mean=max(1.0, book.market_size_mean * mult[2]))
        g = G.fit_family("queue_reactive", frame, b)
        p = _predicted(g, cuts, seed + 100 * (k + 1))
        if not np.isfinite(p).all():
            continue
        brier = float(np.mean((p[T] - Y) ** 2))
        if best is None or brier < best[0]:
            best = (brier, p, b.to_dict())
    return {"stream": stream, "cuts": cuts.tolist(), "p_generic": _probs(p_generic),
            "p_exec": _probs(best[1]) if best else None, "exec_book": best[2] if best else None,
            "train_brier_exec": best[0] if best else None}


def evaluate(cal: dict, day: str, root: str, design: str) -> dict:
    if cal["p_exec"] is None:
        return {"brier_generic": None, "brier_exec": None, "effect": None, "odd": None, "even": None, "probes": 0}
    tercile, y, window = _observed(common.load_cache(day, root, design, "fill evaluation"), np.asarray(cal["cuts"]))
    return score([np.nan if v is None else v for v in cal["p_generic"]],
                 [np.nan if v is None else v for v in cal["p_exec"]], tercile, y, window)


def run(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    t0 = time.time()
    if args.phase == "pilot":
        res = common.units(args, "explore", common.BYBIT_STREAMS, protocol["pools"]["explore"]["count"])
        plan = {}
        for s in common.BYBIT_STREAMS:
            days = sorted(res[s]["datasets"], key=lambda d: d[-10:])
            plan[s] = (days[: len(days) // 2], days[len(days) // 2:])
        design_name = "explore-fills"
    else:
        design = common.design_document(args)
        plan = {s: (list(v["train"]), list(v["test"])) for s, v in design["plan"].items()}
        design_name = args.design
    out = common.start(args)
    tasks = [Task(s, "lob.v09.studies.fills:calibrate_task", {"stream": s, "train": tr, "root": str(root),
                                                              "design": design_name, "seed": args.seed},
                  namespace=args.namespace) for s, (tr, _) in plan.items()]
    cals, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args))
    by = {c["stream"]: c for c in cals}
    units = {}
    for s, (_, te) in plan.items():
        for d in te:
            units[d] = evaluate(by[s], d, str(root), design_name)
    hours = (time.time() - t0) / 3600 / max(len(units), 1)
    caps = common.caps_for(protocol, ("incremental_l2", "trades_with_side", "queue_clearance_observable"))
    vals = list(units.values())
    pilot = {"H6": common.pilot_record([u["effect"] for u in vals], [u["odd"] for u in vals],
                                       [u["even"] for u in vals], caps=caps, hours=hours)}
    result = {"phase": args.phase, "plan": plan, "calibrations": by, "units": units,
              "pilot": pilot if args.phase == "pilot" else None, "records": pilot, "runner": info,
              "label": "BOUNDED clearance target; generator probabilities are SYNTHETIC predictions of real events"}
    return common.finish(args, out, analysis=ANALYSIS, config={"plan": plan, "search": SEARCH, "seed": args.seed},
                         result=result, datasets=[d for tr, te in plan.values() for d in tr + te])
