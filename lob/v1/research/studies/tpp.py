"""v1 marked temporal point-process study (H1, H2 replications; H21, H22, H26).

Unit: one held-out dataset (stream-date); the registered cluster is its calendar date.

For every held-out unit the registered models are fitted on the training units with the unchanged v0.9 rule
(``lob.v09.studies.tpp``): size-class cuts from the training units only; the best classical and the best neural
model are chosen by NLL on an inner validation split (last quarter of the training units in registered order), all
models are refitted on all training units and scored on every aggressive order of the held-out unit.

* H1 (v0.9 H2): NLL(selected classical of {marked Poisson, marked Hawkes}) - NLL(selected neural of {RMTPP,
  Transformer TPP});
* H2 (v0.9 H5): NLL(single model selected by inner validation, over those four) - NLL(fixed-share DMA, share 0.01);
* H21: NLL(unmarked Hawkes + i.i.d. marks) - NLL(marked Hawkes);
* H22: NLL(selected classical of {marked Poisson, unmarked Hawkes, marked Hawkes}) - NLL(selected neural);
* H26: H2's contrast on out-of-time (replicate-pool) units with models fitted on the calibrate pool.

Training scopes: ``pooled`` (all training streams together - the v0.9 rule, used for the replications) or
``per_stream``. Pilot folds leave out the whole calendar date (every stream of it), so no same-date information
reaches the training set (the v0.9 pilot left out one stream-day only).
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from ....v09.generators import point_process as PP
from ....v09.studies.tpp import SHARE, dma
from ...generators.point_process.hawkes import UnmarkedHawkes
from . import common as CM

ANALYSIS = "tpp-v1"
V09_CLASSICAL = ("marked_poisson", "marked_hawkes")
NEURAL = ("rmtpp", "transformer_tpp")


def _neural_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _events(path: str) -> dict:
    from ...data import daycache as DC
    return PP.aggressive_events(DC.load(Path(path)))


def _model(name: str, seed: int):
    if name == "unmarked_hawkes":
        return UnmarkedHawkes()
    if name in PP.NEURAL:
        return PP.NEURAL[name](seed=seed)
    return PP.CLASSICAL[name]()


def fold(held: str, held_path: str, train_paths: list[str], seed: int) -> dict:
    """One held-out unit: fit, select and score with the v0.9 rule plus the v1 contrasts."""
    from ...runtime import cap_threads
    cap_threads()
    t0 = time.time()
    train_ev = [_events(p) for p in train_paths]
    held_ev = _events(held_path)
    cuts = PP.size_cuts(train_ev)
    n_inner = max(1, min(len(train_ev) - 1, int(round(0.75 * len(train_ev)))))
    inner = [s for e in train_ev[:n_inner] for s in PP.segments(e, cuts)]
    val = [s for e in train_ev[n_inner:] for s in PP.segments(e, cuts)]
    full = [s for e in train_ev for s in PP.segments(e, cuts)]
    names = list(V09_CLASSICAL) + ["unmarked_hawkes"] + (list(NEURAL) if _neural_available() else [])
    val_nll = {n: float(_model(n, seed).fit(inner).nll(val).mean()) for n in names}
    fitted = {n: _model(n, seed).fit(full) for n in names}
    segs = PP.segments(held_ev, cuts)
    nll = {n: m.nll(segs) for n, m in fitted.items()}
    neural = [n for n in NEURAL if n in fitted]
    v09_models = [n for n in names if n != "unmarked_hawkes"]
    sel_cls_v09 = min(V09_CLASSICAL, key=val_nll.get)
    sel_cls_v1 = min(("marked_poisson", "unmarked_hawkes", "marked_hawkes"), key=val_nll.get)
    sel_neural = min(neural, key=val_nll.get) if neural else None
    single = min(v09_models, key=val_nll.get)
    mix = dma(np.vstack([nll[n] for n in sorted(v09_models)]), SHARE)
    mean = {n: float(v.mean()) for n, v in nll.items()}
    out = {"held": held, "events": int(len(mix)), "val_nll": val_nll, "nll": mean, "dma_nll": float(mix.mean()),
           "selected_classical_v09": sel_cls_v09, "selected_classical_v1": sel_cls_v1,
           "selected_neural": sel_neural, "selected_single": single,
           "H2": mean[single] - float(mix.mean()),
           "H21": mean["unmarked_hawkes"] - mean["marked_hawkes"],
           "fit_info": {n: getattr(m, "fit_info", {}) for n, m in fitted.items()},
           "seconds": time.time() - t0}
    if sel_neural:
        out["H1"] = mean[sel_cls_v09] - mean[sel_neural]
        out["H22"] = mean[sel_cls_v1] - mean[sel_neural]
    out["H26"] = out["H2"]
    return out


def folds_pilot(datasets: list[str], scope: str) -> list[tuple[str, list[str]]]:
    out = []
    for d in datasets:
        others = [x for x in datasets if CM.date_of(x) != CM.date_of(d)]
        if scope == "per_stream":
            others = [x for x in others if CM.stream_of(x) == CM.stream_of(d)]
        out.append((d, others))
    return out


def run(ctx: CM.Context, out: Path, *, held: list[str], train: list[str] | None, scope: str, seed: int,
        workers: int = 2, hypotheses: tuple[str, ...]) -> dict:
    """``train=None``: leave-one-date-out over ``held`` (pilot); otherwise fixed training units (confirm)."""
    from concurrent.futures import ProcessPoolExecutor
    every = sorted(set(held) | set(train or []), key=lambda d: (held + (train or [])).index(d))
    prep = ctx.prepare(every, use=f"{ANALYSIS} aggressive-order events", workers=workers)
    ok = {d for d, r in prep.items() if r["status"] == "AVAILABLE"}
    rejected = {d: r["status"] for d, r in prep.items() if r["status"] != "AVAILABLE"}
    held_ok = [d for d in held if d in ok]
    if train is None:
        folds = folds_pilot(held_ok, scope)
    else:
        tr = [d for d in train if d in ok]
        folds = [(d, [x for x in tr if scope == "pooled" or CM.stream_of(x) == CM.stream_of(d)]) for d in held_ok]
    rows = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fold, d, prep[d]["cache"], [prep[x]["cache"] for x in tr_], seed): d for d, tr_ in folds}
        for f, d in futs.items():
            rows[d] = f.result()
    hours = sum(r["seconds"] for r in rows.values()) / 3600
    units = {h: {d: rows[d].get(h) for d in rows} for h in hypotheses}
    result = {"analysis": ANALYSIS, "scope": scope, "rows": rows, "units": units,
              "clusters": {h: CM.cluster_values(u) for h, u in units.items()}, "rejected": rejected,
              "folds": {d: tr_ for d, tr_ in folds}, "compute_hours": hours}
    info = CM.finalize(ctx, out, analysis=ANALYSIS, config={"held": held, "train": train, "scope": scope,
                                                            "seed": seed, "hypotheses": list(hypotheses),
                                                            "share": SHARE}, result=result)
    return {**info, "result": result}


__all__ = ["run", "fold", "folds_pilot", "ANALYSIS"]
