"""Exact-estimator replication studies of v0.9 (H3-H12, H45, H46, H48) on ledgered v1 caches.

Every function below runs the unchanged v0.9 task code through ``v09_adapter`` (only the cache loader is redirected
to ledgered v1 caches) and assembles the registered v1 unit values. Training units come from the **calibrate** pool;
evaluation units from the hypothesis's guarded pool. In pilot mode (``train=None``) the v0.9 pilot split is used on
explore units (first half of the dates trains, second half evaluates; leave-one-date-out where v0.9 used it).

Unit values (positive = in the registered direction):

* H3  fills:     Brier(generic) - Brier(execution-aware) per evaluation unit (v0.9 ``fills.evaluate``);
* H4  survival:  IBS(Kaplan-Meier) - IBS(model selected by inner validation on the training units);
* H5  decisions: mean IS(survival passive, KM) - mean IS(survival passive, calibrated) in the F3 twin;
* H9  decisions: sign agreement of the F4 consensus (corrected v1 book) with the F3 directions - 0.5;
* H6  continual: forgetting(naive) - forgetting(replay) per earlier day of each stream's chronological sequence;
* H7  conformal: ACI coverage - 0.90 (+ margin in the decision);
* C1-C3 bounds:  conservative bound over the registered assumption grid of the oriented pairwise difference.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from . import common as CM
from .v09_adapter import call

ANALYSES = {"fills": "fills-v1", "survival": "survival-v1", "continual": "continual-v1",
            "conformal": "conformal-v1", "bounds": "bounds-v1", "decisions": "decisions-v1"}


def _protocol(root: Path) -> dict:
    return json.loads((Path(root) / "configs/v1/protocol.json").read_text(encoding="utf-8"))


def _mapping(prep: dict) -> dict[str, str]:
    return {d: r["cache"] for d, r in prep.items() if r["status"] == "AVAILABLE"}


def _pool(executor, jobs: dict):
    """{key: (fn, mapping, kwargs)} -> {key: result}, executed in worker processes."""
    futs = {executor.submit(call, fn, mapping, **kw): k for k, (fn, mapping, kw) in jobs.items()}
    return {k: f.result() for f, k in futs.items()}


def _split(datasets: list[str]) -> tuple[list[str], list[str]]:
    dates = sorted({CM.date_of(d) for d in datasets})
    half = set(dates[: len(dates) // 2])
    return [d for d in datasets if CM.date_of(d) in half], [d for d in datasets if CM.date_of(d) not in half]


# ----------------------------------------------------------------------------------------------- H3 fills
def fills(ctx: CM.Context, out: Path, *, evaluate: list[str], train: list[str] | None, seed: int,
          workers: int = 2) -> dict:
    t0 = time.time()
    if train is None:
        train, evaluate = _split(evaluate)
    prep = ctx.prepare(train + evaluate, use="fills-v1 (v0.9 estimator)", workers=workers)
    mp = _mapping(prep)
    streams = sorted({CM.stream_of(d) for d in evaluate})
    with ProcessPoolExecutor(workers) as ex:
        cal = _pool(ex, {s: ("lob.v09.studies.fills:calibrate_task", mp,
                             {"stream": s, "train": [d for d in train if CM.stream_of(d) == s and d in mp],
                              "root": str(ctx.root), "design": "v1", "seed": seed}) for s in streams})
        ev = _pool(ex, {d: ("lob.v09.studies.fills:evaluate", mp,
                            {"cal": cal[CM.stream_of(d)], "day": d, "root": str(ctx.root), "design": "v1"})
                        for d in evaluate if d in mp})
    units = {d: r["effect"] for d, r in ev.items()}
    res = {"calibration": cal, "evaluation": ev, "units": {"H3": units}, "clusters": {"H3": CM.cluster_values(units)},
           "per_stream": {s: float(np.nanmean([v for d, v in units.items() if CM.stream_of(d) == s and v is not None]
                                              or [np.nan])) for s in streams},
           "compute_hours": (time.time() - t0) / 3600, "train": train, "evaluate": evaluate}
    return CM.finalize(ctx, out, analysis=ANALYSES["fills"], config={"seed": seed, "train": train,
                                                                     "evaluate": evaluate}, result=res) | {"result": res}


# ----------------------------------------------------------------------------------------------- H4 survival
def survival(ctx: CM.Context, out: Path, *, evaluate: list[str], train: list[str] | None, seed: int,
             workers: int = 2) -> dict:
    from ....v09.queue import survival as S
    from ....v09.studies import survival as SV
    t0 = time.time()
    pilot = train is None
    prep = ctx.prepare((train or []) + evaluate, use="survival-v1 (v0.9 estimator)", workers=workers)
    mp = _mapping(prep)
    with ProcessPoolExecutor(workers) as ex:
        ev = _pool(ex, {d: ("lob.v09.studies.survival:events_task", mp,
                            {"dataset_id": d, "use": "v1", "design": "v1", "root": str(ctx.root)})
                        for d in (train or []) + evaluate if d in mp})
    rows = {}
    if pilot:                                  # leave-one-date-out over the evaluation units (v0.9 lodo by date)
        days = [ev[d] for d in evaluate if d in ev]
        for held in days:
            train_days = [x for x in days if CM.date_of(x["dataset"]) != CM.date_of(held["dataset"])]
            rows[held["dataset"]] = _fit_eval(SV, S, train_days, held, seed)
    else:
        tr = [ev[d] for d in train if d in ev]
        for d in evaluate:
            if d in ev:
                rows[d] = _fit_eval(SV, S, tr, ev[d], seed)
    units = {d: r["effect"] for d, r in rows.items()}
    res = {"rows": rows, "units": {"H4": units}, "clusters": {"H4": CM.cluster_values(units)},
           "compute_hours": (time.time() - t0) / 3600}
    return CM.finalize(ctx, out, analysis=ANALYSES["survival"], config={"seed": seed, "train": train,
                                                                        "evaluate": evaluate}, result=res) | {"result": res}


def _fit_eval(SV, S, train_days: list[dict], held: dict, seed: int) -> dict:
    """The v0.9 lodo body for one held-out unit: inner-validation selection, refit on all training units."""
    cut = max(1, min(len(train_days) - 1, int(round(0.75 * len(train_days)))))
    Xa, Ta, Ea = SV._stack(train_days[:cut], seed)
    Xv, Tv, Ev = SV._stack(train_days[cut:], seed + 1)
    scores = {n: S.integrated_brier(m.fit(Xa, Ta, Ea).predict(Xv), Tv, Ev) for n, m in SV._models(seed).items()}
    best = min(scores, key=scores.get)
    X, T, E = SV._stack(train_days, seed)
    model = SV._models(seed)[best].fit(X, T, E)
    km = S.KaplanMeier().fit(X, T, E)
    ibs_km, ibs_m = SV._eval_day(held, model, km)
    return {"selected": best, "inner_ibs": scores, "ibs_km": ibs_km, "ibs_model": ibs_m, "effect": ibs_km - ibs_m}


# ----------------------------------------------------------------------------------------------- H6 continual
def continual(ctx: CM.Context, out: Path, *, sequence: list[str], seed: int, workers: int = 2,
              method: str = "replay") -> dict:
    t0 = time.time()
    prep = ctx.prepare(sequence, use="continual-v1 (v0.9 estimator)", workers=workers)
    mp = _mapping(prep)
    streams = sorted({CM.stream_of(d) for d in sequence})
    with ProcessPoolExecutor(workers) as ex:
        rows = _pool(ex, {s: ("lob.v09.studies.continual:sequence_task", mp,
                              {"days": sorted([d for d in sequence if CM.stream_of(d) == s and d in mp],
                                              key=CM.date_of), "root": str(ctx.root), "design": "v1",
                               "seed": seed}) for s in streams})
    from ....v09.studies.continual import units_of
    per = units_of(list(rows.values()))
    units = {d: (u["naive"] - u[method]) for d, u in per.items()}
    res = {"rows": rows, "per_unit": per, "method": method, "units": {"H6": units},
           "clusters": {"H6": CM.cluster_values(units)}, "compute_hours": (time.time() - t0) / 3600}
    return CM.finalize(ctx, out, analysis=ANALYSES["continual"], config={"seed": seed, "sequence": sequence,
                                                                         "method": method}, result=res) | {"result": res}


# ----------------------------------------------------------------------------------------------- H7 conformal
def conformal(ctx: CM.Context, out: Path, *, evaluate: list[str], train: list[str] | None, workers: int = 2,
              alpha_target: float = 0.90) -> dict:
    t0 = time.time()
    prep = ctx.prepare((train or []) + evaluate, use="conformal-v1 (v0.9 estimator)", workers=workers)
    mp = _mapping(prep)
    ev_ok = [d for d in evaluate if d in mp]
    if train is None:
        jobs = {d: ("lob.v09.studies.conformal:fold_task", mp,
                    {"held": d, "train": [x for x in ev_ok if CM.date_of(x) != CM.date_of(d)],
                     "root": str(ctx.root), "design": "v1"}) for d in ev_ok}
    else:
        tr = [d for d in train if d in mp]
        jobs = {d: ("lob.v09.studies.conformal:fold_task", mp,
                    {"held": d, "train": tr, "root": str(ctx.root), "design": "v1"}) for d in ev_ok}
    with ProcessPoolExecutor(workers) as ex:
        rows = _pool(ex, jobs)
    units = {d: r["methods"]["aci"]["coverage"] - alpha_target for d, r in rows.items()}
    res = {"rows": rows, "units": {"H7": units}, "clusters": {"H7": CM.cluster_values(units)},
           "compute_hours": (time.time() - t0) / 3600}
    return CM.finalize(ctx, out, analysis=ANALYSES["conformal"], config={"train": train, "evaluate": evaluate},
                       result=res) | {"result": res}


# ----------------------------------------------------------------------------------------------- C1-C3 bounds
def bounds(ctx: CM.Context, out: Path, *, evaluate: list[str], conclusions: list[str], workers: int = 2,
           hidden_grid: tuple[float, ...] | None = None, policies_extra: tuple[str, ...] = ()) -> dict:
    """Bounded replay of the registered policy/mandate combinations on every unit (v0.9 ``bounds.day_task``)."""
    from ....v09.replay import engine as E
    from ....v09.studies import bounds as BD
    t0 = time.time()
    prot = _protocol(ctx.root)
    carried = prot["carried_from_v09"]
    concl = {c: carried["conclusions"][c] for c in conclusions}
    combos = sorted({(c["treatment"], c["mandate"]) for c in concl.values()} |
                    {(c["control"], c["mandate"]) for c in concl.values()} |
                    {(p, "medium") for p in policies_extra})
    a = carried["counterfactual_assumptions"]
    grid = E.grid(latencies=tuple(a["latency_s"]), hidden=tuple(hidden_grid or a["hidden_fraction"]),
                  impacts=tuple(a["impact"]))
    keys = [g.key() for g in grid]
    notionals = {k: v["notional_usd"] for k, v in carried["mandates"]["classes"].items()}
    prep = ctx.prepare(evaluate, use="bounds-v1 (v0.9 replay engine)", workers=workers)
    mp = _mapping(prep)
    with ProcessPoolExecutor(workers) as ex:
        days = _pool(ex, {d: ("lob.v09.studies.bounds:day_task", mp,
                              {"dataset_id": d, "use": "v1", "design": "v1", "root": str(ctx.root),
                               "combos_": combos, "assumption_keys": keys, "notionals": notionals})
                          for d in evaluate if d in mp})
    units, clusters = {}, {}
    for cid, c in concl.items():
        u = {}
        for d, day in days.items():
            r = BD.pair_day(day, f"{c['treatment']}|{c['mandate']}", f"{c['control']}|{c['mandate']}",
                            c["endpoint"], c["direction"])
            if r is not None:
                u[d] = r
        units[cid] = u
        clusters[cid] = CM.cluster_values({d: r["lower"] for d, r in u.items()})
    res = {"conclusions": concl, "units": units, "clusters": clusters, "assumptions": len(keys),
           "hidden_grid": list(hidden_grid or a["hidden_fraction"]), "episodes": {d: day["episodes"]
                                                                                for d, day in days.items()},
           "rejected": {d: r["status"] for d, r in prep.items() if r["status"] != "AVAILABLE"},
           "compute_hours": (time.time() - t0) / 3600,
           "label": "BOUNDED, HYPOTHETICAL, ASSUMPTION_DEPENDENT replay; never actual fills"}
    return CM.finalize(ctx, out, analysis=ANALYSES["bounds"], config={"evaluate": evaluate, "conclusions": conclusions,
                                                                      "combos": [list(x) for x in combos]},
                       result=res) | {"result": res}


__all__ = ["fills", "survival", "continual", "conformal", "bounds", "ANALYSES"]
