"""counterfactual-bounds-v09 (M28, M78; F2): bounded historical execution of the registered policy set.

For every unit (instrument-day) and registered (policy, mandate) combination, every registered window is replayed
under the registered assumption grid (queue x latency x hidden fraction x impact). Policies that never rest passive
orders are invariant to the queue rule and the hidden fraction (tested), so they run once per (latency, impact) and
are looked up for every queue/hidden combination - an exactness-preserving reduction, not a change of the grid.

Pairwise comparisons are made *under the same assumption* (one true world, unknown to us): for a day the oriented
difference d_a (positive = treatment better) is computed for every assumption a, giving the partial-identification
interval [min_a d_a, max_a d_a]. A direction is IDENTIFIED for the day when the whole interval lies on one side of
zero, INDETERMINATE otherwise. The registered decision quantity is the conservative bound min_a d_a, averaged over
units (paired block bootstrap over days in the confirmatory analysis). Windows a source cannot price (L1-only books
beyond the observed touch size) are NOT_IDENTIFIED and excluded from both policies of the pair.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np

from .. import protocol as pr
from ..capabilities import provided
from ..replay import engine as E
from ..runner import Task, run_tasks
from . import common

ANALYSIS = "counterfactual-bounds-v09"
RANKING_POLICIES = ("twap", "vwap", "pov", "static_opt", "sweep", "passive_naive", "passive_deadline",
                    "imbalance_passive", "toxicity_passive", "liquidity_pov", "mpc_stochastic", "mpc_cvar",
                    "mpc_dro", "mpc_chance", "mpc_adaptive")
AGGRESSIVE_ONLY = frozenset({"twap", "vwap", "pov", "static_opt", "sweep", "liquidity_pov"})
FIELDS = ("is_net_bps", "is_gross_bps", "markout5s_bps", "completion", "maker_fraction")


def combos(protocol: dict) -> list[tuple[str, str]]:
    out = {(p, "medium") for p in RANKING_POLICIES}
    for c in protocol["conclusions"]["set"].values():
        out.add((c["treatment"], c["mandate"]))
        out.add((c["control"], c["mandate"]))
    return sorted(out)


def assumptions(protocol: dict) -> list[E.Assumption]:
    a = protocol["counterfactual_assumptions"]
    return E.grid(latencies=tuple(a["latency_s"]), hidden=tuple(a["hidden_fraction"]), impacts=tuple(a["impact"]))


def canonical(a: E.Assumption, policy: str) -> E.Assumption:
    if policy in AGGRESSIVE_ONLY:
        return E.Assumption("neutral", a.latency_s, 0.0, a.impact)
    return a


def day_task(dataset_id: str, use: str, design: str, root: str, combos_: list, assumption_keys: list,
             notionals: dict) -> dict:
    """All registered episodes of one day: {policy|mandate: {assumption key: [window rows]}}."""
    from ..data import store
    from ..execution import policies as P
    t0 = time.time()
    cache, _ = store.prepare(dataset_id, use=use, design=design, root=Path(root), purpose="bounded replay")
    book = E.DayBook(cache)
    instrument = cache_instrument(dataset_id)
    grid_ = [E.Assumption(**json.loads(k)) for k in assumption_keys]
    out: dict = {}
    for policy, mandate in combos_:
        notional = notionals[mandate][instrument]
        seen: dict[str, list] = {}
        rows_by_key = {}
        for a in grid_:
            ca = canonical(a, policy)
            ck = ca.key()
            if ck not in seen:
                rows = []
                for w in range(len(E.WINDOW_STARTS_S)):
                    r = E.episode(book, P.make(policy), window=w, notional_usd=notional, assumption=ca,
                                  context={"seed": w})
                    rows.append(None if r["status"] != "OK" else
                                [r.get(f) for f in FIELDS] + [bool(r.get("identified", True))])
                seen[ck] = rows
            rows_by_key[a.key()] = seen[ck]
        out[f"{policy}|{mandate}"] = rows_by_key
    return {"dataset": dataset_id, "episodes": out, "seconds": time.time() - t0,
            "levels": int(cache.get("levels", 200))}


def cache_instrument(dataset_id: str) -> str:
    return dataset_id.split("-")[2].upper()


# ============================================================================= aggregation


def _endpoint(rows: list, endpoint: str, mask: np.ndarray) -> float | None:
    idx = {"mean_is_bps": 0, "cvar90_is_bps": 0, "markout5s_bps": 2, "completion": 3}[endpoint]
    vals = np.asarray([rows[w][idx] if rows[w] is not None and rows[w][idx] is not None else np.nan
                       for w in range(len(rows))], float)[mask]
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return None
    if endpoint == "cvar90_is_bps":
        k = max(1, int(math.ceil(0.1 * len(vals))))
        return float(np.sort(vals)[-k:].mean())
    return float(vals.mean())


def _mask(a: list, b: list, endpoint: str) -> np.ndarray:
    idx = {"mean_is_bps": 0, "cvar90_is_bps": 0, "markout5s_bps": 2, "completion": 3}[endpoint]
    return np.asarray([x is not None and y is not None and x[5] and y[5] and x[idx] is not None and y[idx] is not None
                       for x, y in zip(a, b)])


def pair_day(day: dict, treatment: str, control: str, endpoint: str, direction: str,
             windows: slice | None = None) -> dict | None:
    """Per-assumption oriented differences for one day; None when the pair is not evaluable."""
    t, c = day["episodes"].get(treatment), day["episodes"].get(control)
    if t is None or c is None:
        return None
    diffs = {}
    for key in t:
        a_rows, b_rows = t[key], c[key]
        if windows is not None:
            a_rows, b_rows = a_rows[windows], b_rows[windows]
        mask = _mask(a_rows, b_rows, endpoint)
        if mask.sum() < 3:
            continue
        x, y = _endpoint(a_rows, endpoint, mask), _endpoint(b_rows, endpoint, mask)
        if x is None or y is None:
            continue
        diffs[key] = (y - x) if direction == "decrease" else (x - y)
    if not diffs:
        return None
    lo, hi = min(diffs.values()), max(diffs.values())
    return {"lower": lo, "upper": hi, "status": "IDENTIFIED" if lo > 0 or hi < 0 else "INDETERMINATE",
            "assumptions": len(diffs)}


def conclusion_units(days: list[dict], conclusion: dict, *, streams=None) -> dict:
    t = f"{conclusion['treatment']}|{conclusion['mandate']}"
    c = f"{conclusion['control']}|{conclusion['mandate']}"
    rows, odd, even = {}, [], []
    for d in days:
        if streams is not None and not streams(d["dataset"]):
            continue
        r = pair_day(d, t, c, conclusion["endpoint"], conclusion["direction"])
        if r is None:
            continue
        rows[d["dataset"]] = r
        o = pair_day(d, t, c, conclusion["endpoint"], conclusion["direction"], slice(1, None, 2))
        e = pair_day(d, t, c, conclusion["endpoint"], conclusion["direction"], slice(0, None, 2))
        if o is not None and e is not None:
            odd.append(o["lower"])
            even.append(e["lower"])
    return {"units": rows, "split_half": (odd, even)}


def reliability(odd: list[float], even: list[float]) -> float | None:
    """Spearman-Brown corrected split-half reliability of the unit-level conservative bound."""
    if len(odd) < 4:
        return None
    from ...v08.statistics import spearman
    r = spearman(np.asarray(odd), np.asarray(even))
    if not np.isfinite(r):
        return None
    return float(2 * r / (1 + r)) if r > -1 else -1.0


def pilot_record(units: dict, split: tuple, *, capabilities: dict, hours: float, threshold: float) -> dict:
    vals = np.asarray([u["lower"] for u in units.values()], float)
    return {"data_available": len(vals) > 0, "capabilities": capabilities,
            "reliability": reliability(*split), "units": int(len(vals)),
            "effect": float(vals.mean()) if len(vals) else None,
            "sd": float(vals.std(ddof=1)) if len(vals) > 1 else None,
            "compute_hours_per_unit": hours, "unit_values": vals.tolist(),
            "identified_fraction": float(np.mean([u["status"] == "IDENTIFIED" and u["lower"] > 0
                                                  for u in units.values()])) if units else 0.0,
            "threshold": threshold}


def run(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    if args.phase == "pilot" and args.namespace not in ("explore", "pilot"):
        raise ValueError("the pilot phase reads explore days (namespace explore or pilot)")
    if args.phase == "confirm" and args.namespace != "confirm":
        raise ValueError("the confirm phase requires --namespace confirm and its sealed design")
    design = common.design_document(args)
    if args.phase == "pilot":
        n = protocol["pools"]["explore"]["count"]
        res = common.units(args, "explore", ["bybit-linear:BTCUSDT", "bybit-linear:ETHUSDT", "binance-um:BTCUSDT",
                                             "binance-um:ETHUSDT"], n)
        datasets = [d for r in res.values() for d in r["datasets"]]
        design_name = "explore-bounds"
    else:
        datasets = list(design["datasets"])
        design_name = args.design
    if args.dataset:
        datasets = [d for d in datasets if d in set(args.dataset)]
    out = common.start(args)
    combos_ = combos(protocol)
    grid_ = [json.dumps(E.assumption_dict(a), sort_keys=True) for a in assumptions(protocol)]
    notionals = {k: v["notional_usd"] for k, v in protocol["mandates"]["classes"].items()}
    l1_combos = sorted({(c[k], c["mandate"]) for c in protocol["conclusions"]["set"].values()
                        if c["identifiability"] == "L1" for k in ("treatment", "control")})
    tasks = [Task(d, "lob.v09.studies.bounds:day_task",
                  {"dataset_id": d, "use": common.use_of(protocol, d), "design": design_name, "root": str(root),
                   "combos_": combos_ if d.startswith("bybit") else l1_combos, "assumption_keys": grid_,
                   "notionals": notionals}, namespace=args.namespace)
             for d in datasets]
    days, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args),
                           checkpoint=root / (args.out + ".checkpoint"),
                           design={"design": design_name, "combos": combos_, "grid": grid_}, inputs=datasets)
    hours = float(np.mean([d["seconds"] for d in days]) / 3600.0) if days else 0.0
    result = summarize(protocol, days, hours=hours, phase=args.phase)
    result["runner"] = info
    return common.finish(args, out, analysis=ANALYSIS,
                         config={"phase": args.phase, "datasets": datasets, "combos": combos_, "assumptions": grid_,
                                 "design": design_name},
                         result=result, datasets=datasets, summary={"conclusions": result["conclusions"]},
                         resources={"compute_hours_per_unit": hours})


def summarize(protocol: dict, days: list[dict], *, hours: float, phase: str) -> dict:
    caps = {c: provided(protocol, "explore", c) for c in ("incremental_l2", "trades_with_side", "l1_quotes",
                                                         "queue_clearance_observable")}
    l2 = lambda d: d.startswith("bybit")                                    # noqa: E731
    l1 = lambda d: d.startswith("binance")                                  # noqa: E731
    btc = lambda d: "btcusdt" in d and d.startswith("bybit")                # noqa: E731
    eth = lambda d: "ethusdt" in d and d.startswith("bybit")                # noqa: E731
    conclusions = {}
    for cid, c in protocol["conclusions"]["set"].items():
        streams = l2 if c["identifiability"] == "L2" else l1
        u = conclusion_units(days, c, streams=streams)
        conclusions[cid] = {"definition": c, "units": u["units"],
                            "pilot": pilot_record(u["units"], u["split_half"], capabilities=caps, hours=hours,
                                                  threshold=c["threshold_bps"])}
        for name, flt in (("btc", btc), ("eth", eth), ("l1", l1)):
            if c["identifiability"] == "L2" and name == "l1":
                continue
            v = conclusion_units(days, c, streams=flt)
            conclusions[cid][f"pilot_{name}"] = pilot_record(v["units"], v["split_half"], capabilities=caps,
                                                             hours=hours, threshold=c["threshold_bps"])
    c5 = protocol["conclusions"]["set"].get("C5")
    if c5 is not None:                         # H9 completion non-inferiority (registered margin 0.02)
        comp = conclusion_units(days, {**c5, "endpoint": "completion", "direction": "increase"}, streams=l2)
        conclusions["C5"]["completion_units"] = comp["units"]
    days_out = {d["dataset"]: {"seconds": d["seconds"], "levels": d["levels"]} for d in days}
    return {"phase": phase, "conclusions": conclusions, "days": days_out,
            "ranking": ranking(days), "label": "BOUNDED, HYPOTHETICAL, ASSUMPTION_DEPENDENT; never actual fills"}


def ranking(days: list[dict]) -> dict:
    """Per-day pairwise identification of the registered medium-mandate policy set (input to H30/H31)."""
    out = {}
    pols = [p for p in RANKING_POLICIES]
    for d in days:
        if not d["dataset"].startswith("bybit"):
            continue
        pairs = {}
        for i, a in enumerate(pols):
            for b in pols[i + 1:]:
                r = pair_day(d, f"{a}|medium", f"{b}|medium", "mean_is_bps", "decrease")
                if r is not None:
                    pairs[f"{a}>{b}"] = {"lower": r["lower"], "upper": r["upper"], "status": r["status"]}
        out[d["dataset"]] = pairs
    return out
