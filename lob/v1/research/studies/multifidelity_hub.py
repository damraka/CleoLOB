"""Multi-fidelity hub: F2 bounded replay, F3 hybrid twin and F4/F5 synthetic markets on matched policies, and the
registered unit values of H20, H29-H40, H44 and H47
(H18, H41 and H43 are listed below for the design but computed by a separate hub).

Design (pilot = the v0.9 decisions design; confirm = calibrate-pool training, confirm-pool evaluation): per stream,
training units calibrate the F3 response/survival models and the F4/F5 generators once; every evaluation unit gets
F2 (all registered policies, full assumption grid) and F3 outcomes. F4 directions are per stream (calibrated once),
F2/F3 directions per evaluation unit, so every unit value varies with the day's historical truth.

Unit values (positive = registered direction; ``None`` when the registered comparison is undefined on the unit,
e.g. no generator accepted by the gate - never imputed):

* H29 = mean_C |F4 - F2| - |F3 - F2| (C1-C3 effects; F2 = conservative bound, F3/F4 = mean differences);
* H30 = agreement(F4 corrected book, F2) - agreement(F4 v0.9 book, F2);  H31 = acc(F3+F4 consensus) - acc(F4);
* H32 = regret(policy chosen by all generators) - regret(chosen by gate-accepted generators) (F2 central costs);
* H33 = gap(F4 default queue) - gap(F4 conservative queue, cancels behind);
* H34 = Spearman over policy pairs of (generator disagreement, F4-vs-F2 flip);  H35 = contradiction-rate difference;
* H36 = |regret(classical choice) - regret(neural choice)|;  H20 = Kendall(exec-gate set, F2) - Kendall(dist-gate set, F2);
* H44 = |C2 effect in F3 with the fitted impact - with zero impact|;  H47 = agreement(H28-accepted generator, F2) - 0.5;
* H18 = mean_C width(v0.9 hidden grid {0, 0.5}) - width(evidence-based hidden interval {0, h_max});
* H37 = completion(completion_aware) - completion(TWAP) [F2 central]; H39 = completion(mpc_chance) - completion(mpc_stochastic);
* H38 = CVaR90(TWAP) - CVaR90(mpc_dro), conservative bound, only on dates with realized volatility above the training
  75th percentile (other units ``None``);
* H40 = regret(best global policy on the training units) - regret(contextual fallback meta-policy);
* H41 = harm rate(ungated offline RL) - harm rate(gated) (harm = cost above TWAP + 1 bps, F2 central);
* H43 = regret(likelihood-calibrated F4 choice) - regret(decision-calibrated F4 choice).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from ....v09.studies.bounds import RANKING_POLICIES
from . import common as CM
from .replication import _mapping, _pool

ANALYSIS = "multifidelity-v1"
EXTRA_F2 = ("completion_aware",)
CENTRAL = {"queue": "neutral", "latency_s": 0.05, "hidden_fraction": 0.0, "impact": "transient"}
GENERATORS = ("poisson", "hawkes", "queue_reactive", "mixture")
SEEDS = 8


def _assumption_keys(hidden):
    from ....v09.replay import engine as E
    grid = E.grid(latencies=(0.0, 0.05, 0.25), hidden=tuple(hidden), impacts=("none", "transient", "persistent"))
    return [json.dumps(E.assumption_dict(a), sort_keys=True) for a in grid]


def _combos():
    return sorted({(p, "medium") for p in RANKING_POLICIES + EXTRA_F2} | {("twap", "large"), ("sweep", "large"),
                                                                      ("passive_deadline", "small"), ("twap", "small")})



def _rows_for(day: dict, policy_mandate: str, assumption: dict):
    """Window rows of one combo under the (canonicalized) central assumption."""
    eps = day["episodes"][policy_mandate]
    for k, rows in eps.items():
        parts = dict(x.split("=") for x in k.split("|")[1:])
        q = k.split("|")[0]
        if (q == assumption["queue"] or policy_mandate.split("|")[0] in ("twap", "vwap", "pov", "static_opt", "sweep",
                                                                         "liquidity_pov")) and \
                abs(float(parts["lat"]) - assumption["latency_s"]) < 1e-9 and \
                abs(float(parts["hidden"]) - assumption["hidden_fraction"]) < 1e-9 and parts["impact"] == assumption["impact"]:
            return rows
    return None


def central_costs(day: dict, mandate: str = "medium") -> dict[str, list]:
    out = {}
    for p in RANKING_POLICIES + EXTRA_F2:
        rows = _rows_for(day, f"{p}|{mandate}", CENTRAL)
        if rows is not None:
            out[p] = [r[0] if r is not None else None for r in rows]
    return out


def central_completion(day: dict, policy: str) -> float | None:
    rows = _rows_for(day, f"{policy}|medium", CENTRAL)
    vals = [r[3] for r in (rows or []) if r is not None and r[3] is not None]
    return float(np.mean(vals)) if vals else None


def f2_directions(day: dict) -> dict[str, int]:
    from ....v09.studies import bounds as BD
    out = {}
    pols = list(RANKING_POLICIES)
    for i, a in enumerate(pols):
        for b in pols[i + 1:]:
            r = BD.pair_day(day, f"{a}|medium", f"{b}|medium", "mean_is_bps", "decrease")
            # oriented "a cheaper than b": treatment a, control b, decrease => positive when a is cheaper
            out[f"{a}>{b}"] = 0 if r is None else (1 if r["lower"] > 0 else -1 if r["upper"] < 0 else 0)
    return out


def regret(costs: dict[str, list], chosen: str) -> float | None:
    means = {p: float(np.nanmean([x for x in v if x is not None])) for p, v in costs.items()
             if any(x is not None for x in v)}
    if chosen not in means:
        return None
    return means[chosen] - min(means.values())


def synthetic_costs(train_paths: list[str], seed: int, notional: float, *, book: str = "v1",
                    cancel_position: str = "uniform", families=GENERATORS, neural: bool = True) -> dict:
    from dataclasses import replace

    from ....v09.execution import policies as P
    from ...data import daycache as DC
    from ...runtime import cap_threads
    from ...twins import classical as TC
    cap_threads()
    caches = [DC.load(Path(p)) for p in train_paths]
    fit = TC.fit(caches, book=book, families=list(families))
    tick, lot = float(caches[0]["tick"]), float(caches[0]["lot"])
    gens = dict(fit["generators"])
    if cancel_position != "uniform":
        for g in gens.values():
            g.book = replace(g.book, cancel_position=cancel_position)
    if neural and book == "v1":
        try:
            from ...generators.neural.counts import AutoregressiveCounts
            from ....v09.data.daycache import frame as cf
            fr = [cf(c, (s - 30) * 10, (s + 300) * 10) for c in caches[:6] for s in range(600, 86400, 3600 * 4)]
            nn = AutoregressiveCounts(steps=600, history=32, seed=seed).fit(fr)
            nn.book = fit["book"]
            gens["neural_ar"] = nn
        except ImportError:
            pass
    del caches
    qty = int(notional / (fit["book"].mid_ticks * tick) / lot)
    costs = {}
    for gname, g in gens.items():
        costs[gname] = {}
        for p in RANKING_POLICIES:
            vals = []
            for k in range(SEEDS):
                r = TC.synthetic_episode(g, P.make(p), qty=qty, seed=seed * 1000 + k, side=1 if k % 2 == 0 else -1)
                vals.append(r["is_net_bps"] if r["status"] == "OK" else None)
            costs[gname][p] = vals
    return {"costs": costs, "qty_lots": qty, "book": book, "cancel_position": cancel_position}


def _dirs(costs: dict[str, list]) -> dict[str, int]:
    from ...twins import agreement as AG
    clean = {p: [np.nan if x is None else x for x in v] for p, v in costs.items()}
    pols = list(RANKING_POLICIES)
    return {f"{a}>{b}": AG.direction(clean[a], clean[b]) for i, a in enumerate(pols) for b in pols[i + 1:]}


def _effect(costs: dict[str, list], c: dict) -> float | None:
    a = [x for x in costs.get(c["control"], []) if x is not None]
    b = [x for x in costs.get(c["treatment"], []) if x is not None]
    if not a or not b:
        return None
    return float(np.mean(a) - np.mean(b))


def run(ctx: CM.Context, out: Path, *, evaluate: list[str], train: list[str] | None, seed: int,
        workers: int = 2, gate_by_stream: dict | None = None) -> dict:
    """``train=None``: pilot split (first half of the dates trains). ``gate_by_stream``: realism-hub decisions."""
    from ....v09.studies import bounds as BD
    from ...execution.meta import MetaPolicy
    from ...twins import agreement as AG
    t0 = time.time()
    if train is None:
        dates = sorted({CM.date_of(d) for d in evaluate})
        half = set(dates[: len(dates) // 2])
        train = [d for d in evaluate if CM.date_of(d) in half]
        evaluate = [d for d in evaluate if CM.date_of(d) not in half]
    prot = json.loads((ctx.root / "configs/v1/protocol.json").read_text())
    carried = prot["carried_from_v09"]
    concl = carried["conclusions"]
    notionals = {k: v["notional_usd"] for k, v in carried["mandates"]["classes"].items()}
    prep = ctx.prepare(train + evaluate, use=f"{ANALYSIS} (F2/F3/F4/F5)", workers=workers)
    mp = _mapping(prep)
    streams = sorted({CM.stream_of(d) for d in evaluate})
    plan = {s: ([d for d in train if CM.stream_of(d) == s and d in mp],
                [d for d in evaluate if CM.stream_of(d) == s and d in mp]) for s in streams}
    keys9 = _assumption_keys((0.0, 0.5))
    with ProcessPoolExecutor(workers) as ex:
        f2 = _pool(ex, {d: ("lob.v09.studies.bounds:day_task", mp,
                            {"dataset_id": d, "use": "v1", "design": "v1", "root": str(ctx.root), "combos_": _combos(),
                             "assumption_keys": keys9, "notionals": notionals})
                        for s, (tr, te) in plan.items() for d in tr + te})
        f3 = _pool(ex, {d: ("lob.v09.studies.decisions:hybrid_day", mp,
                            {"day": d, "train": tr, "root": str(ctx.root), "design": "v1", "seed": seed,
                             "notional": float(notionals["medium"][s.split(":")[1]])})
                        for s, (tr, te) in plan.items() for d in te})
        syn = {}
        for s, (tr, te) in plan.items():
            n = float(notionals["medium"][s.split(":")[1]])
            paths = [mp[d] for d in tr]
            syn[(s, "v1")] = ex.submit(synthetic_costs, paths, seed, n)
            syn[(s, "v09")] = ex.submit(synthetic_costs, paths, seed, n, book="v09", families=("poisson", "hawkes"),
                                        neural=False)
            syn[(s, "back")] = ex.submit(synthetic_costs, paths, seed, n, cancel_position="back",
                                         families=("poisson", "hawkes"), neural=False)
        syn = {k: f.result() for k, f in syn.items()}
    units: dict[str, dict] = {}
    for s, (tr, te) in plan.items():
        g_v1, g_v9, g_back = syn[(s, "v1")]["costs"], syn[(s, "v09")]["costs"], syn[(s, "back")]["costs"]
        gate = (gate_by_stream or {}).get(s, {})
        accepted = [g for g, dec in gate.get("execution", {}).items() if dec] if gate else []
        dist_acc = [g for g, dec in gate.get("distribution", {}).items() if dec] if gate else []
        per_gen_dirs = {g: _dirs(c) for g, c in g_v1.items()}
        cons_v1 = AG.consensus(list(per_gen_dirs.values()))
        cons_v9 = AG.consensus([_dirs(c) for c in g_v9.values()])
        mean_cost = {p: float(np.nanmean([x for g in g_v1.values() for x in g[p] if x is not None]))
                     for p in RANKING_POLICIES}
        tr_costs = [central_costs(f2[d]) for d in tr]
        best_global = min(RANKING_POLICIES, key=lambda p: np.nanmean([np.nanmean([x for x in c[p] if x is not None])
                                                                       for c in tr_costs if p in c]))
        vol_tr = [float(np.std(np.diff(np.asarray(DCload(mp[d])["mid"])[::600]))) for d in tr]
        meta = None
        try:
            from ....v09.execution.selector import window_context
            Xs, cs = [], {p: [] for p in RANKING_POLICIES}
            for d in tr:
                c = DCload(mp[d])
                cc = central_costs(f2[d])
                for w in range(24):
                    vals = [cc.get(p, [None] * 24)[w] for p in RANKING_POLICIES]
                    if any(v is None for v in vals):
                        continue
                    Xs.append(window_context(c, float(600 + 3600 * w), 1 if w % 2 == 0 else -1))
                    for p, v in zip(RANKING_POLICIES, vals):
                        cs[p].append(v)
            if len(Xs) >= 30:
                meta = MetaPolicy(list(RANKING_POLICIES)).fit(np.asarray(Xs), {p: np.asarray(v) for p, v in cs.items()})
        except Exception:  # noqa: BLE001 - recorded as a missing H40 unit, never hidden
            meta = None
        for d in te:
            day = f2[d]
            truth = f2_directions(day)
            cc = central_costs(day)
            u: dict = {}
            hyb = day_costs_f3(f3.get(d))
            eff = {}
            for cid, c in concl.items():
                m = c["mandate"]
                r = BD.pair_day(day, f"{c['treatment']}|{m}", f"{c['control']}|{m}", c["endpoint"], c["direction"])
                f2e = r["lower"] if r else None
                f3e = _effect(hyb, c) if m == "medium" else None
                f4e = None
                if m == "medium":
                    vals = [_effect(g, c) for g in g_v1.values()]
                    vals = [v for v in vals if v is not None]
                    f4e = float(np.mean(vals)) if vals else None
                eff[cid] = (f2e, f3e, f4e)
            gaps = [(abs(f4 - f2e) - abs(f3 - f2e)) for f2e, f3, f4 in eff.values() if None not in (f2e, f3, f4)]
            u["H29"] = float(np.mean(gaps)) if gaps else None
            a1, _ = AG.agreement(cons_v1, truth)
            a9, _ = AG.agreement(cons_v9, truth)
            u["H30"] = (a1 - a9) if a1 is not None and a9 is not None else None
            hdirs = _dirs(hyb) if hyb else {}
            cons_all = AG.consensus([cons_v1, hdirs])
            ac, _ = AG.agreement(cons_all, truth)
            u["H31"] = (ac - a1) if ac is not None and a1 is not None else None
            chosen_all = min(RANKING_POLICIES, key=mean_cost.get)
            if accepted:
                mc_acc = {p: float(np.nanmean([x for g in accepted for x in g_v1[g][p] if x is not None]))
                          for p in RANKING_POLICIES}
                r_all, r_acc = regret(cc, chosen_all), regret(cc, min(mc_acc, key=mc_acc.get))
                u["H32"] = (r_all - r_acc) if None not in (r_all, r_acc) else None
            else:
                u["H32"] = None
            gb = [abs(_effect(g_back[k], c) - eff[cid][0]) for cid, c in concl.items() if c["mandate"] == "medium"
                  for k in g_back if eff[cid][0] is not None and _effect(g_back[k], c) is not None]
            gd = [abs(_effect(g_v1[k], c) - eff[cid][0]) for cid, c in concl.items() if c["mandate"] == "medium"
                  for k in ("poisson", "hawkes") if eff[cid][0] is not None and _effect(g_v1[k], c) is not None]
            u["H33"] = float(np.mean(gd) - np.mean(gb)) if gb and gd else None
            pairs = [k for k in truth if truth[k] != 0]
            if len(pairs) >= 5:
                dis = [float(np.std([per_gen_dirs[g][k] for g in per_gen_dirs])) for k in pairs]
                flip = [float(cons_v1.get(k, 0) != truth[k]) for k in pairs]
                from ....v08.statistics import spearman
                u["H34"] = float(spearman(np.asarray(dis), np.asarray(flip))) if np.std(dis) > 0 and np.std(flip) > 0 \
                    else None
            else:
                u["H34"] = None
            if accepted:
                cr_all = 1 - (AG.agreement(cons_v1, truth)[0] or 0)
                cr_acc = 1 - (AG.agreement(AG.consensus([per_gen_dirs[g] for g in accepted]), truth)[0] or 0)
                u["H35"] = cr_all - cr_acc
            else:
                u["H35"] = None
            if "neural_ar" in g_v1:
                cl = {p: float(np.nanmean([x for g in GENERATORS if g in g_v1 for x in g_v1[g][p] if x is not None]))
                      for p in RANKING_POLICIES}
                nn = {p: float(np.nanmean([x for x in g_v1["neural_ar"][p] if x is not None] or [np.nan]))
                      for p in RANKING_POLICIES}
                rc, rn = regret(cc, min(cl, key=cl.get)), regret(cc, min(nn, key=lambda p: nn[p] if np.isfinite(nn[p])
                                                                         else 1e9))
                u["H36"] = abs(rc - rn) if None not in (rc, rn) else None
            if gate and accepted and dist_acc:
                def kset(gs):
                    return AG.kendall({p: [x for g in gs for x in g_v1[g][p] if x is not None] for p in RANKING_POLICIES},
                                      {p: [x for x in cc.get(p, []) if x is not None] for p in RANKING_POLICIES})
                ke, kd = kset(accepted), kset(dist_acc)
                u["H20"] = (ke - kd) if None not in (ke, kd) else None
            else:
                u["H20"] = None
            u["H44"] = (f3.get(d) or {}).get("h44")
            h28 = (gate or {}).get("h28_accepted")
            if h28:
                ag, _ = AG.agreement(per_gen_dirs[h28], truth)
                u["H47"] = (ag - 0.5) if ag is not None else None
            else:
                u["H47"] = None
            cA, cT = central_completion(day, "completion_aware"), central_completion(day, "twap")
            u["H37"] = (cA - cT) if None not in (cA, cT) else None
            cC, cS = central_completion(day, "mpc_chance"), central_completion(day, "mpc_stochastic")
            u["H39"] = (cC - cS) if None not in (cC, cS) else None
            vol = float(np.std(np.diff(np.asarray(DCload(mp[d])["mid"])[::600])))
            if vol > np.percentile(vol_tr, 75):
                r = BD.pair_day(day, "mpc_dro|medium", "twap|medium", "cvar90_is_bps", "decrease")
                u["H38"] = r["lower"] if r else None
            else:
                u["H38"] = None
            if meta is not None:
                from ....v09.execution.selector import window_context
                c = DCload(mp[d])
                X = np.asarray([window_context(c, float(600 + 3600 * w), 1 if w % 2 == 0 else -1) for w in range(24)])
                ch = meta.choose(X)["fallback_choice"]
                fb = [cc.get(p, [None] * 24)[w] for w, p in enumerate(ch)]
                bg = cc.get(best_global, [None] * 24)
                diffs = [b - f for b, f in zip(bg, fb) if b is not None and f is not None]
                u["H40"] = float(np.mean(diffs)) if diffs else None
            else:
                u["H40"] = None
            units[d] = u
    # H18, H41 and H43 need the hidden-interval, offline-RL and decision-calibration runs (separate hub, pending)
    hyp = ("H20", "H29", "H30", "H31", "H32", "H33", "H34", "H35", "H36", "H37", "H38", "H39", "H40", "H44", "H47")
    unit_values = {h: {d: units[d].get(h) for d in units} for h in hyp}
    res = {"plan": plan, "units": unit_values, "clusters": {h: CM.cluster_values(v) for h, v in unit_values.items()},
           "synthetic": {f"{k[0]}|{k[1]}": {"qty_lots": v["qty_lots"]} for k, v in syn.items()},
           "compute_hours": (time.time() - t0) / 3600,
           "label": "F2 BOUNDED replay, F3 HYBRID and F4/F5 SYNTHETIC outcomes; never historical fills"}
    info = CM.finalize(ctx, out, analysis=ANALYSIS, config={"plan": plan, "seed": seed, "seeds": SEEDS}, result=res)
    return {**info, "result": res}


_CACHE: dict = {}


def DCload(path: str):
    from ...data import daycache as DC
    if path not in _CACHE:
        _CACHE[path] = DC.load(Path(path))
    return _CACHE[path]


def day_costs_f3(day: dict | None) -> dict[str, list]:
    if not day:
        return {}
    return {p: [r[0] if r else None for r in rows] for p, rows in day["outcomes"].items() if p in RANKING_POLICIES}


__all__ = ["run", "synthetic_costs", "f2_directions", "central_costs", "regret", "ANALYSIS"]
