"""Hybrid-twin decision studies and multi-fidelity agreement (F3, F4/F6; H8, H18, H19, H22, H25, H29, H30, H31).

Design (pilot): per instrument, the explore days in chronological order; the first half are training days, the
second half evaluation units (instrument-days). Confirm: training = all explore days of the instrument, evaluation =
the design's confirm days.

Training days provide (refitted identically, seeded, inside each task): the hybrid impact response model (registered
boosted regression on bursts) and persistent fraction, the survival models for the survival-driven passive policy
(Kaplan-Meier = uncalibrated, Weibull AFT = calibrated), and the hybrid outcomes used to fit the contextual cost
models and to select the best global policy. Every day runs the registered policy set on the hybrid twin over the 24
windows with common random numbers (seed = hash(day, window)).

Units (evaluation days):

* H8  = mean IS(survival_passive[km]) - mean IS(survival_passive[calibrated]) (bps);
* H18 = Spearman over windows of (pre-decision disagreement, realized regret of the meta-policy's choice);
* H19 = incorrect-selection rate of always-select minus abstention v3 (correct = within 0.5 bps of the window's
  best realized policy);
* H22 = policy completion(mpc_chance) - policy completion(mpc_stochastic), with the cost non-inferiority reported;
* H25 = mean IS(best global policy on training days) - mean IS(meta-policy);
* H29 = pairwise sign agreement between the synthetic consensus (F4/F6 generators calibrated on the training days)
  and the hybrid directions of the day (determinate pairs only), minus nothing (agreement level 0.5 registered);
* H30 = agreement between hybrid directions and historically identified bounded directions of the day (``--input
  bounds=RUN``);
* H31 = agreement(consensus of synthetic and hybrid, identified bounded) - agreement(synthetic, identified bounded).
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from .. import protocol as pr, twin
from ..execution import policies as P
from ..execution.selector import CostModels, abstain, select, window_context
from ..generators import classical as G, hybrid as Hy
from ..generators.strategic import StrategicEcology
from ..impact import models as IM
from ..queue import clearance as C, survival as S
from ..replay import engine as E
from ..runner import Task, run_tasks
from . import common
from .bounds import RANKING_POLICIES

ANALYSIS = "decisions-v09"
EXTRA = ("survival_passive:km", "survival_passive:calibrated")
SYN_SEEDS = 8


def _seed(day: str, window: int) -> int:
    return int(hashlib.sha256(f"{day}|{window}".encode()).hexdigest()[:8], 16)


def _train_models(train: list[str], root: str, design: str, seed: int):
    caches = [common.load_cache(d, root, design, "decision training") for d in train]
    bursts = [Hy.burst_targets(c) for c in caches]
    X = np.vstack([b["X"] for b in bursts])
    y10 = np.vstack([b["Y"] for b in bursts])[:, 1]          # 5 s signed mid move (registered impact target)
    impact = IM.StateImpact(trees=100).fit(X, y10)
    dec = [IM.decomposition(IM.bursts(c)) for c in caches]
    temp = np.mean([d["temporary_1s_bps"] for d in dec if d])
    pers = np.mean([d["persistent_60s_bps"] for d in dec if d])
    persistent = float(np.clip(pers / temp, 0.0, 1.0)) if temp > 0 else 0.5
    ev = [C.events(c) for c in caches]
    Xs = np.vstack([e["X"] for e in ev])
    T = np.concatenate([e["T"] for e in ev])
    Ev = np.concatenate([e["E"] for e in ev])
    keep = np.sort(np.random.default_rng(seed).choice(len(T), min(60_000, len(T)), replace=False))
    km = S.KaplanMeier().fit(Xs[keep], T[keep], Ev[keep])
    cal = S.WeibullAFT().fit(Xs[keep], T[keep], Ev[keep])
    return impact, persistent, km, cal


def _policy(name: str):
    if name.startswith("survival_passive:"):
        return P.make("survival_passive", model_key=name.split(":")[1])
    return P.make(name)


def hybrid_day(day: str, train: list[str], root: str, design: str, seed: int, notional: float) -> dict:
    impact, persistent, km, cal = _train_models(train, root, design, seed)
    cache = common.load_cache(day, root, design, "hybrid decisions")
    feats = C.features_at(cache)
    survival = {"km": (km, feats), "calibrated": (cal, feats)}
    out, ctx = {}, []
    for w in range(len(E.WINDOW_STARTS_S)):
        ctx.append(window_context(cache, float(E.WINDOW_STARTS_S[w]), E.side_of(w)).tolist())
    for name in RANKING_POLICIES:
        rows = []
        for w in range(len(E.WINDOW_STARTS_S)):
            r = twin.hybrid_episode(cache, _policy(name), window=w, notional=notional, impact_model=impact,
                                    persistent=persistent, seed=_seed(day, w))
            rows.append(None if r["status"] != "OK" else [r["is_net_bps"], r["policy_completion"]])
        out[name] = rows
    for name in EXTRA:                       # survival policies receive the fitted models in the episode context
        rows = []
        for w in range(len(E.WINDOW_STARTS_S)):
            pol = _policy(name)
            book = E.DayBook(cache)
            qty = E.window_quantity(book, w, notional, float(cache["lot"]))
            t0 = float(E.WINDOW_STARTS_S[w])
            if qty <= 0 or not cache["valid"][book.bin_of(t0 - E.WARMUP_S):book.bin_of(t0 + E.HORIZON_S)].all():
                rows.append(None)
                continue
            rep = Hy.HybridReplay(cache, pol, impact_model=impact, persistent=persistent, seed=_seed(day, w),
                                  window=w, qty=qty, side=E.side_of(w), assumption=E.Assumption("neutral"),
                                  context={"seed": _seed(day, w), "survival": survival})
            res = rep.run()
            m = E.metrics(res, rep.book)
            forced = sum(f[2] for f in res.fills if abs(f[0] - (t0 + E.HORIZON_S)) < 1e-9)
            rows.append([m["is_net_bps"], 1.0 - forced / qty])
        out[name] = rows
    return {"day": day, "outcomes": out, "context": ctx, "persistent": persistent}


def synthetic_task(stream: str, train: list[str], root: str, design: str, seed: int, notional: float,
                   lot: float) -> dict:
    caches = [common.load_cache(d, root, design, "synthetic ranking") for d in train]
    frame = G.training_frame(caches)
    tick = float(caches[0]["tick"])
    del caches
    book = G.book_model(frame)
    gens = {n: G.fit_family(n, frame, book) for n in list(G.FAMILIES) + list(G.STATIC_ECOLOGY)}
    strat = StrategicEcology(None, book).fit(frame)
    strat.book = book
    gens["strategic_ecology"] = strat
    qty = int(notional / (book.mid_ticks * tick) / lot)
    costs = {}
    for gname, g in gens.items():
        costs[gname] = {}
        for p in RANKING_POLICIES:
            vals = []
            for k in range(SYN_SEEDS):
                r = twin.synthetic_episode(g, P.make(p), qty=qty, seed=seed * 1000 + k, side=1 if k % 2 == 0 else -1)
                vals.append(r["is_net_bps"] if r["status"] == "OK" else None)
            costs[gname][p] = vals
    return {"stream": stream, "qty_lots": qty, "costs": costs}


def _dirs_from_costs(costs: dict[str, list]) -> dict[str, int]:
    pols = list(costs)
    out = {}
    for i, a in enumerate(pols):
        for b in pols[i + 1:]:
            pa = [x for x in costs[a]]
            pb = [x for x in costs[b]]
            pairs = [(x, y) for x, y in zip(pa, pb) if x is not None and y is not None]
            out[f"{a}>{b}"] = twin.direction([x for x, _ in pairs], [y for _, y in pairs])
    return out


def _synthetic_dirs(syn: dict) -> dict[str, int]:
    per_gen = [_dirs_from_costs({p: syn["costs"][g][p] for p in RANKING_POLICIES}) for g in syn["costs"]]
    keys = per_gen[0].keys()
    return {k: twin.consensus({str(i): d[k] for i, d in enumerate(per_gen)}) for k in keys}


def _hybrid_dirs(day: dict) -> dict[str, int]:
    return _dirs_from_costs({p: [r[0] if r else None for r in day["outcomes"][p]] for p in RANKING_POLICIES})


def _bounded_dirs(bounds_result: dict, day: str) -> dict[str, int]:
    pairs = (bounds_result.get("ranking") or {}).get(day, {})
    return {k: (1 if v["lower"] > 0 else -1 if v["upper"] < 0 else 0) for k, v in pairs.items()}


def analyse(train_days: list[dict], eval_days: list[dict], syn: dict | None, bounds_result: dict | None) -> dict:
    from ...v08.statistics import spearman
    policies = list(RANKING_POLICIES)
    Xtr, costs_tr = [], {p: [] for p in policies}
    for d in train_days:
        for w, c in enumerate(d["context"]):
            vals = [d["outcomes"][p][w] for p in policies]
            if any(v is None for v in vals):
                continue
            Xtr.append(c)
            for p, v in zip(policies, vals):
                costs_tr[p].append(v[0])
    Xtr = np.asarray(Xtr)
    models = CostModels().fit(Xtr, {p: np.asarray(v) for p, v in costs_tr.items()})
    _, dis_tr = select(models.predict(Xtr), policies)
    thr = float(np.quantile(dis_tr, 0.75))
    best_global = min(policies, key=lambda p: float(np.mean(costs_tr[p])))
    syn_dirs = _synthetic_dirs(syn) if syn else {}
    units = {}
    for d in eval_days:
        X = np.asarray(d["context"])
        ok = [w for w in range(len(X)) if all(d["outcomes"][p][w] is not None for p in policies)]
        if len(ok) < 6:
            continue
        real = np.asarray([[d["outcomes"][p][w][0] for p in policies] for w in ok])
        pred = models.predict(X[ok])
        choice, dis = select(pred, policies)
        idx = np.asarray([policies.index(c) for c in choice])
        chosen = real[np.arange(len(ok)), idx]
        best = real.min(1)
        regret = chosen - best
        abst = abstain(dis, thr)
        twap = real[:, policies.index("twap")]
        final = np.where(abst, twap, chosen)
        wrong_always = (chosen - best) > 0.5
        wrong_abst = (final - best) > 0.5
        bg = real[:, policies.index(best_global)]
        sp = d["outcomes"]
        km = [r[0] for r in sp["survival_passive:km"] if r]
        cal = [r[0] for r in sp["survival_passive:calibrated"] if r]
        ch = [r[1] for r in sp["mpc_chance"] if r]
        st = [r[1] for r in sp["mpc_stochastic"] if r]
        hyb = _hybrid_dirs(d)
        u = {"windows": len(ok), "h8": float(np.mean(km) - np.mean(cal)) if km and cal else None,
             "h18": float(spearman(dis, regret)) if np.std(dis) > 0 and np.std(regret) > 0 else None,
             "h19": float(wrong_always.mean() - wrong_abst.mean()),
             "h22": float(np.mean(ch) - np.mean(st)) if ch and st else None,
             "h22_cost_diff": float(np.mean([r[0] for r in sp["mpc_chance"] if r])
                                    - np.mean([r[0] for r in sp["mpc_stochastic"] if r])),
             "h25": float(bg.mean() - chosen.mean()), "best_global": best_global,
             "abstain_rate": float(abst.mean()), "meta_regret": float(regret.mean())}
        half = np.arange(len(ok)) % 2 == 1
        u["h25_odd"], u["h25_even"] = float(bg[half].mean() - chosen[half].mean()), \
            float(bg[~half].mean() - chosen[~half].mean())
        u["h19_odd"] = float(wrong_always[half].mean() - wrong_abst[half].mean())
        u["h19_even"] = float(wrong_always[~half].mean() - wrong_abst[~half].mean())

        def sp_dis(sel):
            if np.std(dis[sel]) > 0 and np.std(regret[sel]) > 0:
                return float(spearman(dis[sel], regret[sel]))
            return None
        u["h18_odd"], u["h18_even"] = sp_dis(half), sp_dis(~half)

        def win_diff(a_, b_, parity):
            d_ = [x - y for i, (x, y) in enumerate(zip(a_, b_)) if x is not None and y is not None and i % 2 == parity]
            return float(np.mean(d_)) if d_ else None
        kmw = [r[0] if r else None for r in sp["survival_passive:km"]]
        calw = [r[0] if r else None for r in sp["survival_passive:calibrated"]]
        chw = [r[1] if r else None for r in sp["mpc_chance"]]
        stw = [r[1] if r else None for r in sp["mpc_stochastic"]]
        u["h8_odd"], u["h8_even"] = win_diff(kmw, calw, 1), win_diff(kmw, calw, 0)
        u["h22_odd"], u["h22_even"] = win_diff(chw, stw, 1), win_diff(chw, stw, 0)
        keys = sorted(hyb)
        odd_k, even_k = set(keys[1::2]), set(keys[0::2])

        def sub(dct, ks):
            return {k: v for k, v in dct.items() if k in ks}
        if syn_dirs:
            a, n = twin.agreement(syn_dirs, hyb)
            u["h29"], u["h29_pairs"] = a, n
            u["h29_odd"] = twin.agreement(sub(syn_dirs, odd_k), sub(hyb, odd_k))[0]
            u["h29_even"] = twin.agreement(sub(syn_dirs, even_k), sub(hyb, even_k))[0]
        if bounds_result is not None:
            bd = _bounded_dirs(bounds_result, d["day"])
            a, n = twin.agreement(hyb, bd)
            u["h30"], u["h30_pairs"] = a, n
            u["h30_odd"] = twin.agreement(sub(hyb, odd_k), sub(bd, odd_k))[0]
            u["h30_even"] = twin.agreement(sub(hyb, even_k), sub(bd, even_k))[0]
            if syn_dirs:
                cons = {k: twin.consensus({"syn": syn_dirs.get(k, 0), "hyb": hyb.get(k, 0)}) for k in hyb}

                def h31(ks):
                    ac, _ = twin.agreement(sub(cons, ks), sub(bd, ks))
                    asyn, _ = twin.agreement(sub(syn_dirs, ks), sub(bd, ks))
                    return (ac - asyn) if ac is not None and asyn is not None else None
                u["h31"], u["h31_odd"], u["h31_even"] = h31(set(keys)), h31(odd_k), h31(even_k)
        units[d["day"]] = u
    return {"units": units, "best_global": best_global, "abstain_threshold": thr}


def run(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    t0 = time.time()
    notional = {k: v["notional_usd"] for k, v in protocol["mandates"]["classes"].items()}["medium"]
    if not getattr(args, "design", None):            # pilot: registered explore slots; else the sealed design
        res = common.units(args, "explore", common.BYBIT_STREAMS, protocol["pools"]["explore"]["count"])
        plan = {}
        for s in common.BYBIT_STREAMS:
            days = sorted(res[s]["datasets"], key=lambda d: d[-10:])
            plan[s] = (days[: len(days) // 2], days[len(days) // 2:])
        design_name = "explore-decisions"
    else:
        design = common.design_document(args)
        plan = {s: (list(v["train"]), list(v["test"])) for s, v in design["plan"].items()}
        design_name = args.design
    bounds_result = common.read_input(args, "bounds") if any(k == "bounds" for k, _ in args.input) else None
    out = common.start(args)
    tasks = []
    for s, (tr, te) in plan.items():
        inst = s.split(":")[1]
        lot = float(next(d for d in protocol["datasets"] if d["instrument"] == inst)["lot_size"])
        for d in tr + te:
            tasks.append(Task(f"h|{d}", "lob.v09.studies.decisions:hybrid_day",
                              {"day": d, "train": tr, "root": str(root), "design": design_name, "seed": args.seed,
                               "notional": float(notional[inst])}, namespace=args.namespace))
        tasks.append(Task(f"s|{s}", "lob.v09.studies.decisions:synthetic_task",
                          {"stream": s, "train": tr, "root": str(root), "design": design_name, "seed": args.seed,
                           "notional": float(notional[inst]), "lot": lot}, namespace=args.namespace))
    rows, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args),
                           checkpoint=root / (args.out + ".checkpoint"), design={"plan": plan}, inputs=list(plan))
    by = {t.task_id: r for t, r in zip(sorted(tasks, key=lambda t: t.task_id), rows)}
    units = {}
    for s, (tr, te) in plan.items():
        a = analyse([by[f"h|{d}"] for d in tr], [by[f"h|{d}"] for d in te], by[f"s|{s}"], bounds_result)
        units.update(a["units"])
    hours = (time.time() - t0) / 3600 / max(len(units), 1)
    caps = common.caps_for(protocol, ("incremental_l2", "trades_with_side", "queue_clearance_observable"))
    vals = list(units.values())

    def rec(key):
        pairs = [(u.get(f"{key}_odd"), u.get(f"{key}_even")) for u in vals
                 if u.get(f"{key}_odd") is not None and u.get(f"{key}_even") is not None]
        return common.pilot_record([u.get(key) for u in vals], [a for a, _ in pairs], [b for _, b in pairs],
                                   caps=caps, hours=hours)

    pilot = {h: rec(h.lower()) for h in ("H8", "H18", "H19", "H22", "H25", "H29", "H30", "H31")}
    result = {"namespace": args.namespace, "plan": plan, "units": units,
              "pilot": pilot if args.namespace in ("explore", "pilot", "dev") else None, "records": pilot,
              "synthetic": {s: {"qty_lots": by[f"s|{s}"]["qty_lots"]} for s in plan}, "runner": info,
              "label": "HYBRID (E3) and SYNTHETIC outcomes; bounded directions from the bounds run (E4/E5)"}
    datasets = [d for tr, te in plan.values() for d in tr + te]
    return common.finish(args, out, analysis=ANALYSIS, config={"plan": plan, "seed": args.seed,
                                                               "synthetic_seeds": SYN_SEEDS}, result=result,
                         datasets=datasets)
