"""offline-rl-v09 (M81, M82; H26, H27): offline RL with OPE support gates and safe improvement (SYNTHETIC, E1).

Per world cluster (strategic ecology calibrated on one day): 200 behaviour episodes with logged propensities (training
data), 60 further behaviour episodes for OPE, five learners (BC, CQL, IQL, TD3+BC, decision policy), an FQE critic,
DR estimates of each learner's improvement over the best classical baseline (TWAP, POV, passive deadline, MPC v2
stochastic; chosen by its own simulated mean on training seeds), and the true performance of every candidate by
simulation on 16 held-out seeds (common random numbers).

* H26 unit: mean cost(best classical) - mean cost(gated deployment), where the gate deploys the candidate with the
  highest DR mean only if the bootstrap 95 % lower bound of its DR improvement exceeds 0, else the best classical;
* H27 unit: harmful-deployment rate without the gate (deploy every candidate whose DR mean improvement > 0) minus
  with the gate (LCB > 0); harmful = deployed and truly worse than the baseline by more than 0.5 bps.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from .. import protocol as pr, rl, synthetic as SY
from ..runner import Task, run_tasks
from . import common

ANALYSIS = "offline-rl-v09"
CLASSICAL = ("twap", "pov", "passive_deadline", "mpc_stochastic")
N_TRAIN, N_OPE, N_EVAL = 200, 60, 16


def _seeds(base, label, n):
    return [int(hashlib.sha256(f"{base}|{label}|{k}".encode()).hexdigest()[:8], 16) for k in range(n)]


def world_task(day: str, root: str, design: str, base_seed: int, notional: float, lot: float) -> dict:
    t0 = time.time()
    cache = common.load_cache(day, root, design, "offline RL world")
    g, tick, _ = SY.calibrate([cache])
    del cache
    qty = SY.qty_for(g, tick, lot, notional)
    train = [rl.run_logged(g, qty, s, 1 if k % 2 == 0 else -1) for k, s in enumerate(_seeds(base_seed, "train", N_TRAIN))]
    ope_eps = [rl.run_logged(g, qty, s, 1 if k % 2 == 0 else -1) for k, s in enumerate(_seeds(base_seed, "ope", N_OPE))]
    learned = rl.learners(train)
    S, A, Pm, R, nxt, done, rtg = rl._stack(train)
    fqe = rl.fit_q(S, A, R, nxt, done, kind="fqe")
    behaviour_value = float(np.mean([sum(e["r"]) for e in ope_eps]))
    classical_train = SY.costs(g, CLASSICAL, qty, _seeds(base_seed, "cls", 8))
    best = SY.mean_order(classical_train)[0]
    base_value = -float(np.mean([v for v in classical_train[best] if v is not None]))
    est = {}
    for name, fn in learned.items():
        o = rl.ope(ope_eps, fn, fqe)
        imp = np.asarray(o["dr_episodes"]) - base_value
        est[name] = {"dr": o["dr"], "pdis": o["pdis"], "wpdis": o["wpdis"], "ood_action_rate": o["ood_action_rate"],
                     "improvement_mean": float(imp.mean()), "improvement_lcb": rl.lcb(imp, base_seed)}
    eval_seeds = _seeds(base_seed, "eval", N_EVAL)
    truth = SY.costs(g, [best], qty, eval_seeds)
    base_true = float(np.mean([v for v in truth[best] if v is not None]))
    true_cost = {}
    for name, fn in learned.items():
        c = []
        for k, s in enumerate(eval_seeds):
            from .. import twin
            r = twin.synthetic_episode(g, rl.Greedy(fn, name), qty=qty, seed=s, side=1 if k % 2 == 0 else -1)
            c.append(r["is_net_bps"])
        true_cost[name] = c
    mean_true = {n: float(np.mean([v for v in c if v is not None])) for n, c in true_cost.items()}
    top = max(est, key=lambda n: est[n]["improvement_mean"])
    gated_name = top if est[top]["improvement_lcb"] > 0 else best
    gated_cost = mean_true[top] if gated_name == top else base_true
    ungated = [n for n in est if est[n]["improvement_mean"] > 0]
    gated = [n for n in est if est[n]["improvement_lcb"] > 0]
    harmful = {n: mean_true[n] - base_true > 0.5 for n in est}
    rate_u = sum(harmful[n] for n in ungated) / len(est)
    rate_g = sum(harmful[n] for n in gated) / len(est)
    states = np.asarray([s for e in ope_eps for s in e["s"]], float)

    def half(parity):
        sel = [k for k in range(N_EVAL) if k % 2 == parity]
        bt = float(np.mean([truth[best][k] for k in sel if truth[best][k] is not None]))
        gt = bt if gated_name == best else float(np.mean([true_cost[top][k] for k in sel
                                                         if true_cost[top][k] is not None]))
        return bt - gt
    def harm_half(parity):
        sel = [k for k in range(N_EVAL) if k % 2 == parity]
        bt = float(np.mean([truth[best][k] for k in sel if truth[best][k] is not None]))
        h = {n: float(np.mean([true_cost[n][k] for k in sel if true_cost[n][k] is not None])) - bt > 0.5 for n in est}
        return (sum(h[n] for n in ungated) - sum(h[n] for n in gated)) / len(est)

    return {"day": day, "qty_lots": qty, "best_classical": best, "estimates": est, "true_mean_cost": mean_true,
            "h27_odd": harm_half(1), "h27_even": harm_half(0),
            "baseline_true_cost": base_true, "gated_choice": gated_name, "h26": base_true - gated_cost,
            "h26_odd": half(1), "h26_even": half(0), "h27": rate_u - rate_g, "harmful": harmful,
            "deployed_ungated": ungated, "deployed_gated": gated, "behaviour_value": behaviour_value,
            "support_fraction": rl.support(train, states), "seconds": time.time() - t0}


def run(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    if args.namespace in ("explore", "pilot", "dev"):
        days, design_name = common.explore_units(args, protocol), "explore-offline-rl"
    else:
        design = common.design_document(args)
        days, design_name = list(design["calibration_days"]), args.design
    base = common.seeds(args, ANALYSIS, len(days))
    out = common.start(args)
    notional = protocol["mandates"]["classes"]["medium"]["notional_usd"]
    tasks = []
    unique = len(set(days)) == len(days)          # confirm designs cycle calibration days over clusters
    for i, (d, s) in enumerate(zip(days, base)):
        inst = d.split("-")[2].upper()
        lot = float(next(x for x in protocol["datasets"] if x["instrument"] == inst)["lot_size"])
        tasks.append(Task(d if unique else f"{i:03d}|{d}", "lob.v09.studies.rl_study:world_task", {"day": d, "root": str(root), "design": design_name,
                                                                     "base_seed": int(s), "notional": float(notional[inst]),
                                                                     "lot": lot}, seed=int(s), namespace=args.namespace))
    rows, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args),
                           checkpoint=root / (args.out + ".checkpoint"), design={"days": days}, inputs=days)
    hours = float(np.mean([r["seconds"] for r in rows])) / 3600 if rows else 0.0
    caps = common.caps_for(protocol, ("propensities",))
    pilot = {"H26": common.pilot_record([r["h26"] for r in rows], [r["h26_odd"] for r in rows],
                                        [r["h26_even"] for r in rows], caps=caps, hours=hours),
             "H27": common.pilot_record([r["h27"] for r in rows], [r["h27_odd"] for r in rows],
                                        [r["h27_even"] for r in rows], caps=caps, hours=hours)}
    result = {"namespace": args.namespace, "worlds": {t.task_id: r for t, r in zip(sorted(tasks, key=lambda t: t.task_id), rows)}, "capabilities": rl.capability_report(),
              "pilot": pilot if args.namespace in ("explore", "pilot", "dev") else None, "records": pilot, "runner": info,
              "label": "SYNTHETIC (E1); OPE valid only because synthetic behaviour propensities are logged"}
    return common.finish(args, out, analysis=ANALYSIS, config={"days": days, "n_train": N_TRAIN, "n_ope": N_OPE,
                                                               "n_eval": N_EVAL}, result=result, datasets=days)
