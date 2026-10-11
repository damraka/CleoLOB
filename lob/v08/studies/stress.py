"""``adversarial-stress-v08``: structured stress worlds and bounded adversarial search (M28).

Every registered scenario is applied to the nominal world and to a subset of selection worlds; every registered
policy is evaluated. Reported: per-scenario cost and completion, ranking changes versus the unstressed world
(Kendall tau) and the policies whose rank changes most. The bounded adversarial search looks for plausible
worlds (Mahalanobis radius at the 99th percentile of plausible-world summaries) that maximize the cost of MPC
relative to TWAP within a registered box of rate multipliers and a fixed evaluation budget. Everything here is
labelled STRESS / SYNTHETIC; it is not empirical evidence.
"""
from __future__ import annotations

import numpy as np

from .. import protocol as pr
from ..execution.simulator import run_episode
from ..market.frame import summaries
from ..ood import Reference
from ..statistics import kendall_tau
from ..stress import adversarial_search, apply
from . import common as C

DEFAULTS = {"tiny": {"worlds": 1, "seeds": 2, "search_budget": 4, "search_seeds": 1},
            "small": {"worlds": 3, "seeds": 4, "search_budget": 12, "search_seeds": 2},
            "medium": {"worlds": 6, "seeds": 12, "search_budget": 40, "search_seeds": 4}}


def run(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    wrun = C.input_run(args, "worlds")
    sel = C.load_world_set(wrun, "selection").worlds
    bases = sel[: d["worlds"]]
    scenarios = pr.load_protocol(root / pr.PROTOCOL_PATH)["worlds"]["stress_scenarios"]
    registered = C.registered_policies(root)
    names = sorted(registered)
    pol = [(p, registered[p]) for p in names]
    m = C.mandate(root)
    seeds = C.seeds(args, "stress", d["seeds"])
    base_rows = C.evaluate(args, out, bases, pol, m, seeds, tag="base")
    base_means = C.world_means(base_rows)
    per_scenario = {}
    for s in scenarios:
        worlds = [apply(b, s) for b in bases]
        rows = C.evaluate(args, out, worlds, pol, m, seeds, tag=f"stress-{s}")
        means = C.world_means(rows)
        taus, costs, comp = [], {}, {}
        for b, w in zip(bases, worlds):
            x = [base_means.get((b.world_id, p)) for p in names]
            y = [means.get((w.world_id, p)) for p in names]
            if all(v is not None for v in x + y):
                taus.append(kendall_tau(x, y))
        for p in names:
            v = [r for r in rows if r["policy"] == p and r["status"] == "VALID"]
            costs[p] = float(np.mean([r["economic_cost_bps"] for r in v])) if v else None
            comp[p] = float(np.mean([r["actual_completion"] for r in v])) if v else None
        per_scenario[s] = {"kendall_tau_vs_unstressed": taus, "mean_tau": float(np.mean(taus)) if taus else None,
                           "cost": costs, "completion": comp, "invalid": C.invalid_count(rows)}
    plausible = [summaries(w.generator.simulate(1200, seed=7 + i, book=w.book)) for i, w in enumerate(sel)]
    ref = Reference(np.asarray(plausible))
    radius = float(np.quantile(ref.mahalanobis(np.asarray(plausible)), 0.99))
    search_seeds = C.seeds(args, "stress.search", d["search_seeds"])

    def objective(world) -> float:
        diffs = []
        for s in search_seeds:
            a = run_episode(world, C.policy("mpc", registered["mpc"]), m, s)
            b = run_episode(world, C.policy("twap"), m, s)
            if a["status"] == "VALID" and b["status"] == "VALID":
                diffs.append(a["economic_cost_bps"] - b["economic_cost_bps"])
        return float(np.mean(diffs)) if diffs else float("-inf")

    search = adversarial_search(sel[0], objective, reference=ref, radius=radius, budget=d["search_budget"],
                                seed=search_seeds[0] % 100000)
    result = {"scenarios": per_scenario, "base_cost": {p: float(np.mean([r["economic_cost_bps"] for r in base_rows
                                                                          if r["policy"] == p and r["status"] == "VALID"]))
                                                       for p in names},
              "adversarial_search": search, "labels": {"evidence": "SYNTHETIC_STRESS"}}
    return C.finish(args, out, analysis="adversarial_stress", config={"design": d}, result=C.json_safe(result),
                    inputs=[wrun.relative_to(root).as_posix()])
