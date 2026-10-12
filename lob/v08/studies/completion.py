"""``completion-study-v08``: completion-constrained execution (M22, H7) and Pareto analysis (M23).

H7 members (pov, heuristic, adaptive_twap, mpc) run on held-out worlds with the same seeds under completion mode
``none`` and ``hard``. Paired per (world, seed): completion increase and economic-cost difference (hard - none),
with hierarchical bootstrap intervals at the registered alpha. The completion summary keeps actual fills,
terminal inventory, hypothetical liquidation, penalty, economic cost and optimization objective separate.

M23: every registered policy's objective vector (economic cost, completion probability, CVaR, participation,
p95 tail, residual inventory, footprint, latency sensitivity from a 250 ms latency rerun) and the Pareto front;
the completion-constrained choice minimizes expected cost subject to the completion lower bound >= 0.95.
"""
from __future__ import annotations

import numpy as np

from ..execution.completion import completion_summary, constrained_choice, objectives, pareto_front
from ..execution.simulator import LatencyProfile
from ..statistics import hierarchical_paired
from . import common as C
from .mpc import paired_matrix

DEFAULTS = {
    "tiny": {"worlds": 3, "seeds": 2, "pareto_worlds": 2, "pareto_seeds": 2, "boot": 200, "mandates": {"primary": {}}},
    "small": {"worlds": 8, "seeds": 8, "pareto_worlds": 4, "pareto_seeds": 4, "boot": 500, "mandates": {"primary": {}}},
    "medium": {"worlds": None, "seeds": 24, "pareto_worlds": 8, "pareto_seeds": 8, "boot": 2000,
               "mandates": {"primary": {}, "large": {"qty": 400}}},
}
MEMBERS = ("pov", "heuristic", "adaptive_twap", "mpc")
ALPHA = 0.01 / len(MEMBERS)
SLOW = LatencyProfile.uniform(0.25)


def run(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    wrun = C.input_run(args, "worlds")
    hold = C.load_world_set(wrun, "heldout").worlds
    if d.get("worlds"):
        hold = hold[: d["worlds"]]
    registered = C.registered_policies(root)
    result = {"mandates": {}}
    for mname, overrides in d["mandates"].items():
        seeds = C.seeds(args, f"completion.{mname}", d["seeds"])
        arms = {}
        for mode in ("none", "hard"):
            m = C.mandate(root, completion=mode, **overrides)
            arms[mode] = C.evaluate(args, out, hold, [(p, registered[p]) for p in MEMBERS], m, seeds,
                                    tag=f"{mname}-{mode}")
        worlds = sorted(w.world_id for w in hold)
        members = {}
        for p in MEMBERS:
            summaries = {mode: completion_summary([r for r in arms[mode] if r["policy"] == p]) for mode in arms}
            a = paired_matrix(arms["hard"], p, worlds, seeds, "actual_completion")
            b = paired_matrix(arms["none"], p, worlds, seeds, "actual_completion")
            ca = paired_matrix(arms["hard"], p, worlds, seeds)
            cb = paired_matrix(arms["none"], p, worlds, seeds)
            if any(x is None for x in (a, b, ca, cb)):
                members[p] = {"status": "INVALID", "reason": "INVALID episodes", "summaries": summaries}
                continue
            members[p] = {"summaries": summaries,
                          "completion_increase": hierarchical_paired(a, b, np.mean, alpha=ALPHA, n_boot=d["boot"],
                                                                     seed=args.seed),
                          "cost_difference_hard_minus_none": hierarchical_paired(ca, cb, np.mean, alpha=2 * ALPHA,
                                                                                 n_boot=d["boot"], seed=args.seed + 1),
                          "unconstrained_completion": summaries["none"].get("completion_probability")}
        # M23: objectives and Pareto front over every registered policy (completion none, plus latency rerun)
        pw, ps = hold[: d["pareto_worlds"]], seeds[: d["pareto_seeds"]]
        m_none = C.mandate(root, completion="none", **overrides)
        prow = C.evaluate(args, out, pw, [(p, registered[p]) for p in sorted(registered)], m_none, ps,
                          tag=f"{mname}-pareto")
        slow = C.evaluate(args, out, pw, [(p, registered[p]) for p in sorted(registered)], m_none, ps,
                          tag=f"{mname}-pareto-slow", latency=SLOW.to_dict())
        points = {p: objectives([r for r in prow if r["policy"] == p],
                                latency_rows=[r for r in slow if r["policy"] == p]) for p in sorted(registered)}
        summaries_all = {p: completion_summary([r for r in prow if r["policy"] == p]) for p in sorted(registered)}
        result["mandates"][mname] = {
            "members": members, "pareto": pareto_front(points),
            "constrained_choice": constrained_choice(summaries_all, threshold=0.95),
            "completion_summaries_none": summaries_all,
            "invalid_episodes": sum(C.invalid_count(x) for x in (*arms.values(), prow, slow))}
    result["labels"] = {"evidence": "SYNTHETIC", "alpha_member": ALPHA,
                        "separation": "actual fill, terminal inventory, hypothetical liquidation, completion penalty, "
                                      "economic cost and optimization objective are separate fields"}
    return C.finish(args, out, analysis="completion_study", config={"design": d}, result=C.json_safe(result),
                    inputs=[wrun.relative_to(root).as_posix()])
