"""``mpc-study-v08``: controlled execution benchmark (brief section 33) and H9 (MPC/adaptive vs static).

Stage 1 (development seeds, selection worlds): every adaptive variant is scored by CVaR(0.8) over worlds of
world-mean economic cost; the lowest becomes the registered ``best_adaptive_dev`` member.
Stage 2 (run namespace seeds, held-out worlds): every registered policy is evaluated; H9 compares
``mpc``, ``mpc_robust`` and ``best_adaptive_dev`` with TWAP by the CVaR difference (hierarchical paired
bootstrap; Holm over the three members). Completion is reported separately from cost; with the registered
hard completion every policy completes unless liquidity is insufficient (counted).
"""
from __future__ import annotations

import numpy as np

from .. import protocol as pr
from ..execution.policies import ADAPTIVE
from ..statistics import cvar, hierarchical_paired, holm, rank_uncertainty
from . import common as C

DEFAULTS = {
    "tiny": {"worlds": 3, "seeds_dev": 2, "seeds_eval": 2, "boot": 200, "mandates": {"primary": {}}},
    "small": {"worlds": 8, "seeds_dev": 6, "seeds_eval": 12, "boot": 500, "mandates": {"primary": {}}},
    "medium": {"worlds": None, "seeds_dev": 16, "seeds_eval": 48, "boot": 2000,
               "mandates": {"primary": {}, "large": {"qty": 400}}},
}
LEVEL = 0.8
H9_ALPHA = 0.01


def benchmark_table(rows: list[dict], policies: list[str], level: float = LEVEL, seed: int = 0) -> dict:
    wm = C.world_means(rows)
    worlds = sorted({r["world_id"] for r in rows})
    episodes = C.by(rows, "world_id", "seed")
    regrets = {p: [] for p in policies}
    for _, group in episodes.items():
        costs = {r["policy"]: r["economic_cost_bps"] for r in group if r["status"] == "VALID"}
        if len(costs) == len(policies):
            best = min(costs.values())
            for p in policies:
                regrets[p].append(costs[p] - best)
    table = {}
    for p in policies:
        prow = [r for r in rows if r["policy"] == p]
        valid = [r for r in prow if r["status"] == "VALID"]
        means = [wm.get((w, p)) for w in worlds]
        clean = [m for m in means if m is not None]
        cost = np.asarray([r["economic_cost_bps"] for r in valid], float)
        table[p] = {"episodes": len(prow), "invalid": len(prow) - len(valid),
                    "economic_cost_mean_bps": float(cost.mean()) if len(cost) else None,
                    "economic_cost_sd_bps": float(cost.std(ddof=1)) if len(cost) > 1 else None,
                    "episode_cvar90_bps": cvar(cost, 0.9) if len(cost) else None,
                    "world_mean_cvar_bps": cvar(clean, level) if clean else None,
                    "worst_world_mean_bps": float(max(clean)) if clean else None,
                    "actual_completion": float(np.mean([r["actual_completion"] for r in valid])) if valid else None,
                    "within_horizon_completion": float(np.mean([r["within_horizon_completion"] for r in valid]))
                    if valid else None,
                    "optimization_cost_mean_bps": float(np.mean([r["optimization_cost_bps"] for r in valid
                                                                 if r["optimization_cost_bps"] is not None]))
                    if valid else None,
                    "participation": float(np.mean([r["participation"] for r in valid])) if valid else None,
                    "children": float(np.mean([r["children"] for r in valid])) if valid else None,
                    "regret_mean_bps": float(np.mean(regrets[p])) if regrets[p] else None,
                    "regret_p95_bps": float(np.quantile(regrets[p], 0.95)) if regrets[p] else None}
    matrix = np.asarray([[wm.get((w, p)) for p in policies] for w in worlds], dtype=object)
    complete = [i for i in range(len(worlds)) if all(x is not None for x in matrix[i])]
    ranks = rank_uncertainty(np.asarray(matrix[complete], float), policies, n_boot=500, seed=seed) if len(complete) > 2 \
        else None
    return {"policies": table, "rank_uncertainty": ranks, "worlds": len(worlds)}


def paired_matrix(rows: list[dict], policy: str, worlds: list[str], seeds: list[int],
                  value: str = "economic_cost_bps") -> np.ndarray | None:
    lookup = {(r["world_id"], r["seed"]): r for r in rows if r["policy"] == policy}
    out = np.full((len(worlds), len(seeds)), np.nan)
    for i, w in enumerate(worlds):
        for j, s in enumerate(seeds):
            r = lookup.get((w, s))
            if r is None or r["status"] != "VALID" or r.get(value) is None:
                return None
            out[i, j] = r[value]
    return out


def run(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    wrun = C.input_run(args, "worlds")
    sel = C.load_world_set(wrun, "selection").worlds
    hold = C.load_world_set(wrun, "heldout").worlds
    if d.get("worlds"):
        sel, hold = sel[: d["worlds"]], hold[: d["worlds"]]
    registered = C.registered_policies(root)
    names = sorted(registered)
    result = {"mandates": {}, "worlds": {"selection": len(sel), "heldout": len(hold)}}
    for mname, overrides in d["mandates"].items():
        m = C.mandate(root, **overrides)
        dev_seeds = pr.seeds("dev", f"mpc.dev.{mname}", d["seeds_dev"])
        dev_rows = C.evaluate(args, out, sel, [(p, registered[p]) for p in ADAPTIVE], m, dev_seeds, tag=f"dev-{mname}")
        dev_wm = C.world_means(dev_rows)
        dev_cvar = {}
        for p in ADAPTIVE:
            vals = [dev_wm.get((w.world_id, p)) for w in sel]
            dev_cvar[p] = cvar([v for v in vals if v is not None], LEVEL) if all(v is not None for v in vals) else None
        best = min((p for p in ADAPTIVE if dev_cvar[p] is not None), key=lambda p: dev_cvar[p])
        eval_seeds = C.seeds(args, f"mpc.eval.{mname}", d["seeds_eval"])
        rows = C.evaluate(args, out, hold, [(p, registered[p]) for p in names], m, eval_seeds, tag=f"eval-{mname}")
        worlds = sorted({w.world_id for w in hold})
        twap = paired_matrix(rows, "twap", worlds, eval_seeds)
        members = {}
        for label, policy in (("mpc", "mpc"), ("mpc_robust", "mpc_robust"), ("best_adaptive_dev", best)):
            mm = paired_matrix(rows, policy, worlds, eval_seeds)
            if twap is None or mm is None:
                members[label] = {"policy": policy, "status": "INVALID", "reason": "INVALID episodes in the comparison"}
                continue
            stats = hierarchical_paired(twap, mm, lambda x: cvar(x, LEVEL), alpha=H9_ALPHA / 3, n_boot=d["boot"],
                                        seed=args.seed + len(members))
            mean_diff = hierarchical_paired(twap, mm, np.mean, alpha=H9_ALPHA / 3, n_boot=d["boot"], seed=args.seed + 7)
            members[label] = {"policy": policy, "cvar_difference_twap_minus_member": stats,
                              "mean_difference_twap_minus_member": mean_diff}
        valid = [k for k, v in members.items() if "cvar_difference_twap_minus_member" in v]
        rejects = holm([members[k]["cvar_difference_twap_minus_member"]["p_value"] for k in valid], H9_ALPHA)
        for k, r in zip(valid, rejects):
            members[k]["holm_reject"] = bool(r)
        result["mandates"][mname] = {"mandate": m.to_dict(), "dev_cvar": dev_cvar, "best_adaptive_dev": best,
                                     "table": benchmark_table(rows, names, seed=args.seed), "h9_members": members,
                                     "invalid_episodes": C.invalid_count(rows), "episodes": len(rows),
                                     "dev_episodes": len(dev_rows)}
    result["labels"] = {"evidence": "SYNTHETIC (plausible worlds anchored to consumed development/calibration days)",
                        "cost_endpoint": "economic_cost_bps; completion reported separately"}
    return C.finish(args, out, analysis="mpc_benchmark", config={"design": d}, result=C.json_safe(result),
                    inputs=[wrun.relative_to(root).as_posix()],
                    seed_info={"namespace": args.namespace, "eval_study": "mpc.eval.<mandate>"})
