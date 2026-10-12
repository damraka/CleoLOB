"""``robust-execution-v08``: robust selection over registered plausible worlds (M24, H8, brief section 36).

Stage 1 (development seeds, selection worlds): the registered parametric schedule family is evaluated and every
criterion (nominal, posterior-expected, worst-case, minimax-regret, CVaR, KL-DRO; with and without the completion
constraint) picks a candidate. Stage 2 (run-namespace seeds, held-out worlds never used for selection): every
candidate plus the registered baselines are evaluated; out-of-sample regret is computed against the best candidate
in each world. H8 compares the nominal and the CVaR choices by CVaR(0.8) over worlds of regret (hierarchical paired
bootstrap that recomputes regrets in every replicate).
"""
from __future__ import annotations

import itertools

import numpy as np

from .. import protocol as pr
from ..robust import select_all
from ..statistics import cvar
from . import common as C
from .mpc import paired_matrix

DEFAULTS = {
    "tiny": {"worlds": 3, "seeds_dev": 2, "seeds_eval": 2, "boot": 200, "mandates": {"primary": {}}},
    "small": {"worlds": 8, "seeds_dev": 6, "seeds_eval": 10, "boot": 500, "mandates": {"primary": {}}},
    "medium": {"worlds": None, "seeds_dev": 16, "seeds_eval": 32, "boot": 2000,
               "mandates": {"primary": {}, "large": {"qty": 400}}},
}
LEVEL, RHO, H8_ALPHA = 0.8, 0.1, 0.01
BASELINES = ("twap", "ac", "mpc", "adaptive_twap")


def candidates(root) -> list[str]:
    grid = pr.load_protocol(root / pr.PROTOCOL_PATH)["policies"]["robust_family"]["param_schedule"]
    return [C.candidate_name(u, p, c) for u, p, c in itertools.product(grid["urgency"], grid["passive"], grid["catchup"])]


def _matrix(rows, names, worlds, value="economic_cost_bps"):
    wm = C.world_means(rows, value)
    return np.asarray([[wm.get((w, n)) if wm.get((w, n)) is not None else np.nan for w in worlds] for n in names], float)


def run(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    wrun = C.input_run(args, "worlds")
    sel_all = C.load_world_set(wrun, "selection").worlds
    hold = C.load_world_set(wrun, "heldout").worlds
    nominal = sel_all[0]
    sel = sel_all[1:]
    if d.get("worlds"):
        sel, hold = sel[: d["worlds"]], hold[: d["worlds"]]
    cands = candidates(root)
    registered = C.registered_policies(root)
    result = {"candidates": cands, "mandates": {}}
    for mname, overrides in d["mandates"].items():
        m = C.mandate(root, **overrides)
        dev_seeds = pr.seeds("dev", f"robust.dev.{mname}", d["seeds_dev"])
        dev_rows = C.evaluate(args, out, [nominal] + sel, [(c, {}) for c in cands], m, dev_seeds, tag=f"dev-{mname}")
        sel_ids = [w.world_id for w in sel]
        cost = _matrix(dev_rows, cands, sel_ids)
        comp = _matrix(dev_rows, cands, sel_ids, "actual_completion")
        nominal_cost = _matrix(dev_rows, cands, [nominal.world_id])[:, 0]
        ok = ~np.isnan(cost).any(1)
        names_ok = [c for c, k in zip(cands, ok) if k]
        weights = np.ones(len(sel_ids))
        choice = select_all(names_ok, cost[ok], weights, level=LEVEL, rho=RHO)
        choice_c = select_all(names_ok, cost[ok], weights, level=LEVEL, rho=RHO, completion=comp[ok],
                              completion_threshold=0.95)
        # the nominal criterion uses the registered nominal world only (never part of the posterior set)
        k = int(np.nanargmin(np.where(ok, nominal_cost, np.inf)))
        nominal_choice = {"choice": cands[k], "value": float(nominal_cost[k]),
                          "values": {c: float(v) for c, v in zip(cands, nominal_cost) if np.isfinite(v)}}
        choice["criteria"]["nominal"] = nominal_choice
        choice_c["criteria"]["nominal"] = nominal_choice
        choice["criteria_agree"] = len({v.get("choice") for v in choice["criteria"].values()}) == 1
        eval_seeds = C.seeds(args, f"robust.eval.{mname}", d["seeds_eval"])
        policies = [(c, {}) for c in cands] + [(b, registered[b]) for b in BASELINES]
        rows = C.evaluate(args, out, hold, policies, m, eval_seeds, tag=f"eval-{mname}")
        hold_ids = sorted(w.world_id for w in hold)
        mats = {c: paired_matrix(rows, c, hold_ids, eval_seeds) for c in cands}
        complete = all(v is not None for v in mats.values())
        oos = {}
        if complete:
            stack = np.stack([mats[c] for c in cands])            # candidates x worlds x seeds
            means = stack.mean(2)
            regret = means - means.min(0, keepdims=True)
            for crit, info in choice["criteria"].items():
                if info.get("choice") is None:
                    continue
                i = cands.index(info["choice"])
                oos[crit] = {"choice": info["choice"], "mean_cost": float(means[i].mean()),
                             "worst_cost": float(means[i].max()), "cvar_cost": cvar(means[i], LEVEL),
                             "mean_regret": float(regret[i].mean()), "worst_regret": float(regret[i].max()),
                             "cvar_regret": cvar(regret[i], LEVEL)}
            h8 = _h8(stack, cands, choice, d["boot"], args.seed)
        else:
            h8 = {"status": "INVALID", "reason": "INVALID episodes among candidates"}
        result["mandates"][mname] = {
            "mandate": m.to_dict(), "dev_selection": choice, "dev_selection_completion_constrained": choice_c,
            "out_of_sample": oos, "h8": h8, "baselines": {b: _summary(rows, b) for b in BASELINES},
            "invalid_episodes": C.invalid_count(rows) + C.invalid_count(dev_rows), "episodes": len(rows) + len(dev_rows)}
    result["labels"] = {"evidence": "SYNTHETIC", "world_source": "registered selection / held-out plausible sets"}
    return C.finish(args, out, analysis="robust_execution", config={"design": d, "level": LEVEL, "rho": RHO},
                    result=C.json_safe(result), inputs=[wrun.relative_to(root).as_posix()])


def _h8(stack: np.ndarray, cands: list[str], choice: dict, n_boot: int, seed: int) -> dict:
    nom, rob = choice["criteria"]["nominal"].get("choice"), choice["criteria"]["cvar"].get("choice")
    if nom is None or rob is None:
        return {"status": "NOT_EVALUABLE", "reason": "a criterion had no feasible candidate"}
    i, j = cands.index(nom), cands.index(rob)
    means = stack.mean(2)
    regret = means - means.min(0, keepdims=True)
    point = cvar(regret[i], LEVEL) - cvar(regret[j], LEVEL)
    rng = np.random.default_rng(seed)
    _, w, s = stack.shape
    boots = np.empty(n_boot)
    for b in range(n_boot):
        wi = rng.integers(0, w, w)
        si = rng.integers(0, s, (w, s))
        m = stack[:, wi[:, None], si].mean(2)
        r = m - m.min(0, keepdims=True)
        boots[b] = cvar(r[i], LEVEL) - cvar(r[j], LEVEL)
    lo, hi = np.quantile(boots, [H8_ALPHA / 2, 1 - H8_ALPHA / 2])
    return {"nominal_choice": nom, "cvar_choice": rob, "same_choice": nom == rob,
            "estimate_cvar_regret_nominal_minus_cvar": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "alpha": H8_ALPHA, "p_value": float(min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))),
            "worst_regret_nominal": float(regret[i].max()), "worst_regret_cvar": float(regret[j].max())}


def _summary(rows, policy) -> dict:
    v = [r for r in rows if r["policy"] == policy and r["status"] == "VALID"]
    return {"economic_cost_mean_bps": float(np.mean([r["economic_cost_bps"] for r in v])) if v else None,
            "actual_completion": float(np.mean([r["actual_completion"] for r in v])) if v else None, "n": len(v)}
