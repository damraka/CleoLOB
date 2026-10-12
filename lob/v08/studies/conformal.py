"""``conformal-study-v08``: conformal coverage of episode economic cost (M20, H19).

Predictor (registered): for each policy, the plausible-world mean cost (mean over selection worlds of the
world-mean cost on development seeds). Split conformal residuals come from held-out worlds with a separate set of
development seeds; evaluation uses run-namespace seeds in the same held-out worlds (exchangeable with the
calibration episodes) and in registered stress worlds (shifted). Mondrian (by world family) and rolling
(worlds ordered by family, a drift-like sequence) variants are reported as secondary results.
"""
from __future__ import annotations

import numpy as np

from .. import conformal as cf
from .. import protocol as pr
from . import common as C
from .cases import POLICIES

DEFAULTS = {"tiny": {"worlds": 3, "seeds_pred": 2, "seeds_cal": 2, "seeds_eval": 2, "mandates": {"primary": {}}},
            "small": {"worlds": 8, "seeds_pred": 4, "seeds_cal": 6, "seeds_eval": 6, "mandates": {"primary": {}}},
            "medium": {"worlds": None, "seeds_pred": 8, "seeds_cal": 16, "seeds_eval": 16,
                       "mandates": {"primary": {}, "large": {"qty": 400}}}}
ALPHA = 0.1
STRESS = ("liquidity_drought", "market_order_burst", "volatility_shock", "asymmetric_order_flow")


def run(args) -> dict:
    from ..stress import apply
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    wrun = C.input_run(args, "worlds")
    sel = C.load_world_set(wrun, "selection").worlds[1:]
    hold = C.load_world_set(wrun, "heldout").worlds
    if d.get("worlds"):
        sel, hold = sel[: d["worlds"]], hold[: d["worlds"]]
    shifted = [apply(w, s) for w in hold for s in STRESS]
    registered = C.registered_policies(root)
    pol = [(p, registered[p]) for p in POLICIES]
    result = {"mandates": {}}
    for mname, overrides in d["mandates"].items():
        m = C.mandate(root, **overrides)
        pred_rows = C.evaluate(args, out, sel, pol, m, pr.seeds("dev", f"conformal.pred.{mname}", d["seeds_pred"]),
                               tag=f"pred-{mname}")
        wm = C.world_means(pred_rows)
        predictor = {p: float(np.mean([v for (w, q), v in wm.items() if q == p and v is not None])) for p in POLICIES}
        cal_rows = C.evaluate(args, out, hold, pol, m, pr.seeds("dev", f"conformal.cal.{mname}", d["seeds_cal"]),
                              tag=f"cal-{mname}")
        eval_seeds = C.seeds(args, f"conformal.eval.{mname}", d["seeds_eval"])
        ev_rows = C.evaluate(args, out, hold, pol, m, eval_seeds, tag=f"eval-{mname}")
        sh_rows = C.evaluate(args, out, shifted, pol, m, eval_seeds, tag=f"shift-{mname}")
        valid = lambda rows: [r for r in rows if r["status"] == "VALID"]  # noqa: E731
        cal, ev, sh = valid(cal_rows), valid(ev_rows), valid(sh_rows)
        per_policy, pooled = {}, {"true": [], "lo": [], "hi": []}
        for p in POLICIES:
            c = [r for r in cal if r["policy"] == p]
            e = [r for r in ev if r["policy"] == p]
            s = [r for r in sh if r["policy"] == p]
            if len(c) < 5 or not e:
                per_policy[p] = {"status": "NOT_EVALUABLE", "reason": "too few calibration or evaluation episodes"}
                continue
            yhat = predictor[p]
            split = cf.split([r["economic_cost_bps"] for r in c], [yhat] * len(c), [yhat] * len(e), ALPHA)
            exch = cf.coverage([r["economic_cost_bps"] for r in e], split["lower"], split["upper"], nominal=1 - ALPHA)
            sp = cf.split([r["economic_cost_bps"] for r in c], [yhat] * len(c), [yhat] * len(s), ALPHA)
            shift = cf.coverage([r["economic_cost_bps"] for r in s], sp["lower"], sp["upper"], nominal=1 - ALPHA)
            groups = [r["generator_family"] for r in c]
            mond = cf.mondrian([r["economic_cost_bps"] for r in c], [yhat] * len(c), groups, [yhat] * len(e),
                               [r["generator_family"] for r in e], ALPHA)
            mond_cov = cf.coverage([r["economic_cost_bps"] for r in e], mond["lower"], mond["upper"], nominal=1 - ALPHA)
            seq = sorted(e, key=lambda r: (r["generator_family"], r["world_id"], r["seed"]))
            roll = cf.rolling([r["economic_cost_bps"] for r in seq], [yhat] * len(seq), ALPHA, window=max(20, len(c)))
            hits = [x for x in roll["covered"] if x is not None]
            per_policy[p] = {"predictor_bps": yhat, "q_bps": split["q"], "exchangeable": exch, "shifted": shift,
                             "mondrian_by_family": mond_cov, "mondrian_fallback_groups": mond["fallback_groups"],
                             "rolling": {"coverage": float(np.mean(hits)) if hits else None, "n": len(hits)}}
            pooled["true"] += [r["economic_cost_bps"] for r in e]
            pooled["lo"] += split["lower"]
            pooled["hi"] += split["upper"]
        pooled_cov = cf.coverage(pooled["true"], pooled["lo"], pooled["hi"], nominal=1 - ALPHA) if pooled["true"] else None
        result["mandates"][mname] = {"per_policy": per_policy, "pooled_exchangeable": pooled_cov,
                                     "invalid_episodes": sum(C.invalid_count(x) for x in (pred_rows, cal_rows, ev_rows,
                                                                                          sh_rows))}
    result["labels"] = {"evidence": "SYNTHETIC", "nominal": 1 - ALPHA, "shifts": list(STRESS),
                        "validity": "distribution-free coverage requires exchangeability; shifted worlds violate it by design"}
    return C.finish(args, out, analysis="conformal_study", config={"design": d}, result=C.json_safe(result),
                    inputs=[wrun.relative_to(root).as_posix()])
