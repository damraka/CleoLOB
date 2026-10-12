"""``calibration-utility-v08``: decision-focused calibration (M21, H6) and the misspecification benchmark (section 34).

Truth worlds: the six development fits (one per core family). For each truth T a 10-minute data frame is simulated;
candidates C are the six families fitted on that frame (one well-specified, five misspecified) plus two variants
per family drawn from that family's real-data posterior. For every (T, C):

* generic and execution-aware calibration losses (summary distances scaled by T's own summary noise);
* predictive fit (one-step log score of a second T frame under C);
* policy costs, completion, ranking, top-policy retention and regret, with C's predictions from simulation in C
  and the truth from simulation in T (same seeds);
* misspecification diagnostics: market-rate bias, fit degradation versus the well-specified fit, completion
  error and an OOD response (Mahalanobis distance of T's summaries to C's replicate cloud).

H6: Spearman correlation between the execution-aware calibration loss and regret across pairs.
"""
from __future__ import annotations

import numpy as np

from ..decision_calibration import EXECUTION_SUMMARIES, associations, pair_metrics, summary_distance
from ..generators.base import MarketGenerator
from ..inference.abc import draw
from ..market.frame import summaries
from ..ood import Reference
from ..runner import Task, run_tasks
from ..worlds import World
from . import common as C

DEFAULTS = {"tiny": {"truths": 2, "variants": 0, "bins": 1200, "reps": 3, "seeds": 2, "boot": 200},
            "small": {"truths": 6, "variants": 1, "bins": 3000, "reps": 4, "seeds": 4, "boot": 500},
            "medium": {"truths": 6, "variants": 2, "bins": 6000, "reps": 6, "seeds": 8, "boot": 2000}}
POLICIES = ("twap", "vwap", "pov", "ac", "heuristic", "adaptive_twap", "mpc", "dynamic_ac")
FAMILIES = ("poisson", "hawkes", "queue_reactive", "regime_switching", "state_dependent", "mixture")
H6_ALPHA = 0.01


def _costs(world: World, mandate, seeds, registered) -> tuple[dict, dict]:
    from ..execution.simulator import run_episode
    cost, comp = {}, {}
    for p in POLICIES:
        rows = [run_episode(world, C.policy(p, registered[p]), mandate, s) for s in seeds]
        valid = [r for r in rows if r["status"] == "VALID"]
        cost[p] = float(np.mean([r["economic_cost_bps"] for r in valid])) if len(valid) == len(rows) else None
        comp[p] = float(np.mean([r["actual_completion"] for r in valid])) if valid else None
    return cost, comp


def truth_task(truth: dict, bins: int, reps: int, seeds: list[int], data_seed: int, registered: dict,
               mandate: dict) -> dict:
    g = MarketGenerator.from_dict(truth)
    data = g.simulate(bins, data_seed)
    rep = np.asarray([summaries(g.simulate(bins, data_seed + 1 + k)) for k in range(reps)])
    cost, comp = _costs(World("truth", g, g.book), C.mandate_from(mandate), seeds, registered)
    # noise floor: summaries that barely vary between replicates must not dominate the distance
    noise = np.maximum(rep.std(0), 1e-3 + 0.05 * np.abs(rep.mean(0)))
    return {"summaries": summaries(data).tolist(), "noise": noise.tolist(), "cost": cost,
            "completion": comp, "market_rate": float(data.counts[:, :2].sum(1).mean())}


def candidate_task(truth: dict, candidate: dict | None, family: str, bins: int, reps: int, seeds: list[int],
                   data_seed: int, registered: dict, mandate: dict, truth_info: dict) -> dict:
    t = MarketGenerator.from_dict(truth)
    data = t.simulate(bins, data_seed)
    test = t.simulate(bins, data_seed + 997)
    if candidate is None:
        from ..generators import get
        template = get(family)(book=t.book) if family != "mixture" else None
        if family == "mixture":
            from ..generators import MixtureGenerator
            template = MixtureGenerator.of([get(f)(book=t.book) for f in FAMILIES if f != "mixture"], book=t.book)
        g = template.fit(data)
        kind = "fit"
    else:
        g = MarketGenerator.from_dict(candidate)
        kind = "posterior_variant"
    sims = [g.simulate(bins, data_seed + 50 + k) for k in range(reps)]
    s = np.asarray([summaries(f) for f in sims])
    observed, noise = np.asarray(truth_info["summaries"]), np.asarray(truth_info["noise"])
    cost, comp = _costs(World("candidate", g, g.book), C.mandate_from(mandate), seeds, registered)
    ll = g.loglik(test)
    try:
        ood = float(Reference(np.vstack([s, s + 1e-9 * np.random.default_rng(0).standard_normal(s.shape)]))
                    .mahalanobis(observed)[0])
    except ValueError:
        ood = None
    return {"family": family, "kind": kind, "generic_loss": summary_distance(s.mean(0), observed, noise),
            "execution_loss": summary_distance(s.mean(0), observed, noise, EXECUTION_SUMMARIES),
            "predictive_fit": float(np.nanmean(ll)), "cost": cost, "completion": comp,
            "market_rate": float(np.mean([f.counts[:, :2].sum(1).mean() for f in sims])), "ood_distance": ood}


def run(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    prun = C.input_run(args, "posterior")
    post = C.read_result(prun)
    registered = C.registered_policies(root)
    reg = {p: registered[p] for p in POLICIES}
    m = C.mandate(root).to_dict()
    seeds = C.seeds(args, "calibration_utility", d["seeds"] + 100)
    policy_seeds, data_seeds = seeds[: d["seeds"]], seeds[d["seeds"]:]
    truths = {f: post["fits"][f] for f in FAMILIES[: d["truths"]]}
    ttasks = [Task(f"truth|{f}", "lob.v08.studies.calibration_utility:truth_task",
                   {"truth": g, "bins": d["bins"], "reps": d["reps"], "seeds": policy_seeds, "data_seed": data_seeds[i],
                    "registered": reg, "mandate": m}) for i, (f, g) in enumerate(truths.items())]
    tres, _ = run_tasks(ttasks, workers=args.workers, checkpoint=out.parent / (out.name + ".ckpt") / "truth",
                        binding={"study": "calibration_utility"})
    tinfo = {t.task_id.split("|")[1]: r for t, r in zip(sorted(ttasks, key=lambda x: x.task_id), tres)}
    ctasks = []
    for i, (tf, tg) in enumerate(truths.items()):
        for fam in FAMILIES:
            ctasks.append(Task(f"{tf}|{fam}|fit", "lob.v08.studies.calibration_utility:candidate_task",
                               {"truth": tg, "candidate": None, "family": fam, "bins": d["bins"], "reps": d["reps"],
                                "seeds": policy_seeds, "data_seed": data_seeds[i], "registered": reg, "mandate": m,
                                "truth_info": tinfo[tf]}))
            pfam = post["posteriors"][fam]
            base = C.generator_from(post["fits"][fam])
            for k, theta in enumerate(draw(pfam, d["variants"], seed=900 + i) if d["variants"] else []):
                ctasks.append(Task(f"{tf}|{fam}|var{k}", "lob.v08.studies.calibration_utility:candidate_task",
                                   {"truth": tg, "candidate": base.with_params(theta).to_dict(), "family": fam,
                                    "bins": d["bins"], "reps": d["reps"], "seeds": policy_seeds,
                                    "data_seed": data_seeds[i], "registered": reg, "mandate": m,
                                    "truth_info": tinfo[tf]}))
    cres, _ = run_tasks(ctasks, workers=args.workers, checkpoint=out.parent / (out.name + ".ckpt") / "candidates",
                        binding={"study": "calibration_utility"})
    rows = []
    for t, r in zip(sorted(ctasks, key=lambda x: x.task_id), cres):
        tf = t.task_id.split("|")[0]
        ti = tinfo[tf]
        if any(v is None for v in r["cost"].values()) or any(v is None for v in ti["cost"].values()):
            rows.append({"truth": tf, "candidate": t.task_id, "status": "INVALID"})
            continue
        pm = pair_metrics(ti["cost"], r["cost"], true_completion=ti["completion"], model_completion=r["completion"])
        well = next((x for x, tt in zip(cres, sorted(ctasks, key=lambda x: x.task_id))
                     if tt.task_id == f"{tf}|{tf}|fit"), None)
        rows.append({"truth": tf, "candidate": t.task_id, "family": r["family"], "kind": r["kind"], "status": "VALID",
                     "misspecified": r["family"] != tf, **pm, "generic_loss": r["generic_loss"],
                     "execution_loss": r["execution_loss"], "predictive_fit": r["predictive_fit"],
                     "fit_degradation": (well["predictive_fit"] - r["predictive_fit"]) if well else None,
                     "market_rate_bias": r["market_rate"] - ti["market_rate"], "ood_distance": r["ood_distance"]})
    valid = [r for r in rows if r["status"] == "VALID"]
    assoc = associations(valid, metrics=("execution_loss", "generic_loss", "cost_error", "completion_error"),
                         target="regret", n_boot=d["boot"], seed=args.seed)
    from ..statistics import spearman
    rng = np.random.default_rng(args.seed)
    x = np.asarray([r["execution_loss"] for r in valid])
    y = np.asarray([r["regret"] for r in valid])
    boots = [spearman(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(d["boot"]))] if len(x) > 4 \
        else []
    h6 = {"spearman": spearman(x, y) if len(x) > 4 else None, "n": len(x),
          "ci": [float(np.quantile(boots, H6_ALPHA / 2)), float(np.quantile(boots, 1 - H6_ALPHA / 2))] if boots else None,
          "alpha": H6_ALPHA}
    mis = {}
    for flag in (False, True):
        group = [r for r in valid if r["misspecified"] == flag and r["kind"] == "fit"]
        if group:
            mis["misspecified" if flag else "well_specified"] = {
                k: float(np.mean([r[k] for r in group if r.get(k) is not None]))
                for k in ("regret", "kendall_tau", "pairwise_accuracy", "top_retention", "cost_error", "completion_error",
                          "fit_degradation", "market_rate_bias", "ood_distance", "generic_loss", "execution_loss")
                if any(r.get(k) is not None for r in group)} | {"n": len(group)}
    result = {"rows": rows, "associations": assoc, "h6": h6, "misspecification": {"summary": mis},
              "truth_costs": {k: v["cost"] for k, v in tinfo.items()},
              "labels": {"evidence": "SYNTHETIC", "question": "does better calibration improve execution decisions?"}}
    return C.finish(args, out, analysis="calibration_utility", config={"design": d}, result=C.json_safe(result),
                    inputs=[prun.relative_to(root).as_posix()])
