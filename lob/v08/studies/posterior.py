"""``posterior-study-v08``: posterior predictive checks 2.0 (M16) and the generator rejection gate (section 13).

Observed: registered windows of the consumed selection day (RETROSPECTIVE). Replicates: frames simulated from
posterior draws of each family. For every family: PPC failure fractions per horizon, discriminator AUC (ridge
logistic regression, 5-fold CV, real versus simulated window summaries), support coverage, variance-ratio and
queue-depletion mismatches and the execution-sensitive check (simulated TWAP cost versus the historical bounded
TWAP interval widened by 2 bps). A family is ACCEPTED_UNDER_GATE only if every registered criterion passes; the
gate never relabels the least-bad family as validated (NO_ACCEPTABLE_MODEL when none passes).
"""
from __future__ import annotations

import numpy as np

from .. import counterfactual as cf, ppc, protocol as pr, realdata
from ..execution.simulator import run_episode
from ..generators.base import MarketGenerator
from ..inference.abc import draw
from ..market.frame import SUMMARY_NAMES, summaries
from ..runner import Task, run_tasks
from ..worlds import World
from . import common as C

DEFAULTS = {"tiny": {"windows": 3, "replicates": 4, "bins": 1200, "policy_seeds": 2},
            "small": {"windows": 8, "replicates": 10, "bins": 3000, "policy_seeds": 4},
            "medium": {"windows": 24, "replicates": 24, "bins": 6000, "policy_seeds": 8}}
DATASET = "deribit-eth-perp-2020-12-01"


def replicate_task(generator: dict, theta: dict, bins: int, seed: int) -> dict:
    g = MarketGenerator.from_dict(generator).with_params(theta)
    f = g.simulate(bins, seed)
    return {k: v.tolist() for k, v in f.to_arrays().items() if not k.startswith("s__")}


def auc_cv(x_real: np.ndarray, x_sim: np.ndarray, folds: int = 5, seed: int = 0) -> float:
    x = np.vstack([x_real, x_sim])
    y = np.r_[np.ones(len(x_real)), np.zeros(len(x_sim))]
    mean, sd = x.mean(0), x.std(0) + 1e-9
    z = np.column_stack([np.ones(len(x)), (x - mean) / sd])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    scores = np.zeros(len(x))
    for k in range(folds):
        test = idx[k::folds]
        train = np.setdiff1d(idx, test)
        w = np.zeros(z.shape[1])
        for _ in range(50):
            p = 1 / (1 + np.exp(-np.clip(z[train] @ w, -30, 30)))
            grad = z[train].T @ (y[train] - p) - 1.0 * w
            hess = (z[train] * (p * (1 - p))[:, None]).T @ z[train] + np.eye(len(w))
            w += np.linalg.solve(hess, grad)
        scores[test] = z[test] @ w
    pos, neg = scores[y == 1], scores[y == 0]
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def run(args) -> dict:
    from ..market.frame import EventFrame
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    prun = C.input_run(args, "posterior")
    post = C.read_result(prun)
    gate_rules = pr.load_protocol(root / pr.PROTOCOL_PATH)["generator_gate"]["criteria"]
    frames, meta = realdata.load_frames(DATASET, use="select", design="posterior-study", root=root,
                                        purpose="posterior predictive checks and generator gate",
                                        windows=realdata.registered_windows()[: d["windows"]])
    real_summ = np.asarray([summaries(f) for f in frames])
    m = C.mandate(root)
    neutral_grid = cf.grid((0.0, 0.1), 0.5, impacts=("none", "persistent"), permanent=(0.0, 0.05))
    twap_bounds = [cf.bounds(f, lambda: C.policy("twap"), m, neutral_grid, start_bin=300) for f in frames]
    valid_b = [b for b in twap_bounds if b.get("status") == "VALID"]
    hist_lo = float(np.mean([b["economic_cost_bps"]["lower"] for b in valid_b])) if valid_b else None
    hist_hi = float(np.mean([b["economic_cost_bps"]["upper"] for b in valid_b])) if valid_b else None
    seeds = C.seeds(args, "posterior_study", 1000)
    families, ppc_summary = {}, {}
    for fi, (name, gd) in enumerate(sorted(post["fits"].items())):
        thetas = draw(post["posteriors"][name], d["replicates"], seed=950 + fi)
        tasks = [Task(f"{name}|{k:03d}", "lob.v08.studies.posterior:replicate_task",
                      {"generator": gd, "theta": th, "bins": len(frames[0]), "seed": seeds[(fi * 50 + k) % 1000]})
                 for k, th in enumerate(thetas)]
        reps, _ = run_tasks(tasks, workers=args.workers, checkpoint=None)
        rep_frames = [EventFrame.from_arrays({k: np.asarray(v) for k, v in r.items()}) for r in reps]
        sim_summ = np.asarray([summaries(f) for f in rep_frames])
        checks = [ppc.check(f, rep_frames) for f in frames]
        failure = float(np.mean([c["failure_fraction"] for c in checks]))
        lo, hi = sim_summ.min(0), sim_summ.max(0)
        pad = 0.05 * (hi - lo + 1e-9)
        support = float(np.mean(np.all((real_summ >= lo - pad) & (real_summ <= hi + pad), 1)))
        auc = auc_cv(real_summ, sim_summ, seed=args.seed)
        vr_i, dep_i = SUMMARY_NAMES.index("vol_ratio"), SUMMARY_NAMES.index("depletion_rate")
        vr_mismatch = float(abs(np.log(max(sim_summ[:, vr_i].mean(), 1e-9) / max(real_summ[:, vr_i].mean(), 1e-9))))
        dep_ratio = float(sim_summ[:, dep_i].mean() / max(real_summ[:, dep_i].mean(), 1e-9))
        g = C.generator_from(gd)
        world = World(f"{name}-fit", g, g.book)
        pseeds = seeds[900:900 + d["policy_seeds"]]
        sim_twap = float(np.mean([run_episode(world, C.policy("twap"), m, s)["economic_cost_bps"] for s in pseeds]))
        crit = {"discriminator_auc": {"value": auc, "pass": auc <= gate_rules["discriminator_auc_max"]},
                "support_coverage": {"value": support, "pass": support >= gate_rules["support_coverage_min"]},
                "ppc_failure_fraction": {"value": failure, "pass": failure <= gate_rules["ppc_failure_fraction_max"]},
                "variance_ratio_log_mismatch": {"value": vr_mismatch,
                                                "pass": vr_mismatch <= gate_rules["variance_ratio_log_mismatch_max"]},
                "queue_depletion_ratio": {"value": dep_ratio, "pass": gate_rules["queue_depletion_ratio_range"][0] <= dep_ratio
                                          <= gate_rules["queue_depletion_ratio_range"][1]},
                "execution_sensitive": {"value": sim_twap, "historical_interval": [hist_lo, hist_hi],
                                        "pass": hist_lo is not None and hist_lo - 2.0 <= sim_twap <= hist_hi + 2.0}}
        status = "ACCEPTED_UNDER_GATE" if all(c["pass"] for c in crit.values()) else "REJECTED"
        families[name] = {"criteria": crit, "status": status}
        ppc_summary[name] = {"failure_fraction": failure,
                             "by_horizon": {h: float(np.mean([c["failures"].get(h, 0) / max(c["checks"].get(h, 1), 1)
                                                              for c in checks])) for h in checks[0]["failures"]}}
    accepted = [k for k, v in families.items() if v["status"] == "ACCEPTED_UNDER_GATE"]
    realdata.mark_evaluated(DATASET, use="select", design="posterior-study", run=out.relative_to(root).as_posix(),
                            root=root)
    result = {"gate": {"families": families, "accepted": accepted,
                       "overall": "ACCEPTED_UNDER_GATE" if accepted else "NO_ACCEPTABLE_MODEL"},
              "ppc": {"summary": ppc_summary, "horizons": list(ppc.HORIZONS)}, "dataset": DATASET, "meta": meta,
              "labels": {"evidence": "RETROSPECTIVE (consumed selection day); EXPLORATORY",
                         "gate_rule": "every criterion must pass; the least-bad family is never relabelled as validated"}}
    return C.finish(args, out, analysis="posterior_study", config={"design": d}, result=C.json_safe(result),
                    datasets=[DATASET], inputs=[prun.relative_to(root).as_posix()])
