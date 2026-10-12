"""``ood-study-v08``: OOD detection (M19).

Reference: pre-trade summaries of in-distribution development cases (selection worlds x regimes). Thresholds at
the registered 5% false-positive rate on a separate in-distribution calibration set. Evaluated: false-positive
rate on held-out in-distribution cases, detection rate for every registered stress scenario, and decision
degradation versus score (Spearman correlation between a case's OOD score and the regret of the regime selector,
using the abstention study's design with fewer seeds). Detectors: support, Mahalanobis, kNN, conformal,
ensemble disagreement of the development fits, posterior predictive surprise under the nominal fit.
"""
from __future__ import annotations

import numpy as np

from .. import protocol as pr
from ..ood import Reference, evaluate, surprise_score, threshold
from ..selection import Selector, evaluate as sel_evaluate
from ..statistics import spearman
from . import common as C
from .cases import POLICIES, build_cases, disagreement, first_seed, prefixes, regime_worlds, stressed_worlds

DEFAULTS = {"tiny": {"worlds": 2, "seeds": 1, "policy_seeds": 1, "mandates": {"primary": {}}},
            "small": {"worlds": 6, "seeds": 1, "policy_seeds": 4, "mandates": {"primary": {}}},
            "medium": {"worlds": None, "seeds": 2, "policy_seeds": 8, "mandates": {"primary": {}}}}
DETECTORS = ("support", "mahalanobis", "knn", "conformal", "ensemble", "surprise")


def _nll(prefix: dict, g) -> float:
    from .cases import disagreement as _  # noqa: F401  (same frame reconstruction)
    from ..market.frame import EventFrame
    f = prefix["frame"]
    n = len(f["counts"])
    frame = EventFrame(np.asarray(f["counts"], np.int64), np.asarray(f["state"], float), np.zeros(n, np.int64),
                       np.zeros(n, np.int64), np.zeros((n, 2, 10)), np.zeros((n, 2)), np.zeros(n + 1),
                       np.asarray(f["valid"], bool))
    return float(-np.nanmean(g.loglik(frame)))


def run(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    wrun = C.input_run(args, "worlds")
    prun = C.input_run(args, "posterior")
    post = C.read_result(prun)
    fits = [C.generator_from(g) for g in post["fits"].values()]
    nominal = C.generator_from(post["fits"]["poisson"])
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    sel = C.load_world_set(wrun, "selection").worlds[1:]
    hold = C.load_world_set(wrun, "heldout").worlds
    if d.get("worlds"):
        sel, hold = sel[: d["worlds"]], hold[: d["worlds"]]
    train_w, id_w = regime_worlds(sel), regime_worlds(hold)
    shifts = {s: stressed_worlds(hold, [s]) for s in protocol["worlds"]["stress_scenarios"]}
    ref_seeds = pr.seeds("dev", "ood.reference", d["seeds"])
    cal_seeds = pr.seeds("dev", "ood.calibration", d["seeds"])
    test_seeds = C.seeds(args, "ood.test", d["seeds"])

    def collect(worlds, seed_list, tag):
        out_list = []
        for k, s in enumerate(seed_list):
            pre = prefixes(args, out, worlds, {w.world_id: s for w in worlds}, f"{tag}-{k}")
            out_list.extend(pre[w.world_id] for w in worlds)
        return out_list

    ref_p, cal_p, id_p = collect(train_w, ref_seeds, "ref"), collect(train_w, cal_seeds, "cal"), \
        collect(id_w, test_seeds, "id")
    shift_p = {s: collect(ws, test_seeds, f"shift-{s}") for s, ws in shifts.items()}
    ref = Reference(np.asarray([p["summaries"] for p in ref_p], float))
    ref_nll = np.asarray([_nll(p, nominal) for p in ref_p])

    def scores(method, plist):
        x = np.asarray([p["summaries"] for p in plist], float)
        if method == "ensemble":
            return np.asarray([disagreement(p, fits) for p in plist])
        if method == "surprise":
            return surprise_score(np.asarray([_nll(p, nominal) for p in plist]), ref_nll)
        return ref.score(method, x)

    detectors = {}
    for method in DETECTORS:
        thr = threshold(scores(method, cal_p), 0.05)
        detectors[method] = evaluate(scores(method, id_p), {s: scores(method, p) for s, p in shift_p.items()}, thr)
    # decision degradation versus OOD score (regime selector, Mahalanobis score)
    registered = C.registered_policies(root)
    m = C.mandate(root)
    pseeds = C.seeds(args, "ood.policies", d["policy_seeds"])
    tseeds = pr.seeds("dev", "ood.policies.train", d["policy_seeds"])
    ood_w = stressed_worlds(hold, protocol["worlds"]["ood_scenarios"])
    pol = [(p, registered[p]) for p in POLICIES]
    train_rows = C.evaluate(args, out, train_w, pol, m, tseeds, tag="policy-train")
    rows = C.evaluate(args, out, id_w + ood_w, pol, m, pseeds, tag="policy-test")
    selector = Selector("regime", list(POLICIES), safe_baseline="twap").fit(build_cases(train_rows, train_w))
    cases = build_cases(rows, id_w + ood_w)
    pre = prefixes(args, out, id_w + ood_w, first_seed(pseeds, id_w + ood_w), "policy-prefix")
    for c in cases:
        c.ood_score = float(ref.mahalanobis(np.asarray(pre[c.case_id]["summaries"], float))[0])
    ev = sel_evaluate(selector, cases, margin=0.5)
    degradation = {"spearman_score_vs_regret": spearman([r["ood_score"] for r in ev["rows"]],
                                                        [r["regret"] for r in ev["rows"]]), "cases": len(cases)}
    result = {"detectors": detectors, "degradation": degradation, "fpr_target": 0.05,
              "labels": {"evidence": "SYNTHETIC", "known_shifts": "registered stress scenarios"}}
    return C.finish(args, out, analysis="ood_study", config={"design": d}, result=C.json_safe(result),
                    inputs=[wrun.relative_to(root).as_posix(), prun.relative_to(root).as_posix()])
