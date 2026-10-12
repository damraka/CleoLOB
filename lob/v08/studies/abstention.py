"""``policy-abstention-v08``: OOD-aware abstention and safe fallback (M27, H11, H12; brief section 37).

In-distribution test cases: held-out worlds in the three regimes. OOD cases: the same worlds under the registered
OOD stress scenarios. The regime selector is trained on development cases. The OOD score is the Mahalanobis
distance of the pre-trade summaries to development (in-distribution) cases; the uncertainty score is the
disagreement of the six development fits on the pre-trade window. Thresholds: 95th percentiles on a separate
in-distribution calibration set (development seeds). Abstention falls back to the registered safe baseline (TWAP).
"""
from __future__ import annotations

import numpy as np

from .. import protocol as pr
from ..ood import Reference, threshold
from ..selection import Abstention, Selector, evaluate
from ..statistics import paired_bootstrap, wilson
from . import common as C
from .cases import POLICIES, build_cases, disagreement, first_seed, prefixes, regime_worlds, stressed_worlds

DEFAULTS = {
    "tiny": {"worlds": 2, "seeds_train": 2, "seeds_test": 2, "boot": 200, "mandates": {"primary": {}}},
    "small": {"worlds": 6, "seeds_train": 4, "seeds_test": 6, "boot": 500, "mandates": {"primary": {}}},
    "medium": {"worlds": None, "seeds_train": 8, "seeds_test": 16, "boot": 2000,
               "mandates": {"primary": {}, "large": {"qty": 400}}},
}
ALPHA = 0.025
MARGIN, LARGE = 0.5, 2.0


def run(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    wrun = C.input_run(args, "worlds")
    prun = C.input_run(args, "posterior")
    fits = [C.generator_from(g) for k, g in C.read_result(prun)["fits"].items()]
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    scenarios = protocol["worlds"]["ood_scenarios"]
    sel = C.load_world_set(wrun, "selection").worlds[1:]
    hold = C.load_world_set(wrun, "heldout").worlds
    if d.get("worlds"):
        sel, hold = sel[: d["worlds"]], hold[: d["worlds"]]
    registered = C.registered_policies(root)
    result = {"mandates": {}, "ood_scenarios": scenarios}
    for mname, overrides in d["mandates"].items():
        m = C.mandate(root, **overrides)
        train_w, id_w, ood_w = regime_worlds(sel), regime_worlds(hold), stressed_worlds(hold, scenarios)
        train_seeds = pr.seeds("dev", f"abstention.train.{mname}", d["seeds_train"])
        cal_seeds = pr.seeds("dev", f"abstention.calibration.{mname}", 1)
        test_seeds = C.seeds(args, f"abstention.test.{mname}", d["seeds_test"])
        pol = [(p, registered[p]) for p in POLICIES]
        train_rows = C.evaluate(args, out, train_w, pol, m, train_seeds, tag=f"train-{mname}")
        id_rows = C.evaluate(args, out, id_w, pol, m, test_seeds, tag=f"id-{mname}")
        ood_rows = C.evaluate(args, out, ood_w, pol, m, test_seeds, tag=f"ood-{mname}")
        train = build_cases(train_rows, train_w)
        id_cases, ood_cases = build_cases(id_rows, id_w), build_cases(ood_rows, ood_w)
        ref_prefix = prefixes(args, out, train_w, first_seed(train_seeds, train_w), f"prefix-train-{mname}")
        cal_prefix = prefixes(args, out, train_w, first_seed(cal_seeds, train_w), f"prefix-cal-{mname}")
        id_prefix = prefixes(args, out, id_w, first_seed(test_seeds, id_w), f"prefix-id-{mname}")
        ood_prefix = prefixes(args, out, ood_w, first_seed(test_seeds, ood_w), f"prefix-ood-{mname}")
        ref = Reference(np.asarray([ref_prefix[c.case_id]["summaries"] for c in train], float))
        cal_scores = ref.mahalanobis(np.asarray([cal_prefix[w.world_id]["summaries"] for w in train_w], float))
        cal_unc = [disagreement(cal_prefix[w.world_id], fits) for w in train_w]
        thr_ood, thr_unc = threshold(cal_scores, 0.05), threshold(np.asarray(cal_unc), 0.05)
        for cases, pre in ((id_cases, id_prefix), (ood_cases, ood_prefix)):
            for c in cases:
                c.ood_score = float(ref.mahalanobis(np.asarray(pre[c.case_id]["summaries"], float))[0])
                c.uncertainty = disagreement(pre[c.case_id], fits)
        selector = Selector("regime", list(POLICIES), safe_baseline="twap", seed=args.seed).fit(train)
        rule = Abstention(thr_ood, thr_unc)
        all_cases = id_cases + ood_cases
        without = evaluate(selector, all_cases, margin=MARGIN)
        with_ = evaluate(selector, all_cases, margin=MARGIN, abstention=rule)
        inc = [float(a["incorrect"]) - float(b["incorrect"]) for a, b in zip(without["rows"], with_["rows"])]
        h11 = paired_bootstrap(inc, alpha=ALPHA, n_boot=d["boot"], seed=args.seed).to_dict()
        ood_ids = {c.case_id for c in ood_cases}
        lw = [r for r in with_["rows"] if r["case"] in ood_ids]
        lo = [r for r in without["rows"] if r["case"] in ood_ids]
        k_with, k_without = sum(r["regret"] > LARGE for r in lw), sum(r["regret"] > LARGE for r in lo)
        red = [float(a["regret"] > LARGE) - float(b["regret"] > LARGE) for a, b in zip(lo, lw)]
        h12 = {"with_fallback_rate": k_with / len(lw) if lw else None,
               "with_fallback_wilson": list(wilson(k_with, len(lw), ALPHA)) if lw else None,
               "without_fallback_rate": k_without / len(lo) if lo else None,
               "without_fallback_wilson": list(wilson(k_without, len(lo), ALPHA)) if lo else None,
               "reduction": paired_bootstrap(red, alpha=ALPHA, n_boot=d["boot"], seed=args.seed).to_dict()
               if len(red) > 1 else None, "n_ood_cases": len(lw), "large_degradation_bps": LARGE}
        result["mandates"][mname] = {
            "thresholds": {"ood": thr_ood, "uncertainty": thr_unc},
            "without_abstention": {k: v for k, v in without.items() if k != "rows"},
            "with_abstention": {k: v for k, v in with_.items() if k != "rows"},
            "abstention_rate_id": float(np.mean([r["abstained"] for r in with_["rows"] if r["case"] not in ood_ids])),
            "abstention_rate_ood": float(np.mean([r["abstained"] for r in lw])) if lw else None,
            "h11": h11, "h12": h12, "cases": {"train": len(train), "id": len(id_cases), "ood": len(ood_cases)},
            "invalid_episodes": sum(C.invalid_count(x) for x in (train_rows, id_rows, ood_rows))}
    result["labels"] = {"evidence": "SYNTHETIC", "safe_baseline": "twap", "margin_bps": MARGIN}
    return C.finish(args, out, analysis="policy_abstention", config={"design": d}, result=C.json_safe(result),
                    inputs=[wrun.relative_to(root).as_posix(), prun.relative_to(root).as_posix()])
