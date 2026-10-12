"""``policy-selection-v08``: contextual / regime-aware selection and mixture of experts (M27, H10).

Training cases: selection worlds in the three registered regimes, development seeds. Test cases: held-out worlds
in the three regimes, run-namespace seeds. Selectors: ``global`` (one policy), ``regime`` (best per known regime),
``moe`` (expected cost under regime probabilities from a nearest-centroid classifier on pre-trade summaries) and
``confident`` (regime choice only with bootstrap P(best) >= 0.6, else the registered safe baseline).
H10: regret(global) - regret(regime) per test case, bootstrap interval at the registered alpha.
"""
from __future__ import annotations

import numpy as np

from .. import protocol as pr
from ..selection import Selector, evaluate
from ..statistics import paired_bootstrap
from . import common as C
from .cases import POLICIES, build_cases, first_seed, prefixes, regime_worlds

DEFAULTS = {
    "tiny": {"worlds": 2, "seeds_train": 2, "seeds_test": 2, "boot": 200, "mandates": {"primary": {}}},
    "small": {"worlds": 6, "seeds_train": 4, "seeds_test": 6, "boot": 500, "mandates": {"primary": {}}},
    "medium": {"worlds": None, "seeds_train": 8, "seeds_test": 16, "boot": 2000,
               "mandates": {"primary": {}, "large": {"qty": 400}}},
}
H10_ALPHA = 0.01
MARGIN = 0.5


def classifier(train_prefix: dict, train_cases, test_prefix: dict, test_cases) -> None:
    """Nearest-centroid regime probabilities (softmax of negative standardized distances) on test cases."""
    x = np.asarray([train_prefix[c.case_id]["summaries"] for c in train_cases], float)
    mean, sd = x.mean(0), x.std(0) + 1e-9
    labels = np.asarray([c.regime for c in train_cases])
    centres = {r: ((x[labels == r] - mean) / sd).mean(0) for r in sorted(set(labels.tolist()))}
    for c in test_cases:
        z = (np.asarray(test_prefix[c.case_id]["summaries"], float) - mean) / sd
        d = np.asarray([np.linalg.norm(z - centres[r]) for r in sorted(centres)])
        p = np.exp(-(d - d.min()))
        c.regime_probs = (p / p.sum()).tolist()


def build(args, out, mname, overrides, d, root):
    wrun = C.input_run(args, "worlds")
    sel = C.load_world_set(wrun, "selection").worlds[1:]
    hold = C.load_world_set(wrun, "heldout").worlds
    if d.get("worlds"):
        sel, hold = sel[: d["worlds"]], hold[: d["worlds"]]
    train_w, test_w = regime_worlds(sel), regime_worlds(hold)
    registered = C.registered_policies(root)
    m = C.mandate(root, **overrides)
    train_seeds = pr.seeds("dev", f"selection.train.{mname}", d["seeds_train"])
    test_seeds = C.seeds(args, f"selection.test.{mname}", d["seeds_test"])
    train_rows = C.evaluate(args, out, train_w, [(p, registered[p]) for p in POLICIES], m, train_seeds,
                            tag=f"train-{mname}")
    test_rows = C.evaluate(args, out, test_w, [(p, registered[p]) for p in POLICIES], m, test_seeds,
                           tag=f"test-{mname}")
    train, test = build_cases(train_rows, train_w), build_cases(test_rows, test_w)
    tp = prefixes(args, out, train_w, first_seed(train_seeds, train_w), f"prefix-train-{mname}")
    sp = prefixes(args, out, test_w, first_seed(test_seeds, test_w), f"prefix-test-{mname}")
    classifier(tp, train, sp, test)
    return wrun, train, test, train_rows, test_rows, tp, sp, train_w, test_w, m


def run(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    result = {"mandates": {}}
    wrun = None
    for mname, overrides in d["mandates"].items():
        wrun, train, test, train_rows, test_rows, *_ = build(args, out, mname, overrides, d, root)
        selectors = {k: Selector(k, list(POLICIES), safe_baseline="twap", seed=args.seed).fit(train)
                     for k in ("global", "regime", "moe", "confident")}
        evals = {k: evaluate(s, test, margin=MARGIN) for k, s in selectors.items()}
        g = {r["case"]: r["regret"] for r in evals["global"]["rows"]}
        rg = {r["case"]: r["regret"] for r in evals["regime"]["rows"]}
        diffs = [g[c] - rg[c] for c in sorted(g)]
        h10 = paired_bootstrap(diffs, alpha=H10_ALPHA, n_boot=d["boot"], seed=args.seed).to_dict() if len(diffs) > 1 \
            else {"status": "NOT_EVALUABLE"}
        result["mandates"][mname] = {
            "selectors": {k: {kk: v for kk, v in e.items() if kk != "rows"} for k, e in evals.items()},
            "global_choice": selectors["global"].global_choice,
            "regime_choices": {str(r): POLICIES[int(np.argmin(v))] for r, v in selectors["regime"].regime_means.items()},
            "h10": h10, "train_cases": len(train), "test_cases": len(test),
            "invalid_episodes": C.invalid_count(train_rows) + C.invalid_count(test_rows)}
    result["labels"] = {"evidence": "SYNTHETIC", "regimes": "calm, active, thin (registered)", "margin_bps": MARGIN}
    return C.finish(args, out, analysis="policy_selection", config={"design": d}, result=C.json_safe(result),
                    inputs=[wrun.relative_to(root).as_posix()])
