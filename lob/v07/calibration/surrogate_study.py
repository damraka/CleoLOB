"""M7 runs: emulator held-out error on the sealed v0.6 development candidates, and active versus random
calibration at equal budget on a synthetic recovery problem (simulator objective, known truth)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ...experiments.registry import PROJECT_ROOT
from ...sim_v2 import SimulatorSpec
from ...v06.calibration import NAMES, spec_from_unit, unit_from_spec
from ...v06.identifiability import clean
from ..evidence.runs import finalize, new_run
from ..posterior.study import _family_inputs, record_derived_use
from ..realism.scoring import run_tasks, simulate_sketches
from ..v06_compat import pool
from . import surrogate as sg

DEVELOP_RUN = "results/v06/m6/develop"
BUDGET, INITIAL, BATCH, REPLICATES = 48, 16, 8, 3
SEED = 77501


def emulator(root: Path) -> dict:
    from ...v06.evidence import verify_run as verify_v06
    report = verify_v06(root / DEVELOP_RUN, root=root)
    if not report["valid"]:
        raise ValueError(f"{DEVELOP_RUN} is not valid: {report['issues']}")
    candidates = json.loads((root / DEVELOP_RUN / "candidates.json").read_text(encoding="utf-8"))
    rows = [c for c in candidates if c["objective"] is not None]
    x = np.asarray([c["unit"] for c in rows], float)
    y = np.asarray([c["objective"] for c in rows], float)
    start = np.asarray([c["start"] for c in rows])
    train, test = start <= 5, start >= 6        # grouped by search start: refinement chains never straddle splits
    out = {"train": int(train.sum()), "test": int(test.sum()), "failed_candidates_excluded": len(candidates) - len(rows),
           "split": "v0.6 search starts 0-5 train, 6-7 test"}
    sub = np.random.default_rng(SEED).choice(np.flatnonzero(train), min(600, int(train.sum())), replace=False)
    out["gaussian_process"] = sg.emulator_report(sg.GaussianProcess(seed=SEED), x[sub], y[sub], x[test], y[test])
    out["regression_forest"] = sg.emulator_report(sg.RegressionForest(seed=SEED), x[train], y[train], x[test], y[test])
    out["neural_surrogate"] = "NOT_AVAILABLE: not justified at this data size (GP and forest suffice for screening)"
    return out


def run(out: str | Path, *, root: Path = PROJECT_ROOT, workers: int | None = None) -> dict:
    inputs = _family_inputs(root)
    frozen = inputs["frozen"]
    out = new_run(root / out)
    record_derived_use(root, "deribit-eth-perp-2020-04-01", "develop", "m7-surrogate",
                       root / DEVELOP_RUN / "candidates.json", "emulator training data (sealed v0.6 development search)")
    result = {"emulator": emulator(root)}
    truth = unit_from_spec(SimulatorSpec(inputs["selection"]["selected"]["config"],
                                         inputs["selection"]["selected"]["extensions"]))
    target = pool(simulate_sketches(spec_from_unit(truth, inputs["base"], inputs["tables"]), [SEED, SEED + 1], 3600.0,
                                    frozen["design"]))
    context = {"design": frozen["design"], "scales": frozen["objective_scales"], "targets": {"synthetic": target},
               "keep_sketches": False}
    counter = {"n": 0}

    def objective(units: np.ndarray) -> np.ndarray:
        tasks = []
        for u in units:
            counter["n"] += 1
            tasks.append((f"e{counter['n']}", spec_from_unit(u, inputs["base"], inputs["tables"]),
                          [SEED + 1000 + 2 * counter["n"], SEED + 1001 + 2 * counter["n"]], 1800.0, "synthetic"))
        return np.asarray([r["objective"] if r.get("objective") is not None else np.inf
                           for r in run_tasks(tasks, context, workers=workers)])

    arms = {"active": [], "random": []}
    for rep in range(REPLICATES):
        for name, fn in (("active", lambda s: sg.active_search(objective, BUDGET, len(NAMES), s, initial=INITIAL,
                                                               batch=BATCH)),
                         ("random", lambda s: sg.random_search(objective, BUDGET, len(NAMES), s))):
            r = fn(SEED + 10 * rep + (0 if name == "active" else 5))
            y = np.where(np.isfinite(r["y"]), r["y"], np.nan)
            arms[name].append({"best": r["best"], "best_distance_to_truth": float(np.max(np.abs(r["best_x"] - truth))),
                               "trace_best": np.fmin.accumulate(np.nan_to_num(y, nan=np.inf)).tolist(),
                               "failures": int(np.sum(~np.isfinite(r["y"])))})
    gains = [r["best"] - a["best"] for a, r in zip(arms["active"], arms["random"])]
    result["active_vs_random"] = {"budget": BUDGET, "initial": INITIAL, "batch": BATCH, "replicates": REPLICATES,
                                  "arms": arms, "mean_gain_objective": float(np.mean(gains)), "gains": gains,
                                  "status": "EXPLORATORY",
                                  "note": "synthetic recovery target (truth = v0.6 selected vector); 3 replicates"}
    finalize(out, analysis="m7-surrogate-active", dataset_ids=["deribit-eth-perp-2020-04-01"],
             config={"budget": BUDGET, "initial": INITIAL, "batch": BATCH, "replicates": REPLICATES, "seed": SEED},
             result=clean(result), root=root, seeds={"seed": SEED})
    return result
