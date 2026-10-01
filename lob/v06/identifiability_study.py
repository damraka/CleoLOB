"""M7 run: identifiability and sensitivity of calibration v3 on development data (H4).

Reads the sealed selection (near-optimal region, distinct set) and evaluates
local sensitivity, profiles and (EXPLORATORY) Morris effects on the development
target with the calibration fit seeds. Never reads a holdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..experiments.registry import PROJECT_ROOT
from . import protocol as pr
from .calibration import NAMES, run_tasks
from .calibration_study import DESIGN_RUN, read_selection, record_use
from .evidence import finalize, new_run, write_json
from .identifiability import (clean, morris_effects, morris_tasks, near_optimal_structure, profile_tasks, profiles,
                              sensitivity_matrix, sensitivity_tasks)
from .observables import pool
from .realism_study import load_sketches, read_design

ANALYSIS = "m7-identifiability"
BUDGET = {"step": 0.1, "profile_draws": 24, "profile_step": 0.12, "morris_trajectories": 12, "seconds": 1800.0}


def run(selection_run: str | Path, develop_run: str | Path, out: str | Path, *, root: Path = PROJECT_ROOT,
        budget: dict | None = None, label: str = "registered") -> dict:
    budget = {**BUDGET, **(budget or {})}
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    seeds = protocol["seeds"]
    selection = read_selection(selection_run, root=root)
    developed = json.loads((Path(develop_run) / "result.json").read_text(encoding="utf-8"))
    frozen = read_design(root / DESIGN_RUN, root=root)
    dev_id = pr.datasets_by_role(protocol, "development")[0]
    out = new_run(out)
    record_use(dev_id, "develop", ANALYSIS, f"identifiability diagnostics ({label}, sealed development sketches)", root)
    target = pool(load_sketches(root / DESIGN_RUN / "development-sketches.npz"))
    context = {"design": frozen["design"], "scales": frozen["objective_scales"], "targets": {"development": target}}
    base, tables = developed["base_config"], developed["size_tables"]
    selected = np.asarray(selection["selected"]["unit"], float)
    common = dict(base=base, tables=tables, seeds=seeds["identifiability"]["sensitivity_seeds"],
                  seconds=budget["seconds"], target="development")
    sens = sensitivity_tasks(selected, step=budget["step"], **common)
    prof = profile_tasks(selected, draws=budget["profile_draws"], step=budget["profile_step"],
                         seed=seeds["identifiability"]["profile_seed"], **common)
    morris, layout = morris_tasks(trajectories=budget["morris_trajectories"],
                                  seed=seeds["identifiability"]["morris_seed"], **common)
    records = {r["key"]: r for r in run_tasks(sens + prof + morris, context)}
    write_json(out, "evaluations.json", sorted(records.values(), key=lambda r: r["key"]))
    noise = records["sens-centre"]["seed_objectives"]
    finite = [v for v in noise if v is not None]
    noise_se = float(np.std(finite, ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else 0.0
    units = {k: np.asarray(m["unit"], float) for k, m in
             zip(selection["ensemble_members"], selection["ensemble"])}
    region_units = {}
    advanced = {c["key"]: c for c in developed["advanced"]}
    for key in selection["near_optimal"]:
        region_units[key] = np.asarray(advanced[key]["unit"], float)
    distinct = selection["distinct_near_optimal"]
    h4 = {"status": "ESTABLISHED" if len(distinct) >= 2 else "NOT_ESTABLISHED",
          "distinct_near_optimal": len(distinct), "near_optimal": len(selection["near_optimal"]),
          "rule": "materially distinct (L-infinity >= 0.25 in the unit box) near-optimal vectors >= 2",
          "pairwise_linf": {f"{a}|{b}": float(np.max(np.abs(units[a] - units[b])))
                            for i, a in enumerate(units) for b in list(units)[i + 1:]}}
    result = {"label": label, "budget": budget, "H4_identifiability": h4,
              "near_optimal_structure": near_optimal_structure(region_units),
              "sensitivity": sensitivity_matrix(records, selected, step=budget["step"]),
              "profiles": profiles(records, draws=budget["profile_draws"], noise_se=noise_se),
              "profile_noise_se": noise_se, "morris": morris_effects(records, layout),
              "failed_evaluations": sum(r["objective"] is None for r in records.values()),
              "parameters": list(NAMES),
              "interpretation": ("Identifiability at the resolution of the sealed objective and seeds; flat directions "
                                 "describe the calibration problem, not real market mechanisms.")}
    result = clean(result)
    finalize(out, analysis=f"{ANALYSIS}-{label}", dataset_ids=[dev_id],
             config={"budget": budget, "selection_run": Path(selection_run).name}, result=result, root=root,
             seeds=seeds["identifiability"])
    return result
