"""M6/M8 runs: calibration-v3 development search, selection, near-optimal region and ensemble.

develop -> ``results/v06/m6/develop`` (development only)
select  -> ``results/v06/m6/select`` (selection day; seals ``m6-calibration-selection``)

The selection seal fixes the selected v3 model, the v0.5 control, the near-optimal
region, the materially distinct set and the ensemble members before any holdout
or policy study. Pilot runs use reduced budgets, are labelled and never seal.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..experiments.registry import PROJECT_ROOT
from . import protocol as pr
from .calibration import (NAMES, PARAMETERS, base_config, distinct_set, near_optimal, pareto_front, run_tasks,
                          search, size_tables, unit_from_spec)
from .data import acquire
from .evidence import finalize, new_run, write_json
from .history import block_raws, load
from .observables import FAMILIES, pool
from .realism_study import load_sketches, read_design

DESIGN_RUN = "results/v06/m1/design"
V05_SELECTION = "results/v05/m3/select/result.json"
DEVELOP_ANALYSIS, SELECT_ANALYSIS, SEAL = "m6-calibration-develop", "m6-calibration-select", "m6-calibration-selection"
REGISTERED = {"starts": 8, "global_draws": 96, "rounds": (0.25, 0.12, 0.06), "per_round": 64, "seconds": 1800.0,
              "rescore": 64, "advance": 32, "rescore_seconds": 3600.0, "selection_seconds": 3600.0,
              "ensemble_limit": 8, "distinct": 0.25, "near_optimal_relative": 0.10}


def record_use(dataset_id: str, use: str, analysis: str, purpose: str, root: Path) -> None:
    """Ledger entry for analyses that reuse sealed sketches of a dataset (no new download)."""
    acquire(dataset_id, use=use, analysis=analysis, purpose=purpose, root=root)


def v05_control(root: Path = PROJECT_ROOT) -> dict:
    selected = json.loads((root / V05_SELECTION).read_text(encoding="utf-8"))["selected"]
    return {"family": selected["family"], "config": selected["config"], "extensions": selected["extensions"],
            "source": V05_SELECTION, "sha256": pr.document_sha256({"config": selected["config"],
                                                                   "extensions": selected["extensions"]})}


def _context(design_run: dict, targets: dict) -> dict:
    return {"design": design_run["design"], "scales": design_run["objective_scales"], "targets": targets}


def develop(out: str | Path, *, root: Path = PROJECT_ROOT, budget: dict | None = None, label: str = "registered",
            design_run: str = DESIGN_RUN) -> dict:
    budget = {**REGISTERED, **(budget or {})}
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    seeds = protocol["seeds"]["calibration"]
    dev_id = pr.datasets_by_role(protocol, "development")[0]
    frozen = read_design(root / design_run, root=root)
    out = new_run(out)
    tape, _ = load(dev_id, use="develop", analysis=DEVELOP_ANALYSIS, purpose=f"calibration v3 search ({label})",
                   root=root, scale=frozen["scale_native_per_lot"])
    tables = size_tables(block_raws(tape)[0])
    del tape
    target = pool(load_sketches(root / design_run / "development-sketches.npz"))
    context = _context(frozen, {"development": target})
    base = base_config(frozen["tick_bps"])
    records = search(starts=budget["starts"], global_draws=budget["global_draws"], rounds=tuple(budget["rounds"]),
                     per_round=budget["per_round"], seed=seeds["search_seed"], base=base, tables=tables,
                     seeds=seeds["fit_seeds"], seconds=budget["seconds"], context=context, target="development")
    ranked = sorted((r for r in records if r["objective"] is not None), key=lambda r: (r["objective"], r["key"]))
    top = ranked[:budget["rescore"]]
    tasks = [(r["key"], {"config": r["config"], "extensions": r["extensions"]}, seeds["rescore_seeds"],
              budget["rescore_seconds"], "development") for r in top]
    rescored = {r["key"]: r for r in run_tasks(tasks, context)}
    order = sorted((k for k in rescored if rescored[k]["objective"] is not None),
                   key=lambda k: (rescored[k]["objective"], k))
    advanced = order[:budget["advance"]]
    write_json(out, "candidates.json", records)
    write_json(out, "rescored.json", list(rescored.values()))
    failures = [r for r in records if r["objective"] is None]
    result = {
        "label": label, "budget": budget, "parameters": [p.__dict__ for p in PARAMETERS], "base_config": base,
        "size_tables": tables, "candidates": len(records), "failed_candidates": len(failures),
        "failure_reasons": sorted({(r["error"] or "non-finite objective").split(":")[0] for r in failures}),
        "best_search": {k: top[0][k] for k in ("key", "objective", "unit", "config", "extensions")} if top else None,
        "rescored": {k: {"search_objective": next(r["objective"] for r in top if r["key"] == k),
                         "rescore_objective": rescored[k]["objective"], "families": rescored[k]["families"],
                         "seed_objectives": rescored[k]["seed_objectives"]} for k in rescored},
        "advanced": [{"key": k, "unit": next(r["unit"] for r in top if r["key"] == k),
                      "config": next(r["config"] for r in top if r["key"] == k),
                      "extensions": next(r["extensions"] for r in top if r["key"] == k),
                      "rescore_objective": rescored[k]["objective"]} for k in advanced],
        "pareto_rescored": pareto_front({k: np.asarray(rescored[k]["families"], float) for k in order}),
        "start_best": {str(s): min((r["objective"] for r in records if r["start"] == s and r["objective"] is not None),
                                   default=None) for s in range(budget["starts"])},
        "interpretation": "Development fit only; never evidence of generalization."}
    finalize(out, analysis=f"{DEVELOP_ANALYSIS}-{label}", dataset_ids=[dev_id],
             config={"budget": budget, "design_run": design_run, "design_sha256": frozen["design_sha256"]},
             result=result, root=root, seeds={k: seeds[k] for k in ("search_seed", "fit_seeds", "rescore_seeds")})
    return result


def select(develop_dir: str | Path, out: str | Path, *, root: Path = PROJECT_ROOT, seal: bool = True,
           design_run: str = DESIGN_RUN) -> dict:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    seeds = protocol["seeds"]["calibration"]
    developed = json.loads((Path(develop_dir) / "result.json").read_text(encoding="utf-8"))
    budget = developed["budget"]
    if seal and developed["label"] != "registered":
        raise ValueError("only a registered development search can be sealed")
    frozen = read_design(root / design_run, root=root)
    sel_id = pr.datasets_by_role(protocol, "selection")[0]
    out = new_run(out)
    record_use(sel_id, "select", SELECT_ANALYSIS, "calibration v3 selection (sealed selection sketches)", root)
    target = pool(load_sketches(root / design_run / "selection-sketches.npz"))
    context = _context(frozen, {"selection": target})
    control = v05_control(root)
    candidates = {c["key"]: c for c in developed["advanced"]}
    tasks = [(k, {"config": c["config"], "extensions": c["extensions"]}, seeds["selection_seeds"],
              budget["selection_seconds"], "selection") for k, c in candidates.items()]
    tasks.append(("v05-control", {"config": control["config"], "extensions": control["extensions"]},
                  seeds["selection_seeds"], budget["selection_seconds"], "selection"))
    scored = {r["key"]: r for r in run_tasks(tasks, context)}
    scores = {k: scored[k]["objective"] for k in candidates}
    valid = {k: v for k, v in scores.items() if v is not None and math.isfinite(v)}
    if not valid:
        raise ValueError("no candidate produced a finite selection objective")
    selected = min(valid, key=lambda k: (valid[k], candidates[k]["rescore_objective"], k))
    per_seed = [v for v in scored[selected]["seed_objectives"] if v is not None]
    se = float(np.std(per_seed, ddof=1) / math.sqrt(len(per_seed))) if len(per_seed) > 1 else 0.0
    region = near_optimal(valid, selected, se, relative=budget["near_optimal_relative"])
    units = {k: np.asarray(candidates[k]["unit"], float) for k in candidates}
    distinct = distinct_set(region, units, distance=budget["distinct"])
    members = distinct[:budget["ensemble_limit"]]
    family_vectors = {k: np.asarray(scored[k]["families"], float) for k in valid}
    result = {
        "selected": {"key": selected, "config": candidates[selected]["config"],
                     "extensions": candidates[selected]["extensions"], "unit": candidates[selected]["unit"],
                     "selection_objective": valid[selected]},
        "selection_objectives": scores, "selection_families": {k: scored[k]["families"] for k in scored},
        "selection_seed_objectives": {k: scored[k]["seed_objectives"] for k in scored},
        "selected_seed_se": se, "near_optimal_threshold": valid[selected] + max(
            budget["near_optimal_relative"] * valid[selected], 2 * se),
        "near_optimal": region, "distinct_near_optimal": distinct, "ensemble_members": members,
        "ensemble": [{"key": k, "config": candidates[k]["config"], "extensions": candidates[k]["extensions"],
                      "unit": candidates[k]["unit"], "selection_objective": valid[k],
                      "development_rescore_objective": candidates[k]["rescore_objective"],
                      "selection_families": dict(zip(FAMILIES, scored[k]["families"])),
                      "config_sha256": pr.document_sha256({"config": candidates[k]["config"],
                                                           "extensions": candidates[k]["extensions"]})}
                     for k in members],
        "control": {**control, "selection_objective": scored["v05-control"]["objective"],
                    "selection_families": dict(zip(FAMILIES, scored["v05-control"]["families"]))},
        "pareto_selection": pareto_front(family_vectors), "families": list(FAMILIES), "parameters": list(NAMES),
        "control_unit_coordinates": unit_from_spec_safe(control),
        "develop_result_sha256": pr.file_sha256(Path(develop_dir) / "result.json"),
        "rule": ("selected = minimum selection objective (ties: lower development rescore); near-optimal = objective <= "
                 "min + max(0.10 min, 2 SE); distinct = greedy L-infinity >= 0.25 in the unit box; ensemble = first 8 "
                 "distinct members (selected first)")}
    finalize(out, analysis=SELECT_ANALYSIS, dataset_ids=[sel_id],
             config={"develop_dir": Path(develop_dir).name, "design_sha256": frozen["design_sha256"]}, result=result,
             root=root, seeds={"selection_seeds": seeds["selection_seeds"]})
    if seal:
        pr.append_event(root / pr.LEDGER_PATH, "seal_design", {
            "analysis": SEAL, "design_sha256": selection_hash(result), "reads": [],
            "run": Path(out).name, "note": ("Selected v3 model, v0.5 control, near-optimal region, distinct set and "
                                            "ensemble fixed from development/selection data before any holdout.")},
            protocol=protocol)
    return result


def unit_from_spec_safe(control: dict) -> list | None:
    from ..sim_v2 import SimulatorSpec
    try:
        values = unit_from_spec(SimulatorSpec(control["config"], control["extensions"]))
    except (KeyError, ValueError, ZeroDivisionError):
        return None
    return [None if not math.isfinite(v) else float(v) for v in values]


def selection_hash(result: dict) -> str:
    return pr.document_sha256({"selected": result["selected"], "ensemble": result["ensemble"],
                               "control": result["control"]["sha256"], "near_optimal": result["near_optimal"]})


def read_selection(run: str | Path, *, root: Path = PROJECT_ROOT) -> dict:
    result = json.loads((Path(run) / "result.json").read_text(encoding="utf-8"))
    state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), pr.load_protocol(root / pr.PROTOCOL_PATH))
    sealed = state.designs.get(SEAL)
    if sealed is None or sealed["design_sha256"] != selection_hash(result):
        raise ValueError("calibration selection is not sealed in the v0.6 ledger")
    return result
