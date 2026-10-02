"""M12/M17 run: regime-conditioned calibration on development (fit) and selection (choose) data.

Development and selection days are split into 5-minute blocks labelled by the
unchanged v0.5 M6 volatility thresholds. For each regime the calibration-v3
search (4 starts, same rounds) fits that regime's development blocks, the 64
best are re-scored and the 32 best are scored on that regime's selection blocks;
the minimum selection objective is the regime model. Sealed before any holdout.
"""
from __future__ import annotations

import json
from pathlib import Path


from ..experiments.registry import PROJECT_ROOT
from . import protocol as pr
from .calibration import run_tasks, search
from .calibration_study import DESIGN_RUN, REGISTERED, read_selection
from .evidence import finalize, new_run, write_json
from .history import load
from .observables import pool
from .realism_study import read_design
from .regimes import load_thresholds, regime_sketches

ANALYSIS = "m12-regime-calibration"


def run(develop_run: str | Path, selection_run: str | Path, out: str | Path, *, root: Path = PROJECT_ROOT,
        budget: dict | None = None, label: str = "registered", seal: bool = True) -> dict:
    budget = {**REGISTERED, "starts": 4, **(budget or {})}
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    seeds = protocol["seeds"]["calibration"]
    frozen = read_design(root / DESIGN_RUN, root=root)
    read_selection(selection_run, root=root)
    developed = json.loads((Path(develop_run) / "result.json").read_text(encoding="utf-8"))
    thresholds, thresholds_sha = load_thresholds(root)
    dev_id = pr.datasets_by_role(protocol, "development")[0]
    sel_id = pr.datasets_by_role(protocol, "selection")[0]
    out = new_run(out)
    scale = frozen["scale_native_per_lot"]
    dev, _ = load(dev_id, use="develop", analysis=ANALYSIS, purpose=f"regime-conditioned fit ({label})", root=root,
                  scale=scale)
    dev_groups = regime_sketches(dev, frozen["design"], thresholds)
    del dev
    sel, _ = load(sel_id, use="select", analysis=ANALYSIS, purpose=f"regime-conditioned selection ({label})",
                  root=root, scale=scale)
    sel_groups = regime_sketches(sel, frozen["design"], thresholds)
    del sel
    targets = {f"dev:{k}": pool(v) for k, v in dev_groups.items()}
    targets.update({f"sel:{k}": pool(v) for k, v in sel_groups.items()})
    context = {"design": frozen["design"], "scales": frozen["objective_scales"], "targets": targets}
    models, details = {}, {}
    for regime in ("high", "low"):
        records = search(starts=budget["starts"], global_draws=budget["global_draws"], rounds=tuple(budget["rounds"]),
                         per_round=budget["per_round"], seed=seeds["regime_search_seeds"][regime],
                         base=developed["base_config"], tables=developed["size_tables"], seeds=seeds["fit_seeds"],
                         seconds=budget["seconds"], context=context, target=f"dev:{regime}", label=f"{regime[0]}")
        ranked = sorted((r for r in records if r["objective"] is not None), key=lambda r: (r["objective"], r["key"]))
        top = ranked[:budget["rescore"]]
        rescored = {r["key"]: r for r in run_tasks(
            [(r["key"], {"config": r["config"], "extensions": r["extensions"]}, seeds["rescore_seeds"],
              budget["rescore_seconds"], f"dev:{regime}") for r in top], context)}
        advanced = sorted((k for k in rescored if rescored[k]["objective"] is not None),
                          key=lambda k: (rescored[k]["objective"], k))[:budget["advance"]]
        write_json(out, f"candidates-{regime}.json", records)
        by_key = {r["key"]: r for r in top}
        scored = {r["key"]: r for r in run_tasks(
            [(k, {"config": by_key[k]["config"], "extensions": by_key[k]["extensions"]}, seeds["selection_seeds"],
              budget["selection_seconds"], f"sel:{regime}") for k in advanced], context)}
        valid = {k: v["objective"] for k, v in scored.items() if v["objective"] is not None}
        if not valid:
            models[regime] = None
            details[regime] = {"status": "INVALID", "reason": "no finite regime selection objective"}
            continue
        chosen = min(valid, key=lambda k: (valid[k], rescored[k]["objective"], k))
        models[regime] = {"key": chosen, "config": by_key[chosen]["config"],
                          "extensions": by_key[chosen]["extensions"], "unit": by_key[chosen]["unit"],
                          "selection_objective": valid[chosen]}
        details[regime] = {"candidates": len(records), "failed": sum(r["objective"] is None for r in records),
                           "best_search": ranked[0]["objective"] if ranked else None, "selection_objectives": valid,
                           "dev_blocks": len(dev_groups.get(regime, [])), "sel_blocks": len(sel_groups.get(regime, []))}
    result = {"label": label, "budget": budget, "models": models, "details": details,
              "thresholds_sha256": thresholds_sha, "dimension": "volatility",
              "interpretation": "Regime-conditioned simulators fitted on development regime blocks only."}
    finalize(out, analysis=f"{ANALYSIS}-{label}", dataset_ids=[dev_id, sel_id], config={"budget": budget},
             result=result, root=root, seeds={"regime_search_seeds": seeds["regime_search_seeds"]})
    if seal and label == "registered":
        pr.append_event(root / pr.LEDGER_PATH, "seal_design", {
            "analysis": ANALYSIS, "design_sha256": pr.document_sha256(models), "reads": [],
            "note": "Regime-conditioned (volatility) models fixed from development/selection blocks before any holdout."},
            protocol=protocol)
    return result


def read_regime(run: str | Path, *, root: Path = PROJECT_ROOT) -> dict:
    result = json.loads((Path(run) / "result.json").read_text(encoding="utf-8"))
    state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), pr.load_protocol(root / pr.PROTOCOL_PATH))
    if state.designs.get(ANALYSIS, {}).get("design_sha256") != pr.document_sha256(result["models"]):
        raise ValueError("regime models are not sealed in the ledger")
    return result
