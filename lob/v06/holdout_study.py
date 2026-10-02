"""M12/M13/M5/M20 runs: simulation bank, sealed holdout design and per-dataset realism evaluation.

bank     -> results/v06/m13/bank: every simulated quantity the holdout analyses need, computed and
            hashed before any holdout access (model sketches, domain-gap windows, support windows,
            long regime paths). Holdout data can therefore never influence a simulated input.
seal     -> ledger ``m13-holdout-design``: bank hash, models, margins, alphas, seeds and rules; reads the
            fresh ETH and cross-instrument BTC holdouts.
evaluate -> results/v06/m13/<dataset>: H1/H2/H3 contrasts, scorecard equivalence, H8 domain gap,
            H9/H10 regime contrasts (fresh ETH), support/OOD and transition realism.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from ..experiments.registry import PROJECT_ROOT
from ..regimes import BLOCK_S
from ..sim_v2 import SimulatorSpec
from . import protocol as pr
from .calibration import EVENTS_PER_SECOND_CAP
from .calibration_study import DESIGN_RUN, read_selection
from .domain_gap import Standardizer, gap_status, support, two_sample_test, window_features
from .evidence import finalize, new_run, write_json
from .history import block_raws, load
from .identifiability import clean
from .observables import measure, pool, sketch
from .realism import bootstrap_realism, improvement_status
from .realism_study import load_sketches, read_design, save_sketches
from .regimes import labelled_blocks, load_thresholds, transition_comparison, transition_profile
from .tape import tape_from_simulator

ANALYSIS = "m13-holdout-design"
REGIME_ANALYSIS = "m12-regime-calibration"


def _capped(spec: dict, seconds: float) -> SimulatorSpec:
    return SimulatorSpec({**spec["config"], "max_events": int(EVENTS_PER_SECOND_CAP * (seconds + 60))},
                         spec["extensions"])


def _bank_task(task: tuple) -> dict:
    kind, name, spec, seed, seconds, design, thresholds = task
    tape = tape_from_simulator(_capped(spec, seconds), seed, seconds=seconds)
    out = {"kind": kind, "name": name, "seed": seed}
    if kind == "sketch":
        out["sketch"] = pool([sketch(measure(b), design) for b in tape.blocks(600.0)])
    elif kind == "windows":
        out["windows"] = window_features(tape)
    elif kind == "path":
        out["blocks"] = labelled_blocks(tape, thresholds)
    return out


def bank(out: str | Path, *, selection_run: str, regime_run: str | None, root: Path = PROJECT_ROOT,
         workers: int = 14, label: str = "registered") -> dict:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    seeds = protocol["seeds"]
    selection = read_selection(root / selection_run, root=root)
    frozen = read_design(root / DESIGN_RUN, root=root)
    thresholds, thresholds_sha = load_thresholds(root)
    models = {"selected": selection["selected"], "control": selection["control"]}
    regime = json.loads((root / regime_run / "result.json").read_text(encoding="utf-8"))["models"] if regime_run else {}
    for name, model in regime.items():
        models[f"regime:{name}"] = model
    members = {m["key"]: m for m in selection["ensemble"]}
    tasks = []
    for name, model in models.items():
        for seed in seeds["calibration"]["holdout_seeds"]:
            tasks.append(("sketch", name, {"config": model["config"], "extensions": model["extensions"]}, seed, 3600.0,
                          frozen["design"], thresholds))
    for seed in seeds["domain_gap"]["simulation_seeds"]:
        tasks.append(("windows", "selected", {"config": models["selected"]["config"],
                                              "extensions": models["selected"]["extensions"]}, seed, 3 * 3600.0,
                      frozen["design"], thresholds))
    for key, member in members.items():
        for seed in seeds["domain_gap"]["simulation_seeds"][:4]:
            tasks.append(("windows", f"member:{key}", {"config": member["config"], "extensions": member["extensions"]},
                          seed, 3600.0, frozen["design"], thresholds))
    for name in ("selected", "control"):
        for seed in seeds["regime_transitions"]["simulation_seeds"]:
            tasks.append(("path", name, {"config": models[name]["config"], "extensions": models[name]["extensions"]},
                          seed, 6 * 3600.0, frozen["design"], thresholds))
    out = new_run(out)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_bank_task, tasks, chunksize=1))
    sketches = {name: [r["sketch"] for r in results if r["kind"] == "sketch" and r["name"] == name] for name in models}
    for name, items in sketches.items():
        save_sketches(out / f"sketches-{name.replace(':', '_')}.npz", items)
    windows = {}
    for r in results:
        if r["kind"] == "windows":
            windows.setdefault(r["name"], []).append(r["windows"])
    np.savez_compressed(out / "windows.npz", **{f"{name.replace(':', '_')}__{i}": w for name, items in windows.items()
                                                for i, w in enumerate(items)})
    paths = {name: [r["blocks"] for r in results if r["kind"] == "path" and r["name"] == name]
             for name in ("selected", "control")}
    write_json(out, "paths.json", clean(paths))
    files = {p.name: pr.file_sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    result = {"label": label, "models": {k: {"config": v["config"], "extensions": v["extensions"]}
                                         for k, v in models.items()},
              "files_sha256": files, "thresholds_sha256": thresholds_sha,
              "transition_profiles": {name: transition_profile(p) for name, p in paths.items()},
              "tasks": len(tasks)}
    finalize(out, analysis=f"m13-simulation-bank-{label}", dataset_ids=[], config={"seeds": seeds},
             result=clean(result), root=root, seeds={"holdout": seeds["calibration"]["holdout_seeds"],
                                              "domain_gap": seeds["domain_gap"]["simulation_seeds"],
                                              "transitions": seeds["regime_transitions"]["simulation_seeds"]})
    return result


def load_bank(run: Path) -> dict:
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    for name, digest in result["files_sha256"].items():
        if pr.file_sha256(run / name) != digest:
            raise ValueError(f"simulation bank file changed: {name}")
    sketches = {name: load_sketches(run / f"sketches-{name.replace(':', '_')}.npz") for name in result["models"]}
    windows: dict[str, list[np.ndarray]] = {}
    with np.load(run / "windows.npz") as stored:
        for key in sorted(stored.files, key=lambda k: (k.split("__")[0], int(k.split("__")[1]))):
            windows.setdefault(key.split("__")[0], []).append(stored[key])
    paths = json.loads((run / "paths.json").read_text(encoding="utf-8"))
    return {"result": result, "sketches": sketches, "windows": windows, "paths": paths}


def history_paths(blocks: list[dict]) -> list[list[dict]]:
    """Split a historical block sequence at gaps so no transition spans a missing block."""
    paths, current = [], []
    for block in blocks:
        if current and abs(block["start_s"] - current[-1]["start_s"] - BLOCK_S) > 1e-6:
            paths.append(current)
            current = []
        current.append(block)
    if current:
        paths.append(current)
    return paths


def design_document(*, bank_run: Path, selection_run: str, regime_run: str | None, root: Path) -> dict:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    frozen = read_design(root / DESIGN_RUN, root=root)
    selection = read_selection(root / selection_run, root=root)
    banked = json.loads((bank_run / "result.json").read_text(encoding="utf-8"))
    families = protocol["statistics"]["families"]
    return {
        "schema": "cleolob-v06-holdout-design-1", "bank_result_sha256": pr.file_sha256(bank_run / "result.json"),
        "bank_files_sha256": banked["files_sha256"], "selection_seal": selection["selected"]["key"],
        "regime_run": regime_run, "margins": frozen["equivalence_margins"], "scales": frozen["objective_scales"],
        "observable_design_sha256": frozen["design_sha256"],
        "H1_H3": {"alpha": 0.05 / 3, "samples": 2000, "seed": protocol["seeds"]["bootstrap"]["realism"],
                  "contrast": ["selected", "control"], "rule": "improvement_status(upper < 0 ESTABLISHED)"},
        "scorecard": {"alpha": families["F10_scorecard_equivalence"]["adjusted_alpha"],
                      "models": ["selected", "control"]},
        "H8": {"classifier": "logistic", "alpha": 0.05, "samples": 2000,
               "seed": protocol["seeds"]["domain_gap"]["classifier_seed"], "window_s": 60, "block_windows": 10},
        "H9_H10": {"alpha": 0.0125, "samples": 2000, "seed": protocol["seeds"]["bootstrap"]["regime"],
                   "dimension": "volatility", "noninferiority_delta_rule": "0.10 x global selection objective",
                   "noninferiority_delta": 0.10 * selection["selected"]["selection_objective"]},
        "support": {"k": 5, "quantile": 0.99, "no_claim_fraction": 0.5},
        "datasets": {"H1": "deribit-eth-perp-2020-06-01", "H2": "deribit-eth-perp-2020-09-01",
                     "H3": "deribit-btc-perp-2020-09-01", "H8": "deribit-eth-perp-2020-09-01",
                     "H9_H10": "deribit-eth-perp-2020-09-01"},
        "descriptive_retrospective": ["deribit-eth-perp-2020-07-01", "deribit-btc-perp-2020-07-01",
                                      "deribit-eth-perp-2020-08-01"]}


def power_report(*, bank_run: Path, evaluation_run: Path | None, document: dict, root: Path) -> dict:
    """Resolution expected before any holdout access: selection-day contrast width and execution MDEs."""
    from .inference import minimum_detectable_effect
    frozen = read_design(root / DESIGN_RUN, root=root)
    banked = load_bank(bank_run)
    selection_blocks = load_sketches(root / DESIGN_RUN / "selection-sketches.npz")
    contrast = bootstrap_realism(selection_blocks, {k: banked["sketches"][k] for k in ("selected", "control")},
                                 frozen["design"], frozen["objective_scales"], samples=500,
                                 seed=document["H1_H3"]["seed"], alpha=document["H1_H3"]["alpha"],
                                 contrasts=(("selected", "control"),))["contrasts"]["selected-control"]
    out = {"realism_selection_day": {"difference": contrast["difference"], "ci_low": contrast["ci_low"],
                                     "ci_high": contrast["ci_high"],
                                     "half_width": (contrast["ci_high"] - contrast["ci_low"]) / 2,
                                     "note": "selection data; expected resolution of H1-H3 contrasts on a full day"}}
    if evaluation_run is not None:
        result = json.loads((evaluation_run / "result.json").read_text(encoding="utf-8"))
        mde = {}
        for key, summary in result["summaries"].items():
            world, agent = key.split("|")
            if world == "selected" and summary.get("sd_cost_bps"):
                mde[agent] = {"sd_cost_bps": summary["sd_cost_bps"],
                              "mde_bps_alpha_0.05_power_0.8": minimum_detectable_effect(
                                  summary["sd_cost_bps"], summary["markets"], alpha=0.05)}
        out["execution_selected_world"] = mde
    return out


def seal(*, bank_run: str, selection_run: str, regime_run: str | None, root: Path = PROJECT_ROOT,
         evaluation_run: str | None = "results/v06/m10/evaluation") -> dict:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    document = design_document(bank_run=root / bank_run, selection_run=selection_run, regime_run=regime_run,
                               root=root)
    document["power"] = power_report(bank_run=root / bank_run,
                                     evaluation_run=root / evaluation_run if evaluation_run else None,
                                     document=document, root=root)
    path = root / "configs/v06/holdout-design.json"
    if path.exists():
        raise FileExistsError("holdout design already written; it is immutable")
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    pr.append_event(root / pr.LEDGER_PATH, "seal_design", {
        "analysis": ANALYSIS, "design_sha256": pr.document_sha256(document),
        "reads": ["deribit-eth-perp-2020-09-01", "deribit-btc-perp-2020-09-01"],
        "note": "Simulation bank, models, margins, alphas, seeds and rules for H1-H3, H8, H9, H10, support and "
                "transition realism sealed before any fresh holdout access."}, protocol=protocol)
    return document


def evaluate(dataset_id: str, out: str | Path, *, bank_run: str, root: Path = PROJECT_ROOT) -> dict:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    document = json.loads((root / "configs/v06/holdout-design.json").read_text(encoding="utf-8"))
    state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), protocol)
    if state.designs.get(ANALYSIS, {}).get("design_sha256") != pr.document_sha256(document):
        raise ValueError("holdout design is not sealed in the ledger")
    declaration = pr.dataset_declaration(protocol, dataset_id)
    use = "retrospective" if declaration["role"] == "retrospective" else "evaluate"
    banked = load_bank(root / bank_run)
    if pr.file_sha256(root / bank_run / "result.json") != document["bank_result_sha256"]:
        raise ValueError("simulation bank differs from the sealed holdout design")
    frozen = read_design(root / DESIGN_RUN, root=root)
    out = new_run(out)
    tape, meta = load(dataset_id, use=use, analysis=ANALYSIS if use == "evaluate" else f"{ANALYSIS}-retrospective",
                      purpose="registered holdout realism evaluation", root=root, scale=frozen["scale_native_per_lot"])
    raws, blocks = block_raws(tape)
    hist = [sketch(r, frozen["design"]) for r in raws]
    sims = {name: banked["sketches"][name] for name in ("selected", "control")}
    realism = bootstrap_realism(hist, sims, frozen["design"], frozen["objective_scales"],
                                samples=document["H1_H3"]["samples"], seed=document["H1_H3"]["seed"],
                                alpha=document["H1_H3"]["alpha"], contrasts=(("selected", "control"),),
                                margins=document["margins"], equivalence_alpha=document["scorecard"]["alpha"])
    contrast = realism["contrasts"]["selected-control"]
    hypothesis = next((h for h, d in document["datasets"].items() if d == dataset_id and h in {"H1", "H2", "H3"}), None)
    result = {"dataset": meta, "blocks": blocks and {"used": sum(b["used"] for b in blocks), "total": len(blocks)},
              "role": declaration["role"], "retrospective": declaration["role"] == "retrospective",
              "realism": realism, "improvement": {"hypothesis": hypothesis, "status": improvement_status(contrast),
                                                  **{k: contrast[k] for k in ("difference", "ci_low", "ci_high",
                                                                              "p_value")}}}
    real_windows = window_features(tape)
    gap = two_sample_test(real_windows, banked["windows"]["selected"], seed=document["H8"]["seed"],
                          samples=document["H8"]["samples"])
    result["domain_gap"] = {**gap, "status": gap_status(gap),
                            "hypothesis": "H8" if document["datasets"]["H8"] == dataset_id else None}
    member_windows = [w for k, v in banked["windows"].items() if k.startswith("member") or k == "selected" for w in v]
    scaler = Standardizer(np.vstack(member_windows))
    result["support"] = support(real_windows, member_windows, scaler, **document["support"])
    thresholds, _ = load_thresholds(root)
    labelled = labelled_blocks(tape, thresholds)
    hist_profile = transition_profile(history_paths(labelled))
    result["transitions"] = {"history": hist_profile, "comparison": {
        name: {dim: transition_comparison(hist_profile[dim], banked["result"]["transition_profiles"][name][dim])
               for dim in hist_profile} for name in ("selected", "control")}}
    if document["datasets"]["H9_H10"] == dataset_id and any(k.startswith("regime:") for k in banked["sketches"]):
        result["regime"] = regime_contrasts(tape, labelled, banked, frozen, document)
    write_json(out, "window_features.json", {"rows": len(real_windows)})
    finalize(out, analysis=f"m13-holdout-{dataset_id}", dataset_ids=[dataset_id],
             config={"design_sha256": pr.document_sha256(document)}, result=clean(result), root=root,
             seeds={"bootstrap": document["H1_H3"]["seed"], "domain_gap": document["H8"]["seed"]})
    return result


def regime_contrasts(tape, labelled: list[dict], banked: dict, frozen: dict, document: dict) -> dict:
    """H9 within-regime and H10 off-regime contrasts on 5-minute historical blocks."""
    by_start = {round(b["start_s"], 6): b["labels"]["volatility"] for b in labelled}
    groups: dict[str, list[dict]] = {"high": [], "low": []}
    for block in tape.blocks(BLOCK_S):
        if len(block.t) and round(float(block.t[0]), 6) in by_start:
            groups[by_start[round(float(block.t[0]), 6)]].append(sketch(measure(block), frozen["design"]))
    spec = document["H9_H10"]
    delta = spec["noninferiority_delta"]
    out = {}
    for regime, other in (("high", "low"), ("low", "high")):
        sims = {"regime": banked["sketches"][f"regime:{regime}"], "global": banked["sketches"]["selected"]}
        within = bootstrap_realism(groups[regime], sims, frozen["design"], frozen["objective_scales"],
                                   samples=spec["samples"], seed=spec["seed"], alpha=spec["alpha"],
                                   contrasts=(("regime", "global"),))
        other_sims = {"regime": banked["sketches"][f"regime:{other}"], "global": banked["sketches"]["selected"]}
        off = bootstrap_realism(groups[regime], other_sims, frozen["design"], frozen["objective_scales"],
                                samples=spec["samples"], seed=spec["seed"] + 1, alpha=spec["alpha"],
                                contrasts=(("regime", "global"),))
        off_contrast = off["contrasts"]["regime-global"]
        h10 = ("ESTABLISHED" if off_contrast["ci_high"] < delta else "FAILED" if off_contrast["ci_low"] > delta
               else "NOT_ESTABLISHED")
        out[regime] = {"blocks": len(groups[regime]), "H9_within": within["contrasts"]["regime-global"],
                       "H9_status": improvement_status(within["contrasts"]["regime-global"]),
                       "H10_off_regime_model": other, "H10_offregime": off_contrast, "H10_status": h10,
                       "noninferiority_delta": delta}
    statuses9 = [v["H9_status"] for v in out.values()]
    statuses10 = [v["H10_status"] for v in out.values()]
    out["H9"] = ("ESTABLISHED" if all(s == "ESTABLISHED" for s in statuses9) else
                 "FAILED" if "FAILED" in statuses9 else "NOT_ESTABLISHED")
    out["H10"] = ("ESTABLISHED" if all(s == "ESTABLISHED" for s in statuses10) else
                  "FAILED" if "FAILED" in statuses10 else "NOT_ESTABLISHED")
    return out
