"""Run the frozen M3 calibration-v2 design in three ledger-ordered phases.

``develop``: fit every family on the development period only.
``select``: score fitted models on the selection period, choose one, and append a
``seal_design`` ledger entry naming the selected and control models.
``evaluate``: refuse unless that seal precedes holdout access; then evaluate the
selected model and control, unchanged, on the internal, external and
cross-instrument holdouts with the paired block bootstrap.

Each phase writes its own write-once, bound and sealed run directory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import pickle

import numpy as np

from .calibration_v2 import (EXTENSION_FAMILIES, HAWKES_DISPERSION_THRESHOLD, base_config,
                             block_bootstrap_loss_difference, block_samples, empirical_size_tables, fit_family,
                             gate_status, simulated_summary, tape_from_tardis)
from .experiments.registry import PROJECT_ROOT
from .observables_v2 import FAMILIES, family_errors, loss, measure, summarize
from . import preregistration as pr
from .sim_v2 import SimulatorSpec
from .v05_data import acquire
from .v05_evidence import LEDGER_PATH, PROTOCOL_PATH, finalize, new_run, write_json

DESIGN_PATH = "configs/v05/m3-design.json"
CACHE_DIR = "data/v05/cache"


def load_design(root: Path = PROJECT_ROOT) -> dict:
    return json.loads((root / DESIGN_PATH).read_text(encoding="utf-8"))


def _tape(dataset_id: str, scale: float | None, root: Path, purpose: str):
    """Tape at native scale (cached locally, never committed), then rescaled."""
    files = acquire(dataset_id, root=root, purpose=purpose)
    cache = root / CACHE_DIR / f"{dataset_id}.tape.pkl"
    if cache.is_file():
        tape, meta = pickle.loads(cache.read_bytes())
    else:
        tape, meta = tape_from_tardis(files["l2"], files["trades"], depth_scale=1.0)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps((tape, meta)))
    if scale is None:
        valid = tape.valid
        scale = float(np.median(np.r_[tape.bq[valid].sum(1), tape.aq[valid].sum(1)])) / 1000.0
    tape.bq = tape.bq / scale
    tape.aq = tape.aq / scale
    tape.trade_q = tape.trade_q / scale
    return tape, meta, scale, files


def _tick_bps(tape) -> float:
    valid = tape.valid
    gaps = np.r_[-np.diff(tape.bp[valid], axis=1).ravel(), np.diff(tape.ap[valid], axis=1).ravel()]
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    tick = float(np.quantile(gaps, 0.05))
    return tick / float(np.median(tape.mid[valid])) * 1e4


def _iid_control(tape, seed: int) -> dict:
    """IID joint resampling of development 1 s rows; only level/return families are expressible."""
    step = 10
    idx = np.arange(0, len(tape.t), step)
    ok = tape.valid[idx]
    rows = idx[ok][1:]
    previous = idx[ok][:-1]
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(rows), len(rows))
    r, p = rows[pick], previous[pick]
    ret = np.log(tape.mid[r] / tape.mid[p]) * 1e4
    spread = (tape.ap[r, 0] - tape.bp[r, 0]) / tape.mid[r] * 1e4
    d5b, d5a = tape.bq[r].sum(1), tape.aq[r].sum(1)
    nan = np.asarray([math.nan])
    samples = {"spread_bps": spread, "l1_depth": np.r_[tape.bq[r, 0], tape.aq[r, 0]], "depth5": np.r_[d5b, d5a],
               "imbalance5": (d5b - d5a) / (d5b + d5a), "return_1s_bps": ret,
               "rv_60s_bps": np.asarray([np.std(ret[i:i + 60]) for i in range(0, len(ret) - 59, 60)]),
               "trade_sizes": np.asarray([]), "addition_sizes": np.asarray([]), "interarrival_s": np.asarray([]),
               "recovery_s": np.asarray([]), "recovered": np.asarray([]), "trade_counts_1s": np.asarray([0.0]),
               "trade_rate": nan, "top_change_rate": nan, "net_cancel_rate": nan, "addition_rate": nan,
               "acf_return": nan, "acf_abs_return": nan, "acf_spread": nan, "corr_spread_depth": nan,
               "corr_imbalance_next_return": nan, "corr_flow_return": nan}
    return summarize(samples)


def _executor():
    # Worker count affects speed only; candidate results are scheduling-independent.
    workers = int(os.environ.get("CLEOLOB_WORKERS", max(1, min(12, (os.cpu_count() or 2) - 2))))
    return ProcessPoolExecutor(max_workers=workers)


def develop(out: str | Path, *, root: Path = PROJECT_ROOT, candidates: int | None = None,
            seconds: float | None = None, label: str = "registered") -> dict:
    design = load_design(root)
    chronology, search = design["chronology"], design["search"]
    candidates = candidates or search["candidates_per_family"]
    seconds = seconds or search["seconds_per_seed"]
    tape, meta, scale, files = _tape(chronology["development"], None, root, "M3 development fit")
    samples = measure(tape)
    target = summarize(samples)
    dispersion = target["trade_dispersion_index"]
    admitted = [f for f in EXTENSION_FAMILIES
                if f != "self_exciting_market_orders" or (dispersion or 0) > HAWKES_DISPERSION_THRESHOLD]
    tick_bps = _tick_bps(tape)
    base = base_config(tick_bps)
    sizes = empirical_size_tables(samples)
    out = new_run(out)
    fits = {}
    with _executor() as pool:
        fits["v04_zi_baseline"] = fit_family("v04_zi_baseline", target, base=base, size_tables=sizes,
                                             baseline_best=None, seeds=search["fit_seeds"], seconds=seconds,
                                             candidates=candidates, search_seed=search["search_seed"], executor=pool)
        control = fits["v04_zi_baseline"]["best"]
        for family in admitted:
            fits[family] = fit_family(family, target, base=base, size_tables=sizes, baseline_best=control,
                                      seeds=search["fit_seeds"], seconds=seconds, candidates=candidates,
                                      search_seed=search["search_seed"], executor=pool)
    for family, fit in fits.items():
        write_json(out, f"fits/{family}.json", fit)
    result = {"label": label, "scale_native_per_lot": scale, "tick_bps": tick_bps, "admitted_families": admitted,
              "excluded_families": sorted(set(EXTENSION_FAMILIES) - set(admitted)),
              "dispersion_index": dispersion, "development_summary": target,
              "size_tables": sizes, "base_config": base, "candidates": candidates, "seconds_per_seed": seconds,
              "training_losses": {f: fit["best"]["loss"] for f, fit in fits.items()},
              "best": {f: {"config": fit["best"]["config"], "extensions": fit["best"]["extensions"],
                           "loss": fit["best"]["loss"], "parameters": fit["parameters"]} for f, fit in fits.items()},
              "interpretation": "Training (development) fit only; never evidence of generalization."}
    finalize(out, analysis=f"m3-develop-{label}", dataset_ids=[chronology["development"]],
             config={"design": design, "design_sha256": pr.document_sha256(design), "files": {k: v.name for k, v in files.items()}},
             result=result, root=root)
    return result


def select(develop_dir: str | Path, out: str | Path, *, root: Path = PROJECT_ROOT, seal: bool = True) -> dict:
    design = load_design(root)
    chronology, search = design["chronology"], design["search"]
    developed = json.loads((Path(develop_dir) / "result.json").read_text(encoding="utf-8"))
    seconds = developed["seconds_per_seed"]
    tape, _, scale, files = _tape(chronology["selection"], developed["scale_native_per_lot"], root, "M3 selection")
    target = summarize(measure(tape))
    out = new_run(out)
    scores = {}
    for family, best in developed["best"].items():
        simulated = simulated_summary(SimulatorSpec(best["config"], best["extensions"]), search["selection_seeds"], seconds)
        scores[family] = {"loss": loss(family_errors(target, simulated)), "parameters": best["parameters"],
                          "errors": family_errors(target, simulated)}
    control_loss = scores["v04_zi_baseline"]["loss"]
    improving = [f for f in developed["admitted_families"] if scores[f]["loss"] < control_loss]
    combined = None
    if len(improving) >= 2:
        dev_tape, *_ = _tape(chronology["development"], developed["scale_native_per_lot"], root, "M3 combined fit")
        dev_target = summarize(measure(dev_tape))
        start = {}
        for family in improving:
            start.update(developed["best"][family]["extensions"])
        with _executor() as pool:
            fit = fit_family("combined", dev_target, base=developed["base_config"], size_tables=developed["size_tables"],
                             baseline_best={"config": developed["best"]["v04_zi_baseline"]["config"]},
                             seeds=search["fit_seeds"], seconds=seconds, candidates=developed["candidates"],
                             search_seed=search["search_seed"], executor=pool, start_extensions=start,
                             extension_families=tuple(f for f in improving if f != "empirical_sizes"))
        if "empirical_sizes" in improving:
            fit["best"]["extensions"].update(developed["size_tables"])
        simulated = simulated_summary(SimulatorSpec(fit["best"]["config"], fit["best"]["extensions"]),
                                      search["selection_seeds"], seconds)
        combined = {"members": improving, "config": fit["best"]["config"], "extensions": fit["best"]["extensions"],
                    "training_loss": fit["best"]["loss"], "parameters": fit["parameters"]}
        scores["combined"] = {"loss": loss(family_errors(target, simulated)), "parameters": fit["parameters"],
                              "errors": family_errors(target, simulated)}
        write_json(out, "combined-fit.json", fit)
    selected = min(scores, key=lambda f: (scores[f]["loss"], scores[f]["parameters"]))
    model = combined if selected == "combined" else developed["best"][selected]
    spec = {"family": selected, "config": model["config"], "extensions": model["extensions"]}
    control = {"family": "v04_zi_baseline", "config": developed["best"]["v04_zi_baseline"]["config"],
               "extensions": developed["best"]["v04_zi_baseline"]["extensions"]}
    result = {"selected": spec, "control": control, "selection_losses": {f: s["loss"] for f, s in scores.items()},
              "selection_errors": {f: s["errors"] for f, s in scores.items()}, "improving_extensions": improving,
              "combined": combined, "scale_native_per_lot": scale, "develop_result_sha256":
              pr.file_sha256(Path(develop_dir) / "result.json")}
    finalize(out, analysis="m3-select", dataset_ids=[chronology["development"], chronology["selection"]],
             config={"design_sha256": pr.document_sha256(design), "develop_dir": Path(develop_dir).name},
             result=result, root=root)
    if seal:
        protocol = pr.load_protocol(root / PROTOCOL_PATH)
        pr.append_event(root / LEDGER_PATH, "seal_design", {
            "analysis": "m3-selection", "design_sha256": pr.document_sha256({"selected": spec, "control": control,
                                                                            "scale": scale}),
            "reads": [chronology["internal_holdout"], chronology["external_holdout"],
                      chronology["cross_instrument_holdout"]],
            "note": "Selected calibration-v2 model and control frozen before any holdout access."}, protocol=protocol)
    return result


def evaluate(select_dir: str | Path, out: str | Path, *, root: Path = PROJECT_ROOT) -> dict:
    design = load_design(root)
    chronology, search, inference = design["chronology"], design["search"], design["inference"]
    selection = json.loads((Path(select_dir) / "result.json").read_text(encoding="utf-8"))
    protocol = pr.load_protocol(root / PROTOCOL_PATH)
    state = pr.replay_ledger(pr.read_ledger(root / LEDGER_PATH), protocol)
    expected = pr.document_sha256({"selected": selection["selected"], "control": selection["control"],
                                   "scale": selection["scale_native_per_lot"]})
    if state.designs.get("m3-selection", {}).get("design_sha256") != expected:
        raise ValueError("selection is not sealed in the ledger; holdout evaluation refused")
    out = new_run(out)
    seconds = search["seconds_per_seed"]
    specs = {name: SimulatorSpec(selection[name]["config"], selection[name]["extensions"])
             for name in ("selected", "control")}
    simulated = {name: simulated_summary(spec, search["holdout_seeds"], seconds) for name, spec in specs.items()}
    holdouts = {}
    alpha = 0.05 / 3
    for role in ("internal_holdout", "external_holdout", "cross_instrument_holdout"):
        dataset = chronology[role]
        tape, meta, _, files = _tape(dataset, selection["scale_native_per_lot"], root, f"M3 {role} evaluation")
        samples = measure(tape)
        target = summarize(samples)
        errors = {name: family_errors(target, simulated[name]) for name in specs}
        valid_fraction = float(tape.valid.mean())
        blocks = block_samples(tape)
        comparison = (block_bootstrap_loss_difference(blocks, simulated["selected"], simulated["control"],
                                                      samples=2000, alpha=alpha, seed=inference["bootstrap_seed"])
                      if selection["selected"]["family"] != "v04_zi_baseline" else
                      {"status": "NOT_APPLICABLE", "reason": "control selected; no contrast"})
        improvement = ("ESTABLISHED" if comparison.get("ci_high", 0) < 0 else "NOT_ESTABLISHED") \
            if "ci_high" in comparison else "NOT_AVAILABLE"
        holdouts[role] = {
            "dataset": dataset, "retrospective": role == "internal_holdout",
            "losses": {name: loss(errors[name]) for name in specs}, "family_errors": errors,
            "generalization_gate": {name: gate_status(errors[name], valid_fraction, int(tape.valid[::10].sum()))
                                    for name in specs},
            "improvement_H3a": {"status": improvement, **comparison}, "target_summary": target,
            "files": {k: v.name for k, v in files.items()}}
    result = {"selected_family": selection["selected"]["family"], "holdouts": holdouts,
              "simulated_summaries": simulated, "families": list(FAMILIES), "alpha_per_holdout": alpha,
              "interpretation": ("June is a retrospective holdout (consumed by v0.4). July ETH is the fresh "
                                 "external holdout. July BTC is a cross-instrument transfer test using the ETH "
                                 "development scale and tick geometry.")}
    finalize(out, analysis="m3-evaluate", dataset_ids=[chronology[r] for r in ("internal_holdout", "external_holdout",
                                                                               "cross_instrument_holdout")],
             config={"design_sha256": pr.document_sha256(design), "select_dir": Path(select_dir).name},
             result=result, root=root)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("develop", "select", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--from", dest="source", type=Path)
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--label", default="registered")
    args = parser.parse_args(argv)
    if args.phase == "develop":
        result = develop(args.out, candidates=args.candidates, seconds=args.seconds, label=args.label)
        print(json.dumps(result["training_losses"], indent=2))
    elif args.phase == "select":
        result = select(args.source, args.out)
        print(json.dumps({"selected": result["selected"]["family"], "losses": result["selection_losses"]}, indent=2))
    else:
        result = evaluate(args.source, args.out)
        print(json.dumps({r: {"losses": h["losses"], "H3a": h["improvement_H3a"].get("status"),
                              "gate": {n: g["status"] for n, g in h["generalization_gate"].items()}}
                          for r, h in result["holdouts"].items()}, indent=2))


if __name__ == "__main__":
    main()
