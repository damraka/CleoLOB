"""M6: regime-conditioned and cross-instrument external validity under frozen designs.

``thresholds`` fits regime thresholds on the development period only. ``evaluate``
reads a holdout (only after its sealed designs), labels 5-minute blocks with the
frozen thresholds, and reports the M3 calibration outcome and the M4 response
agreement per regime label, plus the per-layer summary. Cross-venue layers run
only when the registry deems the datasets comparable; otherwise they are refused.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .calibration_v2 import gate_status, simulated_summary
from .calibration_v2_study import _tape
from .dataset_registry import build, comparable
from .experiments.registry import PROJECT_ROOT
from .impact import event_table
from .impact_validation import compare, simulate_tables
from .observables_v2 import family_errors, loss, measure, summarize
from . import preregistration as pr
from .regimes import BLOCK_S, block_statistics, fit_thresholds, label, regime_counts
from .sim_v2 import SimulatorSpec
from .v05_evidence import LEDGER_PATH, PROTOCOL_PATH, finalize, new_run

DESIGN_PATH = "configs/v05/m6-design.json"
MIN_BLOCKS = 6
POOLED_SCALARS = {"trade_rate", "top_change_rate", "net_cancel_rate", "addition_rate", "acf_return", "acf_abs_return",
                  "acf_spread", "corr_spread_depth", "corr_imbalance_next_return", "corr_flow_return"}


def load_design(root: Path = PROJECT_ROOT) -> dict:
    return json.loads((root / DESIGN_PATH).read_text(encoding="utf-8"))


def fit(select_dir: str | Path, out: str | Path, *, root: Path = PROJECT_ROOT) -> dict:
    selection = json.loads((Path(select_dir) / "result.json").read_text(encoding="utf-8"))
    tape, *_ = _tape("deribit-eth-perp-2020-04-01", selection["scale_native_per_lot"], root, "M6 regime thresholds")
    blocks = block_statistics(tape)
    frozen = fit_thresholds(blocks)
    out = new_run(out)
    result = {"thresholds": frozen, "development_blocks": len(blocks),
              "development_counts": regime_counts(blocks, frozen)}
    finalize(out, analysis="m6-thresholds", dataset_ids=["deribit-eth-perp-2020-04-01"],
             config={"design": load_design(root)}, result=result, root=root)
    return result


def _pool(samples: list[dict]) -> dict:
    merged = {}
    for key in samples[0]:
        values = [s[key] for s in samples]
        if key in POOLED_SCALARS:
            finite = [float(v[0]) for v in values if math.isfinite(float(v[0]))]
            merged[key] = np.asarray([np.mean(finite) if finite else math.nan])
        else:
            merged[key] = np.concatenate(values)
    return merged


def _subset(table: dict, starts: list[float], block_s: float) -> dict:
    keep = np.zeros(len(table["time"]), dtype=bool)
    for start in starts:
        keep |= (table["time"] >= start) & (table["time"] < start + block_s)
    return {k: v[keep] for k, v in table.items()}


def evaluate(dataset_id: str, select_dir: str | Path, m4_dev_dir: str | Path, m6_dir: str | Path, out: str | Path,
             *, root: Path = PROJECT_ROOT) -> dict:
    protocol = pr.load_protocol(root / PROTOCOL_PATH)
    state = pr.replay_ledger(pr.read_ledger(root / LEDGER_PATH), protocol)
    for name in ("m3-selection", "m4-design", "m6-design"):
        if name not in state.designs:
            raise ValueError(f"{name} must be sealed before external-validity evaluation")
    selection = json.loads((Path(select_dir) / "result.json").read_text(encoding="utf-8"))
    edges = json.loads((Path(m4_dev_dir) / "result.json").read_text(encoding="utf-8"))["thresholds"]
    frozen = json.loads((Path(m6_dir) / "result.json").read_text(encoding="utf-8"))["thresholds"]
    from .calibration_v2_study import load_design as m3_design
    from .impact_validation import load_design as m4_design
    search, m4 = m3_design(root)["search"], m4_design(root)
    tape, meta, _, files = _tape(dataset_id, selection["scale_native_per_lot"], root, "M6 external validity")
    blocks = block_statistics(tape)
    labels = {b["start_s"]: label(b, frozen) for b in blocks}
    specs = {name: SimulatorSpec(selection[name]["config"], selection[name]["extensions"]) for name in ("selected", "control")}
    simulated = {name: simulated_summary(spec, search["holdout_seeds"], search["seconds_per_seed"]) for name, spec in specs.items()}
    sim_tables = {name: simulate_tables(spec, m4["simulation_seeds"], m4["simulated_seconds_per_seed"])[0]
                  for name, spec in specs.items()}
    block_samples = {b["start_s"]: measure(tape.block(b["start_s"], b["start_s"] + BLOCK_S)) for b in blocks}
    events = event_table(tape)
    per_regime = {}
    for dimension in list(frozen) + ["stress"]:
        for value in sorted({labels[s][dimension] for s in labels}):
            starts = [s for s in labels if labels[s][dimension] == value]
            key = f"{dimension}={value}"
            if len(starts) < MIN_BLOCKS:
                per_regime[key] = {"blocks": len(starts), "status": "NOT_AVAILABLE", "reason": "fewer than 6 blocks"}
                continue
            target = summarize(_pool([block_samples[s] for s in starts]))
            m3 = {}
            for name in specs:
                errors = family_errors(target, simulated[name])
                m3[name] = {"loss": loss(errors), "gate": gate_status(errors, 1.0, len(starts) * int(BLOCK_S))["status"],
                            "failed_families": gate_status(errors, 1.0, len(starts) * int(BLOCK_S))["failed_families"]}
            subset = _subset(events, starts, BLOCK_S)
            m4_result = {name: compare(subset, sim_tables[name], edges, alpha=m4["statistics"]["alpha_family"],
                                       samples=500, seed=m4["statistics"]["bootstrap_seed"])["H4_by_horizon"]
                         for name in specs}
            per_regime[key] = {"blocks": len(starts), "status": "EVALUATED", "m3": m3,
                               "selected_beats_control": m3["selected"]["loss"] < m3["control"]["loss"],
                               "m4_H4": m4_result}
    evaluated = [r for r in per_regime.values() if r["status"] == "EVALUATED"]
    robust = {
        "selected_passes_gate_in_every_regime": all(r["m3"]["selected"]["gate"] == "ESTABLISHED" for r in evaluated),
        "selected_beats_control_in_every_regime": all(r["selected_beats_control"] for r in evaluated),
        "regimes_evaluated": len(evaluated),
        "regimes_not_available": sorted(k for k, r in per_regime.items() if r["status"] == "NOT_AVAILABLE")}
    registry = build(root)
    venue_check = {other: dict(zip(("comparable", "reasons"), comparable(registry[dataset_id], registry[other])))
                   for other in ("bitstamp-btcusd-mbo-validation",)}
    out = new_run(out)
    result = {"dataset_id": dataset_id, "role": pr.dataset_declaration(protocol, dataset_id)["role"],
              "blocks": len(blocks), "regime_counts": regime_counts(blocks, frozen), "per_regime": per_regime,
              "regime_robustness": robust, "different_venue_layer": venue_check,
              "retrospective": dataset_id in load_design(root)["retrospective"]}
    finalize(out, analysis=f"m6-{dataset_id}", dataset_ids=[dataset_id],
             config={"design": load_design(root), "files": {k: v.name for k, v in files.items()}}, result=result,
             root=root)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("thresholds", "evaluate"))
    parser.add_argument("--dataset")
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--m4-dev", type=Path)
    parser.add_argument("--m6", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == "thresholds":
        print(json.dumps(fit(args.select, args.out), indent=2))
    else:
        result = evaluate(args.dataset, args.select, args.m4_dev, args.m6, args.out)
        print(json.dumps({"robustness": result["regime_robustness"], "counts": result["regime_counts"],
                          "venue": result["different_venue_layer"]}, indent=2))


if __name__ == "__main__":
    main()
