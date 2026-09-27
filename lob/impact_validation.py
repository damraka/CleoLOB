"""Historical versus simulated impact/resilience under frozen horizons and buckets (M4).

Historical means get chronological 10-minute block bootstrap intervals (events
are clustered in time); simulated means use the simulator-seed bootstrap. For
each size bucket and horizon the comparison reports effect sizes, intervals,
sign agreement, log magnitude ratio, a two-sample distribution distance and the
persistence (60 s / 1 s) difference. The frozen H4 gate is evaluated per horizon.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .calibration_v2 import tape_from_simulator
from .calibration_v2_study import _tape
from .experiments.registry import PROJECT_ROOT
from .impact import BUCKET_DIMENSIONS, HORIZONS_S, ac_proxies, buckets, conditional_means, event_table, thresholds
from .observables_v2 import SAMPLE_DT
from . import preregistration as pr
from .resilience import depletion_events, recovery
from .sim_v2 import SimulatorSpec
from .v05_evidence import LEDGER_PATH, PROTOCOL_PATH, finalize, new_run, write_json

DESIGN_PATH = "configs/v05/m4-design.json"
GATE_HORIZONS = (1.0, 10.0, 60.0)
MIN_EVENTS = 100


def load_design(root: Path = PROJECT_ROOT) -> dict:
    return json.loads((root / DESIGN_PATH).read_text(encoding="utf-8"))


def _one_second_returns(tape) -> np.ndarray:
    step = int(round(1 / SAMPLE_DT))
    mid = tape.mid[::step]
    ok = tape.valid[::step]
    r = np.log(mid[1:] / mid[:-1]) * 1e4
    return r[ok[1:] & ok[:-1]]


def _block_ci(values: np.ndarray, times: np.ndarray, *, block: float, samples: int, alpha: float, seed: int):
    keep = np.isfinite(values)
    values, times = values[keep], times[keep]
    if len(values) < MIN_EVENTS:
        return None
    labels = np.floor((times - times.min()) / block).astype(int)
    groups = [values[labels == g] for g in np.unique(labels)]
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for i in range(samples):
        pick = rng.integers(0, len(groups), len(groups))
        draws[i] = sums[pick].sum() / max(1, counts[pick].sum())
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return [float(low), float(high)]


def _seed_ci(per_seed: list[np.ndarray], *, samples: int, alpha: float, seed: int):
    finite = [v[np.isfinite(v)] for v in per_seed]
    if sum(len(v) for v in finite) < MIN_EVENTS or len(finite) < 2:
        return None
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for i in range(samples):
        pick = rng.integers(0, len(finite), len(finite))
        pooled = np.concatenate([finite[j] for j in pick])
        draws[i] = pooled.mean() if len(pooled) else math.nan
    low, high = np.nanquantile(draws, [alpha / 2, 1 - alpha / 2])
    return [float(low), float(high)]


def _ks(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)])
    if len(a) < 10 or len(b) < 10:
        return None
    grid = np.r_[a, b]
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                                - np.searchsorted(b, grid, side="right") / len(b))))


def compare(hist: dict[str, np.ndarray], sims: list[dict[str, np.ndarray]], edges: dict, *, alpha: float,
            samples: int, seed: int) -> dict:
    """Per size bucket and horizon: effect sizes, intervals, sign/magnitude agreement, KS, persistence."""
    hist_labels = buckets(hist, edges, "size")
    sim_labels = [buckets(s, edges, "size") for s in sims]
    cells, per_horizon = {}, {}
    for h in HORIZONS_S:
        name = f"mid_bps_{h:g}s"
        agree = significant = evaluable = 0
        for label in range(len(edges["size"]) + 1):
            hmask = hist["valid_pre"] & (hist_labels == label)
            hv = hist[name][hmask]
            sv = [s[name][s["valid_pre"] & (lab == label)] for s, lab in zip(sims, sim_labels)]
            pooled = np.concatenate(sv) if sv else np.array([])
            hn, sn = int(np.isfinite(hv).sum()), int(np.isfinite(pooled).sum())
            cell = {"horizon_s": h, "size_bucket": label, "hist_n": hn, "sim_n": sn}
            if hn < MIN_EVENTS or sn < MIN_EVENTS:
                cell["status"] = "NOT_AVAILABLE"
                cells[f"{h:g}s|{label}"] = cell
                continue
            hmean, smean = float(np.nanmean(hv)), float(np.nanmean(pooled))
            hci = _block_ci(hv, hist["time"][hmask], block=600.0, samples=samples, alpha=alpha, seed=seed + label)
            sci = _seed_ci(sv, samples=samples, alpha=alpha, seed=seed + 100 + label)
            ratio = (math.log(abs(smean) / abs(hmean)) if hmean and smean and (hmean > 0) == (smean > 0) else None)
            sign_ok = (hmean > 0) == (smean > 0) and hmean != 0 and smean != 0
            magnitude_ok = ratio is not None and abs(ratio) <= math.log(2)
            nonzero = hci is not None and (hci[0] > 0 or hci[1] < 0)
            pooled_sd = math.sqrt((np.nanvar(hv) + np.nanvar(pooled)) / 2) or math.nan
            cell.update(status="EVALUATED", hist_mean_bps=hmean, sim_mean_bps=smean, hist_ci=hci, sim_ci=sci,
                        difference_bps=smean - hmean, standardized_difference=(smean - hmean) / pooled_sd,
                        log_magnitude_ratio=ratio, sign_agreement=sign_ok, magnitude_within_2x=magnitude_ok,
                        hist_nonzero_bonferroni=nonzero, ks_distance=_ks(hv, pooled))
            cells[f"{h:g}s|{label}"] = cell
            if h in GATE_HORIZONS:
                evaluable += 1
                agree += int(sign_ok and magnitude_ok)
                significant += int(nonzero)
        if h in GATE_HORIZONS:
            established = evaluable > 0 and agree >= math.ceil(2 * evaluable / 3) and significant >= math.ceil(evaluable / 2)
            per_horizon[f"{h:g}s"] = {"evaluable_buckets": evaluable, "agreeing_buckets": agree,
                                      "significant_hist_buckets": significant,
                                      "status": ("ESTABLISHED" if established else
                                                 "NOT_AVAILABLE" if evaluable == 0 else "NOT_ESTABLISHED")}
    persistence = {}
    for source, tables in (("hist", [hist]), ("sim", sims)):
        num = np.concatenate([t["mid_bps_60s"][t["valid_pre"]] for t in tables])
        den = np.concatenate([t["mid_bps_1s"][t["valid_pre"]] for t in tables])
        m1, m60 = np.nanmean(den), np.nanmean(num)
        persistence[source] = float(m60 / m1) if m1 else None
    return {"cells": cells, "H4_by_horizon": per_horizon, "persistence_60s_over_1s": persistence,
            "persistence_difference": (persistence["sim"] - persistence["hist"])
            if None not in persistence.values() else None}


def conditional_profile(table: dict[str, np.ndarray], edges: dict) -> dict:
    return {dimension: {response: conditional_means(table, edges, dimension, response)
                        for response in ("mid_bps_1s", "mid_bps_10s", "mid_bps_60s", "spread_change_bps_1s",
                                         "opposite_depth_ratio_1s", "imbalance_change_1s")}
            for dimension in BUCKET_DIMENSIONS}


def simulate_tables(spec: SimulatorSpec, seeds: list[int], seconds: float) -> tuple[list[dict], list]:
    tables, tapes = [], []
    for seed in seeds:
        tape = tape_from_simulator(spec, seed, seconds=seconds)
        tables.append(event_table(tape))
        tapes.append(tape)
    return tables, tapes


def run(dataset_id: str, select_dir: str | Path, out: str | Path, *, thresholds_from: str | Path | None = None,
        root: Path = PROJECT_ROOT, label: str = "registered") -> dict:
    """Development run (thresholds_from=None) freezes bucket edges; later runs must reuse them."""
    design = load_design(root)
    selection = json.loads((Path(select_dir) / "result.json").read_text(encoding="utf-8"))
    protocol = pr.load_protocol(root / PROTOCOL_PATH)
    declaration = pr.dataset_declaration(protocol, dataset_id)
    if declaration["role"] in {"external_holdout", "transfer_holdout", "internal_holdout"}:
        state = pr.replay_ledger(pr.read_ledger(root / LEDGER_PATH), protocol)
        if "m4-design" not in state.designs or thresholds_from is None:
            raise ValueError("holdout impact evaluation requires the sealed M4 design and frozen thresholds")
    tape, meta, _, files = _tape(dataset_id, selection["scale_native_per_lot"], root, f"M4 impact/resilience {label}")
    table = event_table(tape)
    if thresholds_from is None:
        if declaration["role"] != "development":
            raise ValueError("bucket thresholds may only be fitted on the development period")
        edges = thresholds(table)
    else:
        edges = json.loads((Path(thresholds_from) / "result.json").read_text(encoding="utf-8"))["thresholds"]
    stats = design["statistics"]
    sims, sim_tapes = {}, {}
    for name in ("selected", "control"):
        spec = SimulatorSpec(selection[name]["config"], selection[name]["extensions"])
        sims[name], sim_tapes[name] = simulate_tables(spec, design["simulation_seeds"], design["simulated_seconds_per_seed"])
    out = new_run(out)
    comparisons = {name: compare(table, sims[name], edges, alpha=stats["alpha_family"], samples=stats["bootstrap_samples"],
                                 seed=stats["bootstrap_seed"]) for name in sims}
    hist_resilience = recovery(tape, depletion_events(tape))
    sim_resilience = {name: [recovery(t, depletion_events(t)) for t in sim_tapes[name]] for name in sims}
    result = {"dataset_id": dataset_id, "label": label, "thresholds": edges, "events": int(table["valid_pre"].sum()),
              "historical_profile": conditional_profile(table, edges),
              "historical_ac_proxies": ac_proxies(table, _one_second_returns(tape)),
              "simulated_ac_proxies": {name: ac_proxies({k: np.concatenate([t[k] for t in tabs]) for k in tabs[0]},
                                                        np.concatenate([_one_second_returns(t) for t in sim_tapes[name]]))
                                       for name, tabs in sims.items()},
              "comparison": comparisons, "historical_resilience": hist_resilience,
              "simulated_resilience": {name: _merge_resilience(v) for name, v in sim_resilience.items()},
              "selected_family": selection["selected"]["family"],
              "interpretation": ("Responses are associations with signed aggressive flow, not causal effects; "
                                 "net depth cannot separate replenishment from cancellations.")}
    write_json(out, "historical-profile.json", result["historical_profile"])
    finalize(out, analysis=f"m4-{label}-{dataset_id}", dataset_ids=[dataset_id],
             config={"design": design, "design_sha256": pr.document_sha256(design),
                     "select_result_sha256": pr.file_sha256(Path(select_dir) / "result.json"),
                     "files": {k: v.name for k, v in files.items()}},
             result=result, root=root)
    return result


def _merge_resilience(items: list[dict]) -> dict:
    events = sum(i["events"] for i in items)
    recovered = sum(i["recovered"] for i in items)
    curve = {}
    for key in items[0]["net_replenishment_curve_top5"]:
        values = [i["net_replenishment_curve_top5"][key] for i in items if i["net_replenishment_curve_top5"][key] is not None]
        curve[key] = float(np.mean(values)) if values else None
    return {"events": events, "recovered_share": recovered / events if events else None,
            "per_seed_recovery_quantiles_s": [i["recovery_time_quantiles_s"] for i in items],
            "net_replenishment_curve_top5": curve}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id")
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--thresholds-from", type=Path)
    parser.add_argument("--label", default="registered")
    args = parser.parse_args(argv)
    result = run(args.dataset_id, args.select, args.out, thresholds_from=args.thresholds_from, label=args.label)
    print(json.dumps({name: c["H4_by_horizon"] for name, c in result["comparison"].items()}, indent=2))


if __name__ == "__main__":
    main()
