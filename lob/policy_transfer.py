"""M8: historical-versus-synthetic policy transfer under bounded historical fills.

The frozen design is ``configs/v05/m8-design.json``. Policies trained and
registered in M7 run unchanged on historical episodes through
``lob.historical_sim.HistoricalSimulator`` under each declared fill mode. Pairwise
conclusions are classified against the M7 synthetic direction; bounded evidence
is never collapsed into an exact historical point estimate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json
import math
import os
from pathlib import Path

import numpy as np

from .calibration_v2 import tape_from_simulator
from .experiments.registry import PROJECT_ROOT, sha256_file
from .historical_sim import FILL_MODES, HistoricalEpisode
from .observables_v2 import measure, summarize
from . import preregistration as pr
from .policy_study_v05 import ALGORITHMS, COMPLETION, COST, V05Design, episode_params
from .policy_study import CONTROLS, EvaluationPolicy
from .replay.l2 import L2Replay
from .runner import run_episode
from .sim_v2 import SimulatorSpec
from .v05_data import acquire
from .v05_evidence import LEDGER_PATH, PROTOCOL_PATH, finalize, new_run, write_json

DESIGN_PATH = "configs/v05/m8-design.json"
TICKS = {"ETH-PERPETUAL": 0.05, "BTC-PERPETUAL": 0.5}


def load_design(root: Path = PROJECT_ROOT) -> dict:
    return json.loads((root / DESIGN_PATH).read_text(encoding="utf-8"))


def _m7(m7_dir: Path) -> tuple[dict, V05Design, dict, dict]:
    plan = json.loads((m7_dir / "preregistration.json").read_text(encoding="utf-8"))
    design = V05Design.model_validate_json(json.dumps(plan["design"]))
    lock = json.loads((m7_dir / "evaluation_lock.json").read_text(encoding="utf-8"))
    result = json.loads((m7_dir / "result.json").read_text(encoding="utf-8"))
    return plan, design, lock, result


def fit_mapping(m7_dir: Path, m3_develop_dir: Path, *, root: Path = PROJECT_ROOT) -> dict:
    """Clock and lot mapping from development data and the M7 original regime only."""
    design_m8 = load_design(root)["mapping"]
    _, design, _, _ = _m7(m7_dir)
    config = design.config.market.model_dump()
    spec = SimulatorSpec({**config, "record_events": False, "max_events": 5_000_000}, {})
    pooled_rates, depths = [], []
    for seed in design_m8["simulation_seeds"]:
        tape = tape_from_simulator(spec, seed, seconds=design_m8["simulated_seconds_per_seed"])
        pooled_rates.append(summarize(measure(tape))["top_change_rate"])
        valid = tape.valid
        depths.append(np.r_[tape.bq[valid].sum(1), tape.aq[valid].sum(1)])
    sim_rate = float(np.mean(pooled_rates))
    sim_depth = float(np.median(np.concatenate(depths)))
    developed = json.loads((m3_develop_dir / "result.json").read_text(encoding="utf-8"))
    hist_rate = developed["development_summary"]["top_change_rate"]
    hist_depth_native = developed["scale_native_per_lot"] * 1000.0
    return {"clock_ratio": sim_rate / hist_rate, "lots_per_native": sim_depth / hist_depth_native,
            "sim_top_change_rate": sim_rate, "hist_top_change_rate": hist_rate,
            "sim_median_side_depth_lots": sim_depth, "hist_median_side_depth_native": hist_depth_native}


def extract_episodes(dataset_id: str, mapping: dict, window_sim_s: float, *, root: Path = PROJECT_ROOT,
                     purpose: str) -> tuple[list[HistoricalEpisode], dict]:
    design = load_design(root)["episodes"]
    protocol = pr.load_protocol(root / PROTOCOL_PATH)
    instrument = pr.dataset_declaration(protocol, dataset_id)["instrument"]
    tick = TICKS[instrument]
    lots = mapping["lots_per_native"]
    files = acquire(dataset_id, root=root, purpose=purpose)
    window = window_sim_s * mapping["clock_ratio"]
    replay = L2Replay(files["l2"], depth=design["book_depth_levels"])
    starts, windows, first = None, None, None
    previous = None
    for state in replay:
        t = state.local_timestamp_us / 1e6
        if starts is None:
            first = t
            starts = [first + 300.0 + 600.0 * k for k in range(design["count"])]
            windows = [[] for _ in starts]
        snapshot = (t, {int(round(float(p) / tick)): float(q) * lots for p, q in state.bids},
                    {int(round(float(p) / tick)): float(q) * lots for p, q in state.asks})
        k = int((t - first - 300.0) // 600.0)
        for j in (k, k + 1):
            if 0 <= j < len(starts) and starts[j] - 5.0 <= t <= starts[j] + window:
                if not windows[j] and previous is not None:
                    windows[j].append(previous)
                windows[j].append(snapshot)
        previous = snapshot
    if not replay.stats["complete"]:
        raise ValueError("L2 source incomplete")
    prints = [[] for _ in starts]
    opener = gzip.open if str(files["trades"]).endswith(".gz") else open
    with opener(files["trades"], "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            side = row["side"].lower()
            if side not in {"buy", "sell"}:
                continue
            t = int(row["local_timestamp"]) / 1e6
            k = int((t - first - 300.0) // 600.0)
            if 0 <= k < len(starts) and starts[k] < t <= starts[k] + window:
                prints[k].append((t, int(round(float(row["price"]) / tick)), float(row["amount"]) * lots,
                                  "BUY" if side == "buy" else "SELL"))
    episodes = []
    for k, start in enumerate(starts):
        updates = [u for u in windows[k] if u[0] <= start + window]
        valid = updates and updates[0][0] <= start
        if not valid:
            continue
        head = [u for u in updates if u[0] <= start][-1]
        updates = [(start, head[1], head[2])] + [u for u in updates if u[0] > start]
        episodes.append(HistoricalEpisode(start, updates, prints[k], tick, mapping["clock_ratio"],
                                          label=f"{dataset_id}#{k}", meta={"index": k}))
    return episodes, {"episodes": len(episodes), "planned": len(starts), "tick": tick,
                      "files": {k: v.name for k, v in files.items()}}


def _run_chunk(task: tuple) -> list[dict]:
    m7_dir, plan, lock, episodes, dataset_id, policies = task
    import torch
    torch.set_num_threads(1)
    m7_dir = Path(m7_dir)
    design = V05Design.model_validate_json(json.dumps(plan["design"]))
    normalization = json.loads((m7_dir / "normalization.json").read_text(encoding="utf-8"))["normalization"]
    models = {}
    rows = []
    for episode in episodes:
        for agent, seed in policies:
            model = None
            if agent in ALGORITHMS:
                key = (agent, seed)
                if key not in models:
                    from stable_baselines3 import DQN, PPO
                    stem = f"{agent}-main-{seed}"
                    meta = json.loads((m7_dir / "models" / f"{stem}.json").read_text(encoding="utf-8"))
                    path = m7_dir / "models" / f"{stem}.zip"
                    if sha256_file(path) != meta["model_sha256"]:
                        raise ValueError(f"checkpoint changed: {stem}")
                    models[key] = EvaluationPolicy({"ppo": PPO, "dqn": DQN}[agent].load(path, device="cpu"), "main")
                model = models[key]
            for mode in FILL_MODES:
                try:
                    params = episode_params(design, "original", None, 900_000 + episode.meta["index"],
                                            design.config.execution.terminal_penalty_bps, normalization)
                    if agent == "ac":
                        params.update(lock["controls"]["original"]["selected"])
                    params["historical"] = {"episode": episode, "fill_mode": mode}
                    row = run_episode("ppo" if agent in ALGORITHMS else agent, params, model=model)
                    row.pop("audit", None)
                except Exception as exc:
                    row = {"status": "INVALID", "error": f"{type(exc).__name__}: {exc}", COST: None, COMPLETION: None}
                keep = {k: row.get(k) for k in (COST, COMPLETION, "status", "error", "mandate_final_settlement_completion",
                                                 "mandate_fill_fraction_at_horizon", "mandate_realized_fill_cost_bps",
                                                 "mandate_hypothetical_residual_valuation_bps", "mandate_post_horizon_filled_qty")}
                keep.update(dataset=dataset_id, episode=episode.meta["index"], agent=agent, training_seed=seed,
                            fill_mode=mode)
                rows.append(keep)
    return rows


def _paired(rows: list[dict], dataset: str, mode: str, left: str, right: str) -> np.ndarray:
    def per_episode(agent: str) -> dict[int, float]:
        values: dict[int, list[float]] = {}
        for r in rows:
            if (r["dataset"], r["fill_mode"], r["agent"]) == (dataset, mode, agent):
                value = r.get(COST)
                values.setdefault(r["episode"], []).append(float(value) if r.get("status") in {"VALID", "WARNING"}
                                                            and value is not None else math.nan)
        return {k: float(np.mean(v)) for k, v in values.items()}
    a, b = per_episode(left), per_episode(right)
    return np.array([a[k] - b[k] for k in sorted(set(a) & set(b))])


def _interval(delta: np.ndarray, alpha: float, samples: int, seed: int):
    if len(delta) < 10 or not np.isfinite(delta).all():
        return None
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), (samples, len(delta)))].mean(axis=1)
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(delta.mean()), "ci_low": float(low), "ci_high": float(high), "n": int(len(delta))}


def _synthetic(m7_result: dict, left: str, right: str) -> dict | None:
    for c in m7_result["comparisons"]:
        if c["regime"] != "original" or c["cost"]["status"] != "AVAILABLE":
            continue
        if c["agent"] == f"{left}:main" and c["reference"] == right:
            return {"mean": c["cost"]["mean_delta_bps"], "ci_low": c["cost"]["ci_low_bps"],
                    "ci_high": c["cost"]["ci_high_bps"]}
    return None


def classify(synthetic: dict | None, historical: dict[str, dict | None]) -> str:
    if synthetic is None or any(v is None for v in historical.values()):
        return "not_evaluable"
    direction = np.sign(synthetic["mean"])
    if direction == 0:
        return "indeterminate"
    def excludes(interval, sign):
        return interval["ci_high"] < 0 if sign < 0 else interval["ci_low"] > 0
    same = {mode: excludes(v, direction) for mode, v in historical.items()}
    opposite = {mode: excludes(v, -direction) for mode, v in historical.items()}
    if all(same.values()):
        return "agrees_survives_conservative"
    if same.get("optimistic") and not same.get("conservative"):
        return "optimistic_assumption_dependent"
    if all(opposite.values()):
        return "reverses"
    return "indeterminate"


def _kendall(a: list[str], b: list[str]) -> float | None:
    common = [x for x in a if x in b]
    if len(common) < 2:
        return None
    ra, rb = {x: i for i, x in enumerate(a)}, {x: i for i, x in enumerate(b)}
    pairs = [(i, j) for i in range(len(common)) for j in range(i + 1, len(common))]
    score = sum(np.sign(ra[common[i]] - ra[common[j]]) * np.sign(rb[common[i]] - rb[common[j]]) for i, j in pairs)
    return float(score / len(pairs))


def run(m7_dir: str | Path, m3_develop_dir: str | Path, out: str | Path, *, root: Path = PROJECT_ROOT,
        datasets: list[str] | None = None) -> dict:
    protocol = pr.load_protocol(root / PROTOCOL_PATH)
    state = pr.replay_ledger(pr.read_ledger(root / LEDGER_PATH), protocol)
    if "m8-design" not in state.designs:
        raise ValueError("M8 design must be sealed before transfer evaluation")
    m7_dir, design8 = Path(m7_dir), load_design(root)
    plan, design, lock, m7_result = _m7(m7_dir)
    if plan["evidence_level"] != "research":
        raise ValueError("M8 requires the registered (research) M7 study, not a pilot")
    mapping = fit_mapping(m7_dir, Path(m3_develop_dir), root=root)
    window_sim = design.config.execution.warmup_seconds + design.config.execution.horizon + \
        design.config.execution.settlement_timeout + 1.0
    datasets = datasets or [design8["datasets"]["primary_transfer_holdout"], *design8["datasets"]["secondary"]]
    policies = [(c, None) for c in CONTROLS] + [(a, s) for a in ALGORITHMS for s in design.training_seeds]
    out = new_run(out)
    rows, extraction = [], {}
    workers = int(os.environ.get("CLEOLOB_WORKERS", max(1, min(12, (os.cpu_count() or 2) - 2))))
    for dataset in datasets:
        episodes, meta = extract_episodes(dataset, mapping, window_sim, root=root, purpose="M8 policy transfer")
        extraction[dataset] = meta
        chunks = [episodes[i::workers] for i in range(workers)]
        tasks = [(str(m7_dir), plan, lock, chunk, dataset, policies) for chunk in chunks if chunk]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for part in pool.map(_run_chunk, tasks):
                rows.extend(part)
    write_json(out, "episodes.json", rows)
    pairs = [(a, c) for a in ALGORITHMS for c in CONTROLS] + [("ppo", "dqn")]
    family = len(pairs) * len(datasets) * len(FILL_MODES)
    alpha = 0.05 / family
    comparisons = []
    for dataset in datasets:
        for pair_index, (left, right) in enumerate(pairs):
            historical = {mode: _interval(_paired(rows, dataset, mode, left, right), alpha, 5000,
                                          48901 + 10 * pair_index + mode_index)
                          for mode_index, mode in enumerate(FILL_MODES)}
            synthetic = _synthetic(m7_result, left, right) if right in CONTROLS else None
            comparisons.append({"dataset": dataset, "left": left, "right": right, "synthetic": synthetic,
                                "historical": historical, "classification": classify(synthetic, historical)})
    rankings = {}
    synthetic_means = {s["agent"]: s.get("mean_cost_bps") for s in m7_result["summaries"]
                       if s["regime"] == "original" and s["arm"] == "main"}
    synthetic_rank = sorted((a for a in synthetic_means if synthetic_means[a] is not None), key=synthetic_means.get)
    for dataset in datasets:
        for mode in FILL_MODES:
            means = {}
            for agent in list(CONTROLS) + list(ALGORITHMS):
                values = [float(r[COST]) for r in rows if (r["dataset"], r["fill_mode"], r["agent"]) == (dataset, mode, agent)
                          and r.get(COST) is not None and r.get("status") in {"VALID", "WARNING"}]
                completion = [r.get(COMPLETION) for r in rows if (r["dataset"], r["fill_mode"], r["agent"]) == (dataset, mode, agent)]
                means[agent] = {"mean_cost_bps": float(np.mean(values)) if values else None, "n": len(values),
                                "within_horizon_completion_rate": float(np.mean([c is True for c in completion])) if completion else None}
            ranked = sorted((a for a in means if means[a]["mean_cost_bps"] is not None), key=lambda a: means[a]["mean_cost_bps"])
            rankings[f"{dataset}|{mode}"] = {"policies": means, "ranking": ranked,
                                             "kendall_tau_vs_synthetic": _kendall(synthetic_rank, ranked)}
    counts = {}
    for c in comparisons:
        counts[c["classification"]] = counts.get(c["classification"], 0) + 1
    result = {"mapping": mapping, "extraction": extraction, "family_size": family, "alpha": alpha,
              "comparisons": comparisons, "classification_counts": counts, "rankings": rankings,
              "synthetic_ranking_original_regime": synthetic_rank,
              "invalid_rows": sum(r.get("status") == "INVALID" for r in rows), "rows": len(rows),
              "interpretation": ("Bounded historical replay with no market impact of the hypothetical parent; "
                                 "conservative and optimistic modes are separate self-consistent paths; no exact "
                                 "historical fill, profitability or live claim follows.")}
    finalize(out, analysis="m8-transfer", dataset_ids=datasets,
             config={"design": design8, "m7_plan_sha256": pr.document_sha256(plan), "mapping": mapping},
             result=result, root=root)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m7", type=Path, required=True)
    parser.add_argument("--m3-develop", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.m7, args.m3_develop, args.out)
    print(json.dumps({"counts": result["classification_counts"], "mapping": result["mapping"]}, indent=2))


if __name__ == "__main__":
    main()
