"""M15 execution study across the plausible world set (H7, H8, H9; workstreams 35-45). Simulation only.

Worlds (scenario labels in brackets): G0 point [CALIBRATED], 16 posterior draws [POSTERIOR_SAMPLE] and the
point fit of every available richer family [CALIBRATED]. Agents: the 8 primary classical policies plus the
exploratory MPC, each with fixed parameters in every world (VWAP uses one volume profile estimated on the G0
point world). Every (world, agent) cell runs the same 256 market seeds. The worst-plausible search runs the
primary agents on 64 further posterior draws from the 90% lowest-distance region on 64 market seeds.
No historical data is read.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np

from ...experiments.registry import PROJECT_ROOT
from ...sim_v2 import SimulatorSpec
from ...v06.calibration import spec_from_unit
from ...v06.identifiability import clean
from ..evidence.runs import finalize, new_run, verify_run, write_json
from ..execution import episode
from ..generators.study import load_specs, v06_inputs
from ..posterior.study import _family_inputs, posterior_specs
from ..protocol import core as pr
from . import analysis

RICHER = ("G1_state_hawkes", "G2_regime_switching", "G3_conditional_ar", "G4_neural_temporal")
MARKETS = 256
WORST_CANDIDATES, WORST_MARKETS = 64, 64
SAMPLES = 2000


def worlds(*, generator_run: str, posterior_run: str, root: Path) -> tuple[dict, dict]:
    inputs = v06_inputs(root)
    selected = inputs["selection"]["selected"]
    out = {"G0_point": SimulatorSpec(selected["config"], selected["extensions"])}
    labels = {"G0_point": "CALIBRATED"}
    for i, spec in enumerate(posterior_specs(root / posterior_run, root)):
        out[f"post_{i:02d}"] = spec
        labels[f"post_{i:02d}"] = "POSTERIOR_SAMPLE"
    generators = load_specs(root / generator_run)
    for name in RICHER:
        if name in generators:
            out[name] = generators[name]
            labels[name] = "CALIBRATED"
    return out, labels


def _cell(task: tuple) -> dict:
    world, spec, agent, seeds, profile = task
    rows = [episode.run(spec, agent, episode.MANDATE, s, profile=profile) for s in seeds]
    return {"world": world, "agent": agent, "rows": rows}


def run_cells(tasks: list[tuple], workers: int | None) -> list[dict]:
    workers = workers or max(1, min(14, (os.cpu_count() or 2) - 2))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_cell, tasks, chunksize=1))


def cube_from(results: list[dict], agents: tuple[str, ...]) -> tuple[dict, dict]:
    cube, invalid = {}, {}
    for r in results:
        costs = np.asarray([row[episode.COST] if row.get(episode.COST) is not None else np.nan for row in r["rows"]])
        invalid[f"{r['world']}|{r['agent']}"] = float(np.mean(~np.isfinite(costs)))
        cube.setdefault(r["world"], {})[r["agent"]] = costs
    return cube, invalid


def valid_cube(cube: dict, agents: tuple[str, ...], invalid: dict, limit: float = 0.05) -> tuple[dict, list[str]]:
    """Drop worlds with any INVALID cell (> 5% invalid episodes); impute nothing."""
    dropped = [w for w in cube if any(invalid.get(f"{w}|{a}", 1.0) > limit for a in agents)]
    kept = {}
    for w, cells in cube.items():
        if w in dropped:
            continue
        common = np.all([np.isfinite(cells[a]) for a in agents], axis=0)
        kept[w] = {a: cells[a][common] for a in agents}
    return kept, dropped


def run(out: str | Path, *, generator_run: str, posterior_run: str, root: Path = PROJECT_ROOT,
        workers: int | None = None, markets: int = MARKETS, label: str = "registered") -> dict:
    for run_dir in (generator_run, posterior_run):
        report = verify_run(root / run_dir, root=root)
        if not report["valid"]:
            raise ValueError(f"{run_dir} is not valid: {report['issues']}")
    seeds = pr.load_protocol(root / pr.PROTOCOL_PATH)["seeds"]
    start = seeds["execution"]["market_seed_start"]
    market_seeds = list(range(start, start + markets))
    world_specs, labels = worlds(generator_run=generator_run, posterior_run=posterior_run, root=root)
    out = new_run(root / out)
    profile = episode.volume_profile(world_specs["G0_point"], start - 5000, episode.MANDATE)
    agents_all = episode.PRIMARY + episode.EXPLORATORY
    tasks = [(w, spec, a, market_seeds, profile) for w, spec in world_specs.items() for a in agents_all]
    results = run_cells(tasks, workers)
    write_json(out, "episodes.json", clean(results))
    cube, invalid = cube_from(results, agents_all)
    primary_cube, dropped = valid_cube(cube, episode.PRIMARY, invalid)
    # Worst-plausible search inside the 90% lowest-distance region of the pooled posterior.
    inputs = _family_inputs(root)
    with np.load(root / posterior_run / "particles.npz") as stored:
        pooled, distances = stored["pooled"], stored["distances"]
    region = np.flatnonzero(distances <= np.quantile(distances[np.isfinite(distances)], 0.90))
    rng = np.random.default_rng(seeds["worst_plausible"])
    picks = rng.choice(region, min(WORST_CANDIDATES, len(region)), replace=False)
    worst_seeds = list(range(start + 100_000, start + 100_000 + WORST_MARKETS))
    worst_tasks = [(f"cand_{j:02d}", spec_from_unit(pooled[i], inputs["base"], inputs["tables"]), a, worst_seeds,
                    profile) for j, i in enumerate(picks) for a in episode.PRIMARY]
    worst_results = run_cells(worst_tasks, workers)
    write_json(out, "worst-plausible-episodes.json", clean(worst_results))
    worst_cube, worst_invalid = cube_from(worst_results, episode.PRIMARY)
    worst_cube, worst_dropped = valid_cube(worst_cube, episode.PRIMARY, worst_invalid)
    boot = seeds["bootstrap"]["execution"]
    preliminary = analysis.h8(primary_cube, episode.PRIMARY, alpha=0.05, samples=SAMPLES, seed=boot, worst={})
    signs = {k: v["direction"] for k, v in preliminary["edges"].items()}
    worst = analysis.worst_plausible(worst_cube, signs, episode.PRIMARY)
    h8 = analysis.h8(primary_cube, episode.PRIMARY, alpha=0.05, samples=SAMPLES, seed=boot, worst=worst)
    h7 = analysis.h7(primary_cube, episode.PRIMARY, alpha=0.05, samples=SAMPLES, seed=boot + 1)
    groups = {w: ("G0_post" if w.startswith("post_") else w) for w in primary_cube}
    summary = {w: {a: {"mean_cost_bps": float(np.mean(c)), "sd_cost_bps": float(np.std(c, ddof=1)), "n": int(len(c))}
                   for a, c in cells.items()} for w, cells in cube.items()}
    result = {
        "label": label, "worlds": {w: labels[w] for w in world_specs}, "dropped_worlds": dropped,
        "invalid_fraction": invalid, "markets": markets, "agents_primary": list(episode.PRIMARY),
        "agents_exploratory": list(episode.EXPLORATORY), "vwap_profile": profile, "summary": summary,
        "H7": h7, "H8": h8, "H9": analysis.h9(h8),
        "worst_plausible": {"candidates": int(len(picks)), "markets": WORST_MARKETS, "dropped": worst_dropped,
                            "region": "pooled posterior particles with ABC distance <= its 90th percentile"},
        "topology": analysis.ranking_topology(primary_cube, episode.PRIMARY),
        "topology_with_exploratory": analysis.ranking_topology(
            valid_cube(cube, agents_all, invalid)[0], agents_all),
        "decomposition": analysis.decomposition(primary_cube, episode.PRIMARY, groups),
        "mandate": episode.MANDATE, "ac_parameters": episode.AC_V06,
        "interpretation": "Simulator worlds only; no historical, live or profitability statement."}
    finalize(out, analysis=f"m15-execution-model-risk-{label}", dataset_ids=[],
             config={"generator_run": generator_run, "posterior_run": posterior_run, "markets": markets,
                     "worst_candidates": WORST_CANDIDATES, "worst_markets": WORST_MARKETS, "samples": SAMPLES},
             result=clean(result), root=root, seeds={"market_start": start, "worst_plausible": seeds["worst_plausible"],
                                                     "bootstrap": boot})
    return result


def read(run_dir: Path) -> dict:
    report = verify_run(run_dir)
    if not report["valid"]:
        raise ValueError(f"{run_dir} is not valid: {report['issues']}")
    return json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
