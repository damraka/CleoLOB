"""M5: fit the richer generator families on development data and score them (development and selection).

Every family is estimated on the development day only (ETH 2020-04-01, lots at
the v0.6 development depth scale, v0.6 tick geometry). Each fitted world is then
scored with the frozen v0.6 objective on the development target (descriptive
fit quality) and on the selection target (input to the sealed G* rule, applied
in M6 once the G0 posterior exists). No fresh or retrospective data is read.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
import time

import numpy as np

from ...experiments.registry import PROJECT_ROOT
from ...regimes import BLOCK_S
from ...sim_v2 import SimulatorSpec
from ..data import access
from ..evidence.runs import finalize, new_run, write_json
from ..protocol import core as pr
from ..realism.scoring import run_tasks, seed_se
from . import counts as cm
from .engine import EmpiricalTables, GeneratorSpec
from .events import MARKS, Bundles, extract
from .regime import RegimeSpec, exit_rates

DESIGN = "m5-generator-fit"
DESIGN_RUN = "results/v06/m1/design"
SELECTION_RUN = "results/v06/m6/select"
REGIME_RUN = "results/v06/m12/regime"
TABLE_LIMIT = 200_000
SCORE_SECONDS = 3600.0


def v06_inputs(root: Path) -> dict:
    """Frozen v0.6 design (scales, bins, targets), selected point model and regime models (verified runs)."""
    from ...v06.calibration_study import read_selection
    from ...v06.evidence import verify_run
    from ...v06.realism_study import load_sketches, read_design
    from ...v06.calibration import base_config
    for run in (DESIGN_RUN, SELECTION_RUN, REGIME_RUN):
        report = verify_run(root / run, root=root)
        if not report["valid"]:
            raise ValueError(f"v0.6 input run {run} is not valid: {report['issues']}")
    frozen = read_design(root / DESIGN_RUN, root=root)
    selection = read_selection(root / SELECTION_RUN, root=root)
    regime = json.loads((root / REGIME_RUN / "result.json").read_text(encoding="utf-8"))
    from ..v06_compat import pool
    return {"frozen": frozen, "selection": selection, "regime_models": regime["models"],
            "base": base_config(frozen["tick_bps"]),
            "targets": {"development": pool(load_sketches(root / DESIGN_RUN / "development-sketches.npz")),
                        "selection": pool(load_sketches(root / DESIGN_RUN / "selection-sketches.npz"))}}


def _subsample(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return values if len(values) <= TABLE_LIMIT else values[rng.choice(len(values), TABLE_LIMIT, replace=False)]


def fit(out: str | Path, *, root: Path = PROJECT_ROOT, workers: int | None = None, gru_epochs: int = 4) -> dict:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    seeds = protocol["seeds"]
    inputs = v06_inputs(root)
    frozen = inputs["frozen"]
    out = new_run(root / out)
    started = time.time()
    tape, meta = access.load_tape("deribit-eth-perp-2020-04-01", use="develop", design=DESIGN,
                                  purpose="M5 generator estimation (development only)", root=root,
                                  scale=frozen["scale_native_per_lot"])
    bundles = extract(tape)
    np.savez_compressed(out / "development-bundles.npz", **bundles.to_arrays())
    timing = {"extract_s": time.time() - started}
    rng = np.random.default_rng(seeds["generator_fit"]["G3"])
    tables = EmpiricalTables({k: _subsample(v, rng) for k, v in bundles.sizes.items()},
                             {k: _subsample(v, rng) for k, v in bundles.offsets.items()})
    level_sizes = np.r_[tape.bq[tape.valid, :10].ravel(), tape.aq[tape.valid, :10].ravel()]
    target_level_vol = int(max(1, round(float(np.median(level_sizes[level_sizes > 0])))))
    base = {**inputs["base"], "target_level_vol": target_level_vol}

    t = time.time()
    g1 = cm.HawkesCounts.fit(bundles.counts, bundles.state, bundles.valid)
    timing["G1_fit_s"] = time.time() - t
    t = time.time()
    g3 = cm.ConditionalCounts.fit(bundles.counts, bundles.state, bundles.valid)
    timing["G3_fit_s"] = time.time() - t
    t = time.time()
    try:
        g4 = cm.GRUCounts.fit(bundles.counts, bundles.state, bundles.valid, seed=seeds["generator_fit"]["G4"],
                              epochs=gru_epochs)
        g4_error = None
    except ImportError as exc:      # torch is an optional extra
        g4, g4_error = None, f"NOT_AVAILABLE: {exc}"
    timing["G4_fit_s"] = time.time() - t

    from ...v06.regimes import labelled_blocks, load_thresholds
    thresholds, thresholds_sha = load_thresholds(root)
    labels = [b["labels"]["volatility"] for b in labelled_blocks(tape, thresholds)]
    rates = exit_rates(labels, BLOCK_S)
    regime = inputs["regime_models"]
    specs = {
        "G1_state_hawkes": GeneratorSpec("G1_state_hawkes", base, g1, tables, meta={"fit": g1.fit_info}),
        "G2_regime_switching": RegimeSpec("G2_regime_switching", {k: regime[k]["config"] for k in ("low", "high")},
                                          {k: regime[k]["extensions"] for k in ("low", "high")}, rates,
                                          meta={"labels_high_share": float(np.mean([x == "high" for x in labels])),
                                                "thresholds_sha256": thresholds_sha, "source": REGIME_RUN}),
        "G3_conditional_ar": GeneratorSpec("G3_conditional_ar", base, g3, tables),
    }
    if g4 is not None:
        specs["G4_neural_temporal"] = GeneratorSpec("G4_neural_temporal", base, g4, tables, meta={"fit": g4.fit_info})
    selected = inputs["selection"]["selected"]
    specs["G0_point"] = SimulatorSpec(selected["config"], selected["extensions"])
    with (out / "specs.pkl").open("xb") as handle:
        pickle.dump(specs, handle)
    del tape

    context = {"design": frozen["design"], "scales": frozen["objective_scales"], "targets": inputs["targets"]}
    score_seeds = seeds["selection_seeds"]
    tasks = [(f"{name}|{target}", spec, score_seeds, SCORE_SECONDS, target) for name, spec in specs.items()
             for target in ("development", "selection")]
    t = time.time()
    scored = run_tasks(tasks, context, workers=workers)
    timing["scoring_s"] = time.time() - t
    scores = {}
    for r in scored:
        name, target = r["key"].split("|")
        scores.setdefault(name, {})[target] = {
            "objective": r.get("objective"), "families": r.get("families"), "seed_objectives": r.get("seed_objectives"),
            "seed_se": seed_se(r.get("seed_objectives") or []), "error": r["error"], "elapsed_s": r["elapsed_s"]}
    from ..v06_compat import FAMILIES
    result = {
        "label": "DEVELOPMENT (fit) / SELECTION (scores for the sealed G* rule)",
        "families": {name: (spec.describe() if hasattr(spec, "describe") else {"family": "G0_point"})
                     for name, spec in specs.items()},
        "not_available": {"G4_neural_temporal": g4_error} if g4_error else {},
        "diffusion": "NOT_AVAILABLE: no GPU and no development data volume adequate for a diffusion generator",
        "bundles": {"bins": int(len(bundles.counts)), "valid_bins": int(bundles.valid.sum()),
                    "mean_counts_per_bin": dict(zip(MARKS, bundles.counts[bundles.valid].mean(0).tolist()))},
        "target_level_vol": target_level_vol, "exit_rates_per_s": rates, "family_order": list(FAMILIES),
        "scores": scores, "score_seeds": score_seeds, "score_seconds": SCORE_SECONDS, "timing_s": timing,
        "dataset_quality": meta["quality"]["status"]}
    write_json(out, "scores.json", scores)
    access.mark_evaluated("deribit-eth-perp-2020-04-01", use="develop", design=DESIGN,
                          run=out.relative_to(root).as_posix(), root=root)
    finalize(out, analysis="m5-generator-fit", dataset_ids=["deribit-eth-perp-2020-04-01"],
             config={"table_limit": TABLE_LIMIT, "score_seconds": SCORE_SECONDS, "gru_epochs": gru_epochs},
             result=result, root=root, seeds={"selection_seeds": score_seeds, "generator_fit": seeds["generator_fit"]})
    return result


def load_specs(run: Path) -> dict:
    from ..evidence.runs import verify_run
    report = verify_run(run)
    if not report["valid"]:
        raise ValueError(f"generator run {run} is not valid: {report['issues']}")
    with (run / "specs.pkl").open("rb") as handle:
        return pickle.load(handle)


def load_bundles(run: Path) -> Bundles:
    with np.load(run / "development-bundles.npz") as stored:
        return Bundles.from_arrays({k: stored[k] for k in stored.files})
