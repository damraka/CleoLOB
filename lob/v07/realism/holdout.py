"""Fresh-holdout realism: simulation bank, sealed design and registered evaluation (H1-H5, H10, H11).

bank     -> results/v07/m10/bank: every simulated quantity the holdout analyses use, computed and
            hashed before any fresh access (per-seed sketches and 60 s window features of every model).
seal     -> configs/v07/holdout-design.json + ledger ``seal_design`` reading the four fresh realism
            holdouts (ETH 2020-11-01, ETH 2020-12-01, BTC 2020-11-01, BitMEX XBTUSD 2020-11-01).
evaluate -> results/v07/m17/<dataset>: contrasts, domain gap and support contrasts, descriptive metrics.

Historical quantities use the ETH development lot scale and the v0.6 measurement grid unchanged for
every dataset (instrument and venue transfer are deliberate tests). Statuses follow the registered
hypotheses; NOT_AVAILABLE and INVALID are recorded, never substituted.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
import pickle
from pathlib import Path

import numpy as np

from ...experiments.registry import PROJECT_ROOT
from ...sim_v2 import SimulatorSpec
from ...v06.domain_gap import window_features
from ...v06.history import block_raws
from ..calibration.execution_aware import spec as ea_spec
from ..data import access, capability
from ..data.tape import READER_LIMITS
from ..evidence.runs import finalize, new_run, verify_run
from ..generators.study import load_specs, v06_inputs
from ..posterior.study import posterior_specs
from ..protocol import core as pr
from ..v06_compat import measure, pool, sketch, tape_from_simulator
from . import contrast, metrics
from .scoring import capped

DESIGN = "m17-holdout-realism"
DESIGN_PATH = "configs/v07/holdout-design.json"
FRESH = {"deribit-eth-perp-2020-11-01": "FRESH_TEMPORAL", "deribit-eth-perp-2020-12-01": "FRESH_TEMPORAL",
         "deribit-btc-perp-2020-11-01": "FRESH_CROSS_INSTRUMENT", "bitmex-xbtusd-2020-11-01": "FRESH_CROSS_VENUE"}
RICHER = ("G1_state_hawkes", "G2_regime_switching", "G3_conditional_ar", "G4_neural_temporal")
SKETCH_SEEDS, SKETCH_S = 32, 3600.0
WINDOW_SEEDS, WINDOW_S = 16, 10800.0
SAMPLES = 2000


def models(*, generator_run: str, posterior_run: str, ea_run: str, root: Path) -> dict[str, list]:
    """Model name -> list of specs (one per simulated unit family; G0_post has 16 posterior draws)."""
    inputs = v06_inputs(root)
    generators = load_specs(root / generator_run)
    selected = inputs["selection"]["selected"]
    out = {"G0_point": [SimulatorSpec(selected["config"], selected["extensions"])],
           "G0_post": posterior_specs(root / posterior_run, root), "EA": [ea_spec(root / ea_run)]}
    for name in RICHER:
        if name in generators:
            out[name] = [generators[name]]
    return out


def _bank_task(task: tuple) -> dict:
    kind, name, unit, spec, seed, seconds, design = task
    tape = tape_from_simulator(capped(spec, seconds), seed, seconds=seconds)
    if kind == "sketch":
        return {"kind": kind, "name": name, "unit": unit, "sketch": pool([sketch(measure(b), design)
                                                                          for b in tape.blocks(600.0)])}
    return {"kind": kind, "name": name, "unit": unit, "windows": window_features(tape)}


def bank(out: str | Path, *, generator_run: str, posterior_run: str, ea_run: str, root: Path = PROJECT_ROOT,
         workers: int | None = None, label: str = "registered") -> dict:
    seeds = pr.load_protocol(root / pr.PROTOCOL_PATH)["seeds"]
    frozen = v06_inputs(root)["frozen"]
    specs = models(generator_run=generator_run, posterior_run=posterior_run, ea_run=ea_run, root=root)
    base = seeds["holdout_seeds_base"]
    tasks = []
    for name, family in specs.items():
        if len(family) == 1:
            for k in range(SKETCH_SEEDS):
                tasks.append(("sketch", name, k, family[0], base + k, SKETCH_S, frozen["design"]))
            for k in range(WINDOW_SEEDS):
                tasks.append(("windows", name, k, family[0], base + 10_000 + k, WINDOW_S, frozen["design"]))
        else:
            per_draw = SKETCH_SEEDS // len(family)
            for d, spec in enumerate(family):
                for j in range(per_draw):
                    tasks.append(("sketch", name, d * per_draw + j, spec, base + 20_000 + d * per_draw + j, SKETCH_S,
                                  frozen["design"]))
                tasks.append(("windows", name, d, spec, base + 30_000 + d, WINDOW_S, frozen["design"]))
    out = new_run(root / out)
    workers = workers or max(1, min(14, (os.cpu_count() or 2) - 2))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_bank_task, tasks, chunksize=1))
    sketches = {n: [r["sketch"] for r in sorted(results, key=lambda r: r["unit"]) if r["kind"] == "sketch"
                    and r["name"] == n] for n in specs}
    windows = {n: [r["windows"] for r in sorted(results, key=lambda r: r["unit"]) if r["kind"] == "windows"
                   and r["name"] == n] for n in specs}
    with (out / "sketches.pkl").open("xb") as handle:
        pickle.dump(sketches, handle)
    np.savez_compressed(out / "windows.npz", **{f"{n}__{i}": w for n, ws in windows.items() for i, w in enumerate(ws)})
    files = {p.name: pr.file_sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    result = {"label": label, "models": {n: [s.describe() if hasattr(s, "describe") else
                                            {"config": s.config, "extensions": s.extensions} for s in fam]
                                         for n, fam in specs.items()},
              "units": {n: len(v) for n, v in sketches.items()}, "window_units": {n: len(v) for n, v in windows.items()},
              "files_sha256": files, "tasks": len(tasks),
              "sources": {"generator_run": generator_run, "posterior_run": posterior_run, "ea_run": ea_run}}
    finalize(out, analysis=f"m10-simulation-bank-{label}", dataset_ids=[], config={"seeds_base": base},
             result=result, root=root, seeds={"holdout_seeds_base": base})
    return result


def load_bank(run: Path) -> dict:
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    for name, digest in result["files_sha256"].items():
        if pr.file_sha256(run / name) != digest:
            raise ValueError(f"simulation bank file changed: {name}")
    with (run / "sketches.pkl").open("rb") as handle:
        sketches = pickle.load(handle)
    windows: dict[str, list[np.ndarray]] = {}
    with np.load(run / "windows.npz") as stored:
        for key in sorted(stored.files, key=lambda k: (k.split("__")[0], int(k.split("__")[1]))):
            windows.setdefault(key.split("__")[0], []).append(stored[key])
    return {"result": result, "sketches": sketches, "windows": windows}


def design_document(*, bank_run: str, selection_run: str, generator_run: str, root: Path) -> dict:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    families = json.loads((root / pr.CONFIG_DIR / "statistical-families.json").read_text(encoding="utf-8"))["families"]
    selection = json.loads((root / selection_run / "result.json").read_text(encoding="utf-8"))
    g0_selection = json.loads((root / generator_run / "result.json").read_text(encoding="utf-8"))[
        "scores"]["G0_point"]["selection"]["objective"]
    capabilities = {d: capability.check(protocol, d, "realism") for d in FRESH}
    return {"schema": "cleolob-v07-holdout-design-1", "bank_result_sha256": pr.file_sha256(root / bank_run / "result.json"),
            "bank_run": bank_run, "G_star": selection["G_star"], "selection_run": selection_run,
            "H11_delta": 0.10 * g0_selection, "H11_delta_rule": "0.10 x G0 point selection-day objective (M5)",
            "alphas": {f: families[f]["adjusted_alpha"] for f in ("F1_posterior", "F2_distinguishability",
                                                                  "F3_support", "F4_cross_market",
                                                                  "F10_execution_realism", "F11_noninferiority")},
            "samples": SAMPLES, "seeds": protocol["seeds"]["bootstrap"], "datasets": FRESH,
            "es_components": list(contrast.ES_COMPONENTS), "capabilities": capabilities,
            "reader_limits": READER_LIMITS,
            "lot_scale": "ETH development scale (v0.6 design) applied unchanged to every dataset",
            "rules": {"H1": "G0_post - G0_point generic objective; ESTABLISHED per day if CI upper < 0 (alpha F1)",
                      "H2": "AUC(family) - AUC(G0_point), paired real test windows; ESTABLISHED if CI upper < 0 "
                            "(alpha F2); both AUC lower bounds >= 0.99 -> NOT_ESTABLISHED + VACUOUS",
                      "H3": "coverage(family) - coverage(G0_point); ESTABLISHED if CI lower > 0 and estimate >= 0.05",
                      "H4_H5": "G_star - G0_point generic objective on BTC / BitMEX (alpha F4)",
                      "H10": "EA - G0_point ES objective per fresh ETH day (alpha F10)",
                      "H11": "EA - G0_point generic objective one-sided upper < delta (alpha F11)",
                      "quality": "a dataset whose quality status is FAIL makes its results INVALID"}}


def seal(*, bank_run: str, selection_run: str, generator_run: str, root: Path = PROJECT_ROOT) -> dict:
    report = verify_run(root / bank_run, root=root)
    if not report["valid"]:
        raise ValueError(f"bank run is not valid: {report['issues']}")
    path = root / DESIGN_PATH
    if path.exists():
        raise FileExistsError("holdout design already written; it is immutable")
    document = design_document(bank_run=bank_run, selection_run=selection_run, generator_run=generator_run, root=root)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    pr.append_event(root / pr.LEDGER_PATH, "seal_design", design=DESIGN, root=root,
                    protocol=pr.load_protocol(root / pr.PROTOCOL_PATH),
                    reason="holdout realism design sealed before any fresh access (H1-H5, H10, H11)",
                    payload={"design_sha256": pr.document_sha256(document), "reads": sorted(FRESH), "posthoc": False,
                             "design_path": DESIGN_PATH})
    return document


def evaluate(dataset: str, out: str | Path, *, root: Path = PROJECT_ROOT, downloader=None) -> dict:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    document = json.loads((root / DESIGN_PATH).read_text(encoding="utf-8"))
    state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), protocol)
    if state.designs.get(DESIGN, {}).get("design_sha256") != pr.document_sha256(document):
        raise ValueError("holdout design is not sealed in the ledger")
    if pr.file_sha256(root / document["bank_run"] / "result.json") != document["bank_result_sha256"]:
        raise ValueError("simulation bank differs from the sealed design")
    banked = load_bank(root / document["bank_run"])
    frozen = v06_inputs(root)["frozen"]
    out = new_run(root / out)
    run_name = out.relative_to(root).as_posix()
    result: dict = {"dataset": dataset, "label": FRESH[dataset], "design_sha256": pr.document_sha256(document)}
    try:
        tape, meta = access.load_tape(dataset, use="evaluate", design=DESIGN, purpose="registered holdout realism",
                                      root=root, scale=frozen["scale_native_per_lot"],
                                      **({"downloader": downloader} if downloader else {}))
    except access.DataNotAvailable as exc:
        result["status"] = "NOT_AVAILABLE"
        result["reason"] = str(exc)
        finalize(out, analysis=f"m17-holdout-{dataset}", dataset_ids=[dataset], config=document, result=result,
                 root=root)
        return result
    result["quality"] = meta["quality"]
    if meta["quality"]["status"] == "FAIL":
        result["status"] = "INVALID"
        result["reason"] = f"data quality FAIL: {meta['quality']['fails']}"
    raws, blocks = block_raws(tape)
    hist = [sketch(r, frozen["design"]) for r in raws]
    result["blocks"] = {"used": sum(b["used"] for b in blocks), "total": len(blocks)}
    seeds = document["seeds"]
    alphas = document["alphas"]
    sims = banked["sketches"]
    g_star = document["G_star"]
    realism_models = {"G0_point": sims["G0_point"], "G0_post": sims["G0_post"], "EA": sims["EA"]}
    for name in RICHER:
        if name in sims:
            realism_models[name] = sims[name]
    star_key = "G0_post" if g_star == "G0_post" else g_star
    is_eth = FRESH[dataset] == "FRESH_TEMPORAL"
    contrasts = (("G0_post", "G0_point"), ("EA", "G0_point")) if is_eth else ((star_key, "G0_point"),)
    alpha = alphas["F1_posterior"] if is_eth else alphas["F4_cross_market"]
    result["contrasts"] = contrast.bootstrap(hist, realism_models, frozen["design"], frozen["objective_scales"],
                                             contrast.component_scales(frozen), samples=document["samples"],
                                             seed=seeds["realism"], alpha=alpha, contrasts=contrasts)
    hyp: dict = {}
    if is_eth:
        c1 = result["contrasts"]["contrasts"]["G0_post-G0_point"]["objective"]
        hyp["H1"] = {"estimate": c1["difference"], "ci": [c1["ci_low"], c1["ci_high"]], "alpha": alpha,
                     "status": contrast.improvement(c1)}
        # H10/H11 use their own family alphas: re-derive intervals at those alphas from a dedicated bootstrap.
        ea = contrast.bootstrap(hist, {"EA": sims["EA"], "G0_point": sims["G0_point"]}, frozen["design"],
                                frozen["objective_scales"], contrast.component_scales(frozen),
                                samples=document["samples"], seed=seeds["realism"] + 1, alpha=alphas["F10_execution_realism"],
                                contrasts=(("EA", "G0_point"),))["contrasts"]["EA-G0_point"]
        hyp["H10"] = {"estimate": ea["es_objective"]["difference"], "ci": [ea["es_objective"]["ci_low"],
                                                                           ea["es_objective"]["ci_high"]],
                      "alpha": alphas["F10_execution_realism"], "status": contrast.improvement(ea["es_objective"])}
        hyp["H11"] = {"estimate": ea["objective"]["difference"], "one_sided_upper": ea["objective"]["one_sided_upper"],
                      "delta": document["H11_delta"], "alpha": alphas["F11_noninferiority"],
                      "status": contrast.noninferiority(ea["objective"], document["H11_delta"])}
    else:
        c4 = result["contrasts"]["contrasts"][f"{star_key}-G0_point"]["objective"]
        key = "H4" if FRESH[dataset] == "FRESH_CROSS_INSTRUMENT" else "H5"
        hyp[key] = {"G_star": g_star, "estimate": c4["difference"], "ci": [c4["ci_low"], c4["ci_high"]], "alpha": alpha,
                    "status": contrast.improvement(c4)}
    real_windows = window_features(tape)
    result["windows"] = int(len(real_windows))
    gap, cover, describe = {}, {}, {}
    for name in RICHER:
        if name not in banked["windows"]:
            gap[name] = {"status": "NOT_AVAILABLE"}
            continue
        if is_eth:
            g = metrics.auc_contrast(real_windows, banked["windows"][name], banked["windows"]["G0_point"],
                                     alpha=alphas["F2_distinguishability"], samples=document["samples"],
                                     seed=seeds["domain_gap"])
            status, qualifiers = metrics.h2_status(g)
            scores = g.pop("scores", None)
            if scores:
                describe[f"{name}_curves"] = {
                    "roc": metrics.roc_curve(np.asarray(scores["real_family"]), np.asarray(scores["sim_family"])),
                    "pr": metrics.pr_curve(np.asarray(scores["real_family"]), np.asarray(scores["sim_family"])),
                    "calibration": metrics.calibration_curve(np.asarray(scores["real_family"]),
                                                             np.asarray(scores["sim_family"]))}
            gap[name] = {**g, "status_H2": status, "qualifiers": qualifiers}
            c = metrics.coverage_contrast(real_windows, banked["windows"][name], banked["windows"]["G0_point"],
                                          alpha=alphas["F3_support"], samples=document["samples"], seed=seeds["support"])
            status, qualifiers = metrics.h3_status(c)
            cover[name] = {**c, "status_H3": status, "qualifiers": qualifiers}
        sims_all = np.vstack(banked["windows"][name])
        describe[name] = {"mmd": metrics.mmd_rbf(real_windows, sims_all, seed=seeds["domain_gap"]),
                          "precision_recall": metrics.precision_recall(real_windows, sims_all, seed=seeds["domain_gap"])}
    for name in ("G0_point", "G0_post", "EA"):
        sims_all = np.vstack(banked["windows"][name])
        describe[name] = {"mmd": metrics.mmd_rbf(real_windows, sims_all, seed=seeds["domain_gap"]),
                          "precision_recall": metrics.precision_recall(real_windows, sims_all, seed=seeds["domain_gap"])}
    result.update(domain_gap=gap, support=cover, descriptive=describe, hypotheses=hyp)
    if result.get("status") == "INVALID":
        for h in hyp.values():
            h["status"] = "INVALID"
    access.mark_evaluated(dataset, use="evaluate", design=DESIGN, run=run_name, root=root)
    finalize(out, analysis=f"m17-holdout-{dataset}", dataset_ids=[dataset], config=document,
             result=metrics_clean(result), root=root, seeds=seeds)
    access.mark_inspected(dataset, run=run_name, root=root)
    return result


def metrics_clean(value):
    from ...v06.identifiability import clean
    return clean(value)

