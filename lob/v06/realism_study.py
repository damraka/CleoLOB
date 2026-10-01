"""M1-M4 run: freeze the observable design and the real-versus-real references.

``design``: development day only -> lot scale, tick geometry, bins, development
reference marginals, tail thresholds and quintile edges; development first half
versus second half -> per-family objective scales. The selection day (same lot
scale) -> per-family real-versus-real equivalence margins (development vs
selection). The design, scales and margins are sealed in the v0.6 ledger
(``m1-observables-design``) together with the implementation hashes, before any
holdout access. Block sketches are stored locally for later analyses.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..experiments.registry import PROJECT_ROOT
from . import protocol as pr
from .evidence import finalize, new_run, write_json
from .history import block_raws, load, valid_fraction_summary
from .observables import FAMILIES, NOT_AVAILABLE, build_design, pool, registry, sketch
from .realism import compare
from .tape import depth_scale, tick_bps

ANALYSIS = "m1-observables-design"
# Measurement code bound by the seal; orchestration modules may grow without changing measurements.
IMPLEMENTATION = ("lob/v06/tape.py", "lob/v06/observables.py", "lob/v06/realism.py", "lob/v06/history.py")


def implementation_hashes(root: Path = PROJECT_ROOT, files: tuple[str, ...] = IMPLEMENTATION) -> dict[str, str]:
    return {name: pr.text_sha256(root / name) for name in files}


def save_sketches(path: Path, sketches: list[dict]) -> None:
    keys = sorted(sketches[0])
    np.savez_compressed(path, **{key: np.vstack([s[key] for s in sketches]) for key in keys})


def load_sketches(path: Path) -> list[dict]:
    with np.load(path) as stored:
        matrices = {key: stored[key] for key in stored.files}
    rows = len(next(iter(matrices.values())))
    return [{key: matrix[i] for key, matrix in matrices.items()} for i in range(rows)]


def design(out: str | Path, *, root: Path = PROJECT_ROOT, seal: bool = True) -> dict:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    dev_id = pr.datasets_by_role(protocol, "development")[0]
    sel_id = pr.datasets_by_role(protocol, "selection")[0]
    out = new_run(out)
    native, dev_meta = load(dev_id, use="develop", analysis=ANALYSIS, purpose="observable design (bins, scales)",
                            root=root)
    scale = depth_scale(native)
    dev = native.rescaled(scale)
    geometry = tick_bps(dev)
    dev_raws, dev_blocks = block_raws(dev)
    frozen = build_design(dev_raws, tick_bps=geometry)
    dev_sketches = [sketch(r, frozen) for r in dev_raws]
    half = len(dev_sketches) // 2
    halves = compare(pool(dev_sketches[:half]), pool(dev_sketches[half:]), frozen)
    scales = {f: max(halves["families"][f]["error"] or 0.0, 0.05) for f in FAMILIES}
    sel, sel_meta = load(sel_id, use="select", analysis=ANALYSIS, purpose="real-vs-real equivalence margins",
                         root=root, scale=scale)
    sel_raws, sel_blocks = block_raws(sel)
    sel_sketches = [sketch(r, frozen) for r in sel_raws]
    real_vs_real = compare(pool(dev_sketches), pool(sel_sketches), frozen)
    margins = {f: real_vs_real["families"][f]["error"] for f in FAMILIES}
    save_sketches(out / "development-sketches.npz", dev_sketches)
    save_sketches(out / "selection-sketches.npz", sel_sketches)
    write_json(out, "design.json", frozen)
    write_json(out, "registry.json", registry())
    implementation = implementation_hashes(root)
    result = {
        "scale_native_per_lot": scale, "tick_bps": geometry, "development": dev_meta, "selection": sel_meta,
        "development_blocks": valid_fraction_summary(dev_blocks), "selection_blocks": valid_fraction_summary(sel_blocks),
        "objective_scales": scales, "equivalence_margins": margins,
        "development_halves": {"families": halves["families"], "objective_unscaled": halves["objective"]},
        "real_vs_real": {"families": real_vs_real["families"], "objective_unscaled": real_vs_real["objective"],
                         "objective_scaled": compare(pool(dev_sketches), pool(sel_sketches), frozen, scales)["objective"],
                         "components": {k: {"error": v["error"], "family": v["family"]}
                                        for k, v in real_vs_real["components"].items()}},
        "design_sha256": pr.document_sha256(frozen), "not_available": NOT_AVAILABLE,
        "implementation_sha256": implementation,
        "interpretation": ("Development-only design. Scales are development half-vs-half family errors (floor 0.05); "
                           "margins are development-vs-selection (real-vs-real) family errors. Neither is a claim "
                           "about any simulator."),
    }
    finalize(out, analysis=ANALYSIS, dataset_ids=[dev_id, sel_id],
             config={"block_s": 600, "min_block_valid_fraction": 0.5, "datasets": [dev_id, sel_id]}, result=result,
             root=root)
    if seal:
        pr.append_event(root / pr.LEDGER_PATH, "seal_design", {
            "analysis": ANALYSIS, "design_sha256": sealed_hash(result), "reads": [],
            "run": out.relative_to(root).as_posix() if out.is_relative_to(root) else out.name,
            "implementation_sha256": implementation,
            "note": "Observable design, objective scales and equivalence margins frozen from development/selection only."},
            protocol=protocol)
    return result


def sealed_hash(result: dict) -> str:
    return pr.document_sha256({"design_sha256": result["design_sha256"], "scales": result["objective_scales"],
                               "margins": result["equivalence_margins"], "scale": result["scale_native_per_lot"],
                               "implementation": result["implementation_sha256"]})


def read_design(run: str | Path, *, root: Path = PROJECT_ROOT) -> dict:
    """Load a sealed design run and check it against the ledger seal."""
    run = Path(run)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    frozen = json.loads((run / "design.json").read_text(encoding="utf-8"))
    if pr.document_sha256(frozen) != result["design_sha256"]:
        raise ValueError("observable design differs from its recorded hash")
    state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), pr.load_protocol(root / pr.PROTOCOL_PATH))
    sealed = state.designs.get(ANALYSIS)
    if sealed is None or sealed["design_sha256"] != sealed_hash(result):
        raise ValueError("observable design is not sealed in the v0.6 ledger")
    for name, digest in result["implementation_sha256"].items():
        if pr.text_sha256(root / name) != digest:
            raise ValueError(f"sealed observable implementation changed: {name}")
    return {"design": frozen, **result, "run": run}
