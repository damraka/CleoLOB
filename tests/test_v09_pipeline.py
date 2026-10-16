"""End-to-end post-promotion path in an isolated ledger with generated (fake) sealed runs.

promotion run -> freeze -> seal_all -> confirmatory runs -> transfer study -> close -> vault design -> vault run ->
final analysis. No market data are read; unit values are generated so the expected statuses are known.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lob.v09 import artifacts, confirm as CF, promotion as P, protocol as pr
from lob.v09.studies import final as FA, transfer as TR

from .v09_fixtures import v09_root, v09_template  # noqa: F401

RULES = {"min_reliability": 0.5, "min_pilot_units": 6, "min_power": 0.8, "shrinkage_se": 1.0,
         "min_confirm_units": 8, "compute_hours_ceiling": 40.0, "reliability_rule": "split_half_consistency"}


def _pilot(effect, sd, n=16):
    rng = np.random.default_rng(n)
    vals = list(np.linspace(effect - sd, effect + sd, n))
    noise = rng.normal(0.0, 0.1 * sd, n)
    noise -= noise.mean()                       # consistent halves: mean odd-even difference exactly zero
    return {"data_available": True, "units": n, "effect": effect, "sd": sd, "compute_hours_per_unit": 0.01,
            "capabilities": {c: True for c in ("incremental_l2", "trades_with_side", "queue_clearance_observable",
                                               "neural_backend", "l1_quotes")},
            "odd_values": list(np.asarray(vals) + noise), "even_values": list(np.asarray(vals) - noise),
            "unit_values": vals}


def _seal(root: Path, rel: str, analysis: str, result: dict, config: dict | None = None) -> str:
    out = artifacts.new_run(root / rel)
    artifacts.finalize(out, analysis=analysis, config=config or {}, result=result, root=root, check_source=False)
    return rel


def _bounds_units(days, lower):
    return {d: {"lower": lo, "upper": lo + 1.0, "status": "IDENTIFIED" if lo > 0 else "INDETERMINATE",
                "assumptions": 54} for d, lo in zip(days, lower)}


def test_post_promotion_pipeline_end_to_end(v09_root) -> None:
    root = v09_root
    cands = json.loads((root / pr.CANDIDATES_PATH).read_text(encoding="utf-8"))
    fams = json.loads((root / pr.CONFIG_DIR / "statistical-families.json").read_text(encoding="utf-8"))
    pilots = {"H3": _pilot(0.05, 0.02), "H23": _pilot(5.0, 2.0), "H33": _pilot(1.5, 0.5, 8)}
    decision = P.promote(cands, pilots, rules=RULES, families=fams)
    assert {"H3", "H23", "H33"} <= set(decision["promoted"])
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    units = {h: d["required_units"] for h, d in decision["decisions"].items() if d["outcome"] == "PROMOTED"}
    carried = {"C3": {**protocol["conclusions"]["set"]["C3"], "confirm_units": 10}}
    registry = P.confirmatory_registry(cands, decision, extra={"carried_conclusions": carried, "confirm_units": units})
    prun = _seal(root, "results/v09/promotion/run", "hypothesis-promotion-v09",
                 {"promotion": decision, "confirmatory_registry": registry, "carried": ["C3"],
                  "sensitivity_original_gate": {"promoted": [], "not_promoted": {}, "carried": []}})
    P.freeze(root, root / prun)
    sealed = CF.seal_all(root)
    assert {"confirm-sequence-generator", "confirm-bounds", "confirm-data"} <= set(sealed["designs"])

    # confirmatory runs (generated values): H3 clearly positive; bounds: C3 strong on BTC, ETH and late absent
    runs = {}
    seq = pr.sealed_design(root, "confirm-sequence-generator")
    vals = 0.05 + 0.01 * np.sin(np.arange(len(seq["test_datasets"])))
    runs["confirm-sequence-generator"] = _seal(
        root, "results/v09/confirm/confirm-sequence-generator", "sequence-generator-v09",
        {"phase": "confirm", "units": {d: {"effect": float(v)} for d, v in zip(seq["test_datasets"], vals)}})
    bd = pr.sealed_design(root, "confirm-bounds")
    days = bd["datasets"]
    conclusions = {cid: {"units": _bounds_units(days, [(2.0 if cid in ("C1", "C3") else -0.5) + 0.1 * np.cos(i)
                                                       for i in range(len(days))])}
                   for cid in protocol["conclusions"]["set"]}
    runs["confirm-bounds"] = _seal(root, "results/v09/confirm/confirm-bounds", "counterfactual-bounds-v09",
                                   {"phase": "confirm", "conclusions": conclusions}, {"datasets": days})
    targs = SimpleNamespace(root=str(root), out="results/v09/confirm/confirm-transfer", namespace="confirm",
                            design="confirm-bounds", seed=0, input=[("bounds", runs["confirm-bounds"])])
    TR.run(targs)
    runs["confirm-transfer"] = "results/v09/confirm/confirm-transfer"
    CF.close(root, runs)

    # vault: one design, read once
    vd = CF.final_design(root)
    pr.seal_design(root, "final-vault", vd["document"], reads=vd["reads"], namespaces=vd["namespaces"])
    pr.open_namespace(root, "final", design="final-vault", reason="test")
    vdays = vd["document"]["datasets"]
    vconcl = {cid: {"units": _bounds_units(vdays, [1.0 + 0.05 * np.sin(i) for i in range(len(vdays))])}
              for cid in protocol["conclusions"]["set"]}
    vault = _seal(root, "results/v09/final/vault-bounds", "counterfactual-bounds-v09",
                  {"phase": "confirm", "conclusions": vconcl}, {"datasets": vdays})

    fargs = SimpleNamespace(root=str(root), out="results/v09/final/final-analysis", namespace="final",
                            design="final-vault", seed=0, input=[("vault", vault)])
    out = FA.run(fargs)
    assert out["valid"]
    res = json.loads((root / out["run"] / "result.json").read_text(encoding="utf-8"))
    h = res["hypotheses"]
    assert h["H3"]["status"] == "ESTABLISHED" and h["H3"]["evidence_level"] == "E4"
    assert h["H23"]["status"] == "ESTABLISHED"
    assert h["H33"]["status"] == "ESTABLISHED"            # C3 on ETH confirm days
    assert h["H1"]["status"] in ("NOT_ESTABLISHED", "NOT_AVAILABLE") and h["H1"]["promotion"] != "PROMOTED"
    assert res["conclusions"]["status"] == "ESTABLISHED"
    assert "H36" not in registry["hypotheses"] or h["H36"]["status"] in ("ESTABLISHED", "NOT_ESTABLISHED")
    for key in ("data_availability", "realism_gate", "promotion", "ladder", "reality_gap", "incidents"):
        assert key in res
    assert artifacts.verify_run(root / out["run"], root=root)["valid"]
    assert pr.verify_protocol_files(root)["valid"]
