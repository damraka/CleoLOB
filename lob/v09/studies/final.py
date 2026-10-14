"""final-analysis-v09 (M89): registered decisions for H1-H36, the carried conclusions and the reality ladder.

Runs once, after the vault run, as ``--namespace final --design final-vault`` (bootstrap seeds from the final
namespace). It reads only sealed runs:

* the confirmatory runs listed in the ledger's ``close_confirmatory`` payload (design -> run), verified and bound as
  lineage; the transfer run is one of them;
* ``--input vault=RUN`` - the single sealed final-vault bounded-replay run (H36);
* optionally ``--input counterexample=RUN`` (explore-day synthetic counterexample search, ladder rungs only).

Per promoted hypothesis the unit values are read from its study's confirm run, restricted to the first
``confirm_units`` units of the sealed design order (instrument-day units split evenly over the design's streams;
sequence units chronological; synthetic clusters by index) and tested with the registered rule of its test type;
Holm runs over the promoted members of each family. Non-promoted candidates keep their promotion outcome (terminal
NOT_AVAILABLE for data/capability failures, NOT_ESTABLISHED otherwise: they were never tested confirmatorily). The
frozen-screen promotion decisions (amendment 3 sensitivity) are reported beside the amended ones.
"""
from __future__ import annotations

import json
import math

import numpy as np

from .. import confirm as CF, decide as D, protocol as pr
from ..artifacts import read_run
from . import common

ANALYSIS = "final-analysis-v09"
ALPHA = 0.9                                    # conformal nominal coverage (H17 unit = coverage - 0.9)

FIELD = {"H1": "h1", "H12": "h12", "H2": "h2", "H5": "h5", "H3": "effect", "H4": "effect", "H6": "effect",
         "H7": "effect", "H14": "effect", "H15": "effect", "H35": "effect",
         "H8": "h8", "H18": "h18", "H19": "h19", "H22": "h22", "H25": "h25", "H29": "h29", "H30": "h30", "H31": "h31",
         "H26": "h26", "H27": "h27"}
BOUNDS_H = {"H9": "C5", "H23": "C1"}
SYNTHETIC = {"H11", "H13", "H20", "H21", "H24", "H28", "H26", "H27"}


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def unit_values(h: str, result: dict, design: dict) -> dict[str, float | None]:
    """{unit id: oriented value} of hypothesis ``h`` from its study's confirm run."""
    if h in BOUNDS_H:
        return {d: _num(u["lower"]) for d, u in result["conclusions"][BOUNDS_H[h]]["units"].items()}
    if h in ("H26", "H27"):
        return {k: _num(w.get(FIELD[h])) for k, w in result["worlds"].items()}
    if h in SYNTHETIC:
        key = "kendall_distance" if h == "H11" else "effect"
        return {k: _num((c.get(h) or {}).get(key)) for k, c in result["clusters"].items()}
    units = result["units"]
    if h == "H10":
        return {d: _num(u["all"]["effect"]) for d, u in units.items()}
    if h == "H16":
        m = design["method"]
        return {d: _num(u["naive"] - u[m]) if u.get(m) is not None and u.get("naive") is not None else None
                for d, u in units.items()}
    if h == "H17":
        return {d: _num(u["methods"]["aci"]["coverage"] - ALPHA) for d, u in units.items()}
    if h == "H1" or h == "H12":
        return {d: _num(u.get(FIELD[h])) if u.get("status") == "OK" else None for d, u in units.items()}
    return {d: _num(u.get(FIELD[h])) for d, u in units.items()}


def unit_order(h: str, design: dict, values: dict) -> list[str]:
    """Registered unit order of the sealed design."""
    if h in SYNTHETIC:
        return sorted(values)
    if "sequences" in design:
        return [d for s in design["sequences"].values() for d in sorted(s, key=lambda x: x[-10:])[1:]]
    if "plan" in design:
        return [d for v in design["plan"].values() for d in v["test"]]
    if "test_datasets" in design:
        return list(design["test_datasets"])
    if "datasets" in design and "train_datasets" in design:
        train = set(design["train_datasets"])
        return [d for d in design["datasets"] if d not in train]
    return [d for d in design["datasets"] if d.startswith("bybit-linear-btcusdt") or d.startswith("bybit-linear-ethusdt")]


def first_units(order: list[str], k: int, *, synthetic: bool) -> list[str]:
    if synthetic:
        return order[:k]
    streams: dict[str, list[str]] = {}
    for d in order:
        streams.setdefault("-".join(d.split("-")[:3]), []).append(d)
    n = -(-k // max(len(streams), 1))
    return [d for v in streams.values() for d in v[:n]]


def _promotion(root, state) -> dict:
    return read_run(root / state.promotion["promotion_run"])["result"]


def run(args) -> dict:
    root = common.root_of(args)
    protocol, state = pr.state_of(root)
    if state.closed is None:
        raise ValueError("the final analysis runs after close_confirmatory")
    registry = json.loads((root / pr.CONFIRMATORY_PATH).read_text(encoding="utf-8"))
    candidates = json.loads((root / pr.CANDIDATES_PATH).read_text(encoding="utf-8"))["hypotheses"]
    families = json.loads((root / pr.CONFIG_DIR / "statistical-families.json").read_text(encoding="utf-8"))["families"]
    promotion = _promotion(root, state)
    closed = dict(state.closed["runs"])
    given = {k: v for k, v in (args.input or [])}
    args.input = sorted({**{f"confirm:{d}": r for d, r in closed.items()}, **given}.items())
    seeds = common.seeds(args, ANALYSIS, 256)
    results = {d: common.read_input(args, f"confirm:{d}") for d in closed}
    designs, analyses = {}, {}
    for d, run_dir in closed.items():
        analyses[d] = read_run(root / run_dir)["binding"]["analysis"]
        if d in state.designs:
            designs[d], _ = CF.substitute(pr.sealed_design(root, d), protocol, state)
    study_design = {doc.get("study"): name for name, doc in designs.items()
                    if doc.get("study") != "counterfactual-bounds-v09"}
    if "confirm-bounds" in designs:
        study_design["counterfactual-bounds-v09"] = "confirm-bounds"
    out = common.start(args)

    # ---------------------------------------------------------------- promoted hypotheses (non-transfer)
    stats: dict[str, dict] = {}
    for i, (h, reg) in enumerate(sorted(registry["hypotheses"].items(), key=lambda kv: int(kv[0][1:]))):
        if reg["test"] == "transfer":
            continue
        study = "counterfactual-bounds-v09" if h in BOUNDS_H else next(
            (s for s, (hs, _) in CF.STUDIES.items() if h in hs), None)
        dname = study_design.get(study)
        if dname is None:
            stats[h] = {"status": "NOT_EVALUABLE", "reason": f"no sealed confirmatory run of {study}"}
            continue
        values = unit_values(h, results[dname], designs[dname])
        order = [u for u in unit_order(h, designs[dname], values) if u in values]
        chosen = first_units(order, int(reg["confirm_units"]), synthetic=h in SYNTHETIC)
        vals = [values[u] for u in chosen]
        st = D.evaluate(reg["test"], vals, threshold=float(reg["threshold"]),
                        level=float(reg.get("agreement_level") or 0.0), seed=seeds[i])
        stats[h] = {**st, "units_used": chosen, "unit_values": vals, "design": dname, "run": closed[dname]}

    decided: dict[str, dict] = {}
    for fam_name, fam in families.items():
        members = {h: stats[h] for h in fam["members"] if h in stats}
        if members:
            decided.update(D.family(members, alpha=fam["alpha"], correction=fam["correction"]))

    # ---------------------------------------------------------------- carried conclusions (bounded historical rung)
    carried = registry.get("carried_conclusions", {})
    conclusions = {}
    bname = study_design.get("counterfactual-bounds-v09")
    if carried and bname:
        per = {}
        for j, (cid, c) in enumerate(sorted(carried.items())):
            order = [d for d in designs[bname]["datasets"] if d.startswith("bybit-linear-btcusdt")]
            target = order[: int(c["confirm_units"])]
            units = {d: u for d, u in results[bname]["conclusions"][cid]["units"].items() if d in set(target)}
            per[cid] = {**D.conclusion_stats(units, float(c["threshold_bps"]), seed=seeds[100 + j]),
                        "units_used": target}
        conclusions = D.transfer(per)

    # ---------------------------------------------------------------- transfer (H32-H34) and the vault (H36)
    tname = next((n for n, a in analyses.items() if a == "transfer-study-v09"), None)
    transfer = results[tname]["decisions"] if tname else {}
    vault = None
    if "vault" in given:
        vres = common.read_input(args, "vault")
        vdoc = read_run(root / given["vault"])["config"]
        if "H36" in registry["hypotheses"]:
            k = int(registry["confirm_units"]["H36"])
            target = first_units(list(vdoc["datasets"]), k, synthetic=False)
            per = {}
            for j, (cid, c) in enumerate(sorted(carried.items())):
                units = {d: u for d, u in vres["conclusions"][cid]["units"].items() if d in set(target)}
                per[cid] = D.conclusion_stats(units, float(c["threshold_bps"]), seed=seeds[200 + j])
            vault = {**D.transfer(per), "target_units": target, "confirm_units": k}

    # ---------------------------------------------------------------- one table for H1-H36
    table = {}
    for h in sorted(candidates, key=lambda x: int(x[1:])):
        c = candidates[h]
        row = {"title": c["title"], "family": c["family"], "test": c["test"], "threshold": c["threshold"],
               "threshold_units": c["threshold_units"], "fidelity": c["fidelity"],
               "evidence_level_target": c["evidence_level_target"], "headline": c["headline"],
               "promotion": promotion["promotion"]["decisions"][h]["outcome"],
               "promotion_reason": promotion["promotion"]["decisions"][h]["reason"],
               "frozen_screen_promotion": "PROMOTED" if h in promotion["sensitivity_original_gate"]["promoted"]
               else promotion["sensitivity_original_gate"]["not_promoted"].get(h)}
        if h not in registry["hypotheses"]:
            row["status"] = "NOT_AVAILABLE" if row["promotion"] == "NOT_PROMOTED_DATA_CAPABILITY" else "NOT_ESTABLISHED"
            row["note"] = "not promoted; never tested on confirmatory data"
        elif c["test"] == "transfer":
            dec = vault if h == "H36" else transfer.get(h)
            if dec is None:
                row["status"] = "NOT_AVAILABLE"
                row["note"] = "no sealed target-pool run"
            else:
                row.update({"status": dec["status"], "transfer": dec})
        else:
            row.update({k: v for k, v in decided.get(h, stats.get(h, {})).items() if k != "unit_values"})
            row["status"] = decided.get(h, stats.get(h, {})).get("status", "NOT_EVALUABLE")
        row["evidence_level"] = c["evidence_level_target"] if row["status"] == "ESTABLISHED" else None
        table[h] = row

    ladder = _ladder(carried, conclusions, transfer, vault, given, args)
    counts = {}
    for r in table.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    gate = common.read_input(args, "gate") if "gate" in given else None
    result = {"hypotheses": table, "conclusions": conclusions, "ladder": ladder, "counts": counts,
              "promotion": {"promoted": promotion["promotion"]["promoted"],
                            "not_promoted": promotion["promotion"]["not_promoted"],
                            "carried": promotion["carried"], "run": state.promotion["promotion_run"]},
              "data_availability": _availability(protocol, state),
              "realism_gate": {"global": gate.get("global") if gate else "NOT_AVAILABLE",
                               "run": given.get("gate")},
              "reality_gap": _gap(ladder),
              "incidents": [{k: i.get(k) for k in ("index", "kind", "affects_validity", "description")}
                            for i in state.incidents],
              "amendments": [{k: a.get(k) for k in ("index", "number")} for a in state.amendments],
              "established": [h for h, r in table.items() if r["status"] == "ESTABLISHED"],
              "promotion_sensitivity": promotion["sensitivity_original_gate"],
              "statistics": {"bootstrap": "paired percentile, 4000 resamples", "p": "one-sided centred bootstrap",
                             "multiplicity": "Holm within registered families; Holm over carried conclusions",
                             "seeds": "final namespace of the final-vault design"},
              "label": "confirmatory decisions under the frozen protocol; bounded replay results are HYPOTHETICAL; no "
                       "live-trading, profitability, HFT-readiness or exact-fill claim"}
    return common.finish(args, out, analysis=ANALYSIS, config={"closed": closed, "inputs": given},
                         result=result, summary={"counts": counts, "established": result["established"]})


def _availability(protocol: dict, state: pr.LedgerState) -> dict:
    out = {}
    for d in protocol["datasets"]:
        row = out.setdefault(d["pool"], {"declared": 0, "accessed": 0, "not_available": 0})
        row["declared"] += 1
        row["accessed"] += d["id"] in state.first_access
    for a in state.attempts:
        if a.get("outcome") == "NOT_AVAILABLE" and a.get("dataset"):
            pool = pr.dataset_declaration(protocol, a["dataset"])["pool"]
            out[pool]["not_available"] += 1
    return out


def _gap(ladder: dict) -> dict:
    """Synthetic-world vs bounded-historical effect of each carried conclusion (descriptive reality gap)."""
    out = {}
    for cid, rungs in ladder.items():
        syn = (rungs.get("synthetic") or {}).get("mean_nominal_effect_bps")
        hist = (rungs.get("bounded historical") or {}).get("estimate")
        out[cid] = {"synthetic_effect_bps": syn, "bounded_historical_effect_bps": hist,
                    "gap_bps": (syn - hist) if syn is not None and hist is not None else None}
    return out


def _ladder(carried: dict, conclusions: dict, transfer: dict, vault: dict | None, given: dict, args) -> dict:
    cx = common.read_input(args, "counterexample")["summary"] if "counterexample" in given else None
    cx_worlds = common.read_input(args, "counterexample")["worlds"] if "counterexample" in given else {}
    out = {}
    for cid in sorted(carried):
        rungs = {}
        if cx_worlds:
            nominal = [w["conclusions"][cid].get("nominal_effect_bps") for w in cx_worlds.values()
                       if cid in w["conclusions"]]
            nominal = [v for v in nominal if v is not None]
            rungs["synthetic"] = {"status": "SUPPORTS" if nominal and np.mean(nominal) > 0 else "DOES_NOT_SUPPORT",
                                  "mean_nominal_effect_bps": float(np.mean(nominal)) if nominal else None,
                                  "evidence": "E1 (explore-day calibrated ecologies; descriptive)"}
            ff = (cx or {}).get(cid, {}).get("mean_flip_fraction")
            rungs["multi-model synthetic"] = {"status": "SUPPORTS" if ff is not None and ff < 0.5 else
                                              "DOES_NOT_SUPPORT", "flip_fraction": ff, "evidence": "E2 (descriptive)"}
        else:
            rungs["synthetic"] = rungs["multi-model synthetic"] = {"status": "NOT_EVALUATED"}
        rungs["hybrid twin"] = {"status": "NOT_EVALUATED",
                                "note": "no registered hybrid-twin test of individual execution conclusions"}
        c = conclusions.get("conclusions", {}).get(cid, {})
        rungs["bounded historical"] = {"status": c.get("status", "NOT_EVALUABLE"), "estimate": c.get("estimate"),
                                       "threshold": c.get("threshold")}
        for name, h in (("cross-time", "H32"), ("cross-instrument", "H33"), ("cross-venue (L1-identifiable only)", "H34")):
            t = transfer.get(h) or {}
            rungs[name] = {"status": (t.get("conclusions", {}).get(cid) or {}).get("status", t.get("status", "NOT_PROMOTED"))}
        rungs["adversarial stress (reported)"] = {"status": "REPORTED", "flip_fraction":
                                                  (cx or {}).get(cid, {}).get("mean_flip_fraction")}
        rungs["final multi-axis vault"] = {"status": ((vault or {}).get("conclusions", {}).get(cid) or {}).get(
            "status", "NOT_AVAILABLE" if vault is None else "NOT_EVALUABLE")}
        out[cid] = rungs
    return out
