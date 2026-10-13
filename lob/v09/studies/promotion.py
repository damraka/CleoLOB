"""hypothesis-promotion-v09 run (M2): collect the exploratory pilot records and apply the registered promotion gate.

Inputs (``--input KEY=RUN``, every one a sealed explore-namespace pilot run): ``bounds``, ``survival``, ``impact``,
``tpp``, ``sequence``, ``latent``, ``hybrid``, ``inference``, ``meta``, ``continual``, ``conformal``,
``representation``, ``decisions``, ``fills``, ``strategic``, ``active``, ``abstention``, ``dro``, ``mpc``,
``rare_event``, ``rl``. A missing input leaves its candidates without a pilot (NOT_PROMOTED_OTHER_REGISTERED_REASON).

Execution conclusions (protocol ``conclusions``) pass the same gate (their own threshold, the decisions family alpha
divided by the number of conclusions, at most the 40 BTCUSDT confirm days, piloted on the BTCUSDT explore days: the
registered H33 statement selects conclusions on BTCUSDT and re-tests them on ETHUSDT); the promoted ones are
*carried*. Transfer candidates follow
the registered rule: the pilot record of H32 / H33 / H34 / H36 is that of the carried conclusion with the largest
planning effect relative to its own threshold, on the explore streams most like the target (BTCUSDT / ETHUSDT /
Binance L1 / all), with the effect expressed as (conclusion effect - conclusion threshold); no carried conclusion
(L1-identifiable for H34) -> no pilot -> not promoted. H9 = conclusion C5 and H23 = conclusion C1 pilots.
"""
from __future__ import annotations

import json
import math

from .. import promotion as P, protocol as pr
from . import common

ANALYSIS = "hypothesis-promotion-v09"
SOURCES = {"hybrid": ("H1", "H12"), "tpp": ("H2", "H5"), "sequence": ("H3",), "latent": ("H4",), "fills": ("H6",),
           "survival": ("H7",), "decisions": ("H8", "H18", "H19", "H22", "H25", "H29", "H30", "H31"),
           "impact": ("H10",), "strategic": ("H11",), "active": ("H13",), "inference": ("H14",), "meta": ("H15",),
           "continual": ("H16",), "conformal": ("H17",), "abstention": ("H20",), "dro": ("H21",), "mpc": ("H24",),
           "rl": ("H26", "H27"), "rare_event": ("H28",), "representation": ("H35",)}
TRANSFER = {"H32": "pilot_btc", "H33": "pilot_eth", "H34": "pilot_l1", "H36": "pilot"}


def collect(args) -> tuple[dict, dict]:
    pilots, provenance = {}, {}
    runs = common.inputs(args)
    for key, hyps in SOURCES.items():
        if key not in runs:
            continue
        res = common.read_input(args, key)
        for h in hyps:
            rec = (res.get("pilot") or {}).get(h)
            if rec is not None:
                pilots[h] = rec
                provenance[h] = runs[key]
    return pilots, provenance


def conclusions(protocol: dict, bounds: dict, rules: dict, families: dict) -> dict:
    fam = families["families"]["decisions"]
    out = {}
    n = len(protocol["conclusions"]["set"])
    for cid, c in protocol["conclusions"]["set"].items():
        # registered H33 statement: conclusions are selected on BTCUSDT and re-tested on ETHUSDT, so the primary
        # (carried) test of every conclusion uses BTCUSDT confirm days only and is piloted on BTCUSDT explore days
        rec = bounds["conclusions"][cid].get("pilot_btc") or bounds["conclusions"][cid]["pilot"]
        cand = {"test": "superiority", "threshold": float(c["threshold_bps"]), "max_units": 40,
                "required_capabilities": ["incremental_l2", "trades_with_side"], "family": "decisions"}
        out[cid] = P.decide(cid, cand, rec, rules=rules, family=fam, family_size=n)
    return out


def transfer_pilots(protocol: dict, bounds: dict, carried: list[str], rules: dict) -> dict:
    out = {}
    for h, key in TRANSFER.items():
        best = None
        for cid in carried:
            c = protocol["conclusions"]["set"][cid]
            if h == "H34" and c["identifiability"] != "L1":
                continue
            rec = bounds["conclusions"][cid].get(key)
            if not rec or rec.get("effect") is None or rec.get("sd") is None or rec["units"] < 2:
                continue
            se = rec["sd"] / math.sqrt(rec["units"])
            planning = rec["effect"] - c["threshold_bps"] - rules["shrinkage_se"] * se
            if best is None or planning > best[0]:
                best = (planning, cid, rec, c)
        if best is not None:
            _, cid, rec, c = best
            out[h] = {**rec, "effect": rec["effect"] - c["threshold_bps"],
                      "unit_values": [v - c["threshold_bps"] for v in rec.get("unit_values", [])],
                      "conclusion": cid, "capabilities": {**rec.get("capabilities", {}), "l1_quotes": True,
                                                          "incremental_l2": True, "trades_with_side": True}}
    return out


def run(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    candidates = json.loads((root / pr.CANDIDATES_PATH).read_text(encoding="utf-8"))
    families = json.loads((root / pr.CONFIG_DIR / "statistical-families.json").read_text(encoding="utf-8"))
    rules = {k: protocol["promotion_rules"][k] for k in ("min_reliability", "min_pilot_units", "min_power",
                                                         "shrinkage_se", "min_confirm_units", "compute_hours_ceiling")}
    _, state = pr.state_of(root)
    if state.promotion is not None:
        raise ValueError("the confirmatory registry is already promoted; promotion runs exactly once")
    pilots, provenance = collect(args)
    bounds = common.read_input(args, "bounds") if "bounds" in common.inputs(args) else None
    concl = conclusions(protocol, bounds, rules, families) if bounds else {}
    carried = [c for c, d in concl.items() if d["outcome"] == "PROMOTED"]
    if bounds:
        pilots["H9"] = bounds["conclusions"]["C5"]["pilot"]
        pilots["H23"] = bounds["conclusions"]["C1"]["pilot"]
        provenance["H9"] = provenance["H23"] = common.inputs(args)["bounds"]
        for h, rec in transfer_pilots(protocol, bounds, carried, rules).items():
            pilots[h] = rec
            provenance[h] = common.inputs(args)["bounds"]
    out = common.start(args)
    decision = P.promote(candidates, pilots, rules=rules, families=families)
    units = {h: d["required_units"] for h, d in decision["decisions"].items() if d["outcome"] == "PROMOTED"}
    registry = P.confirmatory_registry(candidates, decision, extra={
        "carried_conclusions": {c: {**protocol["conclusions"]["set"][c], "confirm_units": concl[c]["required_units"]}
                                for c in carried},
        "conclusion_decisions": {c: {k: concl[c][k] for k in ("outcome", "reason") if k in concl[c]} for c in concl},
        "confirm_units": units,
        "rule": "each promoted hypothesis reads the first `confirm_units` registered slots (primaries in order, failed "
                "slots replaced by ordered backups) of its pool; designs are sealed before any confirmatory access"})
    result = {"promotion": decision, "confirmatory_registry": registry, "pilot_provenance": provenance,
              "conclusions": concl, "carried": carried, "rules": rules,
              "summary": {"promoted": decision["promoted"], "not_promoted": decision["not_promoted"]}}
    return common.finish(args, out, analysis=ANALYSIS, config={"inputs": common.inputs(args), "rules": rules},
                         result=result)
