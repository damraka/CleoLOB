"""Promotion under the frozen v1 rule (M6, M7): one decision per candidate, recorded once in the ledger.

* Direct replications (``promotion = direct_replication``) are PROMOTED with their registered cluster count; their
  pilots are feasibility reports only.
* Every other candidate is PROMOTED iff (a) its capabilities exist, (b) a pilot with >= ``min_pilot_clusters``
  explore date clusters exists, (c) the planning effect (pilot cluster mean - 1 cluster SE of the registered
  decision quantity) exceeds the threshold, (d) the cluster-t power at that effect reaches ``min_power`` with
  ``min_clusters <= G <= max_confirm_clusters`` (planning alpha = family alpha / family size), and (e) the expected
  compute is within the ceiling. Otherwise the first failing condition is the registered reason.

There is no split-half screen. ``promote`` refuses to run twice (the ledger allows one ``promote`` event).
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import statistics as ST
from ..protocol import document as D
from . import decide as DE

REASONS = ("PROMOTED", "NOT_PROMOTED_DATA_CAPABILITY", "NOT_PROMOTED_NO_PILOT", "NOT_PROMOTED_TOO_FEW_PILOT_CLUSTERS",
           "NOT_PROMOTED_EFFECT_TOO_SMALL", "NOT_PROMOTED_INSUFFICIENT_POWER", "NOT_PROMOTED_COMPUTE")


def _shift(h: dict, rec: dict) -> tuple[float | None, float]:
    """Planning effect of the decision quantity and the threshold it must exceed."""
    pe = rec.get("planning_effect")
    if pe is None:
        return None, 0.0
    if h["test"] == "noninferiority":
        return pe + float(h["margin"]), 0.0
    if h["test"] == "agreement":
        return pe - float(h["level"]), float(h["threshold"])
    if h["test"] == "equivalence":
        return float(h["margin"]) - abs(rec["mean"]) - (rec.get("se") or 0.0), 0.0
    return pe, float(h["threshold"] if h["threshold"] is not None else 0.0)


def decide_one(h: dict, rec: dict | None, *, rules: dict, family_size: int, capability_ok: bool) -> dict:
    if h["promotion"] == "direct_replication":
        return {"outcome": "PROMOTED", "reason": "direct replication (registered, pilot-independent)",
                "confirm_clusters": int(h["confirm_clusters"]), "pilot": rec}
    if not capability_ok:
        return {"outcome": "NOT_PROMOTED_DATA_CAPABILITY", "reason": "required capability absent", "pilot": rec}
    if rec is None:
        return {"outcome": "NOT_PROMOTED_NO_PILOT", "reason": "no exploratory pilot record exists"}
    if rec.get("clusters", 0) < rules["min_pilot_clusters"]:
        return {"outcome": "NOT_PROMOTED_TOO_FEW_PILOT_CLUSTERS",
                "reason": f"{rec.get('clusters', 0)} < {rules['min_pilot_clusters']} pilot clusters", "pilot": rec}
    effect, thr = _shift(h, rec)
    if effect is None or not effect > thr:
        return {"outcome": "NOT_PROMOTED_EFFECT_TOO_SMALL",
                "reason": f"planning effect {effect} does not exceed {thr}", "pilot": rec}
    sd = rec.get("sd_cluster_means") or 0.0
    alpha = 0.05 / max(family_size, 1)
    g = ST.feasible_clusters(effect - thr, sd, alpha=alpha, power=rules["min_power"],
                             max_clusters=rules["max_confirm_clusters"]) if sd > 0 else DE.MIN_CLUSTERS
    if g is None:
        return {"outcome": "NOT_PROMOTED_INSUFFICIENT_POWER",
                "reason": f"power {rules['min_power']} needs more than {rules['max_confirm_clusters']} clusters",
                "pilot": rec}
    g = max(g, DE.MIN_CLUSTERS)
    hours = float(rec.get("compute_hours_per_cluster") or 0.0) * g
    if hours > rules["compute_ceiling_hours"]:
        return {"outcome": "NOT_PROMOTED_COMPUTE", "reason": f"{hours:.1f} h > ceiling", "pilot": rec}
    power = ST.cluster_power(effect - thr, sd, g, alpha=alpha) if sd > 0 else 1.0
    return {"outcome": "PROMOTED", "reason": "all registered promotion conditions pass", "confirm_clusters": int(g),
            "planning_effect": effect, "planning_alpha": alpha, "expected_power": power,
            "expected_compute_hours": hours, "pilot": rec}


def decide_all(root: Path, pilots: dict[str, dict], capabilities: dict[str, bool]) -> dict:
    root = Path(root)
    cands = {h["id"]: h for h in json.loads((root / "configs/v1/hypotheses-candidates.json").read_text())["hypotheses"]}
    fams = json.loads((root / "configs/v1/statistical-families.json").read_text())["families"]
    rules = json.loads((root / "configs/v1/protocol.json").read_text())["promotion_rules"]
    size = {m: len(f["members"]) for f in fams.values() for m in f["members"]}
    out = {}
    for hid, h in cands.items():
        out[hid] = decide_one(h, pilots.get(hid), rules=rules, family_size=size[hid],
                              capability_ok=capabilities.get(hid, True))
    return {"decisions": out, "promoted": sorted((k for k, v in out.items() if v["outcome"] == "PROMOTED"),
                                                 key=lambda k: int(k[1:]))}


def promote(ws: D.Workspace, decision: dict, *, run: str) -> dict:
    return ws.append("promote", reason="the single v1 promotion decision under the frozen rule",
                     payload={"promotion_run": run, "promoted": decision["promoted"],
                              "confirm_clusters": {k: v["confirm_clusters"] for k, v in decision["decisions"].items()
                                                   if v["outcome"] == "PROMOTED"}})


__all__ = ["decide_one", "decide_all", "promote", "REASONS"]
