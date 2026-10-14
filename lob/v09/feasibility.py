"""Pre-promotion feasibility report (M2): access paths, effective sample size, power and compute.

Sealed as ``promotion-feasibility-v09`` in the explore namespace before ``hypothesis-promotion-v09 run``. It never
reads confirmatory or vault market data:

* access paths - a header-only (HTTP HEAD) probe of every registered source URL of the confirm, confirm_late and
  final pools; per stream the number of reachable primary and backup slots;
* effective sample size - for real-data pilots with instrument-day units, the intraclass correlation of the
  BTCUSDT/ETHUSDT units that share a calendar date (design effect 1 + ICC for clusters of two) and the effective
  number of independent units;
* power - per candidate the planning effect, the minimal detectable effect at the registered unit ceiling (power
  0.8, planning alpha), the effective-sample-size-adjusted power at the ceiling, and the compute hours at the
  ceiling from the pilot cost per unit.

The report is descriptive input to the documentation; the registered gate in ``promotion.decide`` is unchanged.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import promotion as P, protocol as pr
from .data import sources


def probe_pools(root: Path, pools=("confirm", "confirm_late", "final"), probe=sources.probe, threads: int = 8) -> dict:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    rows = [(d, name, src["url"]) for d in protocol["datasets"] if d["pool"] in pools
            for name, src in sorted(d["sources"].items())]
    with ThreadPoolExecutor(threads) as ex:
        results = list(ex.map(lambda r: probe(r[2]), rows))
    per: dict[str, dict] = {}
    by_dataset: dict[str, bool] = {}
    for (d, _, _), res in zip(rows, results):
        ok = res.get("status") == 200
        by_dataset[d["id"]] = by_dataset.get(d["id"], True) and ok
    for d in protocol["datasets"]:
        if d["pool"] not in pools:
            continue
        key = f"{d['pool']}|{d['venue']}:{d['instrument']}"
        row = per.setdefault(key, {"primaries": 0, "primaries_reachable": 0, "backups": 0, "backups_reachable": 0})
        kind = "primaries" if d["slot"] < protocol["pools"][d["pool"]]["count"] else "backups"
        row[kind] += 1
        row[f"{kind}_reachable"] += int(by_dataset.get(d["id"], False))
    failures = [{"dataset": d["id"], "source": name, **res} for (d, name, _), res in zip(rows, results)
                if res.get("status") != 200]
    return {"streams": per, "urls": len(rows), "unreachable": failures,
            "method": "HTTP HEAD only; no market data read"}


def icc_pairs(values: dict[str, float]) -> tuple[float | None, int]:
    """ICC of BTCUSDT/ETHUSDT unit values sharing a date (one-way random effects, clusters of two)."""
    by_date: dict[str, dict[str, float]] = {}
    for unit, v in values.items():
        if v is None or not math.isfinite(v):
            continue
        parts = unit.split("-")
        if len(parts) < 6:
            continue
        by_date.setdefault(unit[-10:], {})[parts[2]] = v
    pairs = np.asarray([[d["btcusdt"], d["ethusdt"]] for d in by_date.values() if {"btcusdt", "ethusdt"} <= set(d)])
    if len(pairs) < 4:
        return None, int(len(pairs))
    grand = pairs.mean()
    msb = 2 * ((pairs.mean(1) - grand) ** 2).sum() / (len(pairs) - 1)
    msw = ((pairs - pairs.mean(1, keepdims=True)) ** 2).sum() / len(pairs)
    icc = (msb - msw) / (msb + msw) if (msb + msw) > 0 else 0.0
    return float(np.clip(icc, -1.0, 1.0)), int(len(pairs))


def power_table(candidates: dict, pilots: dict, families: dict, rules: dict, unit_maps: dict) -> dict:
    sizes: dict[str, int] = {}
    for h in candidates.values():
        sizes[h["family"]] = sizes.get(h["family"], 0) + 1
    out = {}
    for hid, c in sorted(candidates.items(), key=lambda kv: int(kv[0][1:])):
        p = pilots.get(hid)
        row = {"family": c["family"], "test": c["test"], "threshold": c["threshold"], "max_units": c["max_units"]}
        if not p or p.get("effect") is None or p.get("sd") is None or not p.get("units"):
            out[hid] = {**row, "status": "NO_PILOT_ESTIMATE"}
            continue
        alpha = P.planning_alpha(families[c["family"]], sizes[c["family"]])
        level = float(c.get("agreement_level") or 0.0) if c["test"] in ("agreement", "transfer") else 0.0
        se = p["sd"] / math.sqrt(max(int(p["units"]), 1))
        planning = (p["effect"] - level) - rules["shrinkage_se"] * se
        n = int(c["max_units"])
        z_a, z_b = P.zq(1 - alpha), P.zq(0.8)
        mde = (z_a + z_b) * p["sd"] / math.sqrt(n) + (c["threshold"] if c["test"] != "noninferiority" else
                                                      -c["threshold"])
        icc, pairs = icc_pairs(unit_maps.get(hid, {}))
        deff = 1.0 + max(icc, 0.0) if icc is not None else 1.0
        row.update({"pilot_units": int(p["units"]), "effect": p["effect"], "sd": p["sd"], "planning_effect": planning,
                    "planning_alpha": alpha, "mde_at_ceiling": mde,
                    "power_at_ceiling": P.power(c["test"], planning, p["sd"], n, alpha=alpha,
                                                threshold=float(c["threshold"])),
                    "icc_date_pairs": icc, "date_pairs": pairs, "design_effect": deff,
                    "effective_units_at_ceiling": n / deff,
                    "power_at_ceiling_ess": P.power(c["test"], planning, p["sd"], max(2, int(n / deff)), alpha=alpha,
                                                    threshold=float(c["threshold"])),
                    "compute_hours_at_ceiling": float(p.get("compute_hours_per_unit") or 0.0) * n})
        out[hid] = row
    return out


def unit_maps(root: Path, runs: dict[str, str]) -> dict[str, dict[str, float]]:
    """{hypothesis: {unit id: pilot value}} for real-data pilots whose units are keyed by dataset id."""
    from .studies import final as F
    from .studies.promotion import SOURCES
    out = {}
    for key, hyps in SOURCES.items():
        if key not in runs:
            continue
        res = json.loads((root / runs[key] / "result.json").read_text(encoding="utf-8"))
        for h in hyps:
            if h in F.SYNTHETIC or h in ("H16",):
                continue
            try:
                out[h] = F.unit_values(h, res, {})
            except (KeyError, TypeError, AttributeError):
                continue
    if "bounds" in runs:
        res = json.loads((root / runs["bounds"] / "result.json").read_text(encoding="utf-8"))
        for h in F.BOUNDS_H:
            out[h] = F.unit_values(h, res, {})
    return out
