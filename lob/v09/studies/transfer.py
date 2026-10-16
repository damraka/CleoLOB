"""transfer-study-v09 (M86; H32-H34): carried execution conclusions re-tested across time, instrument and venue.

Inputs: ``--input bounds=RUN`` (the sealed ``confirm-bounds`` run: Bybit BTCUSDT/ETHUSDT and Binance confirm days)
and, when H32 was promoted, ``--input late=RUN`` (the sealed ``confirm-bounds-late`` run on ``confirm_late``).
Per transfer hypothesis the target units are the first ``confirm_units`` slots of its target stream(s) in the
sealed design order (after the registered backup rule): H32 = BTCUSDT ``confirm_late`` days (time), H33 = ETHUSDT
confirm days (instrument), H34 = Binance L1 confirm days, only for L1-identifiable carried conclusions (venue).
Each carried conclusion is tested on its conservative per-day bound against its own threshold; Holm over the carried
conclusions; the registered transfer rule (``decide.transfer``) gives the hypothesis status. Non-promoted transfer
hypotheses are reported with their promotion outcome and never tested.
"""
from __future__ import annotations

import json

from .. import decide as D, protocol as pr
from ..artifacts import read_run
from . import common

ANALYSIS = "transfer-study-v09"
TARGETS = {"H32": ("late", lambda d: d.startswith("bybit-linear-btcusdt")),
           "H33": ("bounds", lambda d: d.startswith("bybit-linear-ethusdt")),
           "H34": ("bounds", lambda d: d.startswith("binance-um"))}


def ordered(run_dir, predicate) -> list[str]:
    cfg = read_run(run_dir)["config"]
    return [d for d in cfg["datasets"] if predicate(d)]


def take(datasets: list[str], k: int, per_stream: bool) -> list[str]:
    if not per_stream:
        return datasets[:k]
    streams: dict[str, list[str]] = {}
    for d in datasets:
        streams.setdefault("-".join(d.split("-")[:3]), []).append(d)
    n = -(-k // max(len(streams), 1))
    return [d for v in streams.values() for d in v[:n]]


def run(args) -> dict:
    root = common.root_of(args)
    registry = json.loads((root / pr.CONFIRMATORY_PATH).read_text(encoding="utf-8"))
    carried = registry.get("carried_conclusions", {})
    runs = common.inputs(args)
    results = {k: common.read_input(args, k) for k in runs}
    seeds = common.seeds(args, ANALYSIS, 64)
    out = common.start(args)
    decisions, used = {}, []
    for i, (h, (key, pred)) in enumerate(TARGETS.items()):
        if h not in registry["hypotheses"]:
            decisions[h] = {"status": "NOT_PROMOTED", "promotion_outcome": registry["not_promoted"].get(h)}
            continue
        if key not in results:
            decisions[h] = {"status": "NOT_AVAILABLE", "reason": f"missing --input {key}"}
            continue
        k = int(registry["confirm_units"][h])
        target = take(ordered(root / runs[key], pred), k, per_stream=(h == "H34"))
        used += target
        per = {}
        for j, (cid, c) in enumerate(sorted(carried.items())):
            if h == "H34" and c["identifiability"] != "L1":
                continue
            units = {d: u for d, u in results[key]["conclusions"][cid]["units"].items() if d in set(target)}
            per[cid] = D.conclusion_stats(units, float(c["threshold_bps"]), seed=seeds[(8 * i + j) % len(seeds)])
        decisions[h] = {**D.transfer(per), "target_units": target, "confirm_units": k}
    result = {"decisions": decisions, "carried": sorted(carried), "inputs": runs,
              "label": "BOUNDED historical transfer (F2); conservative per-day bounds; HYPOTHETICAL fills"}
    return common.finish(args, out, analysis=ANALYSIS, config={"inputs": runs, "targets": sorted(TARGETS)},
                         result=result, datasets=sorted(set(used)))
