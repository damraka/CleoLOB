"""``transfer-study-v08``: cross-time, cross-instrument and cross-venue transfer (M31/M32; H15-H17) and portfolio
infrastructure.

* ``retrospective`` (RETROSPECTIVE, EXPLORATORY): bounded replays of the historical-replay policies on the
  consumed selection day (ETH 2020-12-01), the next temporal day (ETH 2021-01-01), the cross-instrument day
  (BTC 2020-11-01) and the cross-venue day (BitMEX XBTUSD 2020-11-01). Per cell: pair verdicts (bounds with shared
  assumptions) and neutral-assumption mean costs; across cells: dominance stability and a Kendall-tau matrix.
* ``fresh``: the registered acquisition rule for the three fresh transfer days (H13, H15-H17). The design reading
  them is sealed first; a failed availability probe records NOT_AVAILABLE attempts and no holdout byte is requested.
* ``portfolio``: synthetic multi-venue routing and basket execution with cross-impact sensitivity, plus hypothetical
  routing on the time-aligned consumed Deribit BTC and BitMEX XBTUSD days of 2020-11-01.
"""
from __future__ import annotations

import numpy as np

from .. import protocol as pr, realdata
from ..execution.simulator import Fees
from ..generators.endogenous import EndogenousGenerator
from ..portfolio import ROUTERS, Venue, basket_episode, multi_venue_episode, replay_routing
from ..transfer import dominance_stability, matrix
from ..worlds import World
from . import common as C
from .counterfactual import bounded_windows, grid_from_protocol, pair_table

DEFAULTS = {"tiny": {"windows": 3, "boot": 200, "seeds": 2, "mandates": {"primary": {}}},
            "small": {"windows": 8, "boot": 500, "seeds": 6, "mandates": {"primary": {}}},
            "medium": {"windows": 24, "boot": 2000, "seeds": 16, "mandates": {"primary": {}, "large": {"qty": 400}}}}
CELLS = {"selection_eth": ("deribit-eth-perp-2020-12-01", "select"),
         "temporal_eth": ("deribit-eth-perp-2021-01-01", "retrospective"),
         "instrument_btc": ("deribit-btc-perp-2020-11-01", "retrospective"),
         "venue_bitmex": ("bitmex-xbtusd-2020-11-01", "retrospective")}
FRESH = ("deribit-eth-perp-2021-02-01", "deribit-btc-perp-2021-02-01", "bitmex-xbtusd-2021-02-01")
AXES = {"temporal": ("selection_eth", "temporal_eth"), "instrument": ("selection_eth", "instrument_btc"),
        "venue": ("instrument_btc", "venue_bitmex")}


def run(args) -> dict:
    return {"retrospective": retrospective, "fresh": fresh, "portfolio": portfolio}[args.phase](args)


def retrospective(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    fam = pr.load_protocol(root / pr.PROTOCOL_PATH)["experiment_families"]["historical_replay"]
    registered = C.registered_policies(root)
    names = fam["policies"]
    policies = [(p, registered[p]) for p in names]
    grid = [vars(a) for a in grid_from_protocol(root)]
    windows = realdata.registered_windows()[: d["windows"]]
    result = {"cells": {}, "mandates": {}}
    frames = {}
    for cell, (ds, use) in CELLS.items():
        frames[cell], meta = realdata.load_frames(ds, use=use, design="transfer-retrospective", root=root,
                                                  purpose="retrospective transfer cell", windows=windows)
        result["cells"][cell] = {"dataset": ds, "meta": meta, "windows": len(frames[cell])}
    for mname, overrides in d["mandates"].items():
        m = C.mandate(root, **overrides)
        cells, costs = {}, {}
        for cell in CELLS:
            rows = bounded_windows(args, out, frames[cell], policies, m, grid, f"{mname}-{cell}", fam["start_bin"])
            pairs = pair_table(rows, names, d["boot"], args.seed)
            cells[cell] = {k: v.get("status") for k, v in pairs.items()}
            key = "neutral|lat=0|hidden=0|impact=none|perm=0"
            costs[cell] = {p: float(np.nanmean([r[p]["rows"][key]["economic_cost_bps"] for r in rows
                                                if r[p].get("status") == "VALID" and key in r[p]["rows"]] or [np.nan]))
                           for p in names}
        axes = {}
        for axis, (a, b) in AXES.items():
            stab = dominance_stability({a: cells[a], b: cells[b]})
            both = [p for p in cells[a] if cells[a][p] in ("A_LOWER", "B_LOWER") and cells[b][p] in ("A_LOWER", "B_LOWER")]
            axes[axis] = {"cells": [a, b], "pairs_determinate_in_both": len(both), "stability": stab["stability"],
                          "stable": stab["stable"], "unstable": stab["unstable"],
                          "kendall_tau_neutral_costs": matrix({a: costs[a], b: costs[b]}, names=names)["kendall_tau"][0][1]}
        result["mandates"][mname] = {"pair_verdicts": cells, "neutral_costs": costs, "axes": axes,
                                     "dominance_all_cells": dominance_stability(cells),
                                     "kendall_matrix": matrix(costs, names=names)}
    for cell, (ds, use) in CELLS.items():
        realdata.mark_evaluated(ds, use=use, design="transfer-retrospective", run=out.relative_to(root).as_posix(),
                                root=root)
    result["labels"] = {"evidence": "RETROSPECTIVE (consumed days); EXPLORATORY; HYPOTHETICAL bounded replays",
                        "note": "consumed days cannot establish transfer; the fresh phase is the confirmatory route"}
    return C.finish(args, out, analysis="transfer_retrospective", config={"design": d}, result=C.json_safe(result),
                    datasets=[ds for ds, _ in CELLS.values()])


def fresh(args) -> dict:
    root = C.root_of(args)
    out = C.start(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), protocol)
    design = "fresh-transfer-h13-h17"
    if design not in state.designs:
        pr.seal_design(root, design, {"hypotheses": ["H13", "H15", "H16", "H17"], "reads": list(FRESH),
                                      "rule": "probe, then bounded replays of the historical-replay policies"},
                       reads=list(FRESH), reason="seal the fresh transfer reader before any access")
    outcomes = {}
    for ds in FRESH:
        try:
            realdata.acquire(ds, use="evaluate", design=design, purpose="fresh transfer (H13, H15-H17)", root=root)
            outcomes[ds] = "ACQUIRED"
        except realdata.DataNotAvailable as exc:
            outcomes[ds] = f"NOT_AVAILABLE: {exc}"
    status = "ACQUIRED" if all(v == "ACQUIRED" for v in outcomes.values()) else "NOT_AVAILABLE"
    result = {"status": status, "acquisition": outcomes, "design": design,
              "retrospective_pointer": "transfer-study-v08 retrospective (EXPLORATORY)",
              "labels": {"meaning": "NOT_AVAILABLE means the registered probe failed and no holdout byte was requested"}}
    if status == "ACQUIRED":
        result["note"] = "data acquired; run the registered bounded replays on these days before final analysis"
    return C.finish(args, out, analysis="transfer_fresh", config={"design": design}, result=result)


def portfolio(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    wrun = C.input_run(args, "worlds")
    nominal = C.load_world_set(wrun, "selection").worlds[0]
    deep = nominal.book.replace(refill_mean=nominal.book.refill_mean * 1.5)
    thin = nominal.book.replace(refill_mean=max(1.0, nominal.book.refill_mean * 0.4))

    def venue(name, book, fee, lat):
        from ..generators.base import MarketGenerator
        gd = nominal.generator.to_dict()
        gd["book"] = book.to_dict()
        g = MarketGenerator.from_dict(gd)
        return Venue(name, World(name, g, book), Fees(fee, 0.0), lat)

    venues = [venue("deep_expensive_fast", deep, 1.5, 0.002), venue("thin_cheap_slow", thin, 0.5, 0.05)]
    seeds = C.seeds(args, "portfolio", d["seeds"])
    routing = {}
    for r in ROUTERS:
        eps = [multi_venue_episode(venues, r, qty=400, seed=s) for s in seeds]
        routing[r] = {"economic_cost_mean_bps": float(np.mean([e["economic_cost_bps"] for e in eps])),
                      "completion": float(np.mean([e["completion"] for e in eps])),
                      "venue_share": np.mean([e["venue_share"] for e in eps], 0).tolist()}
    wa = World("a", nominal.generator, nominal.book)
    wb = World("b", EndogenousGenerator.wrap(nominal.generator), nominal.book)
    basket = {}
    for c in (0.0, 0.5, 1.0):
        for sch in ("independent", "sequenced"):
            eps = [basket_episode(wa, wb, qty_a=200, qty_b=200, cross_impact=c, schedule=sch, seed=s) for s in seeds]
            basket[f"{sch}|cross={c:g}"] = {"cost_mean_bps": float(np.mean([e["cost_bps"] for e in eps])),
                                           "cost_b_mean_bps": float(np.mean([e["cost_b_bps"] for e in eps]))}
    windows = realdata.registered_windows()[: d["windows"]]
    fa, _ = realdata.load_frames("deribit-btc-perp-2020-11-01", use="retrospective", design="portfolio-routing",
                                 root=root, purpose="hypothetical cross-venue routing", windows=windows)
    fb, _ = realdata.load_frames("bitmex-xbtusd-2020-11-01", use="retrospective", design="portfolio-routing",
                                 root=root, purpose="hypothetical cross-venue routing", windows=windows)
    real = {}
    fees = [Fees(1.0, 0.0), Fees(1.5, 0.0)]          # illustrative assumptions, not venue fee schedules
    pairs = [(a, b) for a, b in zip(fa, fb) if abs(a.meta.get("window_start_s", 0) - b.meta.get("window_start_s", 0)) < 1]
    for impact in ("none", "persistent"):
        for r in ROUTERS:
            vals = [replay_routing([a, b], r, fees, qty=40, start_bin=300, impact=impact)["economic_cost_bps"]
                    for a, b in pairs]
            real[f"{r}|{impact}"] = {"economic_cost_mean_bps": float(np.mean(vals)) if vals else None, "windows": len(vals)}
    for ds in ("deribit-btc-perp-2020-11-01", "bitmex-xbtusd-2020-11-01"):
        realdata.mark_evaluated(ds, use="retrospective", design="portfolio-routing", run=out.relative_to(root).as_posix(),
                                root=root)
    result = {"routing_synthetic": routing, "basket_synthetic": basket, "routing_real_hypothetical": real,
              "labels": {"synthetic": "SYNTHETIC research infrastructure; not a production router",
                         "real": "RETROSPECTIVE, HYPOTHETICAL market-order routing on aggregate L2 (no fills claimed)",
                         "fees_real": "illustrative taker fees in bps (assumption), not venue fee schedules"}}
    return C.finish(args, out, analysis="portfolio", config={"design": d}, result=C.json_safe(result),
                    datasets=["deribit-btc-perp-2020-11-01", "bitmex-xbtusd-2020-11-01"],
                    inputs=[wrun.relative_to(root).as_posix()])
