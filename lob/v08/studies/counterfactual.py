"""``counterfactual-bounds-v08``: historical execution bounds (M29, H14, brief section 38).

``synthetic``: held-out plausible worlds; for every (world, seed) the strategy-free twin tape is simulated once and
every registered policy is (a) run for real in the world (ground truth, including its own impact) and (b) bounded
on the twin tape under the registered assumption grid. Reported: per-path coverage, expected-value coverage
(bootstrap intervals of mean(truth - lower) and mean(upper - truth)), bound widths, and pair bounds for the
registered pairs (assumptions shared between the two policies of a pair).

``real``: the same bounds on registered windows of the consumed selection day (RETROSPECTIVE, EXPLORATORY): pair
bounds for every pair of the historical-replay policies; nothing is an actual historical fill.
"""
from __future__ import annotations

import itertools

import numpy as np

from .. import counterfactual as cf, protocol as pr, realdata
from ..execution.simulator import Episode
from ..market.frame import BIN_S
from ..market.sim import MarketSim
from ..runner import Task, run_tasks
from ..statistics import paired_bootstrap
from ..worlds import World
from . import common as C

DEFAULTS = {"tiny": {"worlds": 2, "seeds": 2, "windows": 3, "boot": 200, "mandates": {"primary": {}}},
            "small": {"worlds": 4, "seeds": 6, "windows": 8, "boot": 500, "mandates": {"primary": {}}},
            "medium": {"worlds": 8, "seeds": 12, "windows": 24, "boot": 2000,
                       "mandates": {"primary": {}, "large": {"qty": 400}}}}
FIELDS = ("economic_cost_bps", "optimization_cost_bps", "fill_frac", "completion")
H14_ALPHA = 0.025


def grid_from_protocol(root) -> list[cf.Assumption]:
    g = pr.load_protocol(root / pr.PROTOCOL_PATH)["experiment_families"]["counterfactual_benchmark"]["grid"]
    return [cf.Assumption(q, lat, h, i, k) for q, lat, h, i, k in itertools.product(
        g["queue"], g["latency_s"], g["hidden_fraction"], g["impact"], g["permanent_ticks_per_lot"])]


def _compact(b: dict) -> dict:
    if b.get("status") != "VALID":
        return b
    return {"status": "VALID", "rows": {k: {f: r[f] for f in FIELDS} for k, r in b["rows"].items()},
            **{f: b[f] for f in FIELDS}}


def twin_task(world: dict, policies: list, mandate: dict, seed: int, grid: list) -> dict:
    w = World.from_dict(world)
    m = C.mandate_from(mandate)
    assumptions = [cf.Assumption(**a) for a in grid]
    warm = int(round(m.warmup_s / BIN_S))
    total = warm + int(round((m.horizon_s + m.settlement_s) / BIN_S)) + 40
    sim = MarketSim(w.generator, w.book, seed)
    sim.run(total)
    frame = sim.frame()
    out = {"world_id": w.world_id, "seed": seed, "policies": {}}
    for name, params in policies:
        truth = Episode(w, C.policy(name, params), m, seed).run()
        b = cf.bounds(frame, lambda n=name, p=params: C.policy(n, p), m, assumptions, start_bin=warm, seed=seed)
        out["policies"][name] = {"truth": {"economic_cost_bps": truth["economic_cost_bps"],
                                           "optimization_cost_bps": truth["optimization_cost_bps"],
                                           "fill_frac": truth["fill_frac"],
                                           "completion": float(truth["actual_completion"])},
                                 "bounds": _compact(b)}
    return out


def coverage_summary(tasks: list[dict], policy: str, boot: int, seed: int) -> dict:
    out = {}
    for f in FIELDS:
        rows = []
        for t in tasks:
            p = t["policies"][policy]
            b = p["bounds"]
            if b.get("status") != "VALID" or b[f]["lower"] is None or p["truth"][f] is None:
                continue
            rows.append((float(p["truth"][f]), float(b[f]["lower"]), float(b[f]["upper"])))
        if len(rows) < 2:
            out[f] = {"status": "NOT_EVALUABLE", "n": len(rows)}
            continue
        tr, lo, hi = (np.asarray(x) for x in zip(*rows))
        covered = (lo - 1e-9 <= tr) & (tr <= hi + 1e-9)
        below = paired_bootstrap(tr - lo, n_boot=boot, seed=seed) if np.std(tr - lo) > 0 else None
        above = paired_bootstrap(hi - tr, n_boot=boot, seed=seed + 1) if np.std(hi - tr) > 0 else None
        rejected = bool((below is not None and below.high < 0) or (above is not None and above.high < 0))
        out[f] = {"per_path_coverage": float(covered.mean()), "n": len(rows), "mean_width": float((hi - lo).mean()),
                  "mean_truth": float(tr.mean()), "mean_lower": float(lo.mean()), "mean_upper": float(hi.mean()),
                  "truth_minus_lower": below.to_dict() if below else None,
                  "upper_minus_truth": above.to_dict() if above else None,
                  "expected_value_coverage_rejected": rejected}
    return out


def run(args) -> dict:
    return synthetic(args) if args.phase == "synthetic" else real(args)


def synthetic(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    wrun = C.input_run(args, "worlds")
    hold = C.load_world_set(wrun, "heldout").worlds[: d["worlds"]]
    fam = pr.load_protocol(root / pr.PROTOCOL_PATH)["experiment_families"]["counterfactual_benchmark"]
    registered = C.registered_policies(root)
    policies = [(p, registered[p]) for p in fam["policies"]]
    grid = [vars(a) for a in grid_from_protocol(root)]
    result = {"mandates": {}, "grid_size": len(grid), "pairs": fam["pairs"]}
    for mname, overrides in d["mandates"].items():
        m = C.mandate(root, **overrides)
        seeds = C.seeds(args, f"counterfactual.{mname}", d["seeds"])
        tasks = [Task(f"{w.world_id}|{s:010d}", "lob.v08.studies.counterfactual:twin_task",
                      {"world": w.to_dict(), "policies": policies, "mandate": m.to_dict(), "seed": s, "grid": grid})
                 for w in hold for s in seeds]
        rows, _ = run_tasks(tasks, workers=args.workers, checkpoint=out.parent / (out.name + ".ckpt") / mname,
                            binding={"study": "counterfactual", "mandate": mname, "namespace": args.namespace})
        coverage = {p: coverage_summary(rows, p, d["boot"], args.seed) for p, _ in policies}
        pairs = {}
        for pair in fam["pairs"]:
            a, b = pair.split("|")
            ba = [r["policies"][a]["bounds"] for r in rows]
            bb = [r["policies"][b]["bounds"] for r in rows]
            pairs[pair] = cf.pair_bound(ba, bb, alpha=H14_ALPHA, n_boot=d["boot"], seed=args.seed)
            pairs[pair].pop("assumptions", None)
        result["mandates"][mname] = {"coverage": coverage, "pairs": pairs,
                                     "invalid": sum(r["policies"][p]["bounds"].get("status") != "VALID"
                                                    for r in rows for p, _ in policies),
                                     "episodes": len(rows) * len(policies)}
    result["summary"] = {m: {p: {f: v[f].get("per_path_coverage") for f in FIELDS}
                             for p, v in r["coverage"].items()} for m, r in result["mandates"].items()}
    result["labels"] = {"evidence": "SYNTHETIC validation of the bounding method", "fill_label": "HYPOTHETICAL, BOUNDED"}
    return C.finish(args, out, analysis="counterfactual_synthetic", config={"design": d}, result=C.json_safe(result),
                    inputs=[wrun.relative_to(root).as_posix()])


REAL_DATASET = "deribit-eth-perp-2020-12-01"


def replay_window(frame: dict, meta: dict, policies: list, mandate: dict, grid: list, start_bin: int) -> dict:
    from ..market.frame import EventFrame
    f = EventFrame.from_arrays({k: np.asarray(v) for k, v in frame.items()}, meta)
    m = C.mandate_from(mandate)
    assumptions = [cf.Assumption(**a) for a in grid]
    return {name: _compact(cf.bounds(f, lambda n=name, p=params: C.policy(n, p), m, assumptions, start_bin=start_bin))
            for name, params in policies}


def bounded_windows(args, out, frames, policies, m, grid, tag, start_bin) -> list[dict]:
    tasks = [Task(f"{tag}|{i:03d}", "lob.v08.studies.counterfactual:replay_window",
                  {"frame": {k: v.tolist() for k, v in f.to_arrays().items()}, "meta": f.meta, "policies": policies,
                   "mandate": m.to_dict(), "grid": grid, "start_bin": start_bin})
             for i, f in enumerate(frames)]
    rows, _ = run_tasks(tasks, workers=args.workers, checkpoint=out.parent / (out.name + ".ckpt") / tag,
                        binding={"study": "counterfactual-real", "tag": tag})
    return rows


def pair_table(rows: list[dict], names: list[str], boot: int, seed: int, alpha: float = H14_ALPHA) -> dict:
    out = {}
    for a, b in itertools.combinations(names, 2):
        r = cf.pair_bound([x[a] for x in rows], [x[b] for x in rows], alpha=alpha, n_boot=boot, seed=seed)
        r.pop("assumptions", None)
        out[f"{a}|{b}"] = r
    return out


def real(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    fam = pr.load_protocol(root / pr.PROTOCOL_PATH)["experiment_families"]
    registered = C.registered_policies(root)
    names = fam["historical_replay"]["policies"]
    policies = [(p, registered[p]) for p in names]
    grid = [vars(a) for a in grid_from_protocol(root)]
    frames, meta = realdata.load_frames(REAL_DATASET, use="select", design="counterfactual-real", root=root,
                                        purpose="retrospective bounded replays", windows=realdata.registered_windows()[
                                            : d["windows"]])
    result = {"dataset": REAL_DATASET, "meta": meta, "mandates": {}}
    for mname, overrides in d["mandates"].items():
        m = C.mandate(root, **overrides)
        rows = bounded_windows(args, out, frames, policies, m, grid, mname, fam["historical_replay"]["start_bin"])
        pairs = pair_table(rows, names, d["boot"], args.seed)
        result["mandates"][mname] = {
            "pairs": pairs, "determinate_pairs": sorted(k for k, v in pairs.items() if v.get("status") in
                                                        ("A_LOWER", "B_LOWER")),
            "bounds": {p: {f: {"mean_lower": float(np.mean([r[p][f]["lower"] for r in rows if r[p].get("status") == "VALID"
                                                            and r[p][f]["lower"] is not None] or [np.nan])),
                               "mean_upper": float(np.mean([r[p][f]["upper"] for r in rows if r[p].get("status") == "VALID"
                                                            and r[p][f]["upper"] is not None] or [np.nan]))}
                           for f in FIELDS} for p in names},
            "invalid_windows": sum(r[p].get("status") != "VALID" for r in rows for p in names), "windows": len(rows)}
    realdata.mark_evaluated(REAL_DATASET, use="select", design="counterfactual-real",
                            run=out.relative_to(root).as_posix(), root=root)
    result["labels"] = {"evidence": "RETROSPECTIVE (consumed selection day); EXPLORATORY",
                        "fill_label": "HYPOTHETICAL, BOUNDED, ASSUMPTION_DEPENDENT; never actual historical fills"}
    return C.finish(args, out, analysis="counterfactual_real", config={"design": d}, result=C.json_safe(result),
                    datasets=[REAL_DATASET])
