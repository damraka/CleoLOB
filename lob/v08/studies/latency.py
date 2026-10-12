"""``latency-study-v08``: feed latency, stale-book execution and cancel races (M30).

Every registered policy runs in a subset of held-out worlds under registered latency profiles (simulator
settings, never measurements of a real venue): zero; uniform 5, 50 and 250 ms (feed, send, ack, cancel);
heavy-tailed; spiky; and a 1 s stale feed. Reported per (policy, profile): cost change versus zero latency
(paired), latency-cost proxy, cancel races lost, overfill, completion and within-horizon completion.
"""
from __future__ import annotations

import numpy as np

from ...v07.exchange.latency import LatencyModel
from ..execution.simulator import LatencyProfile
from ..statistics import paired_bootstrap
from . import common as C

DEFAULTS = {"tiny": {"worlds": 2, "seeds": 2, "boot": 200},
            "small": {"worlds": 4, "seeds": 6, "boot": 500},
            "medium": {"worlds": 8, "seeds": 16, "boot": 2000}}
HEAVY = LatencyModel("lognormal", 0.002, 0.010, 1.0)
SPIKY = LatencyModel("spiky", 0.005, 0.005, spike_probability=0.05, spike_seconds=0.5)
PROFILES = {
    "zero": LatencyProfile(),
    "uniform_5ms": LatencyProfile.uniform(0.005),
    "uniform_50ms": LatencyProfile.uniform(0.05),
    "uniform_250ms": LatencyProfile.uniform(0.25),
    "heavy_tail": LatencyProfile(feed=HEAVY, send=HEAVY, ack=HEAVY, cancel=HEAVY),
    "spiky": LatencyProfile(send=SPIKY, ack=SPIKY, cancel=SPIKY),
    "stale_feed_1s": LatencyProfile(feed=LatencyModel("fixed", 1.0, 0.0)),
}


def run(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    wrun = C.input_run(args, "worlds")
    hold = C.load_world_set(wrun, "heldout").worlds[: d["worlds"]]
    registered = C.registered_policies(root)
    pol = [(p, registered[p]) for p in sorted(registered)]
    m = C.mandate(root, completion="none")
    seeds = C.seeds(args, "latency", d["seeds"])
    rows = {name: C.evaluate(args, out, hold, pol, m, seeds, tag=name, latency=prof.to_dict())
            for name, prof in PROFILES.items()}
    base = {(r["world_id"], r["seed"], r["policy"]): r for r in rows["zero"]}
    summary = {}
    for name, rs in rows.items():
        for p in sorted(registered):
            prs = [r for r in rs if r["policy"] == p and r["status"] == "VALID"]
            diffs = [r["economic_cost_bps"] - base[(r["world_id"], r["seed"], p)]["economic_cost_bps"]
                     for r in prs if base.get((r["world_id"], r["seed"], p), {}).get("status") == "VALID"]
            summary.setdefault(p, {})[name] = {
                "economic_cost_mean_bps": float(np.mean([r["economic_cost_bps"] for r in prs])) if prs else None,
                "cost_change_vs_zero": paired_bootstrap(diffs, n_boot=d["boot"], seed=args.seed).to_dict()
                if len(diffs) > 1 and np.std(diffs) > 0 else {"estimate": float(np.mean(diffs)) if diffs else None},
                "latency_cost_proxy_bps": float(np.mean([r["latency_cost_proxy_bps"] for r in prs])) if prs else None,
                "cancel_races_lost": int(sum(r["cancel_races_lost"] for r in prs)),
                "overfill_qty": int(sum(r["overfill_qty"] for r in prs)),
                "actual_completion": float(np.mean([r["actual_completion"] for r in prs])) if prs else None,
                "within_horizon_completion": float(np.mean([r["within_horizon_completion"] for r in prs])) if prs else None,
                "invalid": len([r for r in rs if r["policy"] == p]) - len(prs)}
    result = {"summary": summary, "profiles": {k: v.to_dict() for k, v in PROFILES.items()},
              "labels": {"evidence": "SYNTHETIC", "meaning": "simulator latency settings; no real venue latency or HFT claim",
                         "completion_mode": "none (so latency effects on completion are visible)"}}
    return C.finish(args, out, analysis="latency_study", config={"design": d}, result=C.json_safe(result),
                    inputs=[wrun.relative_to(root).as_posix()])
