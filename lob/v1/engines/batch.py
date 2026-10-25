"""Vectorized Monte Carlo episode scheduling (M45).

An experiment is a grid of (scenario, replicate) episodes. ``plan`` assigns every episode a deterministic id and
seed (from the registered seed namespaces), groups episodes of the same scenario into fixed-size batches and
returns the schedule; ``run`` executes batches on the chosen backend (numba kernel, interpreted kernel or torch) and
reduces per-episode summaries. Because inputs are drawn per batch from a seed derived only from the batch's episode
ids, results do not depend on worker count, batch execution order or backend (tested).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from . import accelerated as A


@dataclass(frozen=True)
class Episode:
    id: str
    scenario: str
    replicate: int
    seed: int


@dataclass(frozen=True)
class Batch:
    id: str
    scenario: str
    episodes: tuple[Episode, ...]
    seed: int


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") & ((1 << 63) - 1)


def plan(scenarios: dict[str, A.KernelParams], replicates: int, *, batch_size: int = 256, root: str = "explore") -> list[Batch]:
    out = []
    for name in sorted(scenarios):
        eps = [Episode(f"{name}#{i}", name, i, _seed(root, name, i)) for i in range(replicates)]
        for b in range(0, len(eps), batch_size):
            chunk = tuple(eps[b:b + batch_size])
            out.append(Batch(f"{name}@{b // batch_size}", name, chunk, _seed(root, *[e.id for e in chunk])))
    return out


def inputs_for(batch: Batch, events: int, *, mark_probs, refill_mean: int) -> dict:
    """Per-episode inputs drawn from each episode's own seed, stacked (order-independent)."""
    parts = [A.draw_inputs(1, events, seed=e.seed, mark_probs=mark_probs, refill_mean=refill_mean)
             for e in batch.episodes]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def run(batches: list[Batch], scenarios: dict[str, A.KernelParams], events: int, *, backend: str = "numba",
        mark_probs=(0.1, 0.1, 0.25, 0.25, 0.15, 0.15), device: str | None = None) -> dict:
    results: dict[str, dict] = {}
    for b in batches:
        p = scenarios[b.scenario]
        inp = inputs_for(b, events, mark_probs=mark_probs, refill_mean=p.refill_mean)
        if backend.startswith("torch"):
            from .gpu import simulate_torch
            out = simulate_torch(p, inp, device=device)
        else:
            out = A.simulate(p, inp, backend=backend)
        spread = out["ask"] - out["bid"]
        for k, e in enumerate(b.episodes):
            ok = bool(out["ok"][k])
            s = spread[k][spread[k] > 0]
            results[e.id] = {"scenario": e.scenario, "ok": ok, "mean_spread": float(s.mean()) if len(s) else None,
                             "p_spread_gt1": float((s > 1).mean()) if len(s) else None,
                             "final_mid2": int(out["bid"][k, -1] + out["ask"][k, -1]) if ok else None}
    return dict(sorted(results.items()))


def reduce(results: dict) -> dict:
    by: dict[str, list] = {}
    for r in results.values():
        by.setdefault(r["scenario"], []).append(r)
    out = {}
    for sc, rs in sorted(by.items()):
        ok = [r for r in rs if r["ok"]]
        ms = np.array([r["mean_spread"] for r in ok], float)
        out[sc] = {"episodes": len(rs), "ok": len(ok), "mean_spread": float(ms.mean()) if len(ms) else None,
                   "sd_mean_spread": float(ms.std(ddof=1)) if len(ms) > 1 else None}
    return out


__all__ = ["Episode", "Batch", "plan", "run", "reduce", "inputs_for"]
