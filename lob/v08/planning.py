"""Experiment planning and resource control (M34, brief section 16).

Budgets ``tiny`` < ``small`` < ``medium`` < ``large`` < ``exhaustive`` fix seeds,
worlds and caps. ``estimate`` predicts events, RAM, disk and wall clock from a
measured simulation rate (``measure_rate``) before anything expensive runs.
``seed_plan`` turns a desired interval width or a minimum detectable effect into
the number of seeds, given a pilot standard deviation. ``write_plan`` stores the
planning artifact before execution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

from . import statistics as st

BINS_PER_EPISODE = 1550        # 30 s warm-up + 120 s horizon + 5 s settlement at 100 ms


@dataclass(frozen=True)
class Budget:
    name: str
    seeds: int
    worlds: int
    policies_cap: int
    max_episodes: int
    max_wall_s: float


BUDGETS = {
    "tiny": Budget("tiny", 4, 3, 8, 200, 120.0),
    "small": Budget("small", 16, 6, 16, 3_000, 900.0),
    "medium": Budget("medium", 48, 12, 16, 20_000, 5_400.0),
    "large": Budget("large", 128, 24, 16, 80_000, 21_600.0),
    "exhaustive": Budget("exhaustive", 256, 48, 16, 400_000, 86_400.0),
}


def budget(name: str) -> Budget:
    if name not in BUDGETS:
        raise KeyError(f"unknown budget {name!r}; choose one of {sorted(BUDGETS)}")
    return BUDGETS[name]


def measure_rate(bins: int = 600, seed: int = 0) -> dict:
    """Seconds per simulated bin and events per bin on this machine (a short timing run)."""
    from .generators import PoissonGenerator
    g = PoissonGenerator()
    start = time.perf_counter()
    frame = g.simulate(bins, seed, warmup_bins=0)
    elapsed = time.perf_counter() - start
    return {"seconds_per_bin": elapsed / bins, "events_per_bin": float(frame.counts.sum(1).mean()),
            "label": "local research-workload measurement"}


def estimate(*, episodes: int, workers: int = 1, rate: dict | None = None, bins_per_episode: int = BINS_PER_EPISODE,
             bytes_per_row: int = 900) -> dict:
    rate = rate or measure_rate()
    cpu = episodes * bins_per_episode * rate["seconds_per_bin"]
    return {"episodes": episodes, "events": int(episodes * bins_per_episode * rate["events_per_bin"]),
            "cpu_seconds": cpu, "wall_seconds": cpu / max(1, workers) * 1.15,
            "ram_mb_per_worker": 120 + 0.004 * bins_per_episode, "disk_mb": episodes * bytes_per_row / 1e6,
            "workers": workers, "rate": rate}


def seed_plan(*, sd: float, alpha: float, power: float | None = None, effect: float | None = None,
              ci_width: float | None = None, minimum: int = 8) -> dict:
    """Seeds needed for a target effect (MDE-driven) and/or a target interval width (CI-driven)."""
    out = {"sd": sd, "alpha": alpha}
    candidates = [minimum]
    if effect is not None:
        n = st.required_n(sd, effect, alpha=alpha, power=power or 0.8)
        out["mde_driven"] = {"effect": effect, "power": power or 0.8, "seeds": n}
        candidates.append(n)
    if ci_width is not None:
        n = st.required_n_for_width(sd, ci_width, alpha=alpha)
        out["ci_driven"] = {"width": ci_width, "seeds": n}
        candidates.append(n)
    out["seeds"] = int(max(candidates))
    out["achieved_mde"] = st.mde(sd, out["seeds"], alpha=alpha, power=power or 0.8)
    return out


def within_budget(plan: dict, b: Budget) -> dict:
    ok = plan["episodes"] <= b.max_episodes and plan["wall_seconds"] <= b.max_wall_s
    return {"budget": asdict(b), "within_budget": ok,
            "reason": None if ok else "planned episodes or wall clock exceed the selected budget"}


def write_plan(out: Path, plan: dict) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "plan.json"
    target.write_text(json.dumps(plan, sort_keys=True, indent=1, default=_default) + "\n", encoding="utf-8",
                      newline="\n")
    return target


def _default(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(type(value))
