"""Transient impact kernels and response diagnostics (M11).

Kernels are *model components* used by the endogenous-response generator and
fitted to simulated response curves. They are not established causal impact of
real markets: v0.8 never claims permanent or causal impact from historical data.

* ``exponential``: ``K(t) = a exp(-t / tau)``;
* ``power_law``: ``K(t) = a (1 + t / t0)^(-gamma)``, approximated in recursions by a
  fixed sum of exponentials (``power_law_exponentials``);
* ``multi_timescale``: ``K(t) = a1 exp(-t / tau1) + a2 exp(-t / tau2)``.

``response_curve`` measures the mid response to a single strategy market order
against a no-trade twin (common random numbers, same seeds); ``diagnose`` returns
immediate response, peak, decay, approximate half-life and recovery fraction,
and ``fit_kernels`` fits all three kernels by least squares.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .market.book import BUY
from .market.frame import BIN_S

KERNELS = ("exponential", "power_law", "multi_timescale")


@dataclass(frozen=True)
class Kernel:
    kind: str
    params: dict

    def __post_init__(self) -> None:
        if self.kind not in KERNELS:
            raise ValueError(f"unknown kernel {self.kind!r}")

    def __call__(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        p = self.params
        if self.kind == "exponential":
            return p["a"] * np.exp(-t / p["tau"])
        if self.kind == "power_law":
            return p["a"] * (1.0 + t / p["t0"]) ** (-p["gamma"])
        return p["a1"] * np.exp(-t / p["tau1"]) + p["a2"] * np.exp(-t / p["tau2"])

    def exponentials(self) -> list[tuple[float, float]]:
        """(weight, decay rate per second) pairs whose sum approximates the kernel (for recursive filters)."""
        p = self.params
        if self.kind == "exponential":
            return [(p["a"], 1.0 / p["tau"])]
        if self.kind == "multi_timescale":
            return [(p["a1"], 1.0 / p["tau1"]), (p["a2"], 1.0 / p["tau2"])]
        return power_law_exponentials(p["a"], p["t0"], p["gamma"])

    def half_life(self, horizon: float = 600.0) -> float | None:
        grid = np.linspace(0.0, horizon, 6001)
        values = self(grid)
        below = np.nonzero(values <= 0.5 * values[0])[0]
        return float(grid[below[0]]) if len(below) else None


def power_law_exponentials(a: float, t0: float, gamma: float, n: int = 6) -> list[tuple[float, float]]:
    """Least-squares nonnegative weights on log-spaced decay rates approximating ``a (1 + t/t0)^-gamma``."""
    rates = 1.0 / (t0 * np.logspace(0, 3, n))
    grid = np.linspace(0.0, 200.0 * t0, 800)
    basis = np.exp(-np.outer(grid, rates))
    target = a * (1.0 + grid / t0) ** (-gamma)
    w = np.linalg.lstsq(basis, target, rcond=None)[0]
    for _ in range(50):          # projected refinement keeps weights nonnegative
        w = np.maximum(w, 0.0)
        residual = basis @ w - target
        w -= 0.5 * basis.T @ residual / (np.linalg.norm(basis, 2) ** 2)
    return [(float(max(x, 0.0)), float(r)) for x, r in zip(w, rates)]


def response_curve(world, *, qty: int, seeds: list[int], horizon_s: float = 60.0, warmup_s: float = 30.0,
                   side: int = BUY) -> dict:
    """Mean mid response (ticks) after one strategy market order, versus the no-trade twin with the same seeds."""
    from .market.sim import MarketSim
    warm, n = int(round(warmup_s / BIN_S)), int(round(horizon_s / BIN_S))
    curves = []
    for seed in seeds:
        paths = []
        for trade in (False, True):
            sim = MarketSim(world.generator, world.book, seed)
            for _ in range(warm):
                sim.run_bin()
            start = sim.book.mid()
            schedule = [(sim.t + 1e-6, 0, lambda s=sim: s.book.market(side, qty, s.t, strategy=True))] if trade else None
            own = np.zeros(6, dtype=np.int64)
            if trade:
                own[0 if side == BUY else 1] = 1
            mids = [start]
            sim.run_bin(schedule, own_counts=own if trade else None)
            mids.append(sim.book.mid())
            for _ in range(n - 1):
                sim.run_bin()
                mids.append(sim.book.mid())
            paths.append(np.asarray(mids) - start)
        curves.append(side * (paths[1] - paths[0]))
    curves = np.asarray(curves)
    return {"t": (np.arange(n + 1) * BIN_S).tolist(), "mean": curves.mean(0).tolist(),
            "se": (curves.std(0, ddof=1) / math.sqrt(len(seeds))).tolist() if len(seeds) > 1 else None,
            "qty": qty, "seeds": len(seeds), "unit": "ticks", "label": "SIMULATED (model component)"}


def diagnose(curve: dict) -> dict:
    t, m = np.asarray(curve["t"]), np.asarray(curve["mean"])
    peak_i = int(np.argmax(m))
    peak = float(m[peak_i])
    after = m[peak_i:]
    below = np.nonzero(after <= 0.5 * peak)[0] if peak > 0 else np.asarray([])
    half = float(t[peak_i + below[0]] - t[peak_i]) if len(below) else None
    return {"immediate_ticks": float(m[1]) if len(m) > 1 else None, "peak_ticks": peak,
            "peak_time_s": float(t[peak_i]), "final_ticks": float(m[-1]),
            "recovery_fraction": float(1.0 - m[-1] / peak) if peak > 0 else None,
            "half_life_s": half, "half_life_status": "MEASURED" if half is not None else "NOT_REACHED_IN_HORIZON"}


def fit_kernels(curve: dict) -> dict:
    """Least-squares fits of the three kernels to the post-peak decay; returns parameters and RMSE."""
    t, m = np.asarray(curve["t"]), np.asarray(curve["mean"])
    i = int(np.argmax(m))
    tt, yy = t[i:] - t[i], m[i:]
    out = {}
    if len(tt) < 5 or yy[0] <= 0:
        return {k: {"status": "NOT_EVALUABLE", "reason": "no positive response peak"} for k in KERNELS}

    def best_single(grid, basis_fn):
        best = None
        for g in grid:
            b = basis_fn(g)
            a = float(max(np.dot(b, yy) / max(np.dot(b, b), 1e-12), 0.0))
            rmse = float(np.sqrt(np.mean((a * b - yy) ** 2)))
            if best is None or rmse < best[0]:
                best = (rmse, a, g)
        return best

    rmse, a, tau = best_single(np.geomspace(0.1, 600, 120), lambda g: np.exp(-tt / g))
    out["exponential"] = {"params": {"a": a, "tau": float(tau)}, "rmse": rmse}
    best = None
    for gamma in np.linspace(0.1, 3.0, 30):
        r = best_single(np.geomspace(0.1, 100, 40), lambda g, gamma=gamma: (1 + tt / g) ** (-gamma))
        if best is None or r[0] < best[0]:
            best = (*r, gamma)
    out["power_law"] = {"params": {"a": best[1], "t0": float(best[2]), "gamma": float(best[3])}, "rmse": best[0]}
    best = None
    for tau1 in np.geomspace(0.1, 30, 20):
        for tau2 in np.geomspace(5, 1200, 20):
            if tau2 <= tau1:
                continue
            basis = np.column_stack([np.exp(-tt / tau1), np.exp(-tt / tau2)])
            coef = np.maximum(np.linalg.lstsq(basis, yy, rcond=None)[0], 0.0)
            rmse = float(np.sqrt(np.mean((basis @ coef - yy) ** 2)))
            if best is None or rmse < best[0]:
                best = (rmse, coef, tau1, tau2)
    out["multi_timescale"] = {"params": {"a1": float(best[1][0]), "tau1": float(best[2]), "a2": float(best[1][1]),
                                         "tau2": float(best[3])}, "rmse": best[0]}
    for k, v in out.items():
        v["half_life_s"] = Kernel(k, v["params"]).half_life()
        v["label"] = "model component fitted to simulated responses; not causal real-market impact"
    return out
