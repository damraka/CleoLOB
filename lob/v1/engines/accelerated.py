"""Optional high-performance aggregate-book simulation backend (M41) with exact compatibility (M42).

One registered kernel, ``run_path``, simulates background order flow on an aggregate (level-size) book in integer
arithmetic only. It is executed

* compiled by numba (``backend="numba"``, if numba is installed - an optional dependency, never required), or
* interpreted (``backend="python"``: numba's ``py_func`` when numba is present, the same source otherwise),

and ``lob.v1.engines.gpu`` re-implements it vectorized over paths in torch. All randomness is pre-drawn integers
(``draw_inputs``), and every decision uses integer comparisons and table lookups, so every backend must produce
**identical** integer outputs from the same inputs; the compatibility tests require bitwise equality.

Book semantics (the AggBook rules of v0.8 with the v1 state-dependent improvement, M70):

* per side, sizes on an absolute tick window of width ``W``; the visible window is the ``L`` levels from each touch;
  the touch is never empty: a depleted touch moves outward and the level entering the window is refilled from the
  pre-drawn refill pool; an improving addition moves the touch inward and the deepest levels leave the window;
* marks: 0 market buy, 1 market sell, 2 add bid, 3 add ask, 4 cancel bid, 5 cancel ask;
* sizes: market ``1 + r1 % (2 m - 1)`` (mean m), limit ``1 + r1 % (2 s - 1)``; level of an addition from the integer
  CDF table of the geometric placement law; improvement when ``spread > 1`` and ``r2 % 10000 < improve_bp[class]``
  with depth ``1 + r0 % (spread - 1)``; a cancellation removes ``max(1, q * f // 10000)`` lots at a level drawn from
  the cancel CDF table restricted to non-empty levels;
* a path whose touch reaches the window edge or exhausts its refill pool stops and is flagged (``overflow``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # optional dependency
    import numba
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised on installations without numba
    numba = None
    HAVE_NUMBA = False

U31 = 1 << 31
SCALE = 10_000


@dataclass(frozen=True)
class KernelParams:
    levels: int = 20                       # visible window per side
    width: int = 4096                      # absolute tick window per side
    market_mean: int = 2
    limit_mean: int = 3
    refill_mean: int = 12
    add_level_p: float = 0.45
    cancel_level_p: float = 0.35
    cancel_frac_bp: int = 3500             # mean cancel fraction (basis points)
    improve_bp: tuple = (0, 0, 0)          # spread classes 2, 3, >= 4 (basis points)
    max_walk: int = 200

    def tables(self) -> tuple[np.ndarray, np.ndarray]:
        """Integer CDF tables (out of 2^31) of the geometric add/cancel level laws over the visible window."""
        def cdf(p):
            w = (1 - p) ** np.arange(self.levels) * p
            w = w / w.sum()
            c = np.floor(np.cumsum(w) * U31).astype(np.int64)
            c[-1] = U31
            return c
        return cdf(self.add_level_p), cdf(self.cancel_level_p)

    def improve(self) -> np.ndarray:
        return np.asarray(self.improve_bp, np.int64)


def draw_inputs(paths: int, events: int, *, seed: int, mark_probs=(0.1, 0.1, 0.25, 0.25, 0.15, 0.15),
                refill_pool: int | None = None, refill_mean: int = 12) -> dict:
    """Pre-drawn integer randomness for ``paths`` independent paths (numpy, seed-addressable)."""
    rng = np.random.default_rng(seed)
    p = np.asarray(mark_probs, float)
    marks = rng.choice(len(p), size=(paths, events), p=p / p.sum()).astype(np.int64)
    r = rng.integers(0, U31, size=(paths, events, 3), dtype=np.int64)
    pool = refill_pool or 4 * events + 1000
    refill = 1 + rng.integers(0, 2 * refill_mean - 1, size=(paths, pool), dtype=np.int64)
    return {"marks": marks, "r": r, "refill": refill}


def _lookup(table, u):
    k = 0
    while k < len(table) - 1 and u >= table[k]:
        k += 1
    return k


def run_path(marks, r, refill, init_bid, init_ask, L, W, mm, lm, cfbp, max_walk, add_cdf, cancel_cdf, improve,
             out_bid, out_ask, out_depth):
    """Simulate one path; writes touch prices per event and the visible touch depths; returns (ok, refills used)."""
    bid = np.zeros(W, np.int64)
    ask = np.zeros(W, np.int64)
    bb = W // 2 - 1
    ba = W // 2
    for k in range(L):
        bid[bb - k] = init_bid[k]
        ask[ba + k] = init_ask[k]
    rc = 0
    npool = refill.shape[0]
    n = marks.shape[0]
    for t in range(n):
        m = marks[t]
        r0 = r[t, 0]
        r1 = r[t, 1]
        r2 = r[t, 2]
        if m == 0 or m == 1:
            q = 1 + r1 % (2 * mm - 1)
            walked = 0
            while q > 0 and walked < max_walk:
                if m == 0:
                    take = min(q, ask[ba])
                    ask[ba] -= take
                    q -= take
                    while ask[ba] <= 0:
                        ba += 1
                        if ba + L - 1 >= W or rc >= npool:
                            return False, rc
                        ask[ba + L - 1] = refill[rc]
                        rc += 1
                        walked += 1
                else:
                    take = min(q, bid[bb])
                    bid[bb] -= take
                    q -= take
                    while bid[bb] <= 0:
                        bb -= 1
                        if bb - L + 1 < 0 or rc >= npool:
                            return False, rc
                        bid[bb - L + 1] = refill[rc]
                        rc += 1
                        walked += 1
        elif m == 2 or m == 3:
            size = 1 + r1 % (2 * lm - 1)
            spread = ba - bb
            cls = min(spread, 4) - 2
            if spread > 1 and r2 % SCALE < improve[cls]:
                ticks = 1 + r0 % (spread - 1)
                if m == 2:
                    for j in range(ticks):
                        bid[bb - L + 1 + j] = 0          # deepest levels leave the window
                    bb += ticks
                    bid[bb] = size
                else:
                    for j in range(ticks):
                        ask[ba + L - 1 - j] = 0
                    ba -= ticks
                    ask[ba] = size
            else:
                lev = _lookup(add_cdf, r0)
                if m == 2:
                    bid[bb - lev] += size
                else:
                    ask[ba + lev] += size
        else:
            # cancel: level from the cancel table restricted to non-empty levels (renormalized by rejection-free
            # integer thinning: walk the table from the drawn level outward to the first non-empty level)
            lev = _lookup(cancel_cdf, r0)
            if m == 4:
                j = lev
                while j < L and bid[bb - j] <= 0:
                    j += 1
                if j == L:
                    j = lev
                    while j >= 0 and bid[bb - j] <= 0:
                        j -= 1
                if j >= 0:
                    qv = bid[bb - j]
                    c = max(1, qv * (r1 % (2 * cfbp + 1)) // SCALE)
                    c = min(c, qv)
                    bid[bb - j] -= c
                    while bid[bb] <= 0:
                        bb -= 1
                        if bb - L + 1 < 0 or rc >= npool:
                            return False, rc
                        bid[bb - L + 1] = refill[rc]
                        rc += 1
            else:
                j = lev
                while j < L and ask[ba + j] <= 0:
                    j += 1
                if j == L:
                    j = lev
                    while j >= 0 and ask[ba + j] <= 0:
                        j -= 1
                if j >= 0:
                    qv = ask[ba + j]
                    c = max(1, qv * (r1 % (2 * cfbp + 1)) // SCALE)
                    c = min(c, qv)
                    ask[ba + j] -= c
                    while ask[ba] <= 0:
                        ba += 1
                        if ba + L - 1 >= W or rc >= npool:
                            return False, rc
                        ask[ba + L - 1] = refill[rc]
                        rc += 1
        out_bid[t] = bb
        out_ask[t] = ba
        out_depth[t, 0] = bid[bb]
        out_depth[t, 1] = ask[ba]
    return True, rc


if HAVE_NUMBA:
    _lookup_nb = numba.njit(cache=True)(_lookup)
    _globals = dict(run_path.__globals__)
    _globals["_lookup"] = _lookup_nb
    import types as _types
    _run_path_src = _types.FunctionType(run_path.__code__, _globals, "run_path")
    run_path_nb = numba.njit(cache=False)(_run_path_src)

    @numba.njit(parallel=True, cache=False)
    def _batch_nb(marks, r, refill, init_bid, init_ask, L, W, mm, lm, cfbp, max_walk, add_cdf, cancel_cdf, improve,
                  out_bid, out_ask, out_depth, ok, used):
        for p in numba.prange(marks.shape[0]):
            a, b = run_path_nb(marks[p], r[p], refill[p], init_bid, init_ask, L, W, mm, lm, cfbp, max_walk,
                               add_cdf, cancel_cdf, improve, out_bid[p], out_ask[p], out_depth[p])
            ok[p] = a
            used[p] = b
else:  # pragma: no cover
    run_path_nb = None
    _batch_nb = None


def init_depth(params: KernelParams) -> tuple[np.ndarray, np.ndarray]:
    d = np.full(params.levels, params.refill_mean, np.int64)
    return d.copy(), d.copy()


def simulate(params: KernelParams, inputs: dict, *, backend: str = "numba") -> dict:
    """Run every path; outputs are integer arrays (touch indices relative to the window, touch depths)."""
    marks, r, refill = inputs["marks"], inputs["r"], inputs["refill"]
    P, n = marks.shape
    add_cdf, cancel_cdf = params.tables()
    ib, ia = init_depth(params)
    out_bid = np.full((P, n), -1, np.int64)
    out_ask = np.full((P, n), -1, np.int64)
    out_depth = np.zeros((P, n, 2), np.int64)
    ok = np.zeros(P, np.bool_)
    used = np.zeros(P, np.int64)
    args = (ib, ia, params.levels, params.width, params.market_mean, params.limit_mean, params.cancel_frac_bp,
            params.max_walk, add_cdf, cancel_cdf, params.improve())
    if backend == "numba":
        if not HAVE_NUMBA:
            raise RuntimeError("numba backend requested but numba is not installed (pip install cleolob[accel])")
        _batch_nb(marks, r, refill, *args, out_bid, out_ask, out_depth, ok, used)
    elif backend == "python":
        for p in range(P):
            ok[p], used[p] = run_path(marks[p], r[p], refill[p], *args, out_bid[p], out_ask[p], out_depth[p])
    else:
        raise ValueError(f"unknown backend {backend}")
    return {"bid": out_bid, "ask": out_ask, "depth": out_depth, "ok": ok, "refills": used, "backend": backend}


def summary(out: dict) -> dict:
    ok = out["ok"]
    spread = (out["ask"] - out["bid"])[ok]
    return {"paths": int(len(ok)), "ok": int(ok.sum()), "mean_spread": float(spread.mean()) if spread.size else None,
            "p_spread_gt1": float((spread > 1).mean()) if spread.size else None,
            "mean_touch_depth": float(out["depth"][ok].mean()) if spread.size else None}


__all__ = ["KernelParams", "draw_inputs", "run_path", "simulate", "summary", "HAVE_NUMBA", "init_depth"]
