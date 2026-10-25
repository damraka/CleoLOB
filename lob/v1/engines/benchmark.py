"""Performance benchmark suite (M49): throughput, latency and memory of the v1 engines and data paths.

Every measurement is a real timed run on the current machine (nothing is extrapolated), reported with the
environment: matching-engine throughput (v2 vs the reference engine on the same command stream), aggregate-book
kernel throughput (numba, interpreted, torch CPU and, when present, CUDA, with the exact-equality check of each
against numba), columnar frame write/open/slice times and peak Python memory (tracemalloc) of a streaming window
pass. ``run`` returns the report; ``cleo v1 engine benchmark --out ...`` writes it as an evidence run.
"""
from __future__ import annotations

import time
import tracemalloc

import numpy as np

from . import accelerated as A
from .matching import BUY, SELL, MatchingEngine, Order
from .reference import ReferenceEngine


def _commands(n: int, seed: int) -> list[Order]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(1, n + 1):
        side = BUY if rng.random() < 0.5 else SELL
        if rng.random() < 0.15:
            out.append(Order(oid=i, owner=f"o{i % 7}", side=side, qty=int(rng.integers(1, 10)), kind="market"))
        else:
            base = 1000 - 1 if side == BUY else 1000 + 1
            px = base - side * int(rng.integers(0, 20))
            out.append(Order(oid=i, owner=f"o{i % 7}", side=side, qty=int(rng.integers(1, 10)), price=px))
    return out


def matching_throughput(n: int = 20_000, seed: int = 0) -> dict:
    cmds = _commands(n, seed)
    out = {}
    for name, cls, k in (("v2", MatchingEngine, n), ("reference", ReferenceEngine, min(n, 4_000))):
        e = cls()
        t = time.perf_counter()
        for o in cmds[:k]:
            e.submit(o)
        dt = time.perf_counter() - t
        out[name] = {"commands": k, "seconds": dt, "commands_per_s": k / dt}
    out["speedup_v2_over_reference"] = out["v2"]["commands_per_s"] / out["reference"]["commands_per_s"]
    return out


def kernel_throughput(paths: int = 256, events: int = 5_000, seed: int = 0) -> dict:
    p = A.KernelParams(improve_bp=(2000, 4000, 6000))
    inp = A.draw_inputs(paths, events, seed=seed)
    out, ref = {}, None
    if A.HAVE_NUMBA:
        A.simulate(p, {k: v[:1] for k, v in inp.items()}, backend="numba")       # compile outside the timing
        t = time.perf_counter()
        ref = A.simulate(p, inp, backend="numba")
        dt = time.perf_counter() - t
        out["numba"] = {"paths": paths, "events": events, "seconds": dt, "events_per_s": paths * events / dt}
    k = max(1, paths // 32)
    sub = {kk: v[:k] for kk, v in inp.items()}
    t = time.perf_counter()
    py = A.simulate(p, sub, backend="python")
    dt = time.perf_counter() - t
    out["python"] = {"paths": k, "events": events, "seconds": dt, "events_per_s": k * events / dt,
                     "equal_to_numba": (bool(all(np.array_equal(py[x], ref[x][:k]) for x in ("bid", "ask", "depth")))
                                        if ref is not None else None)}
    from . import gpu
    av = gpu.available()
    if av["torch"]:
        for dev in (["cpu", "cuda"] if av["cuda"] else ["cpu"]):
            t = time.perf_counter()
            res = gpu.simulate_torch(p, inp if dev == "cuda" else sub, device=dev)
            dt = time.perf_counter() - t
            npaths = paths if dev == "cuda" else k
            if ref is not None:
                cmp = ref if dev == "cuda" else {x: ref[x][:k] for x in ("bid", "ask", "depth", "ok", "refills")}
            else:
                cmp = py if dev == "cpu" else None
            out[f"torch_{dev}"] = {"paths": npaths, "events": events, "seconds": dt,
                                   "events_per_s": npaths * events / dt,
                                   "equal_to_cpu_kernel": bool(gpu.equal(res, cmp)) if cmp is not None else None,
                                   "device": av.get("device") if dev == "cuda" else "cpu"}
    return out


def columnar_io(tmp_dir, bins: int = 864_000) -> dict:
    from pathlib import Path

    from ..data import columnar
    rng = np.random.default_rng(1)
    frames = {"bid_px": np.cumsum(rng.integers(-1, 2, bins)) + 100_000, "valid": rng.random(bins) > 0.001,
              "depth": rng.integers(0, 100, (bins, 2, 10)).astype(np.float32),
              "counts": rng.integers(0, 5, (bins, 6)).astype(np.int32), "mid": rng.random(bins + 1)}
    frames["ask_px"] = frames["bid_px"] + 1
    path = Path(tmp_dir) / "bench.arrow"
    t = time.perf_counter()
    info = columnar.write_frames(frames, path)
    t_write = time.perf_counter() - t
    tracemalloc.start()
    t = time.perf_counter()
    f = columnar.open_frames(path)
    w = f.window(600_000, 603_300)
    s = float(w["depth"].sum())
    t_open = time.perf_counter() - t
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"bins": bins, "file_bytes": info["bytes"], "write_seconds": t_write, "open_and_slice_seconds": t_open,
            "slice_checksum": s, "peak_python_mb_during_window": peak / 2**20,
            "note": "memory-mapped; the window slice copies nothing until used"}


def run(tmp_dir, *, quick: bool = False) -> dict:
    from ..evidence.runs import environment
    return {"environment": environment(),
            "matching": matching_throughput(4_000 if quick else 20_000),
            "kernel": kernel_throughput(64 if quick else 256, 1_000 if quick else 5_000),
            "columnar": columnar_io(tmp_dir, 50_000 if quick else 864_000),
            "label": "measured on this machine; not a claim about other hardware"}


__all__ = ["matching_throughput", "kernel_throughput", "columnar_io", "run"]
