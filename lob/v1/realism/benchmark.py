"""Generator realism and identifiability benchmark (M79).

``benchmark(generators, real_test_frames, reference_metrics)`` simulates every generator over the registered
number of windows (seeded), computes the v1 metrics (``metrics.frame_metrics``), the real-vs-generated classifier AUC
(v0.9 classifier, unchanged), and the gate decisions per capability (``gate.decide``, plus the distribution-only
arm). ``identifiability`` fits a family, simulates from the fit, refits on the simulation and reports the relative
parameter recovery error - a family whose own parameters cannot be recovered from data of its own kind cannot be
calibrated meaningfully to history.
"""
from __future__ import annotations

import numpy as np

from ...v09.realism.benchmark import classifier_auc
from . import gate as GT, metrics as MT


def simulate_windows(generator, n: int, bins: int, seed: int, *, simulate=None) -> list:
    from ..twins import classical as TC
    sim = simulate or TC.simulate
    return [sim(generator, bins, seed + k) for k in range(n)]


def benchmark(generators: dict, real_test_frames: list, reference_metrics: list[dict], *, windows: int = 8,
              bins: int = 3300, seed: int = 0, simulate=None) -> dict:
    real = MT.frame_metrics(real_test_frames)
    se = MT.window_bootstrap_se(real_test_frames, draws=20, seed=seed)
    band = GT.bands(reference_metrics, se)
    out = {"real": real, "band": {k: list(v) for k, v in band.items()}, "generators": {}}
    for name, g in generators.items():
        try:
            fr = simulate_windows(g, windows, bins, seed, simulate=simulate)
            m = MT.frame_metrics(fr)
            auc = classifier_auc(real_test_frames, fr, seed=seed)
            out["generators"][name] = {"metrics": m, "auc": auc, "decisions": GT.decide(m, band, auc=auc),
                                       "marginal_only": GT.decide(m, band, auc=None, marginal_only=True)}
        except Exception as exc:  # noqa: BLE001 - a generator that cannot simulate is rejected, with the reason
            out["generators"][name] = {"error": f"{type(exc).__name__}: {exc}",
                                       "decisions": {c: {"status": "REJECTED", "metrics": {}}
                                                     for c in MT.CAPABILITIES}}
    out["global"] = GT.global_decision({k: v["decisions"] for k, v in out["generators"].items()})
    return out


def identifiability(family_cls, frame, book, *, bins: int = 20_000, seed: int = 0) -> dict:
    """Fit -> simulate -> refit parameter recovery (relative error per scalar parameter)."""
    from ..twins import classical as TC

    def flat(p, pre=""):
        out = {}
        for k, v in p.items():
            if isinstance(v, dict):
                out.update(flat(v, f"{pre}{k}."))
            elif isinstance(v, (list, tuple)):
                for i, x in enumerate(v):
                    if isinstance(x, (int, float)):
                        out[f"{pre}{k}[{i}]"] = float(x)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                out[f"{pre}{k}"] = float(v)
        return out

    g = family_cls(None, book).fit(frame)
    g.book = book
    sim = TC.simulate(g, bins, seed)
    g2 = family_cls(None, book).fit(sim)
    a, b = flat(g.params), flat(g2.params)
    errs = {k: abs(b[k] - a[k]) / max(abs(a[k]), 1e-9) for k in a if k in b}
    return {"family": getattr(g, "name", family_cls.__name__), "parameters": len(errs),
            "median_relative_error": float(np.median(list(errs.values()))) if errs else None,
            "max_relative_error": float(max(errs.values())) if errs else None, "errors": errs}


__all__ = ["benchmark", "identifiability", "simulate_windows"]
