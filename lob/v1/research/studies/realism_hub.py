"""Realism hub: one fold computes the registered unit values of H13, H16, H19, H23, H24, H25 and H28.

Fold = (held-out unit, training units of the same stream). Training units provide the generators and the gate's
reference band (>= 8 training days, registered); the held-out day provides the real windows (the 24 registered
windows). Generators: the classical families Poisson, Hawkes, queue-reactive and mixture with the corrected v1 book
(and Poisson/Hawkes also with the v0.9 book for H13), the F5 autoregressive neural count generator on the corrected
book, and the neural SDE for paths.

* H13 = log(|spread_v09 - real| / |spread_v1 - real|), mean over Poisson and Hawkes (same counts, same seeds);
* H16 = NLL(constant-rate cancel model) - NLL(state-dependent Poisson GLM), nats per bin, on the held-out day;
* H19 = share of (generator x capability) cells accepted (ACCEPTABLE) by the marginal-only gate but REJECTED by the
  registered gate;
* H23 = log(mean relative dispersion error at 1/10/60 s of the best classical / that of the neural AR generator);
* H24 = log(signature MMD of 5 min paths: best classical / neural SDE);
* H25 = log(max normalized spread+depth error: corrected classical / neural AR);
* H28 = 1 if the generator with the lowest training-day realism distance is ACCEPTABLE on every execution
  capability on the held-out day, else 0 (the decision quantity subtracts the 0.5 level).
"""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np

CLASSICAL = ("poisson", "hawkes", "queue_reactive", "mixture")
WINDOWS = 6
BINS = 3300


def _windows(cache, starts=None):
    from ....v09.data.daycache import frame as cf
    starts = starts or [600 + 3600 * h for h in range(24)]
    return [cf(cache, (s - 30) * 10, (s + 300) * 10) for s in starts]


def _sim_frames(g, n, seed, simulate):
    return [simulate(g, BINS, seed + k) for k in range(n)]


def glm_cancel_nll(train_frames, test_frames) -> float:
    """H16: NLL(constant cancel rate) - NLL(Poisson GLM on log1p touch depth, imbalance, spread) per bin."""
    def design(frames):
        X, y = [], []
        for f in frames:
            v = f.valid
            st = f.state[v]
            X.append(np.column_stack([np.ones(v.sum()), np.log1p(np.maximum(st[:, 3], 0)), st[:, 1], st[:, 0]]))
            y.append((f.counts[v, 4] + f.counts[v, 5]).astype(float))
        return np.vstack(X), np.concatenate(y)
    Xtr, ytr = design(train_frames)
    Xte, yte = design(test_frames)
    w = np.zeros(Xtr.shape[1])
    w[0] = math.log(max(ytr.mean(), 1e-6))
    for _ in range(50):                                  # IRLS for the Poisson GLM
        mu = np.exp(np.clip(Xtr @ w, -20, 20))
        H = (Xtr * mu[:, None]).T @ Xtr + 1e-6 * np.eye(len(w))
        step = np.linalg.solve(H, Xtr.T @ (ytr - mu))
        w += step
        if np.abs(step).max() < 1e-8:
            break
    lam0 = max(ytr.mean(), 1e-9)
    mu = np.exp(np.clip(Xte @ w, -20, 20))
    from math import lgamma
    lgy = np.asarray([lgamma(v + 1) for v in yte])
    nll_glm = float((mu - yte * np.log(np.maximum(mu, 1e-12)) + lgy).mean())
    nll_const = float((lam0 - yte * math.log(lam0) + lgy).mean())
    return nll_const - nll_glm


def signature_features(paths: list[np.ndarray]) -> np.ndarray:
    """Level-2 signature features of time-augmented, normalized log-mid paths (increment, Levy area, QV)."""
    out = []
    for p in paths:
        x = np.log(np.maximum(np.asarray(p, float), 1e-9))
        x = (x - x[0]) * 1e4
        t = np.linspace(0, 1, len(x))
        dx, dt = np.diff(x), np.diff(t)
        area = 0.5 * float(((t[:-1] - t[0]) * dx - (x[:-1] - x[0]) * dt).sum())
        out.append([x[-1], area, float((dx ** 2).sum()), float(np.abs(dx).sum()), float(x.max() - x.min())])
    return np.asarray(out)


def mmd(a: np.ndarray, b: np.ndarray) -> float:
    z = np.vstack([a, b])
    mu, sd = z.mean(0), z.std(0) + 1e-9
    a, b = (a - mu) / sd, (b - mu) / sd
    def k(x, y):
        d = ((x[:, None, :] - y[None]) ** 2).sum(-1)
        return np.exp(-d / (2 * a.shape[1]))
    return float(k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean())


def fold(held: str, held_path: str, train_paths: list[str], seed: int) -> dict:
    from ...data import daycache as DC
    from ...realism import gate as GT, metrics as MT
    from ...runtime import cap_threads
    from ...twins import classical as TC
    cap_threads()
    t0 = time.time()
    train = [DC.load(Path(p)) for p in train_paths]
    test = DC.load(Path(held_path))
    real_w = _windows(test)
    real = MT.frame_metrics(real_w)
    ref = [MT.frame_metrics(_windows(c, [600 + 3600 * h for h in range(0, 24, 3)])) for c in train]
    band = GT.bands(ref, MT.window_bootstrap_se(real_w, draws=10, seed=seed))
    fit1 = TC.fit(train, book="v1", families=list(CLASSICAL))
    fit9 = TC.fit(train, book="v09", families=["poisson", "hawkes"])
    sims, mets, dec, marg = {}, {}, {}, {}
    for name, g in fit1["generators"].items():
        sims[name] = _sim_frames(g, WINDOWS, seed, TC.simulate)
        mets[name] = MT.frame_metrics(sims[name])
    out = {"held": held, "real": real}
    # H13: spread mismatch of the v0.9 vs the corrected book, same families and seeds
    ratios = []
    for name in ("poisson", "hawkes"):
        s9 = MT.frame_metrics(_sim_frames(fit9["generators"][name], WINDOWS, seed, TC.simulate))["mean_spread"]
        s1 = mets[name]["mean_spread"]
        ratios.append(math.log(max(abs(s9 - real["mean_spread"]), 1e-6) / max(abs(s1 - real["mean_spread"]), 1e-6)))
    out["H13"] = float(np.mean(ratios))
    # H16: state-dependent cancellation
    out["H16"] = glm_cancel_nll([w for c in train[:6] for w in _windows(c, [600 + 3600 * h for h in range(0, 24, 4)])],
                                real_w)
    # neural F5 generator (autoregressive counts on the corrected book)
    try:
        from ...generators.neural.counts import AutoregressiveCounts
        tr_frames = [w for c in train[:6] for w in _windows(c, [600 + 3600 * h for h in range(0, 24, 4)])]
        nn = AutoregressiveCounts(steps=600, history=32, seed=seed).fit(tr_frames)
        nn.book = fit1["book"]
        sims["neural_ar"] = _sim_frames(nn, WINDOWS, seed, TC.simulate)
        mets["neural_ar"] = MT.frame_metrics(sims["neural_ar"])
    except ImportError:
        pass
    for name, m in mets.items():
        dec[name] = GT.decide(m, band, auc=None)
        marg[name] = GT.decide(m, band, auc=None, marginal_only=True)
    cells = [(n, c) for n in dec for c in dec[n] if marg[n][c]["status"] != "NOT_ASSESSED"]
    out["H19"] = float(np.mean([marg[n][c]["status"] == "ACCEPTABLE" and dec[n][c]["status"] == "REJECTED"
                                for n, c in cells])) if cells else None

    def disp_err(m):
        return float(np.mean([abs(m[k] - real[k]) / max(abs(real[k]), 1e-9)
                              for k in ("disp_market_1s", "disp_market_10s", "disp_market_60s")]))

    def sd_err(m):
        keys = ("mean_spread", "p_spread_gt1", "median_log_touch", "imbalance_sd")
        return float(max(abs(m[k] - real[k]) / max(abs(real[k]), 1e-3) for k in keys))
    best_cls = min(CLASSICAL, key=lambda n: disp_err(mets[n]))
    if "neural_ar" in mets:
        out["H23"] = math.log(max(disp_err(mets[best_cls]), 1e-9) / max(disp_err(mets["neural_ar"]), 1e-9))
        best_sd = min(CLASSICAL, key=lambda n: sd_err(mets[n]))
        out["H25"] = math.log(max(sd_err(mets[best_sd]), 1e-9) / max(sd_err(mets["neural_ar"]), 1e-9))
    # H24: path realism of 5 min mid paths (classical simulated vs neural SDE)
    try:
        from ...generators.neural.dynamics import NeuralSDE, sde_states
        states = [p for c in train[:6] for p in sde_states(c, [600 + 3600 * h for h in range(0, 24, 2)])]
        sde = NeuralSDE(steps=800, seed=seed).fit(states)
        allst = np.concatenate(states)
        lo, hi = np.percentile(allst, 0.5, axis=0), np.percentile(allst, 99.5, axis=0)
        y0 = np.median(allst, axis=0).astype(np.float32)
        real_paths = [w.mid[::10][:300] for w in real_w]
        sde_paths, diverged = [], 0
        for k in range(len(real_paths)):
            # registered safeguard: the sampled state is kept inside the training support; divergence is counted
            y = sde.sample(y0, 299, seed + k)
            diverged += int(not np.isfinite(y).all() or (y < lo - 10 * (hi - lo)).any() or (y > hi + 10 * (hi - lo)).any())
            y = np.clip(np.nan_to_num(y, nan=0.0), lo, hi)
            sde_paths.append(np.exp(np.cumsum(y[:, 0]) / 1e4))
        out["sde_diverged_paths"] = diverged
        mm_n = mmd(signature_features(sde_paths), signature_features(real_paths))
        mm_c = min(mmd(signature_features([f.mid[::10][:300] for f in sims[n]]), signature_features(real_paths))
                   for n in CLASSICAL)
        out["H24"] = math.log(max(mm_c, 1e-9) / max(mm_n, 1e-9))
    except ImportError:
        pass
    # H28: the generator pre-selected on the training days by realism distance, accepted on the held-out day?
    from ...calibration.api import realism_distance
    tr_band = GT.bands(ref)
    pre = min(mets, key=lambda n: realism_distance(mets[n], tr_band))
    out["H28"] = float(all(dec[pre][c]["status"] == "ACCEPTABLE" for c in GT.EXECUTION))
    out.update({"preselected": pre, "decisions": {n: {c: d["status"] for c, d in v.items()} for n, v in dec.items()},
                "metrics": mets, "seconds": time.time() - t0})
    return out


__all__ = ["fold", "glm_cancel_nll", "signature_features", "mmd", "CLASSICAL"]
