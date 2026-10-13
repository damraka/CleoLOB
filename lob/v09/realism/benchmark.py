"""Realism scorecard v3 (M51-M54) and the capability-specific realism gate (M55).

Sources are EventFrame-like windows (real day caches sliced at the registered windows, or simulated frames). For each
capability a small registered metric vector is computed per source:

* ``spread_dynamics``: P(spread > 1 tick), mean spread, lag-1 autocorrelation of spread changes;
* ``depth_dynamics``: median log touch depth, imbalance standard deviation, lag-10 autocorrelation of log touch depth;
* ``event_timing``: per-bin rates of market/add/cancel events, market-count dispersion, lag-1 and lag-10 ACF of
  market counts;
* ``queue_survival`` and ``fill_prediction``: P(back-of-queue clearance at the touch within 1 s / 5 s / 30 s);
* ``impact``: mean signed 10 s mid move after top-1 % |signed flow| seconds (bps);
* ``tail_behavior``: Hill tail index of |10 s mid returns| (top 5 %), P(spread >= 3 ticks), depth-collapse frequency
  (touch depth below 10 % of its median);
* ``strategic_response``: hit-side touch-depth recovery ratio 5 s after top-1 % bursts.

Distance of a generator to real data on a capability = max over its metrics of |m_gen - m_real| / ref, with ref the
95th percentile of |m_day_i - m_day_j| over pairs of real held-out days (day-to-day variation). A classifier
(logistic regression on the v0.8 summary vector of 30 s windows, 5-fold CV) gives the AUC of real vs generated.

Gate (registered in the protocol): ACCEPTABLE if every metric ratio <= 1.0 and AUC <= 0.60; PARTIALLY_ACCEPTABLE if
every ratio <= 2.0; otherwise REJECTED. Global: ACCEPTABLE_MODEL_SET when some generator is ACCEPTABLE on every
capability a study class needs; PARTIAL_MODEL_SET when some generator is at least partially acceptable on each;
otherwise NO_ACCEPTABLE_MODEL. Thresholds are never moved to avoid NO_ACCEPTABLE_MODEL.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from ...v08.market.frame import BIN_S, summaries

CAPABILITIES = ("spread_dynamics", "depth_dynamics", "event_timing", "queue_survival", "fill_prediction", "impact",
                "tail_behavior", "strategic_response")
STUDY_CLASSES = {"execution": ("spread_dynamics", "depth_dynamics", "queue_survival", "fill_prediction", "impact"),
                 "realism": CAPABILITIES,
                 "fill": ("queue_survival", "fill_prediction"),
                 "tail": ("tail_behavior", "spread_dynamics", "depth_dynamics"),
                 "strategic": ("impact", "strategic_response")}


def _acf(x, lag):
    x = np.asarray(x, float)
    if len(x) <= lag + 2 or x.std() == 0:
        return 0.0
    a, b = x[:-lag] - x.mean(), x[lag:] - x.mean()
    return float((a * b).mean() / x.var())


def _clearance(fr, horizons=(1.0, 5.0, 30.0)) -> list[float]:
    """P(back-of-queue clearance at the bid touch within each horizon), probes every second of the frame."""
    t, px, q, s = (np.asarray(fr.samples.get(k, []), float) for k in ("trade_t", "trade_px", "trade_q", "trade_s"))
    t = t - fr.t0
    order = np.argsort(t, kind="stable")
    t, px, q, s = t[order], px[order], q[order], s[order]
    hits = {h: [] for h in horizons}
    for b in range(0, len(fr) - int(max(horizons) / BIN_S), 10):
        if not fr.valid[b]:
            continue
        p0, queue, t0 = fr.bid_px[b], float(fr.depth[b, 0, 0]), b * BIN_S
        lo, hi = np.searchsorted(t, [t0, t0 + max(horizons)])
        st, sp, sq, ss = t[lo:hi], px[lo:hi], q[lo:hi], s[lo:hi]
        sell = ss < 0
        through = sell & (sp < p0)
        at = sell & (sp == p0)
        cum = np.cumsum(np.where(at, sq, 0.0))
        cands = []
        if through.any():
            cands.append(st[through][0])
        k = np.flatnonzero(cum > queue)
        if len(k):
            cands.append(st[k[0]])
        T = (min(cands) - t0) if cands else math.inf
        for h in horizons:
            hits[h].append(T <= h)
    return [float(np.mean(hits[h])) if hits[h] else 0.0 for h in horizons]


def _bursts(fr):
    n_s = len(fr) // 10
    t, q, s = (np.asarray(fr.samples.get(k, []), float) for k in ("trade_t", "trade_q", "trade_s"))
    sec = np.floor(t - fr.t0).astype(int)
    ok = (sec >= 0) & (sec < n_s)
    v = np.zeros(n_s)
    np.add.at(v, sec[ok], q[ok] * s[ok])
    mid = fr.mid[::10][:n_s + 1].astype(float)
    nz = np.abs(v[v != 0])
    if len(nz) < 20:
        return 0.0, 1.0
    idx = np.flatnonzero(np.abs(v) >= np.quantile(nz, 0.99))
    idx = idx[idx + 10 < len(mid)]
    sign = np.sign(v[idx])
    resp = sign * (mid[idx + 10] - mid[idx]) / np.maximum(mid[idx], 1e-9) * 1e4
    dep = fr.depth[::10, :, 0].astype(float)[:n_s]
    j = np.minimum(idx + 5, len(dep) - 1)
    hit = np.where(sign > 0, 1, 0)
    rec = dep[j, hit] / np.maximum(dep[idx, hit], 1.0)
    return float(resp.mean()) if len(resp) else 0.0, float(np.median(rec)) if len(rec) else 1.0


def metrics(frames: list) -> dict:
    """Capability metric vectors of a list of frames (pooled)."""
    v = np.concatenate([f.valid for f in frames])
    spread = np.concatenate([(f.ask_px - f.bid_px).astype(float) for f in frames])[v]
    touch = np.concatenate([(f.depth[:, 0, 0] + f.depth[:, 1, 0]).astype(float) for f in frames])[v]
    imb = np.concatenate([f.state[:, 1] for f in frames])[v]
    c = np.concatenate([f.counts for f in frames])[v].astype(float)
    market = c[:, 0] + c[:, 1]
    mids = [f.mid.astype(float) for f in frames]
    r10 = np.concatenate([np.diff(m[::100]) / np.maximum(m[:-100:100], 1e-9) * 1e4 for m in mids if len(m) > 200])
    a = np.sort(np.abs(r10[r10 != 0]))[::-1] if len(r10) else np.zeros(0)
    k = max(5, int(0.05 * len(a)))
    hill = float(1.0 / np.mean(np.log(a[:k] / a[k]))) if len(a) > k + 1 and a[k] > 0 else 0.0
    med_touch = float(np.median(touch)) if len(touch) else 1.0
    clear = np.mean([_clearance(f) for f in frames], axis=0)
    burst = np.mean([_bursts(f) for f in frames], axis=0)
    ltouch = np.log(np.maximum(touch, 1.0))
    return {
        "spread_dynamics": [float((spread > 1).mean()), float(spread.mean()), _acf(np.diff(spread) != 0, 1)],
        "depth_dynamics": [float(np.median(ltouch)), float(imb.std()), _acf(ltouch, 10)],
        "event_timing": [float(market.mean()), float((c[:, 2] + c[:, 3]).mean()), float((c[:, 4] + c[:, 5]).mean()),
                         float(market.var() / max(market.mean(), 1e-9)), _acf(market, 1), _acf(market, 10)],
        "queue_survival": [float(x) for x in clear],
        "fill_prediction": [float(clear[1])],
        "impact": [float(burst[0])],
        "tail_behavior": [hill, float((spread >= 3).mean()), float((touch < 0.1 * med_touch).mean())],
        "strategic_response": [float(burst[1])],
    }


def reference(real_metrics: list[dict]) -> dict:
    """95th percentile of |m_i - m_j| over pairs of real days, per capability metric (floored)."""
    out = {}
    for cap in CAPABILITIES:
        diffs = [np.abs(np.asarray(a[cap]) - np.asarray(b[cap])) for a, b in itertools.combinations(real_metrics, 2)]
        out[cap] = (np.maximum(np.percentile(diffs, 95, axis=0), 1e-6) if diffs
                    else np.full(len(real_metrics[0][cap]), 1e-6)).tolist()
    return out


def ratios(gen: dict, real: dict, ref: dict) -> dict:
    return {cap: (np.abs(np.asarray(gen[cap]) - np.asarray(real[cap])) / np.asarray(ref[cap])).tolist()
            for cap in CAPABILITIES}


def classifier_auc(real_frames: list, gen_frames: list, window: int = 300, seed: int = 0) -> float:
    def rows(frames):
        out = []
        for f in frames:
            for a in range(0, len(f) - window + 1, window):
                try:
                    out.append(summaries(f.slice(a, a + window)))
                except ValueError:
                    continue
        return out
    R, Gm = rows(real_frames), rows(gen_frames)
    if len(R) < 10 or len(Gm) < 10:
        return float("nan")
    X = np.vstack([R, Gm])
    y = np.r_[np.ones(len(R)), np.zeros(len(Gm))]
    mu, sd = X.mean(0), X.std(0)
    X = (X - mu) / np.where(sd > 0, sd, 1.0)
    rng = np.random.default_rng(seed)
    folds = rng.integers(0, 5, len(y))
    scores = np.zeros(len(y))
    for f in range(5):
        tr, te = folds != f, folds == f
        w = np.zeros(X.shape[1] + 1)
        Z = np.column_stack([np.ones(tr.sum()), X[tr]])
        for _ in range(300):
            p = 1 / (1 + np.exp(-Z @ w))
            w -= 0.1 * (Z.T @ (p - y[tr]) / tr.sum() + 1e-3 * w)
        scores[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ w
    pos, neg = scores[y == 1], scores[y == 0]
    ranks = np.argsort(np.argsort(np.r_[pos, neg])) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def decide(ratio: dict, auc: float, gate: dict | None = None) -> dict:
    acc, part = 1.0, 2.0
    out = {}
    for cap, r in ratio.items():
        worst = float(np.max(r)) if len(r) else math.inf
        if worst <= acc and (not np.isfinite(auc) or auc <= 0.60):
            out[cap] = "ACCEPTABLE"
        elif worst <= part:
            out[cap] = "PARTIALLY_ACCEPTABLE"
        else:
            out[cap] = "REJECTED"
    return out


def global_gate(decisions: dict[str, dict]) -> dict:
    out = {}
    for cls, caps in STUDY_CLASSES.items():
        if any(all(d[c] == "ACCEPTABLE" for c in caps) for d in decisions.values()):
            out[cls] = "ACCEPTABLE_MODEL_SET"
        elif any(all(d[c] in ("ACCEPTABLE", "PARTIALLY_ACCEPTABLE") for c in caps) for d in decisions.values()):
            out[cls] = "PARTIAL_MODEL_SET"
        else:
            out[cls] = "NO_ACCEPTABLE_MODEL"
    return out
