"""Reconstruction, liquidity, regime and continual hub: registered unit values of H14, H17, H18, H27 and H42.

Each function takes one held-out (or evaluation) unit and the training units of the same stream as day-cache
paths; nothing here touches a unit outside the arguments. Operational definitions (sealed with the design):

* H14 = logloss(1 s aggregated order-flow features) - logloss(100 ms event-level features) of a ridge logistic
  classifier of the sign of the next-1 s mid move (non-zero moves, sampled every second). Both feature sets carry
  the same information budget over the last second (OFI, signed volume, market-order imbalance) plus the current
  touch imbalance and spread; the fine set keeps the ten 100 ms lags, the coarse set their sums.
* H17 = |coverage(point model, hidden = 0) - 0.9| - |coverage(evidence-based hidden interval) - 0.9|. Clearance of
  a touch level within 30 s is observed as a print strictly through the touch price. Training units estimate, per
  touch-depth quintile, the probability that the volume printed at or through the touch within 30 s reaches the
  effective queue Q / (1 - h); the point model uses h = 0 with a 90% Wilson interval, the evidence model the union
  over h in [0, h_max] (h_max from ``reconstruction.hidden`` refill evidence on the training units). Coverage is the
  share of held-out (hour x quintile) cells, >= 10 samples, whose trade-through frequency lies in the interval.
* H18 = mean over C1-C3 of width(F2 identified set, hidden grid {0, 0.5}) - width(hidden grid {0, h_max}).
* H27 = NLL(pooled per-column Poisson rates) - NLL(rates conditioned on the 5 min volatility regime), nats per
  100 ms bin, on held-out units whose majority volatility regime differs from the training majority (else
  ``None``); regime cuts come from the training units only.
* H42 = NLL(frozen GLM fitted on the calibrate units) - NLL(continually adapted GLM), nats per event, on each
  evaluation unit in chronological order; the adapted model is refitted prequentially (only earlier units, day
  weights 0.5^age), and the forgetting check reports NLL(adapted, calibrate units) - NLL(frozen, calibrate units)
  against the registered non-inferiority margin 0.01.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

Z90 = 1.6448536269514722
HORIZON_BINS = 10            # 1 s
CLEAR_S = 30.0
FORGETTING_MARGIN = 0.01


def _load(path):
    from ...data import daycache as DC
    return DC.load(Path(path))


# --------------------------------------------------------------------------------------------------------- H14
def _fine_coarse(cache) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from ...microstructure import features as F
    bid, ask = np.asarray(cache["bid_px"]), np.asarray(cache["ask_px"])
    depth = np.asarray(cache["depth"], float)
    valid = np.asarray(cache["valid"], bool)
    mid = np.asarray(cache["mid"], float)[: len(bid)]
    counts = np.asarray(cache["counts"], float)
    vol = np.asarray(cache["volume"], float)
    per_bin = np.column_stack([np.where(valid, F.ofi(bid, ask, depth, 1), 0.0), vol[:, 0] - vol[:, 1],
                               counts[:, 0] - counts[:, 1]])
    imb = F.imbalance(depth, 1)
    spread = (ask - bid).astype(float)
    t = np.arange(HORIZON_BINS, len(bid) - HORIZON_BINS, HORIZON_BINS)
    win = np.stack([per_bin[t - HORIZON_BINS + j + 1] for j in range(HORIZON_BINS)], 1)       # (n, 10, 3)
    ok = np.stack([valid[t - HORIZON_BINS + j + 1] for j in range(HORIZON_BINS + 1)], 1).all(1) & valid[t + HORIZON_BINS]
    dm = mid[t + HORIZON_BINS] - mid[t]
    ok &= dm != 0
    now = np.column_stack([imb[t], spread[t]])
    fine = np.column_stack([win.reshape(len(t), -1), now])[ok]
    coarse = np.column_stack([win.sum(1), now])[ok]
    return fine, coarse, (dm[ok] > 0).astype(float)


def _logit_fit(X, y, l2: float = 1.0, iters: int = 30):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    reg = l2 * np.eye(len(w))
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        H = (Z * (p * (1 - p))[:, None]).T @ Z + reg
        step = np.linalg.solve(H, Z.T @ (y - p) - reg @ w)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w, mu, sd


def _logloss(model, X, y) -> float:
    w, mu, sd = model
    p = 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones(len(X)), (X - mu) / sd]) @ w, -30, 30)))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def h14(held_path: str, train_paths: list[str], *, cap: int = 200_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    parts = [_fine_coarse(_load(p)) for p in train_paths]
    Ff, Fc, Y = (np.concatenate([p[i] for p in parts]) for i in range(3))
    if len(Y) > cap:
        k = rng.choice(len(Y), cap, replace=False)
        Ff, Fc, Y = Ff[k], Fc[k], Y[k]
    tf, tc, ty = _fine_coarse(_load(held_path))
    if len(ty) < 100:
        return {"H14": None, "test_samples": int(len(ty))}
    lf, lc = _logloss(_logit_fit(Ff, Y), tf, ty), _logloss(_logit_fit(Fc, Y), tc, ty)
    return {"H14": lc - lf, "logloss_fine": lf, "logloss_coarse": lc, "test_samples": int(len(ty)),
            "train_samples": int(len(Y))}


# --------------------------------------------------------------------------------------------- H17 / H18 evidence
def touch_prints(cache) -> list[tuple[float, float, float]]:
    """(visible touch size before, printed size, touch size after) per (bin, side) of prints at the touch price."""
    bid, ask = np.asarray(cache["bid_px"]), np.asarray(cache["ask_px"])
    depth = np.asarray(cache["depth"], float)
    t, px, q, s = (np.asarray(cache[k]) for k in ("trade_t", "trade_px", "trade_q", "trade_s"))
    b = np.minimum((t * 10).astype(np.int64), len(bid) - 1)
    out = []
    keys = b * 2 + (s > 0)
    order = np.argsort(keys, kind="stable")
    uk, start = np.unique(keys[order], return_index=True)
    ends = np.r_[start[1:], len(order)]
    for k, a, e in zip(uk, start, ends):
        bb, buy = int(k // 2), bool(k % 2)
        if bb == 0:
            continue
        side = 1 if buy else 0                                        # a buy consumes the ask
        touch = ask if buy else bid
        p0 = touch[bb - 1]
        idx = order[a:e]
        at = idx[px[idx] == p0]
        if p0 <= 0 or len(at) == 0:
            continue
        before = depth[bb - 1, side, 0]
        after = depth[bb, side, 0] if touch[bb] == p0 else 0.0
        out.append((float(before), float(q[at].sum()), float(after)))
    return out


def h_max(train_paths: list[str]) -> dict:
    from ...data.reconstruction import hidden as HD
    rows = [r for p in train_paths for r in touch_prints(_load(p))]
    if not rows:
        return HD.hidden_interval({"executed_volume": 0.0, "unseen_volume_lower_bound": 0.0, "prints": 0})
    a = np.asarray(rows)
    return HD.hidden_interval(HD.refill_evidence(a[:, 0], a[:, 1], a[:, 2]))


def _clearance_samples(cache, *, every_s: float = 10.0):
    """(log Q, hour, volume at-or-through within 30 s, Q, trade-through indicator) for both touch sides."""
    bid, ask = np.asarray(cache["bid_px"]), np.asarray(cache["ask_px"])
    depth = np.asarray(cache["depth"], float)
    valid = np.asarray(cache["valid"], bool)
    t, px, q, s = (np.asarray(cache[k]) for k in ("trade_t", "trade_px", "trade_q", "trade_s"))
    rows = []
    for t0 in np.arange(60.0, 86400.0 - CLEAR_S, every_s):
        b = int(t0 * 10)
        if not valid[b]:
            continue
        lo, hi = np.searchsorted(t, t0), np.searchsorted(t, t0 + CLEAR_S)
        for side, touch, aggr in ((0, bid[b], -1), (1, ask[b], 1)):
            Q = depth[b, side, 0]
            if Q <= 0 or touch <= 0:
                continue
            sl = slice(lo, hi)
            m = s[sl] == aggr
            p = px[sl][m]
            at_or_through = (p <= touch) if side == 0 else (p >= touch)
            through = (p < touch) if side == 0 else (p > touch)
            rows.append((math.log(Q), int(t0 // 3600), float(q[sl][m][at_or_through].sum()), Q, float(through.any())))
    return np.asarray(rows)


def _wilson(k: float, n: float, z: float = Z90) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def h17(held_path: str, train_paths: list[str], *, hm: dict | None = None) -> dict:
    hm = hm or h_max(train_paths)
    tr = np.vstack([_clearance_samples(_load(p)) for p in train_paths])
    te = _clearance_samples(_load(held_path))
    edges = np.quantile(tr[:, 0], [0.2, 0.4, 0.6, 0.8])
    btr, bte = np.searchsorted(edges, tr[:, 0]), np.searchsorted(edges, te[:, 0])
    point, evid = {}, {}
    for k in range(5):
        r = tr[btr == k]
        n = len(r)
        c0 = float((r[:, 2] >= r[:, 3]).sum())
        ch = float((r[:, 2] >= r[:, 3] / max(1 - hm["upper"], 1e-6)).sum())
        point[k] = _wilson(c0, n)
        evid[k] = (_wilson(ch, n)[0], _wilson(c0, n)[1])
    hits_p, hits_e, cells = 0, 0, 0
    for k in range(5):
        for hr in range(24):
            r = te[(bte == k) & (te[:, 1] == hr)]
            if len(r) < 10:
                continue
            f = float(r[:, 4].mean())
            cells += 1
            hits_p += point[k][0] <= f <= point[k][1]
            hits_e += evid[k][0] <= f <= evid[k][1]
    if cells == 0:
        return {"H17": None, "cells": 0}
    cp, ce = hits_p / cells, hits_e / cells
    return {"H17": abs(cp - 0.9) - abs(ce - 0.9), "coverage_point": cp, "coverage_evidence": ce, "cells": cells,
            "h_max": hm["upper"]}


def h18(day_point: dict, day_evidence: dict, conclusions: dict) -> dict:
    """Inputs: v0.9 ``bounds.day_task`` outputs of one unit under the {0, 0.5} and {0, h_max} hidden grids."""
    from ....v09.studies import bounds as BD
    widths = []
    for c in conclusions.values():
        a = BD.pair_day(day_point, f"{c['treatment']}|{c['mandate']}", f"{c['control']}|{c['mandate']}",
                        c["endpoint"], c["direction"])
        b = BD.pair_day(day_evidence, f"{c['treatment']}|{c['mandate']}", f"{c['control']}|{c['mandate']}",
                        c["endpoint"], c["direction"])
        if a and b:
            widths.append((a["upper"] - a["lower"]) - (b["upper"] - b["lower"]))
    return {"H18": float(np.mean(widths)) if widths else None, "conclusions": len(widths)}


def hidden_keys(h: float) -> list[str]:
    from ....v09.replay import engine as E
    grid = E.grid(latencies=(0.0, 0.05, 0.25), hidden=(0.0, round(float(h), 2)), impacts=("none", "transient",
                                                                                         "persistent"))
    return [json.dumps(E.assumption_dict(a), sort_keys=True) for a in grid]


# --------------------------------------------------------------------------------------------------------- H27
def _poisson_nll(y: np.ndarray, lam: np.ndarray) -> np.ndarray:
    yi = np.asarray(y).astype(np.int64)                         # counts are integers: exact log(y!) table
    lg = np.asarray([math.lgamma(k + 1) for k in range(int(yi.max(initial=0)) + 1)])
    lam = np.maximum(lam, 1e-9)
    return lam - y * np.log(lam) + lg[yi]


def _regime_bins(cache, cuts=None):
    from ...microstructure import features as F
    from ...regimes import regimes as RG
    blocks = RG.block_features(F.compute(cache, interval_ms=1000))
    return blocks, (RG.label(blocks, cuts) if cuts is not None else None)


def h27(held_path: str, train_paths: list[str]) -> dict:
    from ...regimes import regimes as RG
    tr_c = [_load(p) for p in train_paths]
    tr_blocks = [_regime_bins(c)[0] for c in tr_c]
    cuts = RG.cuts_from(tr_blocks)
    per = 3000                                                     # 100 ms bins per 5 min block
    rates_reg, pooled_n, pooled_s = {}, 0, np.zeros(6)
    tr_labels = []
    for c, bl in zip(tr_c, tr_blocks):
        lab = RG.label(bl, cuts)
        tr_labels += lab
        cnt, v = np.asarray(c["counts"], float), np.asarray(c["valid"], bool)
        for i, lb in enumerate(lab):
            if lb is None:
                continue
            sl = slice(i * per, (i + 1) * per)
            m = v[sl]
            r = lb.split("|")[0]
            n, s = rates_reg.get(r, (0, np.zeros(6)))
            rates_reg[r] = (n + int(m.sum()), s + cnt[sl][m].sum(0))
            pooled_n += int(m.sum())
            pooled_s += cnt[sl][m].sum(0)
    te = _load(held_path)
    te_blocks, te_lab = _regime_bins(te, cuts)
    maj_tr, maj_te = RG.majority(tr_labels), RG.majority(te_lab)
    if maj_te is None or maj_te == maj_tr:
        return {"H27": None, "train_majority": maj_tr, "held_majority": maj_te}
    pooled = pooled_s / max(pooled_n, 1)
    cnt, v = np.asarray(te["counts"], float), np.asarray(te["valid"], bool)
    nll_p, nll_r, bins = 0.0, 0.0, 0
    for i, lb in enumerate(te_lab):
        if lb is None:
            continue
        sl = slice(i * per, (i + 1) * per)
        y = cnt[sl][v[sl]]
        r = lb.split("|")[0]
        lam_r = rates_reg[r][1] / rates_reg[r][0] if r in rates_reg and rates_reg[r][0] > 0 else pooled
        nll_p += float(_poisson_nll(y, pooled[None]).sum())
        nll_r += float(_poisson_nll(y, lam_r[None]).sum())
        bins += len(y)
    return {"H27": (nll_p - nll_r) / bins if bins else None, "train_majority": maj_tr, "held_majority": maj_te,
            "bins": bins}


# --------------------------------------------------------------------------------------------------------- H42
def _glm_design(cache, *, stride: int = 5):
    v = np.asarray(cache["valid"], bool)
    st = np.asarray(cache["state"], float)
    idx = np.flatnonzero(v)[::stride]
    X = np.column_stack([np.ones(len(idx)), np.log1p(np.maximum(st[idx, 3], 0)), st[idx, 1], st[idx, 0]])
    y = np.asarray(cache["counts"], float)[idx].sum(1)
    return X, y


def _glm_fit(parts, weights, iters: int = 50):
    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    w_obs = np.concatenate([np.full(len(p[1]), wt) for p, wt in zip(parts, weights)])
    w = np.zeros(X.shape[1])
    w[0] = math.log(max(np.average(y, weights=w_obs), 1e-6))
    for _ in range(iters):
        mu = np.exp(np.clip(X @ w, -20, 20))
        H = (X * (w_obs * mu)[:, None]).T @ X + 1e-6 * np.eye(len(w))
        step = np.linalg.solve(H, X.T @ (w_obs * (y - mu)))
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def _nll_per_event(w, part) -> float:
    X, y = part
    lam = np.exp(np.clip(X @ w, -20, 20))
    return float(_poisson_nll(y, lam).sum() / max(y.sum(), 1.0))


def h42(eval_paths: list[str], calibrate_paths: list[str], *, decay: float = 0.5) -> dict:
    """``eval_paths`` in chronological order; one unit value per evaluation unit (prequential, no look-ahead)."""
    cal = [_glm_design(_load(p)) for p in calibrate_paths]
    frozen = _glm_fit(cal, [1.0] * len(cal))
    seen: list = []
    units, adapted = [], frozen
    for p in eval_paths:
        part = _glm_design(_load(p))
        units.append(_nll_per_event(frozen, part) - _nll_per_event(adapted, part))
        seen.append(part)
        hist = cal + seen
        ages = list(range(len(hist) - 1, -1, -1))
        adapted = _glm_fit(hist, [decay ** a for a in ages])
    forget = float(np.mean([_nll_per_event(adapted, c) - _nll_per_event(frozen, c) for c in cal]))
    return {"H42": units, "forgetting": forget, "forgetting_ok": forget <= FORGETTING_MARGIN}


__all__ = ["h14", "h17", "h18", "h27", "h42", "h_max", "hidden_keys", "touch_prints"]
