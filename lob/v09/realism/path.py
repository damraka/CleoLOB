"""Path-level realism (M52): truncated path signatures, signature MMD, spectral and first-passage diagnostics.

A 60 s path is (normalized time, mid in half-ticks relative to its start / scale, spread in ticks - 1) sampled every
100 ms (600 increments). The level-3 truncated signature of the piecewise-linear path (3 + 9 + 27 terms) is computed
exactly with Chen's identity. Two path sets are compared with the unbiased MMD^2 of a Gaussian kernel on standardized
signature features (bandwidth: median pairwise distance of the reference set). Also reported: the power spectrum of
mid increments (log-spaced bands) and the first-passage time distribution of |mid move| >= 2 half-ticks.
"""
from __future__ import annotations

import numpy as np

from ..data.daycache import BIN_MS, HORIZON_S, WARMUP_S, WINDOW_STARTS_S

LENGTH = 600
STRIDE_S = 30


def real_paths(cache: dict) -> np.ndarray:
    """[n, 601, 2] (mid half-ticks relative to start, spread ticks) paths inside the registered windows."""
    out = []
    mid2 = np.rint(2 * cache["mid"]).astype(np.int64)
    spread = (cache["ask_px"] - cache["bid_px"]).astype(float)
    for s in WINDOW_STARTS_S:
        for t0 in range(int(s - WARMUP_S), int(s + HORIZON_S) - 60 + 1, STRIDE_S):
            b = int(t0 * 1000 // BIN_MS)
            if not cache["valid"][b:b + LENGTH + 1].all():
                continue
            m = mid2[b:b + LENGTH + 1] - mid2[b]
            out.append(np.column_stack([m, spread[b:b + LENGTH + 1]]))
    return np.asarray(out, float).reshape(-1, LENGTH + 1, 2)


def signature(paths: np.ndarray, scale: float) -> np.ndarray:
    """Level-3 signatures of (t, mid/scale, spread-1) paths, vectorized over paths. Returns [n, 39]."""
    n, L, _ = paths.shape
    t = np.linspace(0.0, 1.0, L)
    X = np.stack([np.broadcast_to(t, (n, L)), paths[:, :, 0] / scale, paths[:, :, 1] - 1.0], axis=-1)
    d = 3
    S1 = np.zeros((n, d))
    S2 = np.zeros((n, d, d))
    S3 = np.zeros((n, d, d, d))
    for k in range(1, L):
        D = X[:, k] - X[:, k - 1]
        DD = D[:, :, None] * D[:, None, :]
        DDD = DD[:, :, :, None] * D[:, None, None, :]
        S3 = S3 + S2[:, :, :, None] * D[:, None, None, :] + S1[:, :, None, None] * DD[:, None, :, :] / 2 + DDD / 6
        S2 = S2 + S1[:, :, None] * D[:, None, :] + DD / 2
        S1 = S1 + D
    return np.concatenate([S1, S2.reshape(n, -1), S3.reshape(n, -1)], axis=1)


def mmd2(A: np.ndarray, B: np.ndarray, ref: np.ndarray | None = None) -> float:
    """Unbiased MMD^2 with a Gaussian kernel on features standardized by the reference set (default A)."""
    ref = A if ref is None else ref
    mu, sd = ref.mean(0), ref.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    A, B = (A - mu) / sd, (B - mu) / sd
    d_ref = np.sqrt(((A[:, None] - A[None]) ** 2).sum(-1))
    h = float(np.median(d_ref[np.triu_indices(len(A), 1)])) or 1.0

    def k(X, Y):
        return np.exp(-((X[:, None] - Y[None]) ** 2).sum(-1) / (2 * h * h))

    m, n = len(A), len(B)
    kxx, kyy, kxy = k(A, A), k(B, B), k(A, B)
    return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
                 - 2 * kxy.mean())


def spectrum(paths: np.ndarray, bands: int = 8) -> np.ndarray:
    inc = np.diff(paths[:, :, 0], axis=1)
    p = np.abs(np.fft.rfft(inc - inc.mean(1, keepdims=True), axis=1)) ** 2
    edges = np.unique(np.geomspace(1, p.shape[1] - 1, bands + 1).astype(int))
    return np.asarray([p[:, a:b].mean() for a, b in zip(edges[:-1], edges[1:])])


def first_passage(paths: np.ndarray, level: int = 2) -> np.ndarray:
    hit = np.abs(paths[:, :, 0]) >= level
    first = np.where(hit.any(1), hit.argmax(1), paths.shape[1])
    return first * BIN_MS / 1000.0
