"""Neural temporal point processes with exact sampling (M80) and the marked event Transformer (M81).

The v0.9 neural TPPs are reused unchanged for fitting and scoring (``lob.v09.generators.point_process``), so v1
likelihoods equal v0.9's. Their intensity after event j is

    lambda(d) = exp(a + w d) + c,   w < 0 (decaying burst), c > 0 (floor),   d = time since event j,

with (a, w, c) and the mark distribution read from the encoder state. v1 adds **exact sampling** by inversion of
the compensator, which is strictly increasing because lambda > 0:

    Lambda(d) = (exp(a + w d) - exp(a)) / w + c d = E,   E ~ Exp(1),

solved by bracketed bisection to 1e-9 relative precision; the next mark is drawn from the softmax head. The encoder
is rolled forward on its own samples (features: log1p(10 dt) of the previous gap and the mark embedding, as in
fitting). ``RMTPP`` (GRU encoder, M80) and ``EventTransformer`` (causal Transformer encoder, M81) share the head.
"""
from __future__ import annotations

import math

import numpy as np

from ....v09.generators import point_process as PP


def _torch():
    import torch
    return torch


def invert_compensator(a: float, w: float, c: float, E: float) -> float:
    """Smallest d > 0 with Lambda(d) = E for lambda(d) = exp(a + w d) + c (w < 0, c > 0)."""
    def lam_int(d):
        return (math.exp(a + w * d) - math.exp(a)) / w + c * d
    hi = 1.0
    while lam_int(hi) < E:
        hi *= 2.0
        if hi > 1e12:
            return math.inf
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if lam_int(mid) < E:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-9 * max(hi, 1e-12):
            break
    return 0.5 * (lo + hi)


class SamplingMixin:
    def head_params(self, h):
        T = _torch()
        out = self.head(h)
        a = float(T.clamp(out[..., 0], max=20.0))
        w = float(-T.nn.functional.softplus(out[..., 1]) - 1e-3)
        c = float(T.nn.functional.softplus(out[..., 2]) + 1e-6)
        p = T.softmax(self.head_mark(h), -1).numpy().astype(float)
        return a, w, c, p

    def sample(self, horizon: float, seed: int, *, max_events: int = 20_000):
        T = _torch()
        rng = np.random.default_rng(seed)
        t, times, marks = 0.0, [], []
        feats, mks = [0.0], [0]
        with T.no_grad():
            while len(times) < max_events:
                lo = max(0, len(feats) - self.context)
                x = T.as_tensor(np.asarray(feats[lo:], np.float32))[None, :, None]
                mk = T.as_tensor(np.asarray(mks[lo:], np.int64))[None]
                h = self.encode(x, mk)[0, -1]
                a, w, c, p = self.head_params(h)
                d = invert_compensator(a, w, c, rng.exponential())
                t += d
                if not math.isfinite(t) or t >= horizon:
                    break
                m = int(rng.choice(len(p), p=p / p.sum()))
                times.append(t)
                marks.append(m)
                feats.append(math.log1p(d * 10.0))
                mks.append(m)
        return np.asarray(times), np.asarray(marks, dtype=int)


class RMTPP(SamplingMixin, PP.RMTPP):
    """M80: recurrent marked TPP (GRU encoder) - v0.9 fit/score, v1 exact sampling."""


class EventTransformer(SamplingMixin, PP.TransformerTPP):
    """M81: marked event Transformer (causal self-attention encoder) - v0.9 fit/score, v1 exact sampling."""


__all__ = ["RMTPP", "EventTransformer", "SamplingMixin", "invert_compensator"]
