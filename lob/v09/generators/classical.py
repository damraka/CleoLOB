"""Classical generators calibrated to real days (F4, M44) and probe-response tables (H1, H12, realism).

The v0.8 bin-level families (Poisson, Hawkes, queue-reactive, regime switching, state dependent, mixture) and the
v0.8 static ecology are reused unchanged and fitted on real training frames:

* ``training_frame`` concatenates the registered-window slices of several day caches into one EventFrame and adds
  book-model samples inferred from consecutive 100 ms frames (add level and size from visible increases; cancel
  level and fraction from decreases not explained by same-bin prints), so ``BookModel.fit`` sees real order sizes;
* ``fit_family`` fits the family and a ``BookModel`` with the real price scale (``mid_ticks`` = median mid in ticks);
* ``probe_responses`` injects market orders of registered relative sizes (multiples of the median touch depth) into
  a simulated market and records the signed mid move and the hit-side touch-depth recovery at +1 s, +5 s, +30 s.
"""
from __future__ import annotations

import numpy as np

from ...v08.generators import families as F
from ...v08.generators.ecology import EcologyGenerator
from ...v08.market.book import BUY
from ...v08.market.frame import EventFrame
from ...v08.market.sim import BookModel, MarketSim
from ..data.daycache import BIN_MS, HORIZON_S, WARMUP_S, WINDOW_STARTS_S, frame as cache_frame

FAMILIES = {"poisson": F.PoissonGenerator, "hawkes": F.HawkesGenerator, "queue_reactive": F.QueueReactiveGenerator,
            "regime_switching": F.RegimeSwitchingGenerator, "state_dependent": F.StateDependentGenerator,
            "mixture": F.MixtureGenerator}
STATIC_ECOLOGY = {"ecology_static": EcologyGenerator}
REL_SIZES = (0.25, 0.5, 1.0, 2.0, 4.0)
HORIZONS_S = (1, 5, 30)


def book_samples(cache: dict, windows=WINDOW_STARTS_S) -> dict:
    """Add/cancel size and level samples inferred from consecutive frames inside the windows (touch-aligned)."""
    add_level, add_size, cancel_level, cancel_frac = [], [], [], []
    for s in windows:
        a, b = int((s - WARMUP_S) * 1000 // BIN_MS), int((s + HORIZON_S) * 1000 // BIN_MS)
        bp, ap, d, v, ok = (cache[k][a:b] for k in ("bid_px", "ask_px", "depth", "volume", "valid"))
        for side, px in ((0, bp), (1, ap)):
            same = (px[1:] == px[:-1]) & ok[1:] & ok[:-1]
            prev, nxt = d[:-1, side].astype(float), d[1:, side].astype(float)
            diff = nxt - prev
            traded = v[1:, 1 - side].astype(float)
            for k in range(d.shape[2]):
                inc = same & (diff[:, k] > 0)
                add_level.extend([k] * int(inc.sum()))
                add_size.extend(diff[inc, k].tolist())
                dec = same & (diff[:, k] < 0)
                if k == 0:
                    unexplained = np.maximum(-diff[dec, k] - traded[dec], 0.0)
                else:
                    unexplained = -diff[dec, k]
                keep = unexplained > 0
                cancel_level.extend([k] * int(keep.sum()))
                cancel_frac.extend(np.clip(unexplained[keep] / np.maximum(prev[dec, k][keep], 1e-9), 0, 1).tolist())
    cap = 200_000
    return {"add_level": np.asarray(add_level[:cap], float), "add_size": np.asarray(add_size[:cap], float),
            "cancel_level": np.asarray(cancel_level[:cap], float), "cancel_frac": np.asarray(cancel_frac[:cap], float)}


def training_frame(caches: list[dict]) -> EventFrame:
    parts, samples = [], {"add_level": [], "add_size": [], "cancel_level": [], "cancel_frac": [], "market_size": []}
    for c in caches:
        for s in WINDOW_STARTS_S:
            a, b = int((s - WARMUP_S) * 1000 // BIN_MS), int((s + HORIZON_S) * 1000 // BIN_MS)
            parts.append(cache_frame(c, a, b))
        bs = book_samples(c)
        for k, v in bs.items():
            samples[k].append(v)
        samples["market_size"].append(np.asarray(c["trade_q"], float)[: 200_000])
    def cat(name):
        return np.concatenate([getattr(p, name) for p in parts])
    mid = np.concatenate([p.mid[:-1] for p in parts] + [parts[-1].mid[-1:]])
    valid = cat("valid").copy()
    edges = np.cumsum([len(p) for p in parts])[:-1]
    valid[edges] = False                        # never model a jump across window boundaries
    frame = EventFrame(cat("counts"), cat("state"), cat("bid_px"), cat("ask_px"), cat("depth"), cat("volume"), mid,
                       valid, samples={k: np.concatenate(v) if v else np.zeros(0) for k, v in samples.items()},
                       meta={"source": "historical-training", "tick": float(caches[0]["tick"])})
    return frame


def book_model(frame: EventFrame) -> BookModel:
    mid_ticks = int(np.median(frame.mid[frame.mid > 0])) if np.any(frame.mid > 0) else 4000
    return BookModel.fit(frame, mid_ticks=max(mid_ticks, 100))


def fit_family(name: str, frame: EventFrame, book: BookModel | None = None):
    cls = {**FAMILIES, **STATIC_ECOLOGY}[name]
    book = book or book_model(frame)
    g = cls(None, book).fit(frame)
    g.book = book
    return g


def median_touch(frame: EventFrame) -> float:
    d = frame.depth[frame.valid, :, 0].astype(float)
    return float(np.median(d[d > 0])) if (d > 0).any() else 1.0


def probe_responses(generator, *, touch: float, seed: int, probes: int = 40, warmup: int = 600,
                    spacing: int = 400) -> dict:
    """Mean signed mid move (bps) and hit-side touch-depth recovery fraction after injected buy market orders."""
    out = {}
    for r in REL_SIZES:
        sim = MarketSim(generator, generator.book, seed=int(seed * 1000 + r * 10), record=True)
        qty = max(1, int(round(r * touch)))
        times = [warmup + k * spacing for k in range(probes)]
        resp = {h: [] for h in HORIZONS_S}
        rec = {h: [] for h in HORIZONS_S}
        pending = []
        for b in range(warmup + probes * spacing + 400):
            schedule = None
            if b in times:
                pre_mid, pre_depth = sim.book.mid(), float(sim.book.aq[0])
                schedule = [(sim.t + 1e-6, 0, (lambda q=qty: sim.book.market(BUY, q, sim.t, strategy=True)))]
                pending.append((b, pre_mid, pre_depth))
            sim.run_bin(schedule)
            for b0, m0, d0 in pending:
                for h in HORIZONS_S:
                    if b == b0 + int(h * 1000 // BIN_MS):
                        resp[h].append((sim.book.mid() - m0) / max(m0, 1e-9) * 1e4)
                        rec[h].append(float(sim.book.aq[0]) / max(d0, 1e-9))
        out[str(r)] = {"mid_bps": {str(h): float(np.mean(resp[h])) if resp[h] else None for h in HORIZONS_S},
                       "depth_recovery": {str(h): float(np.mean(rec[h])) if rec[h] else None for h in HORIZONS_S}}
    return out
