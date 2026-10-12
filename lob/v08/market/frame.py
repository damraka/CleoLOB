"""EventFrame: the binned (100 ms) representation shared by simulation and historical data.

Rows are bins. For each bin the frame stores the six aggregate event counts
(``MARKS``), the book state at the bin start (``FEATURES``), touch prices and the
visible depth snapshot (what an aggregate-L2 observer sees), traded volume per
aggressor side and the mid at every bin boundary. Book-model samples (add levels
and sizes, cancel levels and fractions, market sizes) are kept for fitting the
book model. ``valid`` marks bins whose endpoints are valid; invalid bins are
never dropped silently: every consumer receives the mask and counts them.

``summaries`` maps a frame to the registered summary vector used by SBI,
posterior predictive checks, OOD scores and regime features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

BIN_S = 0.1
MARKS = ("market_buy", "market_sell", "add_bid", "add_ask", "cancel_bid", "cancel_ask")
FEATURES = ("spread_ticks", "imbalance5", "log_l1_depth", "bid_q0", "ask_q0", "abs_ret10")
VISIBLE = 10
SUMMARY_NAMES = (
    "rate_market", "rate_add", "rate_cancel", "market_imbalance",
    "disp_market", "disp_add", "disp_cancel",
    "acf1_market", "acf10_market",
    "spread_mean", "p_spread1", "imbalance_sd", "log_depth_mean",
    "vol_1s", "vol_10s", "vol_ratio",
    "trade_cluster", "mean_market_size", "depletion_rate", "ret_acf1")


@dataclass
class EventFrame:
    counts: np.ndarray                 # (n, 6) int
    state: np.ndarray                  # (n, F) features at bin start
    bid_px: np.ndarray                 # (n,) int touch at bin start
    ask_px: np.ndarray                 # (n,)
    depth: np.ndarray                  # (n, 2, VISIBLE) int visible sizes at bin start (bid, ask)
    volume: np.ndarray                 # (n, 2) traded volume (buy aggressor, sell aggressor)
    mid: np.ndarray                    # (n + 1,) mid in ticks at bin boundaries
    valid: np.ndarray                  # (n,) bool
    samples: dict = field(default_factory=dict)   # book-model samples
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.counts)
        if self.counts.shape != (n, len(MARKS)):
            raise ValueError("counts must have shape (n, 6)")
        for name, shape in (("state", (n, len(FEATURES))), ("bid_px", (n,)), ("ask_px", (n,)),
                            ("depth", (n, 2, VISIBLE)), ("volume", (n, 2)), ("mid", (n + 1,)), ("valid", (n,))):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")

    def __len__(self) -> int:
        return len(self.counts)

    @property
    def duration(self) -> float:
        return len(self) * BIN_S

    def slice(self, start: int, stop: int) -> EventFrame:
        stop = min(stop, len(self))
        return EventFrame(self.counts[start:stop], self.state[start:stop], self.bid_px[start:stop],
                          self.ask_px[start:stop], self.depth[start:stop], self.volume[start:stop],
                          self.mid[start:stop + 1], self.valid[start:stop], self.samples,
                          {**self.meta, "t0": float(self.meta.get("t0", 0.0)) + start * BIN_S})

    @property
    def t0(self) -> float:
        """Time of the first bin start on the clock of ``samples['trade_t']``."""
        return float(self.meta.get("t0", 0.0))

    def trades(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(time, price ticks, qty, aggressor sign) of prints inside this frame, if recorded."""
        s = self.samples
        if "trade_t" not in s:
            return (np.zeros(0),) * 4
        t = np.asarray(s["trade_t"], float)
        keep = (t >= self.t0) & (t < self.t0 + len(self) * BIN_S)
        return t[keep], np.asarray(s["trade_px"], float)[keep], np.asarray(s["trade_q"], float)[keep],             np.asarray(s["trade_s"], float)[keep]

    def windows(self, size: int) -> list[EventFrame]:
        return [self.slice(i, i + size) for i in range(0, len(self) - size + 1, size)]

    def to_arrays(self) -> dict:
        out = {k: getattr(self, k) for k in ("counts", "state", "bid_px", "ask_px", "depth", "volume", "mid", "valid")}
        out.update({f"s__{k}": np.asarray(v) for k, v in self.samples.items()})
        return out

    @classmethod
    def from_arrays(cls, arrays, meta: dict | None = None) -> EventFrame:
        samples = {k[3:]: np.asarray(arrays[k]) for k in arrays if k.startswith("s__")}
        return cls(*(np.asarray(arrays[k]) for k in ("counts", "state", "bid_px", "ask_px", "depth", "volume", "mid",
                                                     "valid")), samples=samples, meta=meta or {})


def features(bid_px: int, ask_px: int, bq, aq, mids_recent: list[float]) -> np.ndarray:
    """Book state at a bin start (same definition for simulation and history)."""
    b5, a5 = float(sum(bq[:5])), float(sum(aq[:5]))
    imbalance = (b5 - a5) / (b5 + a5) if b5 + a5 > 0 else 0.0
    l1 = max(float(bq[0] + aq[0]), 1e-6)
    ret = abs(mids_recent[-1] - mids_recent[0]) if len(mids_recent) > 1 else 0.0
    return np.asarray([min(float(ask_px - bid_px), 10.0), imbalance, math.log(l1), float(bq[0]), float(aq[0]), ret])


def _acf(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 2:
        return 0.0
    a, b = x[:-lag] - x.mean(), x[lag:] - x.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator > 0 else 0.0


def _dispersion(x: np.ndarray) -> float:
    m = float(x.mean())
    return float(x.var() / m) if m > 0 else 1.0


def summaries(frame: EventFrame) -> np.ndarray:
    """Registered summary vector (``SUMMARY_NAMES``) over the valid bins of a frame."""
    v = frame.valid.astype(bool)
    if v.sum() < 20:
        raise ValueError("frame has fewer than 20 valid bins")
    c = frame.counts[v].astype(float)
    market, add, cancel = c[:, 0] + c[:, 1], c[:, 2] + c[:, 3], c[:, 4] + c[:, 5]
    s = frame.state[v]
    mid = frame.mid.astype(float)
    r1 = mid[10::10] - mid[:-10:10] if len(mid) > 20 else np.zeros(2)
    r10 = mid[100::100] - mid[:-100:100] if len(mid) > 200 else r1 * math.sqrt(10)
    vol1, vol10 = float(np.std(r1)), float(np.std(r10))
    trade = (market > 0).astype(float)
    sizes = frame.samples.get("market_size")
    mean_size = float(np.mean(sizes)) if sizes is not None and len(sizes) else float(
        frame.volume[v].sum() / max(market.sum(), 1.0))
    touch_moves = np.abs(np.diff(frame.bid_px.astype(float))) + np.abs(np.diff(frame.ask_px.astype(float)))
    step = np.diff(mid)
    out = [market.mean(), add.mean(), cancel.mean(),
           float((c[:, 0].sum() - c[:, 1].sum()) / max(c[:, :2].sum(), 1.0)),
           _dispersion(market), _dispersion(add), _dispersion(cancel),
           _acf(market, 1), _acf(market, 10),
           float(s[:, 0].mean()), float((s[:, 0] <= 1.0).mean()), float(s[:, 1].std()), float(s[:, 2].mean()),
           vol1, vol10, float(vol10 / (vol1 * math.sqrt(10))) if vol1 > 0 else 1.0,
           _acf(trade, 1), mean_size, float((touch_moves > 0).mean()) if len(touch_moves) else 0.0,
           _acf(step[step != 0], 1) if np.count_nonzero(step) > 5 else 0.0]
    return np.nan_to_num(np.asarray(out, float), nan=0.0, posinf=0.0, neginf=0.0)


def summary_dict(frame: EventFrame) -> dict:
    return dict(zip(SUMMARY_NAMES, (float(x) for x in summaries(frame))))
