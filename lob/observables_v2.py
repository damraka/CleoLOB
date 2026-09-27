"""One measurement operator for historical and simulated books (calibration v2 / M3).

Both sources are reduced to the same *tape*: top-5 levels sampled every
``SAMPLE_DT`` seconds plus aggressive trade events (prints grouped by source
time and aggressor). Every observable is computed from the tape by the same
code, so historical and simulated values are comparable by construction. Net
level changes between samples are the only order-flow evidence used; aggregate
data cannot separate simultaneous additions and cancellations, so the
cancellation and addition observables are explicitly *net* proxies.

Quantities are in lots: native depth divided by a train-only scale mapping the
development median top-5 side depth to ``DEPTH_LOTS`` lots.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

SAMPLE_DT = 0.1
DEPTH_LOTS = 1000.0
QUANTILES = (0.1, 0.5, 0.9)
FAMILIES = ("spread", "depth", "imbalance", "volatility", "trade_intensity", "update_intensity",
            "cancellation_intensity", "sizes", "interarrival", "resilience", "autocorrelation", "dependence")
GATE = 0.60
ERROR_CAP = 10.0
RECOVERY_CENSOR_S = 30.0


@dataclass
class Tape:
    """Top-5 samples and aggressive events. Prices in native units; quantities in lots."""

    t: np.ndarray            # sample times, seconds (float64), regular grid
    bp: np.ndarray           # (n, 5) bid prices, NaN where absent
    bq: np.ndarray           # (n, 5) bid quantities (lots), 0 where absent
    ap: np.ndarray
    aq: np.ndarray
    trade_t: np.ndarray      # aggressive event times (s)
    trade_p: np.ndarray      # event VWAP price (native)
    trade_q: np.ndarray      # event size (lots)
    trade_s: np.ndarray      # +1 buy aggressor, -1 sell aggressor

    def __post_init__(self) -> None:
        n = len(self.t)
        for name in ("bp", "bq", "ap", "aq"):
            if getattr(self, name).shape != (n, 5):
                raise ValueError(f"{name} must have shape (n, 5)")
        if n > 1 and not np.allclose(np.diff(self.t), SAMPLE_DT, atol=1e-6):
            raise ValueError("tape samples must be on the regular grid")

    @property
    def valid(self) -> np.ndarray:
        return (np.isfinite(self.bp[:, 0]) & np.isfinite(self.ap[:, 0]) & (self.bq[:, 0] > 0)
                & (self.aq[:, 0] > 0) & (self.bp[:, 0] < self.ap[:, 0]))

    @property
    def mid(self) -> np.ndarray:
        return (self.bp[:, 0] + self.ap[:, 0]) / 2

    def block(self, start: float, stop: float) -> Tape:
        s = (self.t >= start) & (self.t < stop)
        e = (self.trade_t >= start) & (self.trade_t < stop)
        return Tape(self.t[s], self.bp[s], self.bq[s], self.ap[s], self.aq[s],
                    self.trade_t[e], self.trade_p[e], self.trade_q[e], self.trade_s[e])


def group_prints(times: np.ndarray, prices: np.ndarray, sizes: np.ndarray, sides: np.ndarray, keys: np.ndarray):
    """Group prints sharing one key (source time + aggressor or taker order) into events."""
    if not len(times):
        return times, prices, sizes, sides
    order = np.lexsort((sides, keys))
    keys_sorted, sides_sorted = keys[order], sides[order]
    boundary = np.r_[True, (keys_sorted[1:] != keys_sorted[:-1]) | (sides_sorted[1:] != sides_sorted[:-1])]
    group = np.cumsum(boundary) - 1
    count = group[-1] + 1
    size = np.bincount(group, weights=sizes[order], minlength=count)
    notional = np.bincount(group, weights=sizes[order] * prices[order], minlength=count)
    first = np.flatnonzero(boundary)
    event_t = times[order][first]
    event_s = sides_sorted[first]
    result = np.argsort(event_t, kind="stable")
    return event_t[result], (notional / size)[result], size[result], event_s[result]


# ----------------------------------------------------------------------------- measurement

def _one_second(tape: Tape) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the 1-second subgrid and their validity."""
    step = int(round(1 / SAMPLE_DT))
    index = np.arange(0, len(tape.t), step)
    return index, tape.valid[index]


def measure(tape: Tape) -> dict[str, np.ndarray]:
    """Raw observable samples; empty arrays when a quantity cannot be formed."""
    valid = tape.valid
    mid = tape.mid
    out: dict[str, np.ndarray] = {}
    idx, ok = _one_second(tape)
    m1 = mid[idx]
    spread = (tape.ap[:, 0] - tape.bp[:, 0]) / mid * 1e4
    out["spread_bps"] = spread[idx][ok]
    out["l1_depth"] = np.r_[tape.bq[idx][ok, 0], tape.aq[idx][ok, 0]]
    d5b, d5a = tape.bq.sum(axis=1), tape.aq.sum(axis=1)
    out["depth5"] = np.r_[d5b[idx][ok], d5a[idx][ok]]
    imbalance = (d5b - d5a) / np.where(d5b + d5a > 0, d5b + d5a, np.nan)
    out["imbalance5"] = imbalance[idx][ok]
    pair = ok[1:] & ok[:-1]
    returns = np.full(len(idx), np.nan)
    returns[1:] = np.where(pair, np.log(m1[1:] / m1[:-1]) * 1e4, np.nan)
    out["return_1s_bps"] = returns[np.isfinite(returns)]
    blocks = []
    for start in range(0, len(returns) - 59, 60):
        window = returns[start:start + 60]
        if np.isfinite(window).sum() >= 50:
            blocks.append(float(np.nanstd(window)))
    out["rv_60s_bps"] = np.asarray(blocks)
    duration = max(len(tape.t) * SAMPLE_DT, SAMPLE_DT)
    valid_seconds = max(float(valid.sum()) * SAMPLE_DT, SAMPLE_DT)
    seconds = int(math.ceil(duration))
    if len(tape.trade_t):
        bins = np.clip(np.floor(tape.trade_t - tape.t[0]).astype(int), 0, seconds - 1)
        counts = np.bincount(bins, minlength=seconds)
    else:
        counts = np.zeros(seconds)
    out["trade_counts_1s"] = counts.astype(float)
    out["trade_rate"] = np.asarray([len(tape.trade_t) / duration])
    # top-of-book change rate on the sampling grid
    both = valid[1:] & valid[:-1]
    changed = ((tape.bp[1:, 0] != tape.bp[:-1, 0]) | (tape.bq[1:, 0] != tape.bq[:-1, 0])
               | (tape.ap[1:, 0] != tape.ap[:-1, 0]) | (tape.aq[1:, 0] != tape.aq[:-1, 0])) & both
    out["top_change_rate"] = np.asarray([changed.sum() / max(both.sum() * SAMPLE_DT, SAMPLE_DT)])
    # net same-price level changes in the top five, minus prints at that price
    add_sizes, cancel_volume = [], 0.0
    trade_bin = np.searchsorted(tape.t, tape.trade_t, side="left")
    printed: dict[tuple[int, float], float] = {}
    for b, price, size in zip(trade_bin, tape.trade_p, tape.trade_q):
        printed[(int(b), round(float(price), 10))] = printed.get((int(b), round(float(price), 10)), 0.0) + float(size)
    for k in np.flatnonzero(both):
        for prices, qtys in ((tape.bp, tape.bq), (tape.ap, tape.aq)):
            before = dict(zip(np.round(prices[k], 10), qtys[k]))
            after = dict(zip(np.round(prices[k + 1], 10), qtys[k + 1]))
            for price in before.keys() & after.keys():
                if not math.isfinite(price):
                    continue
                delta = after[price] - before[price]
                if delta > 0:
                    add_sizes.append(delta)
                elif delta < 0:
                    traded = printed.get((k + 1, price), 0.0)
                    cancel_volume += max(0.0, -delta - traded)
    out["addition_sizes"] = np.asarray(add_sizes)
    out["net_cancel_rate"] = np.asarray([cancel_volume / valid_seconds])
    out["addition_rate"] = np.asarray([len(add_sizes) / valid_seconds])
    out["trade_sizes"] = tape.trade_q.astype(float)
    gaps = np.diff(tape.trade_t)
    out["interarrival_s"] = gaps[gaps > 0]
    out["recovery_s"], out["recovered"] = _recovery(tape)
    r = returns
    out["acf_return"] = np.asarray([_acf(r, 1)])
    out["acf_abs_return"] = np.asarray([_acf(np.abs(r), 1)])
    sp1 = spread[idx]
    out["acf_spread"] = np.asarray([_acf(np.where(ok, sp1, np.nan), 1)])
    depth_total = (d5b + d5a)[idx]
    out["corr_spread_depth"] = np.asarray([_corr(np.where(ok, sp1, np.nan), np.where(ok, depth_total, np.nan))])
    next_return = np.r_[returns[1:], np.nan]
    out["corr_imbalance_next_return"] = np.asarray([_corr(np.where(ok, imbalance[idx], np.nan), next_return)])
    signed = np.zeros(len(idx))
    if len(tape.trade_t):
        second = np.floor(tape.trade_t - tape.t[0]).astype(int)
        np.add.at(signed, np.clip(second, 0, len(idx) - 1), tape.trade_s * tape.trade_q)
    out["corr_flow_return"] = np.asarray([_corr(signed, returns)])
    return out


def _acf(x: np.ndarray, lag: int) -> float:
    a, b = x[lag:], x[:-lag]
    keep = np.isfinite(a) & np.isfinite(b)
    return _corr(a[keep], b[keep])


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 10:
        return float("nan")
    a, b = x[keep], y[keep]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _recovery(tape: Tape) -> tuple[np.ndarray, np.ndarray]:
    """Time for opposite L1 depth to regain 90% of its pre-event level after a depleting event."""
    times, recovered = [], []
    if not len(tape.trade_t) or len(tape.t) < 2:
        return np.asarray(times), np.asarray(recovered)
    valid = tape.valid
    index = np.searchsorted(tape.t, tape.trade_t, side="right") - 1
    for k, size, sign, when in zip(index, tape.trade_q, tape.trade_s, tape.trade_t):
        if k < 0 or not valid[k]:
            continue
        depth = tape.aq[:, 0] if sign > 0 else tape.bq[:, 0]
        pre = depth[k]
        if pre <= 0 or size < 0.5 * pre:
            continue
        stop = min(len(tape.t), k + 1 + int(RECOVERY_CENSOR_S / SAMPLE_DT))
        window = depth[k + 1:stop]
        hit = np.flatnonzero((window >= 0.9 * pre) & valid[k + 1:stop])
        if len(hit):
            times.append(tape.t[k + 1 + hit[0]] - when)
            recovered.append(1.0)
        else:
            recovered.append(0.0)
    return np.asarray(times), np.asarray(recovered)


# ----------------------------------------------------------------------------- summaries & errors

def _quantiles(values: np.ndarray) -> list[float] | None:
    values = values[np.isfinite(values)]
    return np.quantile(values, QUANTILES).tolist() if len(values) else None


def summarize(samples: dict[str, np.ndarray]) -> dict:
    s: dict = {}
    for name in ("spread_bps", "l1_depth", "depth5", "imbalance5", "return_1s_bps", "rv_60s_bps", "trade_sizes",
                 "addition_sizes", "interarrival_s", "recovery_s"):
        s[name] = {"quantiles": _quantiles(samples[name]), "n": int(np.isfinite(samples[name]).sum())}
    returns = samples["return_1s_bps"]
    s["return_std_bps"] = float(returns.std()) if len(returns) else None
    s["return_nonzero_fraction"] = float(np.mean(returns != 0)) if len(returns) else None
    counts = samples["trade_counts_1s"]
    s["trade_rate"] = float(samples["trade_rate"][0])
    s["trade_dispersion_index"] = float(counts.var() / counts.mean()) if counts.mean() > 0 else None
    for name in ("top_change_rate", "net_cancel_rate", "addition_rate", "acf_return", "acf_abs_return", "acf_spread",
                 "corr_spread_depth", "corr_imbalance_next_return", "corr_flow_return"):
        value = float(samples[name][0])
        s[name] = value if math.isfinite(value) else None
    s["recovered_fraction"] = float(samples["recovered"].mean()) if len(samples["recovered"]) else None
    s["recovery_events"] = int(len(samples["recovered"]))
    return s


def _q_error(obs: dict, sim: dict, floor: float, absolute: bool = False) -> float | None:
    if obs["quantiles"] is None:
        return None
    if sim["quantiles"] is None:
        return ERROR_CAP
    o, m = np.asarray(obs["quantiles"]), np.asarray(sim["quantiles"])
    scale = 1.0 if absolute else max(abs(o[1]), floor)
    return float(min(ERROR_CAP, np.mean(np.abs(o - m)) / scale))


def _rel(obs: float | None, sim: float | None, floor: float) -> float | None:
    if obs is None:
        return None
    if sim is None:
        return ERROR_CAP
    return float(min(ERROR_CAP, abs(sim - obs) / max(abs(obs), floor)))


def _abs(obs: float | None, sim: float | None) -> float | None:
    if obs is None:
        return None
    if sim is None:
        return ERROR_CAP
    return float(min(ERROR_CAP, abs(sim - obs)))


def family_errors(target: dict, simulated: dict) -> dict:
    """Dimensionless per-family errors (max over components); None where history lacks the quantity."""
    components = {
        "spread": [_q_error(target["spread_bps"], simulated["spread_bps"], 0.05)],
        "depth": [_q_error(target["l1_depth"], simulated["l1_depth"], 1.0),
                  _q_error(target["depth5"], simulated["depth5"], 1.0)],
        "imbalance": [_q_error(target["imbalance5"], simulated["imbalance5"], 1.0, absolute=True)],
        "volatility": [_rel(target["return_std_bps"], simulated["return_std_bps"], 0.05),
                       _abs(target["return_nonzero_fraction"], simulated["return_nonzero_fraction"]),
                       _q_error(target["rv_60s_bps"], simulated["rv_60s_bps"], 0.05)],
        "trade_intensity": [_rel(target["trade_rate"], simulated["trade_rate"], 1e-3),
                            _rel(target["trade_dispersion_index"], simulated["trade_dispersion_index"], 0.1)],
        "update_intensity": [_rel(target["top_change_rate"], simulated["top_change_rate"], 1e-3)],
        "cancellation_intensity": [_rel(target["net_cancel_rate"], simulated["net_cancel_rate"], 1e-3),
                                   _rel(target["addition_rate"], simulated["addition_rate"], 1e-3)],
        "sizes": [_q_error(target["trade_sizes"], simulated["trade_sizes"], 0.01),
                  _q_error(target["addition_sizes"], simulated["addition_sizes"], 0.01)],
        "interarrival": [_q_error(target["interarrival_s"], simulated["interarrival_s"], 0.01)],
        "resilience": [_q_error(target["recovery_s"], simulated["recovery_s"], 0.5),
                       _abs(target["recovered_fraction"], simulated["recovered_fraction"])],
        "autocorrelation": [_abs(target["acf_return"], simulated["acf_return"]),
                            _abs(target["acf_abs_return"], simulated["acf_abs_return"]),
                            _abs(target["acf_spread"], simulated["acf_spread"])],
        "dependence": [_abs(target["corr_spread_depth"], simulated["corr_spread_depth"]),
                       _abs(target["corr_imbalance_next_return"], simulated["corr_imbalance_next_return"]),
                       _abs(target["corr_flow_return"], simulated["corr_flow_return"])],
    }
    result = {}
    for family in FAMILIES:
        values = [v for v in components[family] if v is not None]
        error = max(values) if values else None
        result[family] = {"error": error, "passed": None if error is None else error <= GATE,
                          "components": components[family]}
    return result


def loss(errors: dict, families: tuple[str, ...] = FAMILIES) -> float:
    values = [errors[f]["error"] for f in families if errors[f]["error"] is not None]
    return float(np.mean(values)) if values else float("nan")
