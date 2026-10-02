"""v0.6 realism observables: one measurement operator, additive block sketches.

``measure(tape)`` reduces one block tape (normally 600 s) to raw samples. A
*design* frozen from development data alone (``build_design``) fixes, for every
component, the transform, histogram bins, normalization scale, reference
marginals, thresholds and quintile edges. ``sketch(raw, design)`` turns raw
samples into additive sufficient statistics (histogram counts, moment sums,
count matrices), so pooling blocks or seeds and block-bootstrap resampling are
exact sums. History and simulation pass through the same code.

Capability: every observable needs only aggregate L2 (top 10 displayed levels)
plus aggressor-signed trade prints. Order-level quantities (individual order
size, modifies, queue position, hidden size) are NOT_AVAILABLE and are listed in
``NOT_AVAILABLE``. Additions and cancellations are *net* displayed-level
changes between 100 ms samples; they are proxies, not order events.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .tape import SAMPLE_DT, BookTape

FAMILIES = ("book_state", "event_activity", "sizes", "returns", "tail", "temporal", "event_process", "dependence",
            "resilience")
BINS = 256
STATES = ("quiet", "trade", "add", "cancel", "add_cancel")
RECOVERY_CENSOR_S = 30.0
NOT_AVAILABLE = {
    "modify_intensity": "aggregate L2 cannot separate modifies from cancel+add",
    "order_size": "individual order sizes require order-level (MBO) data",
    "queue_position": "aggregate L2 has no order identity or FIFO position",
    "hidden_liquidity": "not displayed in L2",
}


@dataclass(frozen=True)
class Component:
    name: str
    family: str
    kind: str            # dist | prob | exc | acf | pit | corr | ratio | moment | markov | cond | mi | hill | tailq | pacf
    source: str
    transform: str = "linear"   # linear | log | log1p (dist only)
    floor: float | str = 0.1    # normalization floor (transformed units); "tick" / "quarter_tick" resolved at design time
    min_n: int = 50
    error: str = "w1"           # w1 | abs | logratio | tv | rows | cond | mi_logratio | tailq
    units: str = ""
    definition: str = ""

    @property
    def capability(self) -> str:
        return "aggregate_l2 + trade prints"


def _c(*args, **kwargs) -> Component:
    return Component(*args, **kwargs)


COMPONENTS: tuple[Component, ...] = (
    # book state ------------------------------------------------------------------------------------
    _c("spread", "book_state", "dist", "spread_bps", floor="tick", units="bps",
       definition="(best ask - best bid) / mid on the 1 s grid"),
    _c("depth_l1", "book_state", "dist", "depth_l1", "log", units="log lots", definition="best-level quantity per side"),
    _c("depth5", "book_state", "dist", "depth5", "log", units="log lots", definition="top-5 side depth"),
    _c("depth10", "book_state", "dist", "depth10", "log", units="log lots", definition="top-10 side depth"),
    _c("side_asymmetry", "book_state", "dist", "side_asymmetry", units="log ratio", definition="log(bid top-5 / ask top-5)"),
    _c("imbalance5", "book_state", "dist", "imbalance5", floor=0.05, units="fraction",
       definition="(bid5 - ask5) / (bid5 + ask5)"),
    _c("concentration", "book_state", "dist", "concentration", floor=0.02, units="fraction",
       definition="(L1 bid + L1 ask) / (top-10 bid + top-10 ask)"),
    _c("book_slope", "book_state", "dist", "book_slope", units="log ratio", definition="log(top-5 side depth / L1 side depth)"),
    _c("level_gap", "book_state", "dist", "level_gap_ticks", floor=0.5, units="ticks",
       definition="price gap between consecutive displayed levels 1-5 per side"),
    _c("microprice_deviation", "book_state", "dist", "microprice_dev_bps", floor="quarter_tick", units="bps",
       definition="(L1-weighted microprice - mid) / mid"),
    # event activity --------------------------------------------------------------------------------
    _c("trade_count_10s", "event_activity", "dist", "trade_count_10s", "log1p", units="log1p count",
       definition="aggressive events per 10 s window (>= 80% valid samples)"),
    _c("book_change_count_10s", "event_activity", "dist", "change_count_10s", "log1p", units="log1p count",
       definition="100 ms steps with any top-10 change per 10 s window"),
    _c("add_count_10s", "event_activity", "dist", "add_count_10s", "log1p", units="log1p count",
       definition="net displayed additions per 10 s window"),
    _c("cancel_count_10s", "event_activity", "dist", "cancel_count_10s", "log1p", units="log1p count",
       definition="net displayed decreases not explained by prints per 10 s window"),
    _c("trade_interarrival", "event_activity", "dist", "trade_interarrival_s", "log", units="log s",
       definition="positive gaps between aggressive events within a block"),
    # sizes -----------------------------------------------------------------------------------------
    _c("trade_size", "sizes", "dist", "trade_size", "log", units="log lots", definition="aggressive event size"),
    _c("add_size", "sizes", "dist", "add_size", "log", units="log lots", definition="net displayed addition size"),
    _c("cancel_size", "sizes", "dist", "cancel_size", "log", units="log lots",
       definition="net displayed decrease not explained by prints"),
    # returns ---------------------------------------------------------------------------------------
    _c("return_1s", "returns", "dist", "r1_bps", floor="tick", units="bps", definition="log mid return over 1 s"),
    _c("return_10s", "returns", "dist", "r10_bps", floor="tick", units="bps", definition="non-overlapping 10 s mid returns"),
    _c("return_60s", "returns", "dist", "r60_bps", floor="tick", units="bps", definition="non-overlapping 60 s mid returns"),
    _c("realized_vol_60s", "returns", "dist", "rv60_bps", "log1p", units="log1p bps",
       definition="std of 1 s returns in non-overlapping 60 s windows (>= 50 returns)"),
    _c("nonzero_return_fraction", "returns", "prob", "nonzero_r1", min_n=300, error="abs", units="probability",
       definition="fraction of nonzero 1 s returns"),
    # tail ------------------------------------------------------------------------------------------
    _c("exceed_spread_p99", "tail", "exc", "spread_bps:gt:0.99", min_n=300, error="logratio", units="probability"),
    _c("exceed_thin_depth5_p1", "tail", "exc", "depth5:lt:0.01", min_n=300, error="logratio", units="probability"),
    _c("exceed_vol_p99", "tail", "exc", "rv60_bps:gt:0.99", min_n=30, error="logratio", units="probability"),
    _c("exceed_trade_size_p99", "tail", "exc", "trade_size:gt:0.99", min_n=50, error="logratio", units="probability"),
    _c("exceed_cancel_burst_p99", "tail", "exc", "cancel_count_10s:gt:0.99", min_n=60, error="logratio",
       units="probability"),
    _c("exceed_imbalance_p99", "tail", "exc", "abs_imbalance5:gt:0.99", min_n=300, error="logratio", units="probability"),
    _c("exceed_return_10s_p999", "tail", "exc", "abs_r10_bps:gt:0.999", min_n=300, error="logratio",
       units="probability", definition="large 10 s price moves (jumps)"),
    _c("tailq_spread", "tail", "tailq", "spread", error="tailq", units="normalized"),
    _c("tailq_depth5", "tail", "tailq", "depth5", error="tailq", units="normalized"),
    _c("tailq_trade_size", "tail", "tailq", "trade_size", error="tailq", units="normalized"),
    _c("tailq_return_10s", "tail", "tailq", "return_10s", error="tailq", units="normalized"),
    _c("tailq_realized_vol", "tail", "tailq", "realized_vol_60s", error="tailq", units="normalized"),
    _c("hill_trade_size", "tail", "hill", "trade_size", min_n=50, error="logratio", units="tail index",
       definition="Hill index above the development 95th percentile"),
    _c("hill_return_10s", "tail", "hill", "abs_r10_bps", min_n=50, error="logratio", units="tail index"),
    # temporal --------------------------------------------------------------------------------------
    *(_c(f"acf_return_l{k}", "temporal", "acf", f"r1:{k}", min_n=300, error="abs", units="correlation")
      for k in (1, 2, 5, 10)),
    *(_c(f"acf_abs_return_l{k}", "temporal", "acf", f"abs_r1:{k}", min_n=300, error="abs", units="correlation")
      for k in (1, 5, 10, 30)),
    *(_c(f"acf_spread_l{k}", "temporal", "acf", f"spread:{k}", min_n=300, error="abs", units="correlation")
      for k in (1, 10)),
    *(_c(f"acf_depth5_l{k}", "temporal", "acf", f"depth5:{k}", min_n=300, error="abs", units="correlation")
      for k in (1, 10, 60)),
    *(_c(f"acf_imbalance_l{k}", "temporal", "acf", f"imbalance:{k}", min_n=300, error="abs", units="correlation")
      for k in (1, 10, 60)),
    *(_c(f"acf_signed_volume_10s_l{k}", "temporal", "acf", f"flow10:{k}", min_n=60, error="abs", units="correlation")
      for k in (1, 2, 5)),
    *(_c(f"acf_event_count_10s_l{k}", "temporal", "acf", f"count10:{k}", min_n=60, error="abs", units="correlation")
      for k in (1, 6)),
    _c("pacf_return_l2", "temporal", "pacf", "r1", min_n=300, error="abs", units="partial correlation"),
    _c("tight_spread_duration", "temporal", "dist", "tight_run_s", "log", units="log s",
       definition="completed runs of spread <= 1.5 ticks on the 100 ms grid"),
    # event process ---------------------------------------------------------------------------------
    _c("event_state_proportions", "event_process", "markov", "markov", min_n=300, error="tv", units="TV distance"),
    _c("event_transition_rows", "event_process", "markov", "markov", min_n=300, error="rows", units="weighted TV"),
    _c("p_cancel_after_trade", "event_process", "markov", "markov:trade>cancel", min_n=30, error="abs",
       units="probability"),
    _c("p_trade_after_cancel", "event_process", "markov", "markov:cancel>trade", min_n=30, error="abs",
       units="probability"),
    _c("fano_trade_1s", "event_process", "ratio", "trade_count_1s", min_n=300, error="logratio", units="var/mean"),
    _c("fano_trade_10s", "event_process", "ratio", "trade_count_10s", min_n=30, error="logratio", units="var/mean"),
    _c("burstiness", "event_process", "moment", "trade_interarrival_s", min_n=50, error="abs",
       units="(sd - mean)/(sd + mean)"),
    _c("depletion_hazard", "event_process", "prob", "depletion", min_n=300, error="logratio", units="probability per s",
       definition="best level moved away or L1 fell below 50% within 1 s"),
    _c("replenishment_probability", "event_process", "prob", "replenished", min_n=30, error="abs", units="probability",
       definition="best level back to the pre-depletion price with >= 90% of its quantity 1 s after depletion"),
    _c("cross_excitation", "event_process", "corr", "trade1_cancel_next1", min_n=300, error="abs", units="correlation",
       definition="corr(trade count in second t, net cancellation count in second t+1); descriptive only"),
    # dependence ------------------------------------------------------------------------------------
    _c("dep_volatility_spread", "dependence", "pit", "w_rv|w_spread", min_n=30, error="abs", units="rank correlation"),
    _c("dep_volatility_depth", "dependence", "pit", "w_rv|w_depth", min_n=30, error="abs", units="rank correlation"),
    _c("dep_activity_volatility", "dependence", "pit", "w_count|w_rv", min_n=30, error="abs", units="rank correlation"),
    _c("dep_cancellation_volatility", "dependence", "pit", "w_cancel|w_rv", min_n=30, error="abs",
       units="rank correlation"),
    _c("dep_imbalance_spread", "dependence", "pit", "w_absimb|w_spread", min_n=30, error="abs", units="rank correlation"),
    _c("dep_imbalance_next_return", "dependence", "pit", "imbalance|next_r1", min_n=300, error="abs",
       units="rank correlation"),
    _c("dep_microprice_next_return", "dependence", "pit", "microdev|next_r1", min_n=300, error="abs",
       units="rank correlation"),
    _c("dep_flow_return", "dependence", "pit", "flow1|r1", min_n=300, error="abs", units="rank correlation"),
    _c("dep_depth_next_abs_return", "dependence", "pit", "depth5_total|next_abs_r1", min_n=300, error="abs",
       units="rank correlation"),
    _c("cond_return_by_imbalance", "dependence", "cond", "imbalance|next_r1", min_n=30, error="cond",
       units="development return sd"),
    _c("mi_imbalance_next_sign", "dependence", "mi", "imbalance|next_r1", min_n=300, error="mi_logratio", units="nats"),
    # resilience ------------------------------------------------------------------------------------
    _c("depth_recovery_time", "resilience", "dist", "recovery_s", "log", units="log s",
       definition="time for the hit side's L1 to regain 90% after an event removing >= 50% (30 s censoring)"),
    _c("recovered_fraction", "resilience", "prob", "recovered", min_n=30, error="abs", units="probability"),
    _c("spread_recovery_time", "resilience", "dist", "wide_run_s", "log", units="log s",
       definition="completed runs of spread > 1.5 ticks"),
    _c("depth5_recovery_ratio_5s", "resilience", "dist", "depth5_ratio_5s", units="log ratio",
       definition="log(top-5 hit-side depth 5 s after / before a depleting event)"),
)
BY_NAME = {c.name: c for c in COMPONENTS}
DIST_SOURCES = sorted({c.source for c in COMPONENTS if c.kind == "dist"})
PIT_SOURCES = sorted({part for c in COMPONENTS if c.kind == "pit" for part in c.source.split("|")})


_SAMPLING = {
    "dist": "pooled samples on development-frozen bins (256 bins in the transformed space, overflow at its mean)",
    "prob": "event and opportunity counts pooled over blocks",
    "exc": "fraction of samples beyond a development-frozen quantile threshold",
    "acf": "lagged pairs within each block (no pair crosses a block or an invalid sample), pooled moment sums",
    "pacf": "Durbin-Levinson lag-2 partial autocorrelation from pooled lag-1 and lag-2 autocorrelations",
    "pit": "Pearson correlation after the mid-rank transform of each variable by its development marginal",
    "corr": "Pearson correlation of paired samples, pooled moment sums",
    "ratio": "variance / mean of counts from pooled sums",
    "moment": "(sd - mean) / (sd + mean) from pooled sums",
    "markov": "5-state chain of consecutive valid 100 ms steps, pooled transition counts",
    "cond": "mean next 1 s return per development imbalance quintile",
    "mi": "plug-in mutual information of imbalance quintile and next-return sign (5 x 3 table)",
    "hill": "Hill estimator k / sum log(x / u) above the development 95th percentile u",
    "tailq": "1% and 99% quantile errors of the referenced distribution, normalized by its scale",
}


def _default_definition(c: Component) -> str:
    if c.definition:
        return c.definition
    if c.kind == "acf":
        series, lag = c.source.split(":")
        grid = "10 s" if series.endswith("10") else "1 s"
        return f"lag-{lag} autocorrelation of {series} on the {grid} grid"
    if c.kind == "exc":
        source, direction, q = c.source.split(":")
        return f"P({source} {'>' if direction == 'gt' else '<'} development {float(q):.1%} quantile)"
    if c.kind == "pit":
        return f"rank correlation of {c.source.replace('|', ' and ')}"
    if c.kind == "tailq":
        return f"1%/99% quantile error of {c.source}"
    if c.kind == "markov":
        if c.error == "tv":
            return "total variation between event-state occupancy distributions"
        if c.error == "rows":
            return "occupancy-weighted total variation between transition rows"
        now, nxt = c.source.split(":")[1].split(">")
        return f"P(next 100 ms state = {nxt} | current state = {now})"
    if c.kind == "ratio":
        return f"Fano factor (variance / mean) of {c.source}"
    if c.kind == "pacf":
        return "lag-2 partial autocorrelation of 1 s returns"
    if c.kind in {"cond", "mi"}:
        return _SAMPLING[c.kind]
    return c.source


def registry() -> list[dict]:
    """Machine-readable definition, units, capability, sampling and invalidity of every component."""
    rows = []
    for c in COMPONENTS:
        rows.append({**asdict(c), "definition": _default_definition(c), "capability": c.capability,
                     "sampling": _SAMPLING[c.kind],
                     "invalid_when": (f"fewer than {c.min_n} samples on either side, an undefined statistic, or a "
                                      "missing development reference; reported NOT_EVALUABLE, never imputed"),
                     "historical_validity": "100 ms samples older than 5 s or crossed books are invalid"})
    return rows


# ----------------------------------------------------------------------------- raw measurement


def _ticks(prices: np.ndarray, tick: float) -> np.ndarray:
    return np.rint(prices / tick)


def _side_changes(p0: np.ndarray, q0: np.ndarray, p1: np.ndarray, q1: np.ndarray, *, bid: bool):
    """Additions and removals between consecutive samples for one side (prices in ticks).

    Returns (additions (steps, 2L) qty, removals (steps, L) qty aligned with p0 levels).
    Only changes inside the displayed window are attributed; levels entering or
    leaving through the deep edge of the window are ignored (window shifts).
    """
    eq = p0[:, :, None] == p1[:, None, :]
    m0, m1 = eq.any(2), eq.any(1)
    q1_at_p0 = (eq * q1[:, None, :]).sum(2)
    delta = np.where(m0, q1_at_p0 - q0, 0.0)
    lo0, hi0 = np.fmin.reduce(p0, axis=1), np.fmax.reduce(p0, axis=1)
    lo1, hi1 = np.fmin.reduce(p1, axis=1), np.fmax.reduce(p1, axis=1)
    with np.errstate(invalid="ignore"):
        if bid:   # best = highest; deep edge = lowest
            new_inside = np.isfinite(p1) & ~m1 & (p1 >= lo0[:, None])
            old_inside = np.isfinite(p0) & ~m0 & (p0 >= lo1[:, None])
        else:     # best = lowest; deep edge = highest
            new_inside = np.isfinite(p1) & ~m1 & (p1 <= hi0[:, None])
            old_inside = np.isfinite(p0) & ~m0 & (p0 <= hi1[:, None])
    additions = np.concatenate([np.where(delta > 0, delta, 0.0), np.where(new_inside, q1, 0.0)], axis=1)
    removals = np.where(delta < 0, -delta, 0.0) + np.where(old_inside, q0, 0.0)
    return additions, removals


def _acf_pairs(x: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= lag:
        return np.empty(0), np.empty(0)
    a, b = x[:-lag], x[lag:]
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep]


def _runs(flag: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Lengths (samples) of completed runs of ``flag``: preceded and ended by valid non-flag samples."""
    lengths, start, after_nonflag = [], None, False
    for i, (f, ok) in enumerate(zip(flag, valid)):
        if not ok:
            start, after_nonflag = None, False
        elif f:
            if start is None and after_nonflag:
                start = i
            after_nonflag = False
        else:
            if start is not None:
                lengths.append(i - start)
            start, after_nonflag = None, True
    return np.asarray(lengths, dtype=float)


def _window_slices(n: int, step: int) -> list[slice]:
    return [slice(i, i + step) for i in range(0, n - step + 1, step)]


def measure(tape: BookTape) -> dict:
    """Raw samples of one block; every array is empty when its quantity cannot be formed."""
    n = len(tape.t)
    out: dict = {}
    if n < 20:
        raise ValueError("block too short to measure")
    valid = tape.valid
    mid = tape.mid
    tick = tape.tick
    step1 = int(round(1 / SAMPLE_DT))
    idx = np.arange(0, n, step1)
    ok = valid[idx]
    spread = (tape.ap[:, 0] - tape.bp[:, 0]) / mid * 1e4
    d1b, d1a = tape.bq[:, 0], tape.aq[:, 0]
    d5b, d5a = tape.bq[:, :5].sum(1), tape.aq[:, :5].sum(1)
    d10b, d10a = tape.bq.sum(1), tape.aq.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        imbalance = (d5b - d5a) / (d5b + d5a)
        micro = (tape.bp[:, 0] * d1a + tape.ap[:, 0] * d1b) / (d1b + d1a)
        microdev = (micro - mid) / mid * 1e4
        out["spread_bps"] = spread[idx][ok]
        out["depth_l1"] = np.r_[d1b[idx][ok], d1a[idx][ok]]
        out["depth5"] = np.r_[d5b[idx][ok], d5a[idx][ok]]
        out["depth10"] = np.r_[d10b[idx][ok], d10a[idx][ok]]
        out["side_asymmetry"] = np.log(d5b / d5a)[idx][ok]
        out["imbalance5"] = imbalance[idx][ok]
        out["abs_imbalance5"] = np.abs(out["imbalance5"])
        out["concentration"] = ((d1b + d1a) / (d10b + d10a))[idx][ok]
        out["book_slope"] = np.r_[np.log(d5b / d1b)[idx][ok], np.log(d5a / d1a)[idx][ok]]
        gaps_b = (tape.bp[:, :4] - tape.bp[:, 1:5]) / tick
        gaps_a = (tape.ap[:, 1:5] - tape.ap[:, :4]) / tick
    gaps = np.r_[gaps_b[idx][ok].ravel(), gaps_a[idx][ok].ravel()]
    out["level_gap_ticks"] = gaps[np.isfinite(gaps)]
    out["microprice_dev_bps"] = microdev[idx][ok]

    # 100 ms step changes ---------------------------------------------------------------------------
    both = valid[1:] & valid[:-1]
    pb, pa = _ticks(tape.bp, tick), _ticks(tape.ap, tick)
    add_b, rem_b = _side_changes(pb[:-1], tape.bq[:-1], pb[1:], tape.bq[1:], bid=True)
    add_a, rem_a = _side_changes(pa[:-1], tape.aq[:-1], pa[1:], tape.aq[1:], bid=False)
    # prints explain removals at their (event VWAP) price on the hit side within (t_k, t_k+1]
    step_of_trade = np.searchsorted(tape.t, tape.trade_t, side="left") - 1
    trade_ticks = _ticks(tape.trade_p, tick)
    traded_step = np.zeros(n - 1, dtype=bool)
    for k, price, size, sign in zip(step_of_trade, trade_ticks, tape.trade_q, tape.trade_s):
        if not 0 <= k < n - 1:
            continue
        traded_step[k] = True
        prices, removals = (pa[k], rem_a) if sign > 0 else (pb[k], rem_b)
        remaining = float(size)
        order = np.argsort(prices if sign > 0 else -prices)
        for j in order:
            if remaining <= 0 or not np.isfinite(prices[j]):
                break
            if (sign > 0 and prices[j] > price) or (sign < 0 and prices[j] < price):
                break
            take = min(removals[k, j], remaining)
            removals[k, j] -= take
            remaining -= take
    adds = np.concatenate([add_b, add_a], axis=1)
    cancels = np.concatenate([rem_b, rem_a], axis=1)
    adds[~both] = 0.0
    cancels[~both] = 0.0
    eps = 1e-9
    out["add_size"] = adds[adds > eps]
    out["cancel_size"] = cancels[cancels > eps]
    add_count = (adds > eps).sum(1)
    cancel_count = (cancels > eps).sum(1)
    changed = both & ((np.nan_to_num(tape.bp[1:], nan=-1) != np.nan_to_num(tape.bp[:-1], nan=-1)).any(1)
                      | (tape.bq[1:] != tape.bq[:-1]).any(1)
                      | (np.nan_to_num(tape.ap[1:], nan=-1) != np.nan_to_num(tape.ap[:-1], nan=-1)).any(1)
                      | (tape.aq[1:] != tape.aq[:-1]).any(1))
    state = np.where(traded_step, 1, np.where((add_count > 0) & (cancel_count > 0), 4,
                                              np.where(add_count > 0, 2, np.where(cancel_count > 0, 3, 0))))
    pair = both[:-1] & both[1:]
    markov = np.zeros((5, 5))
    np.add.at(markov, (state[:-1][pair], state[1:][pair]), 1.0)
    out["markov"] = markov.ravel()

    # trades ------------------------------------------------------------------------------------------
    rel_t = tape.trade_t - tape.t[0]
    out["trade_size"] = tape.trade_q[tape.trade_q > 0]
    gaps_t = np.diff(tape.trade_t)
    out["trade_interarrival_s"] = gaps_t[gaps_t > 0]

    # 10 s windows (>= 80% valid samples) ------------------------------------------------------------
    w10 = int(round(10 / SAMPLE_DT))
    t_counts = np.bincount(np.clip((rel_t // 10).astype(int), 0, None), minlength=n // w10 + 1)
    flows = np.bincount(np.clip((rel_t // 10).astype(int), 0, None), weights=tape.trade_s * tape.trade_q,
                        minlength=n // w10 + 1)
    windows = _window_slices(n, w10)
    win_ok = np.asarray([valid[s].mean() >= 0.8 for s in windows], dtype=bool)
    step_slices = [slice(s.start, min(s.stop, n - 1)) for s in windows]
    tc10 = t_counts[:len(windows)].astype(float)
    out["trade_count_10s"] = tc10[win_ok]
    out["change_count_10s"] = np.asarray([changed[s].sum() for s in step_slices], float)[win_ok]
    out["add_count_10s"] = np.asarray([add_count[s].sum() for s in step_slices], float)[win_ok]
    out["cancel_count_10s"] = np.asarray([cancel_count[s].sum() for s in step_slices], float)[win_ok]
    flow10 = np.where(win_ok, flows[:len(windows)], np.nan)
    count10 = np.where(win_ok, tc10, np.nan)

    # 1 s grid series -------------------------------------------------------------------------------
    m1 = mid[idx]
    r1 = np.full(len(idx), np.nan)
    pair1 = ok[1:] & ok[:-1]
    r1[1:] = np.where(pair1, np.log(m1[1:] / np.where(pair1, m1[:-1], 1.0)) * 1e4, np.nan)
    finite_r1 = r1[np.isfinite(r1)]
    out["r1_bps"] = finite_r1
    out["nonzero_r1"] = np.asarray([float(np.sum(finite_r1 != 0)), float(len(finite_r1))])

    def horizon_returns(h: int) -> np.ndarray:
        values = []
        for i in range(0, len(idx) - h, h):
            if ok[i] and ok[i + h]:
                values.append(math.log(m1[i + h] / m1[i]) * 1e4)
        return np.asarray(values, float)

    out["r10_bps"] = horizon_returns(10)
    out["abs_r10_bps"] = np.abs(out["r10_bps"])
    out["r60_bps"] = horizon_returns(60)
    rv = []
    for i in range(0, len(r1) - 59, 60):
        window = r1[i:i + 60]
        if np.isfinite(window).sum() >= 50:
            rv.append(float(np.nanstd(window)))
    out["rv60_bps"] = np.asarray(rv, float)
    seconds = len(idx)
    sec_of_trade = np.ceil(rel_t - 1e-9).astype(int)
    keep = (sec_of_trade >= 1) & (sec_of_trade < seconds)
    count1 = np.bincount(sec_of_trade[keep], minlength=seconds).astype(float)
    flow1 = np.bincount(sec_of_trade[keep], weights=(tape.trade_s * tape.trade_q)[keep], minlength=seconds)
    cancel_steps = cancel_count.astype(float)
    step_end_sec = np.ceil((tape.t[1:] - tape.t[0]) - 1e-9).astype(int)
    cancel1 = np.bincount(np.clip(step_end_sec, 0, seconds - 1), weights=cancel_steps, minlength=seconds)
    sec_ok = ok.copy()
    sec_ok[0] = False
    spread1 = np.where(ok, spread[idx], np.nan)
    depth1 = np.where(ok, (d5b + d5a)[idx], np.nan)
    imb1 = np.where(ok, imbalance[idx], np.nan)
    micro1 = np.where(ok, microdev[idx], np.nan)
    trade_count_1s = np.where(sec_ok, count1, np.nan)
    out["trade_count_1s"] = count1[sec_ok]
    for lag in (1, 2, 5, 10):
        out[f"acf:r1:{lag}"] = _acf_pairs(r1, lag)
    for lag in (1, 5, 10, 30):
        out[f"acf:abs_r1:{lag}"] = _acf_pairs(np.abs(r1), lag)
    for lag in (1, 10):
        out[f"acf:spread:{lag}"] = _acf_pairs(spread1, lag)
    for lag in (1, 10, 60):
        out[f"acf:depth5:{lag}"] = _acf_pairs(depth1, lag)
        out[f"acf:imbalance:{lag}"] = _acf_pairs(imb1, lag)
    for lag in (1, 2, 5):
        out[f"acf:flow10:{lag}"] = _acf_pairs(flow10, lag)
    for lag in (1, 6):
        out[f"acf:count10:{lag}"] = _acf_pairs(count10, lag)
    next_r1 = np.r_[r1[1:], np.nan]
    out["pair:imbalance|next_r1"] = _paired(imb1, next_r1)
    out["pair:microdev|next_r1"] = _paired(micro1, next_r1)
    out["pair:flow1|r1"] = _paired(np.where(sec_ok, flow1, np.nan), r1)
    out["pair:depth5_total|next_abs_r1"] = _paired(depth1, np.abs(next_r1))
    out["pair:trade1_cancel_next1"] = _paired(trade_count_1s, np.r_[np.where(sec_ok, cancel1, np.nan)[1:], np.nan])

    # 60 s windows for dependence ----------------------------------------------------------------------
    w_rows = []
    for i in range(0, seconds - 59, 60):
        window = r1[i:i + 60]
        oks = ok[i:i + 60]
        if np.isfinite(window).sum() >= 50 and oks.mean() >= 0.8:
            w_rows.append((float(np.nanstd(window)), float(np.nanmean(spread1[i:i + 60])),
                           float(np.nanmean(depth1[i:i + 60])), float(np.nansum(count1[i:i + 60])),
                           float(np.nansum(cancel1[i:i + 60])), float(np.nanmean(np.abs(imb1[i:i + 60])))))
    w = np.asarray(w_rows, float).reshape(-1, 6)
    names = ("w_rv", "w_spread", "w_depth", "w_count", "w_cancel", "w_absimb")
    for j, name in enumerate(names):
        out[f"win:{name}"] = w[:, j]

    # spread runs ------------------------------------------------------------------------------------
    spread_ticks = (tape.ap[:, 0] - tape.bp[:, 0]) / tick
    wide = spread_ticks > 1.5
    out["wide_run_s"] = _runs(wide, valid) * SAMPLE_DT
    out["tight_run_s"] = _runs(~wide, valid) * SAMPLE_DT

    # resilience after depleting events ------------------------------------------------------------------
    recovery, recovered, ratios = [], [], []
    horizon = int(RECOVERY_CENSOR_S / SAMPLE_DT)
    sample_of_trade = np.searchsorted(tape.t, tape.trade_t, side="right") - 1
    for k, size, sign, when in zip(sample_of_trade, tape.trade_q, tape.trade_s, tape.trade_t):
        if k < 0 or k + horizon >= n or not valid[k]:
            continue
        depth = d1a if sign > 0 else d1b
        pre = depth[k]
        if pre <= 0 or size < 0.5 * pre:
            continue
        window = depth[k + 1:k + 1 + horizon]
        hit = np.flatnonzero((window >= 0.9 * pre) & valid[k + 1:k + 1 + horizon])
        if len(hit):
            recovery.append(float(tape.t[k + 1 + hit[0]] - when))
            recovered.append(1.0)
        else:
            recovered.append(0.0)
        side5 = d5a if sign > 0 else d5b
        later = k + int(5 / SAMPLE_DT)
        if valid[later] and side5[k] > 0 and side5[later] > 0:
            ratios.append(math.log(side5[later] / side5[k]))
    out["recovery_s"] = np.asarray([x for x in recovery if x > 0], float)
    out["recovered"] = np.asarray([float(sum(recovered)), float(len(recovered))])
    out["depth5_ratio_5s"] = np.asarray(ratios, float)

    # depletion / replenishment hazards on the 1 s grid ---------------------------------------------------
    deplete, eligible, replenished, depleted_n = 0.0, 0.0, 0.0, 0.0
    bb, ba = tape.bp[idx, 0], tape.ap[idx, 0]
    qb, qa = d1b[idx], d1a[idx]
    for i in range(1, seconds - 1):
        if not (ok[i - 1] and ok[i]):
            continue
        for best, qty, worse in ((bb, qb, -1.0), (ba, qa, 1.0)):
            eligible += 1
            moved = (best[i] - best[i - 1]) * worse > 0
            same = best[i] == best[i - 1]
            if moved or (same and qty[i] < 0.5 * qty[i - 1]):
                deplete += 1
                if ok[i + 1]:
                    depleted_n += 1
                    improved = (best[i + 1] - best[i - 1]) * worse < 0
                    restored = best[i + 1] == best[i - 1] and qty[i + 1] >= 0.9 * qty[i - 1]
                    replenished += float(improved or restored)
    out["depletion"] = np.asarray([deplete, eligible])
    out["replenished"] = np.asarray([replenished, depleted_n])
    return out


def _paired(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def concatenate(raws: list[dict], key: str) -> np.ndarray:
    values = [r[key] for r in raws if key in r and np.size(r[key])]
    if not values:
        return np.empty(0)
    if isinstance(values[0], tuple):
        return np.concatenate([np.column_stack(v) for v in values])
    return np.concatenate(values)


# ----------------------------------------------------------------------------- design (development only)


def _transform(values: np.ndarray, how: str) -> tuple[np.ndarray, int]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if how == "log":
        positive = values > 0
        return np.log(values[positive]), int((~positive).sum())
    if how == "log1p":
        nonneg = values >= 0
        return np.log1p(values[nonneg]), int((~nonneg).sum())
    return values, 0


class Reference:
    """Development marginal used for rank-type correlations: mid-rank CDF of a frozen sample."""

    def __init__(self, values: np.ndarray, cdf_lo: np.ndarray, cdf_hi: np.ndarray) -> None:
        self.values, self.cdf_lo, self.cdf_hi = values, cdf_lo, cdf_hi

    @classmethod
    def fit(cls, sample: np.ndarray, max_points: int = 2001) -> Reference:
        sample = np.sort(np.asarray(sample, float)[np.isfinite(sample)])
        if not len(sample):
            raise ValueError("empty reference sample")
        values, counts = np.unique(sample, return_counts=True)
        if len(values) > max_points:
            values = np.unique(np.quantile(sample, np.linspace(0, 1, max_points)))
            lo = np.searchsorted(sample, values, side="left") / len(sample)
            hi = np.searchsorted(sample, values, side="right") / len(sample)
        else:
            cumulative = np.cumsum(counts) / len(sample)
            hi = cumulative
            lo = cumulative - counts / len(sample)
        return cls(values, lo, hi)

    def pit(self, x: np.ndarray) -> np.ndarray:
        """Mid-rank probability integral transform; linear between reference points."""
        x = np.asarray(x, float)
        mid = (self.cdf_lo + self.cdf_hi) / 2
        if len(self.values) == 1:
            return np.where(x < self.values[0], 0.0, np.where(x > self.values[0], 1.0, 0.5))
        u = np.interp(x, self.values, mid, left=0.0, right=1.0)
        exact = np.searchsorted(self.values, x)
        inside = exact < len(self.values)
        hit = np.zeros(len(x), bool)
        hit[inside] = self.values[exact[inside]] == x[inside]
        u[hit] = mid[exact[hit]]
        return u

    def to_dict(self) -> dict:
        return {"values": self.values.tolist(), "cdf_lo": self.cdf_lo.tolist(), "cdf_hi": self.cdf_hi.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> Reference:
        return cls(np.asarray(d["values"], float), np.asarray(d["cdf_lo"], float), np.asarray(d["cdf_hi"], float))


def build_design(dev_raws: list[dict], *, tick_bps: float) -> dict:
    """Freeze bins, scales, references and thresholds from development raw samples only."""
    floors = {"tick": tick_bps, "quarter_tick": 0.25 * tick_bps}
    dist = {}
    for source in DIST_SOURCES:
        component = next(c for c in COMPONENTS if c.kind == "dist" and c.source == source)
        values, _ = _transform(concatenate(dev_raws, source), component.transform)
        if len(values) < component.min_n:
            raise ValueError(f"development sample too small to bin {source}: {len(values)}")
        lo, hi = np.quantile(values, [0.001, 0.999])
        span = max(hi - lo, 1e-6)
        q25, q75 = np.quantile(values, [0.25, 0.75])
        floor = floors.get(component.floor, component.floor) if isinstance(component.floor, str) else component.floor
        dist[source] = {"transform": component.transform, "lo": float(lo - 0.25 * span), "hi": float(hi + 0.25 * span),
                        "bins": BINS, "scale": float(max(q75 - q25, floor)), "iqr": float(q75 - q25),
                        "floor": float(floor)}
    thresholds = {}
    for component in COMPONENTS:
        if component.kind == "exc":
            source, direction, q = component.source.split(":")
            values = concatenate(dev_raws, source)
            thresholds[component.name] = {"source": source, "direction": direction, "q": float(q),
                                          "value": float(np.quantile(values[np.isfinite(values)], float(q)))}
    hill = {}
    for component in COMPONENTS:
        if component.kind == "hill":
            values = concatenate(dev_raws, component.source)
            values = values[np.isfinite(values) & (values > 0)]
            hill[component.name] = {"source": component.source, "u": float(np.quantile(values, 0.95))}
    references = {}
    for key in PIT_SOURCES:
        references[key] = Reference.fit(_pit_sample(dev_raws, key)).to_dict()
    imbalance = concatenate(dev_raws, "imbalance5")
    r1 = concatenate(dev_raws, "r1_bps")
    return {"schema": "cleolob-v06-observable-design-1", "tick_bps": float(tick_bps), "dist": dist,
            "thresholds": thresholds, "hill": hill, "references": references,
            "quintiles": np.quantile(imbalance, [0.2, 0.4, 0.6, 0.8]).tolist(),
            "return_scale_bps": float(max(np.std(r1), 0.1 * tick_bps)),
            "components": [c.name for c in COMPONENTS]}


def _pit_sample(raws: list[dict], key: str) -> np.ndarray:
    """Development marginal of one variable appearing in a rank-correlation pair."""
    if key.startswith("w_"):
        return concatenate(raws, f"win:{key}")
    for name in ("pair:imbalance|next_r1", "pair:microdev|next_r1", "pair:flow1|r1", "pair:depth5_total|next_abs_r1"):
        left, right = name[5:].split("|")
        if key in (left, right):
            stacked = concatenate(raws, name)
            return stacked[:, 0 if key == left else 1] if stacked.size else stacked
    raise KeyError(key)


# ----------------------------------------------------------------------------- sketches


def _hist(values: np.ndarray, spec: dict) -> np.ndarray:
    """[counts(BINS), under_n, under_sum, over_n, over_sum, dropped] on the transformed scale."""
    x, dropped = _transform(values, spec["transform"])
    lo, hi, bins = spec["lo"], spec["hi"], spec["bins"]
    under, over = x < lo, x > hi
    inside = x[~under & ~over]
    counts = np.histogram(inside, bins=bins, range=(lo, hi))[0].astype(float)
    return np.r_[counts, under.sum(), x[under].sum(), over.sum(), x[over].sum(), dropped]


def _moments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if not len(x):
        return np.zeros(6)
    return np.asarray([len(x), x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()], float)


def sketch(raw: dict, design: dict) -> dict[str, np.ndarray]:
    """Additive sufficient statistics of one block under a frozen design."""
    out: dict[str, np.ndarray] = {}
    for source, spec in design["dist"].items():
        out[f"dist:{source}"] = _hist(raw[source], spec)
    for name, spec in design["thresholds"].items():
        values = np.asarray(raw[spec["source"]], float)
        values = values[np.isfinite(values)]
        k = (values > spec["value"]).sum() if spec["direction"] == "gt" else (values < spec["value"]).sum()
        out[f"exc:{name}"] = np.asarray([k, len(values)], float)
    for name, spec in design["hill"].items():
        values = np.asarray(raw[spec["source"]], float)
        tail = values[np.isfinite(values) & (values > spec["u"])]
        out[f"hill:{name}"] = np.asarray([len(tail), np.log(tail / spec["u"]).sum()], float)
    for component in COMPONENTS:
        if component.kind == "acf":
            out[f"acf:{component.source}"] = _moments(*raw[f"acf:{component.source}"])
        elif component.kind == "prob":
            out[f"prob:{component.source}"] = np.asarray(raw[component.source], float)
        elif component.kind == "corr":
            out[f"corr:{component.source}"] = _moments(*raw[f"pair:{component.source}"])
        elif component.kind == "pit":
            left, right = component.source.split("|")
            if left.startswith("w_"):
                x, y = raw[f"win:{left}"], raw[f"win:{right}"]
            else:
                x, y = raw[f"pair:{component.source}"]
            refs = design["references"]
            out[f"pit:{component.source}"] = _moments(Reference.from_dict(refs[left]).pit(x),
                                                      Reference.from_dict(refs[right]).pit(y))
    out["markov"] = np.asarray(raw["markov"], float)
    for source in ("trade_count_1s", "trade_count_10s"):
        x = np.asarray(raw[source], float)
        out[f"ratio:{source}"] = np.asarray([len(x), x.sum(), (x * x).sum()], float)
    x = np.asarray(raw["trade_interarrival_s"], float)
    out["moment:trade_interarrival_s"] = np.asarray([len(x), x.sum(), (x * x).sum()], float)
    x, y = raw["pair:imbalance|next_r1"]
    bins = np.searchsorted(np.asarray(design["quintiles"]), x, side="right")
    cond = np.zeros(10)
    np.add.at(cond, 2 * bins, 1.0)
    np.add.at(cond, 2 * bins + 1, y)
    out["cond:imbalance|next_r1"] = cond
    table = np.zeros((5, 3))
    np.add.at(table, (bins, (np.sign(y) + 1).astype(int)), 1.0)
    out["mi:imbalance|next_r1"] = table.ravel()
    return out


def add(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: a[key] + b[key] for key in a}


def pool(sketches: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not sketches:
        raise ValueError("nothing to pool")
    keys = sketches[0].keys()
    return {key: np.sum([s[key] for s in sketches], axis=0) for key in keys}


def stack(sketches: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Blocks x statistics matrices, for fast resampled sums (weights @ matrix)."""
    keys = sketches[0].keys()
    return {key: np.vstack([s[key] for s in sketches]) for key in keys}


def weighted(stacked: dict[str, np.ndarray], weights: np.ndarray) -> dict[str, np.ndarray]:
    return {key: weights @ matrix for key, matrix in stacked.items()}
