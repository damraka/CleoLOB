"""Meta-order experiments, transaction-cost decomposition and the impact model zoo (workstreams 6, 28-30).

``run_metaorder`` executes one parent order (TWAP children or POV-style participation)
in any world, recording the mid every 0.5 s from arrival until ``recovery_s`` after
the horizon. It reports implementation shortfall, completion, temporary impact
(mid at the end of trading minus arrival mid, signed), bounded residual impact
(mid ``recovery_s`` later minus arrival) and recovery (fraction of the peak move
reverted). Simulator worlds only; impact is the simulator's endogenous response.

``tca`` decomposes the buy-side shortfall into components that sum exactly to it:
spread crossing (fill price - contemporaneous mid), drift-and-impact (contemporaneous
mid - arrival mid), fees and the opportunity/completion cost of the unfilled
remainder (valued at the final mid plus the mandate penalty). Adverse selection of
passive fills (mid change 5 s after a fill) is reported separately.

The zoo fits, on meta-order outcomes: linear (I = a Q), square-root
(I = Y sigma sqrt(Q / V)) and a power-law propagator kernel on child-trade paths,
and reports in-sample fit and cross-world transfer. No law is treated as universal.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ...engine import Side

SAMPLE_S = 0.5


@dataclass(frozen=True)
class MetaOrder:
    side: str = "buy"
    quantity: int = 14
    horizon_s: float = 120.0
    style: str = "twap"            # twap | pov
    participation: float = 0.1     # pov only
    urgency: float = 0.0           # 0: uniform schedule; > 0 front-loads (sinh profile, kappa T = urgency)
    recovery_s: float = 60.0
    warmup_s: float = 60.0
    slices: int = 20

    def schedule(self) -> list[int]:
        """Cumulative target quantity at the start of each slice (front-loaded when urgency > 0)."""
        out = []
        for j in range(self.slices + 1):
            tau = j / self.slices
            remaining = (math.sinh(self.urgency * (1 - tau)) / math.sinh(self.urgency) if self.urgency > 0
                         else 1 - tau)
            out.append(int(round(self.quantity * (1 - remaining))))
        return out


def run_metaorder(world, order: MetaOrder, seed: int) -> dict:
    sim = world.build(seed)
    sim.step(order.warmup_s)
    side = Side.BUY if order.side == "buy" else Side.SELL
    sign = 1 if side is Side.BUY else -1
    m0 = sim.book.mid()
    t0 = sim.t
    path_t, path_m = [0.0], [m0]
    fills_cursor = len(sim.book.trades)
    targets = order.schedule()
    sent, market_volume = 0, 0.0
    tau = order.horizon_s / order.slices
    for j in range(order.slices):
        if order.style == "twap":
            child = targets[j + 1] - sent
        else:
            child = max(0, int(round(order.participation / (1 - order.participation) * market_volume)) - sent)
            if j == order.slices - 1:
                child = order.quantity - sent
        child = min(child, order.quantity - sent)
        if child > 0:
            sim.submit(side, int(child), "META")
            sent += child
        end = sim.t + tau
        while sim.t < end - 1e-9:
            trades = sim.step(min(SAMPLE_S, end - sim.t))
            market_volume += sum(tr.qty for tr in trades if "META" not in (tr.taker_owner, tr.maker_owner))
            path_t.append(sim.t - t0)
            path_m.append(sim.book.mid())
    m_end = sim.book.mid()
    while sim.t - t0 < order.horizon_s + order.recovery_s - 1e-9:
        sim.step(SAMPLE_S)
        path_t.append(sim.t - t0)
        path_m.append(sim.book.mid())
    fills = [tr for tr in sim.book.trades[fills_cursor:] if "META" in (tr.taker_owner, tr.maker_owner)]
    filled = sum(tr.qty for tr in fills)
    tick = sim.cfg.tick_size
    mids = np.asarray(path_m)
    moves = sign * (mids - m0)
    peak = float(moves.max()) if len(moves) else 0.0
    temporary = sign * (m_end - m0)
    residual = sign * (path_m[-1] - m0)
    shortfall = sum(sign * (tr.price - m0) * tr.qty for tr in fills)
    return {"seed": seed, "filled": int(filled), "completion": filled / order.quantity if order.quantity else 1.0,
            "shortfall_bps": (shortfall / max(filled, 1)) / m0 * 1e4 if filled else 0.0,
            "temporary_impact_bps": temporary / m0 * 1e4, "residual_impact_bps": residual / m0 * 1e4,
            "peak_move_bps": peak / m0 * 1e4,
            "recovery_fraction": (1 - residual / peak) if peak > 0 else None,
            "market_volume": market_volume, "tick": tick, "arrival_mid_ticks": m0,
            "fills": [(tr.time - t0, tr.price, tr.qty, tr.maker_owner == "META") for tr in fills],
            "path": {"t": path_t, "mid": path_m}}


def tca(result: dict, *, side: str = "buy", fee_bps: float = 1.0, quantity: int | None = None,
        penalty_bps: float = 25.0) -> dict:
    """Additive decomposition of the arrival-mid shortfall (bps of arrival notional)."""
    sign = 1 if side == "buy" else -1
    m0 = result["arrival_mid_ticks"]
    t, m = np.asarray(result["path"]["t"]), np.asarray(result["path"]["mid"])
    q_total = quantity if quantity is not None else sum(f[2] for f in result["fills"])
    notional = q_total * m0
    spread = impact = adverse = 0.0
    for time, price, qty, passive in result["fills"]:
        mid_at = float(m[max(0, np.searchsorted(t, time, side="right") - 1)])
        spread += sign * (price - mid_at) * qty
        impact += sign * (mid_at - m0) * qty
        if passive:
            later = float(m[min(len(m) - 1, np.searchsorted(t, time + 5.0))])
            adverse += sign * (mid_at - later) * qty
    filled = sum(f[2] for f in result["fills"])
    unfilled = max(q_total - filled, 0)
    fees = fee_bps / 1e4 * sum(f[1] * f[2] for f in result["fills"])
    opportunity = sign * (float(m[-1]) - m0) * unfilled + penalty_bps / 1e4 * m0 * unfilled
    components = {"spread_crossing": spread, "drift_and_impact": impact, "fees": fees, "rebates": 0.0,
                  "opportunity_and_completion": opportunity}
    total = sum(components.values())
    direct = sum(sign * (f[1] - m0) * f[2] for f in result["fills"]) + fees + opportunity
    scale = 1e4 / notional if notional else 0.0
    return {"components_bps": {k: v * scale for k, v in components.items()}, "total_bps": total * scale,
            "identity_error": abs(total - direct), "adverse_selection_passive_bps": adverse * scale,
            "residual_valuation_bps": sign * (float(m[-1]) - m0) * unfilled * scale, "unfilled": unfilled}


# ----------------------------------------------------------------------------- impact zoo


def fit_linear(q: np.ndarray, impact: np.ndarray) -> dict:
    a = float(np.sum(q * impact) / np.sum(q * q)) if np.any(q) else 0.0
    return {"model": "linear", "a": a, "predict": lambda x: a * x}


def fit_sqrt(q: np.ndarray, impact: np.ndarray, sigma: np.ndarray, volume: np.ndarray) -> dict:
    x = sigma * np.sqrt(q / np.maximum(volume, 1.0))
    y = float(np.sum(x * impact) / np.sum(x * x)) if np.any(x) else 0.0
    return {"model": "square_root", "Y": y, "predict": lambda xq, s, v: y * s * np.sqrt(xq / np.maximum(v, 1.0))}


def propagator_kernel(lags: np.ndarray, g0: float, tau0: float, beta: float) -> np.ndarray:
    return g0 / (1 + lags / tau0) ** beta


def fit_propagator(paths: list[dict], *, grid: tuple = (0.2, 0.5, 1.0, 2.0)) -> dict:
    """Least squares over a beta grid: mid change explained by past child-trade flow through a power-law kernel."""
    best = None
    for beta in grid:
        xs, ys = [], []
        for r in paths:
            t, m = np.asarray(r["path"]["t"]), np.asarray(r["path"]["mid"])
            flows = [(f[0], f[2]) for f in r["fills"] if not f[3]]
            if not flows:
                continue
            x = np.asarray([sum(q * propagator_kernel(np.asarray([ti - s]), 1.0, 1.0, beta)[0]
                                for s, q in flows if s <= ti) for ti in t])
            xs.append(x)
            ys.append(m - m[0])
        if not xs:
            return {"model": "propagator", "status": "NOT_AVAILABLE", "reason": "no aggressive child trades"}
        x, y = np.concatenate(xs), np.concatenate(ys)
        g0 = float(np.sum(x * y) / np.sum(x * x)) if np.any(x) else 0.0
        sse = float(np.sum((y - g0 * x) ** 2))
        if best is None or sse < best["sse"]:
            best = {"model": "propagator", "g0": g0, "tau0": 1.0, "beta": beta, "sse": sse,
                    "r2": float(1 - sse / np.sum((y - y.mean()) ** 2)) if np.var(y) > 0 else None}
    return best


def r2(y: np.ndarray, yhat: np.ndarray) -> float | None:
    v = float(np.sum((y - y.mean()) ** 2))
    return float(1 - np.sum((y - yhat) ** 2) / v) if v > 0 else None
