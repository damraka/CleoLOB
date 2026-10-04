"""World-agnostic execution episodes for v0.7 (workstreams 28, 31, 50, 78).

``run(world, agent, mandate, seed)`` is the v0.5 ``run_baseline`` loop (warmup,
decision steps, shared completion/urgency rule, settlement, report and
execution metrics, all imported unchanged) with one difference: the market is
``world.build(seed)``, so any v0.7 world (v0.6 ``SimulatorSpec``, generator
spec, regime-switching spec) can host any classical agent. Agent parameters are
fixed across worlds: AC uses the frozen v0.6 parameters and VWAP a flat profile
unless one is supplied (re-fitting either per world would turn a world
comparison into a policy comparison).
"""
from __future__ import annotations

from dataclasses import asdict
import math

from ...completion import completion_constraint, execution_metrics, urgency_order
from ...engine import Side
from ...execution import AlmgrenChrissAgent, POVAgent, TWAPAgent, VWAPAgent, build_report
from ...settlement import DEFAULT_SETTLEMENT_POLL_DT, settle_orders
from ..policies.classical import AGENTS as NEW_AGENTS, N_SLICES

CLASSICAL_V06 = ("twap", "vwap", "pov", "ac")
PRIMARY = CLASSICAL_V06 + ("liquidity_sensitive", "urgency", "imbalance_aware", "spread_aware")
EXPLORATORY = ("mpc",)
COST = "mandate_completion_adjusted_cost_bps"
MANDATE = {"side": "buy", "quantity": 14, "horizon_s": 120.0, "decision_dt_s": 6.0, "warmup_s": 60.0,
           "settlement_timeout_s": 5.0, "fees": {"maker_bps": 0.0, "taker_bps": 1.0}, "terminal_penalty_bps": 25.0,
           "completion_urgency_fraction": 0.8, "pov_participation": 0.3}
AC_V06 = {"temp_impact": 0.0003894251582053461, "sigma": 0.013986590092346119,
          "risk_aversion": 0.00013824130302010369}


def make_agent(name: str, mandate: dict, start: float, *, volume_profile=None, ac: dict | None = None):
    qty, horizon = int(mandate["quantity"]), float(mandate["horizon_s"])
    side = Side.BUY if mandate["side"] == "buy" else Side.SELL
    kwargs = {"fees": mandate["fees"], "risk": None}
    if name == "twap":
        return TWAPAgent(qty, horizon, n_slices=N_SLICES, side=side, start_time=start, **kwargs)
    if name == "vwap":
        return VWAPAgent(qty, horizon, n_slices=N_SLICES, side=side, start_time=start,
                         volume_profile=volume_profile or [1.0] * N_SLICES, **kwargs)
    if name == "pov":
        return POVAgent(qty, horizon, side=side, start_time=start, participation=float(mandate["pov_participation"]),
                        decision_dt=float(mandate["decision_dt_s"]), **kwargs)
    if name == "ac":
        params = ac or AC_V06
        return AlmgrenChrissAgent(qty, horizon, n_slices=N_SLICES, side=side, risk_aversion=params["risk_aversion"],
                                  temp_impact=params["temp_impact"], sigma=params["sigma"], start_time=start, **kwargs)
    if name in NEW_AGENTS:
        return NEW_AGENTS[name](qty, horizon, side, start, float(mandate["decision_dt_s"]), **kwargs)
    raise KeyError(f"unknown agent {name!r}")


def volume_profile(world, seed: int, mandate: dict, days: int = 5, warmup: float = 60.0) -> list[float]:
    """Traded volume per slice over ``days`` 'previous days' of the same world (shifted seeds)."""
    tau = float(mandate["horizon_s"]) / N_SLICES
    acc = [0.0] * N_SLICES
    for d in range(days):
        sim = world.build(seed + 1000 + d)
        sim.step(warmup)
        for j in range(N_SLICES):
            acc[j] += float(sum(tr.qty for tr in sim.step(tau)))
    total = sum(acc)
    return [v / total for v in acc] if total > 0 else [1.0 / N_SLICES] * N_SLICES


def run(world, agent_name: str, mandate: dict, seed: int, *, profile=None, ac: dict | None = None) -> dict:
    """One episode row (cost in bps; lower is better). Exceptions are returned as INVALID rows."""
    try:
        return _run(world, agent_name, mandate, seed, profile=profile, ac=ac)
    except (ValueError, RuntimeError, AssertionError, ZeroDivisionError, OverflowError) as exc:
        return {"agent": agent_name, "seed": seed, "status": "INVALID", COST: None,
                "error": f"{type(exc).__name__}: {exc}"}


def _run(world, agent_name, mandate, seed, *, profile, ac) -> dict:
    qty, horizon, dt = int(mandate["quantity"]), float(mandate["horizon_s"]), float(mandate["decision_dt_s"])
    sim = world.build(seed)
    sim.step(float(mandate["warmup_s"]))
    t0, arrival = sim.t, sim.book.mid()
    first_trade = len(sim.book.trades)
    agent = make_agent(agent_name, mandate, t0, volume_profile=profile, ac=ac)
    completion = completion_constraint({"enabled": True, "urgency_fraction": mandate["completion_urgency_fraction"]})
    owner = agent_name.upper()
    steps = max(1, int(math.ceil(horizon / dt - 1e-12)))
    for _ in range(steps):
        if completion.active(sim.t - t0, horizon, dt):
            agent.poll_fills(sim, owner)
            oid = urgency_order(sim, agent.risk, agent._ensure_ledger(sim), owner, agent.filled,
                                elapsed=sim.t - t0, horizon=horizon, decision_dt=dt)
            if oid is not None:
                agent.child_orders += 1
        else:
            agent.on_step(sim, owner)
        sim.step(min(dt, max(0.0, horizon - (sim.t - t0))))
        agent.poll_fills(sim, owner)
        if agent.remaining == 0:
            break
    decision_duration = sim.t - t0
    ledger = agent._ensure_ledger(sim)
    filled_at_stop, fees_at_stop = agent.filled, ledger.total_fees
    settlement = settle_orders(sim, agent.risk, lambda: agent.poll_fills(sim, owner),
                               timeout=float(mandate["settlement_timeout_s"]), poll_dt=DEFAULT_SETTLEMENT_POLL_DT)
    liq, fillable = sim.book.walk_cost(agent.side, agent.remaining) if agent.remaining else (None, 0)
    report = build_report(agent.label, agent.side, qty, agent.fills, arrival, sim.cfg.tick_size, agent.child_orders,
                          decision_duration, liq, leftover_fillable_qty=fillable, fees=agent.fees,
                          total_fees=ledger.total_fees, terminal_penalty_bps=float(mandate["terminal_penalty_bps"]),
                          strict_terminal=True, outstanding_qty=agent.risk.outstanding(sim),
                          accounting=asdict(ledger.snapshot(sim.book.mid() * sim.cfg.tick_size)),
                          risk_rejections=sum(e["event"] == "rejected" for e in agent.risk.events),
                          decision_horizon=horizon, settlement_duration=settlement.duration,
                          settlement_complete=settlement.complete, late_filled_qty=agent.filled - filled_at_stop,
                          late_fees=ledger.total_fees - fees_at_stop,
                          settlement_pending_order_ids=settlement.pending_order_ids)
    raw = execution_metrics(sim, agent.fills, owner=owner, start_time=t0, first_trade=first_trade, report=report)
    row = {"agent": agent_name, "seed": seed, "status": report.status, "shortfall_bps": float(report.shortfall_bps),
           "fill_frac": report.filled_qty / max(1, report.target_qty), "children": int(report.n_child_orders),
           "fee_bps": report.fee_bps, "completion_penalty_bps": report.completion_penalty_bps,
           "events": int(sim.event_count)}
    row.update({k: v for k, v in raw.items() if k.startswith("mandate_") or k in (
        "realized_slippage_bps", "participation", "time_to_completion", "actual_completion")})
    return row
