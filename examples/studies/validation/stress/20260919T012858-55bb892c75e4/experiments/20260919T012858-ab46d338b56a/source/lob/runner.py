"""Run orchestration shared by the web UI, the evaluation harness and notebooks.

Two kinds of agents run on markets built from the same seed:

* **baselines** — schedule/participation agents from ``lob.execution``
  (``ac``, ``twap``, ``vwap``, ``pov``), executing with market orders;
* **policies**  — decision-makers driving ``LOBExecutionEnv``
  (``ppo`` model, ``heuristic`` rule, ``random`` floor).

``run_pair`` returns the rich payload the CLEO UI renders; ``run_episode`` returns a
compact metrics row for large-scale evaluation.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .engine import ExchangeSimulator, Side, SimConfig
from .execution import (AlmgrenChrissAgent, ExecutionAgent, ExecutionReport, POVAgent,
                        TWAPAgent, VWAPAgent, build_report, estimate_volume_profile)
from .rl_env import LOBExecutionEnv
from .settlement import (DEFAULT_SETTLEMENT_POLL_DT, DEFAULT_SETTLEMENT_TIMEOUT,
                         settle_orders, validate_settlement)

WARMUP_S = 5.0
DEPTH_LEVELS = 12
N_SLICES = 20
ProgressFn = Callable[[str, float], None]

BASELINES: Tuple[str, ...] = ("ac", "twap", "vwap", "pov")
POLICIES: Tuple[str, ...] = ("ppo", "heuristic", "random")
AGENTS: Tuple[str, ...] = BASELINES + POLICIES


def _side_from_params(p: Dict[str, Any]) -> Side:
    value = p.get("side", "sell")
    if isinstance(value, Side):
        return value
    if not isinstance(value, str) or value.lower() not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    return Side.BUY if value.lower() == "buy" else Side.SELL


# ------------------------------------------------------------------ config
def cfg_from_params(p: Dict[str, Any]) -> SimConfig:
    base = SimConfig(
        seed=int(p["seed"]),
        latency_base=float(p["latency_ms"]) / 2000.0,
        latency_jitter=float(p["latency_ms"]) / 2000.0,
        resilience=float(p["resilience"]),
        market_rate=float(p["market_rate"]),
    )
    overrides = dict(p.get("sim") or {})
    unknown = set(overrides) - set(base.__dict__)
    if unknown:
        raise KeyError(f"unknown SimConfig fields: {sorted(unknown)}")
    return SimConfig(**{**base.__dict__, **overrides})


@lru_cache(maxsize=64)
def _volume_profile(cfg_items: Tuple[Tuple[str, Any], ...], horizon: float) -> Tuple[float, ...]:
    return tuple(estimate_volume_profile(SimConfig(**dict(cfg_items)), horizon, N_SLICES))


def make_baseline(name: str, p: Dict[str, Any], cfg: SimConfig, start_time: float) -> ExecutionAgent:
    qty, horizon = int(p["qty"]), float(p["horizon"])
    side = _side_from_params(p)
    kwargs = {"fees": p.get("fees"), "risk": p.get("risk")}
    if name == "ac":
        return AlmgrenChrissAgent(qty, horizon, n_slices=N_SLICES, side=side,
                                  risk_aversion=float(p["risk_aversion"]), start_time=start_time, **kwargs)
    if name == "twap":
        return TWAPAgent(qty, horizon, n_slices=N_SLICES, side=side, start_time=start_time, **kwargs)
    if name == "vwap":
        profile = _volume_profile(tuple(sorted(cfg.__dict__.items())), horizon)
        return VWAPAgent(qty, horizon, n_slices=N_SLICES, side=side,
                         start_time=start_time, volume_profile=list(profile), **kwargs)
    if name == "pov":
        return POVAgent(qty, horizon, side=side, start_time=start_time,
                        participation=float(p.get("pov_rate", 0.3)),
                        decision_dt=float(p["dt"]), **kwargs)
    raise KeyError(f"unknown baseline '{name}' (choose from {', '.join(BASELINES)})")


# ------------------------------------------------------------------ series
def _new_series() -> Dict[str, List[Any]]:
    return {"t": [], "mid": [], "best_bid": [], "best_ask": [], "spread": [],
            "remaining": [], "bids": [], "asks": []}


def _push(series: Dict[str, List[Any]], t: float, sim: ExchangeSimulator, remaining: int) -> None:
    book = sim.book
    bids, asks = book.depth(DEPTH_LEVELS)
    bb, ba = book.best_bid(), book.best_ask()
    series["t"].append(round(t, 3))
    series["mid"].append(book.mid())
    series["best_bid"].append(bb)
    series["best_ask"].append(ba)
    series["spread"].append(None if bb is None or ba is None else ba - bb)
    series["remaining"].append(int(remaining))
    series["bids"].append([[int(p), int(v)] for p, v in bids])
    series["asks"].append([[int(p), int(v)] for p, v in asks])


def _run_dict(label: str, series: Dict[str, List[Any]], fills: List[Any], t0: float,
              arrival_ticks: float, report: ExecutionReport) -> Dict[str, Any]:
    return {
        "label": label,
        "arrival_ticks": arrival_ticks,
        "series": series,
        "fills": [{"t": round(tr.time - t0, 3), "px": int(tr.price), "qty": int(tr.qty)}
                  for tr in fills],
        "report": report.as_dict(),
        "raw": {
            "arrival_price": report.arrival_price, "vwap": report.vwap,
            "shortfall_bps": report.shortfall_bps,
            "effective_bps": report.effective_shortfall_bps, "total_cost": report.total_cost,
            "filled": report.filled_qty, "target": report.target_qty,
            "children": report.n_child_orders, "duration": report.duration,
            "gross_cost": report.gross_cost, "total_fees": report.total_fees,
            "fee_bps": report.fee_bps,
            "gross_effective_bps": report.gross_effective_bps,
            "net_effective_bps": report.net_effective_bps,
            "hypothetical_liquidation_cost": report.hypothetical_liquidation_cost,
            "hypothetical_liquidation_fees": report.hypothetical_liquidation_fees,
            "completion_penalty_bps": report.completion_penalty_bps,
            "optimization_cost_bps": report.optimization_cost_bps,
            "unpriced_leftover_qty": report.unpriced_leftover_qty,
            "fillable_leftover_qty": report.fillable_leftover_qty,
            "outstanding_qty": report.outstanding_qty,
            "status": report.status, "invalid_reasons": list(report.invalid_reasons),
            "accounting": report.accounting,
            "decision_horizon": report.decision_horizon,
            "decision_duration": report.decision_duration,
            "settlement_duration": report.settlement_duration,
            "total_duration": report.total_duration,
            "settlement_complete": report.settlement_complete,
            "late_filled_qty": report.late_filled_qty, "late_fees": report.late_fees,
            "settlement_pending_order_ids": list(report.settlement_pending_order_ids),
        },
    }


# ------------------------------------------------------------------ baselines
def run_baseline(p: Dict[str, Any], progress: Optional[ProgressFn] = None,
                 name: Optional[str] = None, record: bool = True) -> Dict[str, Any]:
    name = name or str(p.get("baseline", "ac"))
    qty, horizon, dt = int(p["qty"]), float(p["horizon"]), float(p["dt"])
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    settlement_timeout = p.get("settlement_timeout", DEFAULT_SETTLEMENT_TIMEOUT)
    settlement_poll_dt = p.get("settlement_poll_dt", DEFAULT_SETTLEMENT_POLL_DT)
    validate_settlement(settlement_timeout, settlement_poll_dt)
    cfg = cfg_from_params(p)
    if qty % getattr(cfg, "lot_size", 1):
        raise ValueError("qty must be a multiple of lot_size")
    sim = ExchangeSimulator(cfg)
    for _ in range(int(WARMUP_S / 0.1)):
        sim.step(0.1)
    t0, arrival = sim.t, sim.book.mid()
    agent = make_baseline(name, p, cfg, t0)
    owner = name.upper()
    series = _new_series()
    if record:
        _push(series, 0.0, sim, qty)
    steps = max(1, int(math.ceil(horizon / dt - 1e-12)))
    for i in range(steps):
        agent.on_step(sim, owner)
        sim.step(min(dt, max(0.0, horizon - (sim.t - t0))))
        agent.poll_fills(sim, owner)
        if record:
            _push(series, sim.t - t0, sim, agent.remaining)
        if progress:
            progress("baseline", (i + 1) / steps)
        if agent.remaining == 0:
            break
    decision_duration = sim.t - t0
    ledger = agent._ensure_ledger(sim)
    filled_at_stop, fees_at_stop = agent.filled, ledger.total_fees
    settlement = settle_orders(sim, agent.risk, lambda: agent.poll_fills(sim, owner),
                               timeout=settlement_timeout, poll_dt=settlement_poll_dt,
                               on_step=(lambda: _push(series, sim.t - t0, sim, agent.remaining)) if record else None)
    liq, fillable = sim.book.walk_cost(agent.side, agent.remaining) if agent.remaining else (None, 0)
    report = build_report(agent.label, agent.side, qty, agent.fills, arrival,
                          sim.cfg.tick_size, agent.child_orders, decision_duration, liq,
                          leftover_fillable_qty=fillable, fees=agent.fees,
                          total_fees=ledger.total_fees,
                          terminal_penalty_bps=float(p.get("terminal_penalty_bps", 25.0)),
                          strict_terminal=True, outstanding_qty=agent.risk.outstanding(sim),
                          accounting=asdict(ledger.snapshot(sim.book.mid() * sim.cfg.tick_size)),
                          risk_rejections=sum(e["event"] == "rejected" for e in agent.risk.events),
                          decision_horizon=horizon, settlement_duration=settlement.duration,
                          settlement_complete=settlement.complete,
                          late_filled_qty=agent.filled - filled_at_stop,
                          late_fees=ledger.total_fees - fees_at_stop,
                          settlement_pending_order_ids=settlement.pending_order_ids)
    out = _run_dict(agent.label, series, agent.fills, t0, arrival, report)
    out["implementation"] = name
    if p.get("record_audit"):
        out["audit"] = _audit(sim, agent.risk, agent.fills, owner, t0)
    return out


# ------------------------------------------------------------------ policies
def load_policy_model(model_path: str, *, strict: bool = False) -> Tuple[Optional[Any], str]:
    """(model, label). Falls back to the heuristic with a clear label when no model exists."""
    mp = Path(str(model_path))
    if mp.with_suffix(".zip").exists() or mp.exists():
        try:
            from stable_baselines3 import PPO
            return PPO.load(str(mp)), "RL Agent (PPO)"
        except Exception as exc:
            if strict:
                raise RuntimeError(f"PPO model could not be loaded: {mp}") from exc
            return None, "RL Agent (heuristic — model load failed)"
    if strict:
        raise FileNotFoundError(f"PPO model not found: {mp}; explicitly select heuristic to evaluate it")
    return None, "RL Agent (untrained heuristic)"


def heuristic_action(env: LOBExecutionEnv) -> int:
    """Passive join; turn aggressive when behind a linear schedule by >10%."""
    behind = (env.step_i / env.n_steps) > (1.0 - env.remaining / env.total_qty) + 0.1
    return 4 if behind else 1


def run_policy(p: Dict[str, Any], policy: str = "ppo", progress: Optional[ProgressFn] = None,
               model: Optional[Any] = None, record: bool = True) -> Dict[str, Any]:
    qty, horizon, dt = int(p["qty"]), float(p["horizon"]), float(p["dt"])
    env = LOBExecutionEnv(total_qty=qty, horizon=horizon, decision_dt=dt,
                          warmup=WARMUP_S, cfg=cfg_from_params(p),
                          side=_side_from_params(p),
                          fees=p.get("fees"), risk=p.get("risk"),
                          terminal_penalty_bps=float(p.get("terminal_penalty_bps", 25.0)),
                          settlement_timeout=p.get("settlement_timeout", DEFAULT_SETTLEMENT_TIMEOUT),
                          settlement_poll_dt=p.get("settlement_poll_dt", DEFAULT_SETTLEMENT_POLL_DT),
                          record=bool(p.get("record_audit")))
    obs, _ = env.reset(seed=int(p["seed"]))

    if policy == "ppo":
        if model is None:
            model, label = load_policy_model(str(p.get("model_path", "models/ppo_lob")),
                                             strict=bool(p.get("strict_model", False)))
        else:
            label = "RL Agent (PPO)"
    elif policy == "heuristic":
        model, label = None, "Heuristic"
    elif policy == "random":
        model, label = None, "Random"
    else:
        raise KeyError(f"unknown policy '{policy}' (choose from {', '.join(POLICIES)})")
    rng = np.random.default_rng(int(p["seed"]) + 7)

    series = _new_series()
    if record:
        _push(series, 0.0, env.sim, env.remaining)
    done = False
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        elif policy == "random":
            action = int(rng.integers(0, env.action_space.n))
        else:
            action = heuristic_action(env)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if record:
            _push(series, info["exchange_time"], env.sim, env.remaining)
        if progress:
            progress("rl", min(1.0, env.step_i / env.n_steps))
    out = _run_dict(label, series, env.fills, env.t0, env.arrival, env.report(label))
    out["implementation"] = policy if model is not None or policy != "ppo" else "heuristic"
    out["raw"]["episode_reward"] = env.total_reward
    if p.get("record_audit"):
        out["audit"] = _audit(env.sim, env.risk, env.fills, env.OWNER, env.t0)
        out["audit"]["decisions"] = env.decisions
    return out


def run_rl(p: Dict[str, Any], progress: Optional[ProgressFn] = None) -> Dict[str, Any]:
    """Backwards-compatible alias used by the web server."""
    return run_policy(p, "ppo", progress)


# ------------------------------------------------------------------ entry points
def run_pair(p: Dict[str, Any], progress: Optional[ProgressFn] = None) -> Dict[str, Any]:
    """Selected baseline and the RL policy on markets built from the same seed."""
    base = run_baseline(p, progress)
    rl = run_rl(p, progress)
    return {"tick": cfg_from_params(p).tick_size, "params": p, "decision_dt": float(p["dt"]),
            "runs": {"baseline": base, "rl": rl}}


def run_episode(agent: str, p: Dict[str, Any], model: Optional[Any] = None) -> Dict[str, Any]:
    """One compact metrics row for evaluation (no per-step series)."""
    if agent in BASELINES:
        out = run_baseline(p, name=agent, record=False)
    elif agent in POLICIES:
        out = run_policy(p, agent, model=model, record=False)
    else:
        raise KeyError(f"unknown agent '{agent}' (choose from {', '.join(AGENTS)})")
    raw = out["raw"]
    row = {
        "agent": agent, "label": out["label"], "seed": int(p["seed"]),
        "implementation": out["implementation"],
        "shortfall_bps": float(raw["shortfall_bps"]),
        "effective_bps": raw["effective_bps"],
        "fill_frac": raw["filled"] / max(1, raw["target"]),
        "duration": float(raw["duration"]), "children": int(raw["children"]),
        "vwap": float(raw["vwap"]), "arrival": float(raw["arrival_price"]),
    }
    for key in ("gross_cost", "total_cost", "total_fees", "fee_bps", "gross_effective_bps",
                "net_effective_bps", "hypothetical_liquidation_cost", "hypothetical_liquidation_fees",
                "completion_penalty_bps", "optimization_cost_bps", "unpriced_leftover_qty",
                "fillable_leftover_qty", "outstanding_qty", "status", "invalid_reasons", "accounting",
                "decision_horizon", "decision_duration", "settlement_duration", "total_duration",
                "settlement_complete", "late_filled_qty", "late_fees", "settlement_pending_order_ids"):
        row[key] = raw[key]
    if "audit" in out:
        row["audit"] = out["audit"]
    if "episode_reward" in raw:
        row["episode_reward"] = raw["episode_reward"]
    return row


def _audit(sim: ExchangeSimulator, risk: Any, fills: List[Any], owner: str, t0: float) -> Dict[str, Any]:
    """JSON-safe order/fill/risk evidence; strategy orders only."""
    orders = []
    for oid in risk.order_ids:
        order = sim.orders[oid]
        orders.append({"order_id": oid, "side": order.side.name.lower(), "qty": order.qty,
                       "remaining": order.remaining, "price_ticks": order.price,
                       "submitted_at": order.ts_submit - t0, "active_at": order.ts_active - t0,
                       "status": order.status.value,
                       "status_history": [{"time": t - t0, "status": state.value}
                                          for t, state in order.status_history]})
    return {"orders": orders,
            "seed_manifest": sim.seed_manifest,
            "event_count": sim.event_count,
            "fills": [{"time": tr.time - t0, "price_ticks": tr.price, "qty": tr.qty,
                       "maker": tr.maker_owner == owner,
                       "maker_order_id": tr.maker_order_id, "taker_order_id": tr.taker_order_id}
                      for tr in fills],
            "risk_events": [dict(event, t=event["t"] - t0) for event in risk.events]}
