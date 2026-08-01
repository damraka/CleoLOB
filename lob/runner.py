"""Headless run orchestration: executes baseline & RL runs, returns JSON-serializable payloads.

Shared by the CLEO web UI (server.py) and any notebook/script usage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .engine import ExchangeSimulator, Side, SimConfig
from .execution import AlmgrenChrissAgent, ExecutionReport, build_report
from .rl_env import LOBExecutionEnv

WARMUP_S = 5.0
DEPTH_LEVELS = 12
ProgressFn = Callable[[str, float], None]


def _cfg_from_params(p: Dict[str, Any]) -> SimConfig:
    return SimConfig(
        seed=int(p["seed"]),
        latency_base=float(p["latency_ms"]) / 2000.0,
        latency_jitter=float(p["latency_ms"]) / 2000.0,
        resilience=float(p["resilience"]),
        market_rate=float(p["market_rate"]),
    )


def _new_series() -> Dict[str, List[Any]]:
    return {"t": [], "mid": [], "best_bid": [], "best_ask": [], "spread": [],
            "remaining": [], "bids": [], "asks": []}


def _push(series: Dict[str, List[Any]], t: float, sim: ExchangeSimulator,
          remaining: int) -> None:
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


def _run_dict(label: str, series: Dict[str, List[Any]], fills: List[Any],
              t0: float, arrival_ticks: float, report: ExecutionReport) -> Dict[str, Any]:
    return {
        "label": label,
        "arrival_ticks": arrival_ticks,
        "series": series,
        "fills": [{"t": round(tr.time - t0, 3), "px": int(tr.price), "qty": int(tr.qty)}
                  for tr in fills],
        "report": report.as_dict(),
        "raw": {
            "arrival_price": report.arrival_price,
            "vwap": report.vwap,
            "shortfall_bps": report.shortfall_bps,
            "total_cost": report.total_cost,
            "filled": report.filled_qty,
            "target": report.target_qty,
            "children": report.n_child_orders,
            "duration": report.duration,
        },
    }


def run_baseline(p: Dict[str, Any], progress: Optional[ProgressFn] = None) -> Dict[str, Any]:
    qty, horizon, dt = int(p["qty"]), float(p["horizon"]), float(p["dt"])
    sim = ExchangeSimulator(_cfg_from_params(p))
    for _ in range(int(WARMUP_S / 0.1)):
        sim.step(0.1)
    t0, arrival = sim.t, sim.book.mid()
    agent = AlmgrenChrissAgent(qty, horizon, n_slices=20, side=Side.SELL,
                               risk_aversion=float(p["risk_aversion"]), start_time=t0)
    owner = "AC"
    series = _new_series()
    _push(series, 0.0, sim, qty)
    steps = max(1, int(round(horizon / dt)))
    for i in range(steps):
        agent.on_step(sim, owner)
        sim.step(dt)
        agent.poll_fills(sim, owner)
        _push(series, sim.t - t0, sim, qty - agent.filled)
        if progress:
            progress("baseline", (i + 1) / steps)
        if agent.filled >= qty:
            break
    report = build_report("Almgren-Chriss", Side.SELL, qty, agent.fills, arrival,
                          sim.cfg.tick_size, agent.child_orders, sim.t - t0)
    return _run_dict("Almgren-Chriss", series, agent.fills, t0, arrival, report)


def run_rl(p: Dict[str, Any], progress: Optional[ProgressFn] = None) -> Dict[str, Any]:
    qty, horizon, dt = int(p["qty"]), float(p["horizon"]), float(p["dt"])
    env = LOBExecutionEnv(total_qty=qty, horizon=horizon, decision_dt=dt,
                          warmup=WARMUP_S, cfg=_cfg_from_params(p))
    obs, _ = env.reset(seed=int(p["seed"]))

    model = None
    label = "RL Agent (untrained heuristic)"
    mp = Path(str(p.get("model_path", "models/ppo_lob")))
    if mp.with_suffix(".zip").exists() or mp.exists():
        try:
            from stable_baselines3 import PPO
            model = PPO.load(str(mp))
            label = "RL Agent (PPO)"
        except Exception:
            label = "RL Agent (heuristic — model load failed)"

    series = _new_series()
    _push(series, 0.0, env.sim, env.remaining)
    done = False
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:  # passive, turn aggressive when behind schedule
            behind = (env.step_i / env.n_steps) > (1.0 - env.remaining / env.total_qty) + 0.1
            action = 4 if behind else 1
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        _push(series, info["t"], env.sim, env.remaining)
        if progress:
            progress("rl", min(1.0, env.step_i / env.n_steps))
    return _run_dict(label, series, env.fills, env.t0, env.arrival, env.report(label))


def run_pair(p: Dict[str, Any], progress: Optional[ProgressFn] = None) -> Dict[str, Any]:
    """Baseline and RL agent on markets built from the same seed."""
    base = run_baseline(p, progress)
    rl = run_rl(p, progress)
    return {"tick": SimConfig().tick_size, "params": p,
            "decision_dt": float(p["dt"]),
            "runs": {"baseline": base, "rl": rl}}
