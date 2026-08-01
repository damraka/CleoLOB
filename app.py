"""LOB Execution Lab — Streamlit dashboard.

Runs the Almgren-Chriss baseline and the RL agent on markets built from the same seed,
then compares depth, execution trajectories and implementation-shortfall metrics.

    streamlit run app.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lob.engine import ExchangeSimulator, Side, SimConfig
from lob.execution import AlmgrenChrissAgent, ExecutionReport, build_report
from lob.rl_env import LOBExecutionEnv

st.set_page_config(page_title="LOB Execution Lab", page_icon="📈", layout="wide")

WARMUP_S = 5.0


@dataclass
class RunResult:
    label: str
    snapshots: List[Dict[str, Any]]
    trajectory: List[Tuple[float, int]]   # (t, remaining)
    report: ExecutionReport


# ------------------------------------------------------------------ runners
def run_baseline(qty: int, horizon: float, dt: float, seed: int, cfg: SimConfig,
                 risk_aversion: float) -> RunResult:
    sim = ExchangeSimulator(SimConfig(**{**cfg.__dict__, "seed": seed}))
    for _ in range(int(WARMUP_S / 0.1)):
        sim.step(0.1)
    t0, arrival = sim.t, sim.book.mid()
    agent = AlmgrenChrissAgent(qty, horizon, n_slices=20, side=Side.SELL,
                               risk_aversion=risk_aversion, start_time=t0)
    owner = "AC"
    snaps: List[Dict[str, Any]] = []
    traj: List[Tuple[float, int]] = [(0.0, qty)]

    def snap() -> None:
        bids, asks = sim.book.depth(10)
        snaps.append({"t": sim.t - t0, "bids": bids, "asks": asks,
                      "mid": sim.book.mid(), "remaining": qty - agent.filled})

    snap()
    for _ in range(int(round(horizon / dt))):
        agent.on_step(sim, owner)
        sim.step(dt)
        agent.poll_fills(sim, owner)
        snap()
        traj.append((sim.t - t0, qty - agent.filled))
        if agent.filled >= qty:
            break
    report = build_report("Almgren-Chriss", Side.SELL, qty, agent.fills, arrival,
                          sim.cfg.tick_size, agent.child_orders, sim.t - t0)
    return RunResult("Almgren-Chriss", snaps, traj, report)


def run_rl(qty: int, horizon: float, dt: float, seed: int, cfg: SimConfig,
           model_path: str) -> RunResult:
    env = LOBExecutionEnv(total_qty=qty, horizon=horizon, decision_dt=dt,
                          warmup=WARMUP_S, cfg=cfg, record=True)
    obs, info = env.reset(seed=seed)
    model = None
    label = "RL Agent (PPO)"
    if Path(model_path).with_suffix(".zip").exists() or Path(model_path).exists():
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
    else:
        label = "RL Agent (untrained heuristic)"
    traj: List[Tuple[float, int]] = [(0.0, qty)]
    done = False
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:  # fallback: passive, turn aggressive when behind schedule
            behind = (env.step_i / env.n_steps) > (1.0 - env.remaining / env.total_qty) + 0.1
            action = 4 if behind else 1
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        traj.append((info["t"], info["remaining"]))
    return RunResult(label, env.snapshots, traj, env.report(label))


# ------------------------------------------------------------------- charts
def depth_figure(snap: Dict[str, Any], tick: float) -> go.Figure:
    fig = go.Figure()
    for name, levels, color in (("Bids", snap["bids"], "#16a34a"),
                                ("Asks", snap["asks"], "#dc2626")):
        if levels:
            xs = [p * tick for p, _ in levels]                       # best level first
            ys = list(pd.Series([v for _, v in levels]).cumsum())    # cumulative depth
            fig.add_trace(go.Scatter(x=xs, y=ys, name=name, line_shape="hv",
                                     fill="tozeroy", line=dict(color=color, width=2)))
    fig.add_vline(x=snap["mid"] * tick, line_dash="dot", line_color="gray",
                  annotation_text="mid")
    fig.update_layout(title=f"Order book depth · t = {snap['t']:.1f}s",
                      xaxis_title="Price ($)", yaxis_title="Cumulative size",
                      height=380, margin=dict(l=10, r=10, t=45, b=10),
                      legend=dict(orientation="h", y=1.12))
    return fig


def trajectory_figure(runs: List[RunResult]) -> go.Figure:
    fig = go.Figure()
    for run, color in zip(runs, ("#2563eb", "#f59e0b")):
        xs = [p[0] for p in run.trajectory]
        ys = [p[1] for p in run.trajectory]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=run.label,
                                 line=dict(color=color, width=2)))
    fig.update_layout(title="Execution trajectory · remaining inventory vs time",
                      xaxis_title="Time (s)", yaxis_title="Remaining (shares)",
                      height=380, margin=dict(l=10, r=10, t=45, b=10),
                      legend=dict(orientation="h", y=1.12))
    return fig


# ----------------------------------------------------------------------- UI
st.title("📈 LOB Execution Lab")
st.caption("Price-time priority matching · latency & book-resilience simulation · "
           "Almgren-Chriss baseline vs PPO execution agent, head-to-head on the same seed.")

with st.sidebar:
    st.header("Execution task")
    qty = st.number_input("Parent order (shares to SELL)", 1_000, 100_000, 10_000, step=1_000)
    horizon = st.slider("Horizon T (seconds)", 20, 180, 60, step=10)
    dt = st.select_slider("Decision interval (s)", options=[0.25, 0.5, 1.0], value=0.5)
    seed = st.number_input("Seed — same seed ⇒ same market conditions", 0, 100_000, 42)

    st.header("Market microstructure")
    latency_ms = st.slider("Mean order latency (ms)", 0, 50, 10)
    resilience = st.slider("Book resilience (refill intensity)", 0.0, 2.0, 0.8, 0.1)
    market_rate = st.slider("Background market-order rate (/s per side)", 1.0, 20.0, 6.0)

    st.header("Agents")
    risk_aversion = st.select_slider("Almgren-Chriss risk aversion λ",
                                     options=[1e-7, 1e-6, 1e-5, 1e-4], value=1e-6,
                                     format_func=lambda x: f"{x:.0e}")
    model_path = st.text_input("PPO model path", "models/ppo_lob")
    run_btn = st.button("▶ Run head-to-head", type="primary", use_container_width=True)

if run_btn:
    cfg = SimConfig(latency_base=latency_ms / 2000.0, latency_jitter=latency_ms / 2000.0,
                    resilience=float(resilience), market_rate=float(market_rate))
    with st.spinner("Simulating Almgren-Chriss baseline…"):
        base = run_baseline(int(qty), float(horizon), float(dt), int(seed), cfg,
                            float(risk_aversion))
    with st.spinner("Simulating RL agent…"):
        rl = run_rl(int(qty), float(horizon), float(dt), int(seed), cfg, model_path)
    st.session_state["runs"] = [base, rl]

runs: Optional[List[RunResult]] = st.session_state.get("runs")
if runs is None:
    st.info("Configure the simulation in the sidebar and press **Run head-to-head**. "
            "Both agents will liquidate the same parent order on markets built "
            "from the same seed (paths diverge only through their own impact).")
    st.stop()

base, rl = runs
tick = SimConfig().tick_size

if "untrained" in rl.label:
    st.warning("No trained PPO model found — the RL side is running a simple heuristic. "
               "Train one with `python train_rl.py --timesteps 200000` and rerun.")

m1, m2, m3 = st.columns(3)
m1.metric("Arrival price", f"${base.report.arrival_price:,.2f}")
m2.metric(f"{base.label} · shortfall", f"{base.report.shortfall_bps:.1f} bps")
m3.metric(f"{rl.label} · shortfall", f"{rl.report.shortfall_bps:.1f} bps",
          delta=f"{base.report.shortfall_bps - rl.report.shortfall_bps:+.1f} bps vs AC",
          delta_color="normal")

st.subheader("Cost & slippage comparison")
st.dataframe(pd.DataFrame([base.report.as_dict(), rl.report.as_dict()]),
             use_container_width=True, hide_index=True)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(trajectory_figure(runs), use_container_width=True)
with col2:
    which = st.radio("Book view", [base.label, rl.label], horizontal=True)
    run = base if which == base.label else rl
    idx = st.slider("Snapshot (scrub through time)", 0, len(run.snapshots) - 1,
                    len(run.snapshots) // 2)
    st.plotly_chart(depth_figure(run.snapshots[idx], tick), use_container_width=True)

with st.expander("How to read this"):
    st.markdown(
        "- **Implementation shortfall** = signed difference between the arrival mid-price "
        "and the execution VWAP, in basis points. Positive = cost.\n"
        "- **Trajectory** shows how each strategy spends its inventory; Almgren-Chriss "
        "follows a precomputed sinh-schedule, the RL agent reacts to book state.\n"
        "- **Depth scrubber**: watch liquidity get consumed by child orders and refill "
        "via the resilience process.\n"
        "- Cancels and orders both travel with latency, so fills can race cancels — "
        "exactly the risk real execution desks manage."
    )
