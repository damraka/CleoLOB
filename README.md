# CLEO · LOB Execution Lab

Event-driven **limit order book simulator** with realistic market microstructure, an
**Almgren-Chriss** optimal-execution baseline, and a **PPO reinforcement-learning agent**
competing head-to-head on the same simulated market — visualised in a custom web app
with a 3D "liquidity canyon" view.

## Features

**Simulation core**
- **Matching engine** — strict price-time priority; limit, market and cancel messages.
  Book stored as `dict[price] → deque` (FIFO queues) with `bisect`-sorted price ladders:
  O(1) best-of-book, O(log n) level insertion.
- **Latency simulation** — every strategy message travels through a global event heap and
  arrives after `base + Exp(jitter)` delay. Queue position is earned at *arrival*, and
  cancels can lose the race against incoming fills.
- **Slippage & market impact** — market orders consume real depth, so slippage is
  emergent, not modelled. A resilience process refills depleted top levels toward a
  target depth, reproducing post-impact book recovery.
- **Background flow** — zero-intelligence Poisson limit/market/cancel arrivals with a
  per-order cancel hazard keep the book stationary and the spread dynamic.

**Agents**
- **Almgren-Chriss baseline** — closed-form implementation-shortfall schedule
  `x(t) = X·sinh(κ(T−t))/sinh(κT)`, released as market-order slices.
- **RL agent** — `gymnasium` env (24-dim book state: top-5 levels, micro-price,
  imbalance, inventory, time) with a 5-action passive→aggressive ladder, trained with
  `stable-baselines3` PPO on an IS-based reward with a leftover-inventory penalty.

**CLEO web app** (`server.py` — FastAPI + custom Three.js/canvas frontend, no UI framework)
- 3D **liquidity canyon**: bid/ask cumulative-depth terraces over time, mid-price path on
  the valley floor, fill spikes — orbit, zoom, PNG export.
- DAW-style transport: play/pause, speed, time scrubber synchronised across every panel;
  click any time chart to seek.
- Execution-trajectory and mid-price/fills charts, live 2D depth at the cursor,
  a scrolling fill tape, shortfall metric cards and a full comparison table.
- Market presets (calm / stressed / low-latency / impatient-AC), background jobs with
  progress, JSON export of the whole run.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m lob                            # 30 s smoke test of the raw engine
python train_rl.py --timesteps 200000    # optional; ~15–30 min on CPU
python server.py                         # CLEO app -> http://127.0.0.1:8000
```

The app works without a trained model (RL side falls back to a labelled heuristic),
but the comparison only gets interesting after training.

`app.py` is a legacy Streamlit dashboard kept for reference: `streamlit run app.py`.

## Project structure

```
lob/
  engine.py      # order book, matching, latency, ZI flow, resilience
  execution.py   # Almgren-Chriss agent + implementation-shortfall reporting
  rl_env.py      # gymnasium LOBExecutionEnv
  runner.py      # headless orchestration -> JSON payloads
server.py        # FastAPI backend for the CLEO web app
static/          # index.html · css/app.css · js/app.js (Three.js canyon + canvas charts)
train_rl.py      # PPO training pipeline (stable-baselines3)
app.py           # legacy Streamlit dashboard
```

## Screenshots

<!-- Add: docs/canyon.png (3D view) and docs/overview.png (metrics + charts) -->

## Roadmap

- Rust core (`pyo3` bindings) for a 100–1000× faster inner loop
- Avellaneda-Stoikov market-making agent as a second baseline
- Continuous action space (SAC) — price offset + child size
- Historical replay from LOBSTER / crypto L2 feeds
