# Archived README from the pre-foundation working tree

The content below is preserved as project history. Its results describe the old
simulator; its scientific claims and seed-to-market mapping have not been
revalidated under the current engine. Use the current README and research
methodology instead. This file is not evidence for current performance.

# CLEO · LOB Execution Lab

Event-driven **limit order book simulator** with realistic market microstructure, a
**zoo of execution baselines** (TWAP, VWAP, POV, Almgren-Chriss), a **PPO
reinforcement-learning agent**, a **paired multi-seed evaluation harness**, and a custom
web app with a 3D "liquidity canyon" view.

> Design principle: the platform is not optimised to make PPO look good — it is
> optimised to make PPO *capable of being proven wrong*. Every agent runs on identical
> markets, unfilled inventory is charged at liquidation cost, and every reported number
> comes with a bootstrap confidence interval and a paired significance test.

## Features

**Simulation core** (`lob/engine.py`)
- **Matching engine** — strict price-time priority; limit, market and cancel messages.
  `dict[price] → deque` FIFO queues with `bisect`-sorted price ladders: O(1) best-of-book,
  O(log n) level insertion.
- **Latency** — every strategy message travels through a global event heap and arrives
  after `base + Exp(jitter)`. Queue position is earned at *arrival*; cancels can lose the
  race against incoming fills.
- **Slippage & impact** — market orders consume real depth, so slippage is emergent. A
  resilience process refills depleted levels toward a target depth (post-impact recovery).
- **Background flow** — zero-intelligence Poisson limit/market/cancel arrivals with a
  per-order cancel hazard; the book is stationary and fully seed-reproducible.

**Agents** (`lob/execution.py`, `lob/rl_env.py`)
| agent | type | idea |
|---|---|---|
| TWAP | schedule | equal slices at equal intervals |
| VWAP | schedule | slices follow a forecast volume profile (averaged "previous days") |
| Almgren-Chriss | schedule | `x(t) = X·sinh(κ(T−t))/sinh(κT)`, κ from impact/risk-aversion |
| POV | reactive | keep own fills ≈ p/(1−p) × observed market volume, catch-up near T |
| PPO | policy | `gymnasium` env, 24-dim book state, 5-action passive→aggressive ladder |
| heuristic / random | policy | passive-then-aggressive rule · uniform random (floor) |

**Evaluation** (`evaluate.py`, `lob/scenarios.py`, `lob/stats.py`)
- Five standard scenarios: `calm`, `thin` (large order / thin book), `volatile`,
  `high_latency`, `stressed`. Every agent sees the same scenario **and the same seed**.
- **Effective implementation shortfall**: unfilled inventory at the horizon is priced at
  the walk-the-book liquidation VWAP, so no agent can look cheap by not finishing.
- Mean / median / std / p5 / p95 / worst case, 95 % bootstrap CIs, and **paired-by-seed**
  Δ vs a reference agent with win rate and an exact sign-test p-value.
- Outputs `episodes.csv`, `summary.csv`, `paired.csv`, `report.md`, `report.html`
  (distribution plots) and `meta.json` (seeds, git commit, package versions).

**CLEO web app** (`server.py` + `static/`, FastAPI + Three.js/canvas, no UI framework)
- 3D **liquidity canyon**: bid/ask cumulative-depth terraces over time, mid-price path,
  fill spikes — orbit, zoom, PNG export. Baseline selectable in the sidebar.
- DAW-style transport: play/pause, speed, time scrubber synchronised across every panel.
- Trajectory and mid-price/fills charts, live 2D depth, fill tape, metric cards, table.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

pytest -q                                # 64 tests, ~5 s
python -m lob                            # engine smoke test: flow + liquidity shock + refill
python train_rl.py --timesteps 200000    # PPO, ~15–30 min CPU; curves: tensorboard --logdir logs
python evaluate.py --episodes 200        # all agents × 5 scenarios × 200 seeds → results/<stamp>/
python server.py                         # CLEO app → http://127.0.0.1:8000
```

`evaluate.py --agents ac,twap,pov,ppo --scenarios calm,thin --reference twap` narrows the
run; `--workers N` parallelises across seeds. Without a trained model the `ppo` rows fall
back to the heuristic and say so.

## Example results — baselines, 100 paired seeds per scenario

Effective implementation shortfall in bps (positive = cost), mean with 95 % bootstrap CI;
best per scenario in bold. Reproduce with
`python evaluate.py --episodes 100 --agents ac,twap,vwap,pov,heuristic,random`
(raw episodes and paired tests in `results/baselines-100seeds/`).

| agent | calm | thin | volatile | high_latency | stressed |
|---|---:|---:|---:|---:|---:|
| TWAP | **+3.8 [+3.5, +4.0]** | +17.1 [+16.0, +18.2] | +6.1 [+5.2, +6.9] | +4.1 [+3.9, +4.3] | +5.8 [+5.2, +6.4] |
| VWAP | +3.8 [+3.5, +4.0] | **+16.7 [+15.5, +17.9]** | +5.2 [+4.5, +5.9] | +3.7 [+3.5, +4.0] | +5.7 [+5.0, +6.4] |
| Almgren-Chriss | +3.8 [+3.5, +4.0] | +34.1 [+31.6, +36.8] | +5.5 [+4.7, +6.3] | **+3.6 [+3.4, +3.9]** | +6.1 [+5.6, +6.7] |
| POV | +4.1 [+3.9, +4.3] | +33.2 [+31.4, +35.3] | +8.5 [+7.8, +9.1] | +3.9 [+3.7, +4.1] | +7.7 [+7.1, +8.3] |
| Heuristic | +3.9 [+3.7, +4.1] | +18.1 [+17.3, +18.8] | **+4.5 [+2.4, +6.0]** | +3.8 [+3.6, +4.0] | **+5.7 [+4.9, +6.4]** |
| Random | +4.4 [+4.2, +4.7] | +21.7 [+21.0, +22.4] | +5.7 [+3.7, +7.2] | +4.2 [+3.8, +4.5] | +5.8 [+3.9, +7.4] |

What the numbers say:

- **Calm book: the algorithm barely matters.** TWAP, VWAP and Almgren-Chriss are
  indistinguishable (paired Δ ≈ 0.0 bps, sign-test p ≈ 0.6).
- **Thin book: Almgren-Chriss is punished.** Its impact parameter η was calibrated for the
  calm regime, so the schedule front-loads into a book that cannot absorb it — TWAP/VWAP
  are ~17 bps cheaper on 85–88 % of seeds (p < 10⁻¹²). A textbook case of a model being
  right *given its parameters* and wrong given the market.
- **High latency: front-loading wins.** The same Almgren-Chriss schedule is the cheapest
  agent and beats TWAP on a paired basis (Δ = −0.5 bps, p = 0.012): shortening the
  exposure window matters more when queue position degrades.
- The "passive until behind" heuristic looks cheap on filled shares but pays for its ~9 %
  leftovers once they are priced at liquidation — which is exactly why the effective
  metric exists (and why the RL reward carries the same leftover penalty).

Numbers for the trained PPO policy depend on your checkpoint; run
`python evaluate.py --episodes 200 --agents ac,twap,vwap,ppo` after `train_rl.py` to get
them with the same confidence intervals and paired tests.

## Project structure

```
lob/
  engine.py      # order book, matching, latency, ZI flow, resilience
  execution.py   # ExecutionAgent → TWAP / VWAP / Almgren-Chriss / POV, reporting
  rl_env.py      # gymnasium LOBExecutionEnv
  runner.py      # any agent on identical markets; UI payloads + compact episodes
  scenarios.py   # standard benchmark scenarios
  stats.py       # bootstrap CIs, paired-by-seed comparison, sign test
evaluate.py      # multi-seed evaluation harness → CSV / Markdown / HTML reports
train_rl.py      # PPO training (stable-baselines3, TensorBoard logging)
server.py        # FastAPI backend for the CLEO web app
static/          # index.html · css/app.css · js/app.js
tests/           # 64 pytest tests: engine invariants, agents, env, runner, stats
app.py           # legacy Streamlit dashboard
```

## Roadmap

- Historical order-book replay (LOBSTER / crypto L2) and calibration against real data
- Hawkes-process order flow
- Continuous action space (SAC): price offset + child size
- Transaction-cost decomposition (spread / impact / timing / opportunity)
- Rust core via `pyo3`
