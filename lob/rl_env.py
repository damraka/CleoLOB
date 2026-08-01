"""Gymnasium environment: optimal execution on the simulated LOB."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .engine import ExchangeSimulator, Side, SimConfig
from .execution import ExecutionReport, build_report


class LOBExecutionEnv(gym.Env):
    """Liquidate ``total_qty`` within ``horizon`` seconds, minimising implementation shortfall.

    Observation (24,):
        top-5 bids  : (price − mid)/10 ticks, volume/500        -> 10 features
        top-5 asks  : same                                       -> 10
        micro-price − mid (ticks), book imbalance (top-5),
        remaining inventory fraction, remaining time fraction    -> 4

    Actions (Discrete 5):
        0  wait (keep nothing working)
        1  passive  : limit joining own-side best quote
        2  improve  : limit one tick inside the spread
        3  cross    : marketable limit at the opposite touch
        4  aggress  : market order (1.5× child size)

    Reward: per-fill implementation shortfall in bps (sign-adjusted, share-weighted),
    plus a terminal penalty pricing leftovers at their walk-the-book liquidation VWAP
    and an extra fixed penalty per leftover share.
    """

    metadata: Dict[str, List[str]] = {"render_modes": []}
    OWNER = "RL"

    def __init__(self, total_qty: int = 10_000, horizon: float = 60.0,
                 decision_dt: float = 0.5, side: Side = Side.SELL,
                 warmup: float = 5.0, cfg: Optional[SimConfig] = None,
                 record: bool = False) -> None:
        super().__init__()
        self.total_qty = total_qty
        self.horizon = horizon
        self.decision_dt = decision_dt
        self.side = side
        self.warmup = warmup
        self.base_cfg = cfg or SimConfig()
        self.record = record
        self.n_steps = max(1, int(round(horizon / decision_dt)))
        self.child = max(1, total_qty // max(1, self.n_steps // 2))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        sim_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        cfg = SimConfig(**{**self.base_cfg.__dict__, "seed": sim_seed})
        self.sim = ExchangeSimulator(cfg)
        for _ in range(int(self.warmup / 0.1)):
            self.sim.step(0.1)
        self.t0 = self.sim.t
        self.arrival = self.sim.book.mid()
        self.remaining = self.total_qty
        self.fills: List[Any] = []
        self._cursor = len(self.sim.book.trades)
        self._live_order: Optional[int] = None
        self.n_children = 0
        self.step_i = 0
        self.snapshots: List[Dict[str, Any]] = []
        if self.record:
            self._snap()
        return self._obs(), self._info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        sim, book = self.sim, self.sim.book
        if self._live_order is not None:      # cancel/replace previous child
            sim.cancel(self._live_order)      # cancel travels with latency too
            self._live_order = None
        qty = min(self.child, self.remaining)
        bb, ba = book.best_bid(), book.best_ask()
        if qty > 0 and bb is not None and ba is not None:
            if action == 1:                                   # passive join
                px = ba if self.side is Side.SELL else bb
                self._live_order = sim.submit(self.side, qty, self.OWNER, price_ticks=px)
                self.n_children += 1
            elif action == 2:                                 # improve inside spread
                px = (ba - 1) if self.side is Side.SELL else (bb + 1)
                if not ((self.side is Side.SELL and px > bb) or
                        (self.side is Side.BUY and px < ba)):
                    px = ba if self.side is Side.SELL else bb  # spread=1: fall back to join
                self._live_order = sim.submit(self.side, qty, self.OWNER, price_ticks=px)
                self.n_children += 1
            elif action == 3:                                 # marketable limit at touch
                px = bb if self.side is Side.SELL else ba
                self._live_order = sim.submit(self.side, qty, self.OWNER, price_ticks=px)
                self.n_children += 1
            elif action == 4:                                 # aggressive market
                sim.submit(self.side, min(self.remaining, int(1.5 * self.child)), self.OWNER)
                self.n_children += 1

        sim.step(self.decision_dt)
        self.step_i += 1

        new, self._cursor = sim.fills_for(self.OWNER, self._cursor)
        sign = 1.0 if self.side is Side.SELL else -1.0
        reward = 0.0
        for tr in new:
            self.fills.append(tr)
            self.remaining = max(0, self.remaining - tr.qty)
            reward += sign * (tr.price - self.arrival) / self.arrival * 1e4 \
                * (tr.qty / self.total_qty)

        terminated = self.remaining <= 0
        if self.step_i >= self.n_steps and not terminated:
            # Price leftovers at their immediate walk-the-book liquidation VWAP …
            vwap_est, _ = book.walk_cost(self.side, self.remaining)
            reward += sign * (vwap_est - self.arrival) / self.arrival * 1e4 \
                * (self.remaining / self.total_qty)
            # … plus a hard penalty so "do nothing" is never optimal.
            reward -= 25.0 * (self.remaining / self.total_qty)
            terminated = True
        if self.record:
            self._snap()
        return self._obs(), float(reward), terminated, False, self._info()

    # ------------------------------------------------------------ helpers
    def _obs(self) -> np.ndarray:
        book = self.sim.book
        mid = book.mid()
        bids, asks = book.depth(5)
        feats: List[float] = []
        for i in range(5):
            p, v = bids[i] if i < len(bids) else (mid - (i + 1), 0)
            feats += [(p - mid) / 10.0, v / 500.0]
        for i in range(5):
            p, v = asks[i] if i < len(asks) else (mid + (i + 1), 0)
            feats += [(p - mid) / 10.0, v / 500.0]
        feats += [
            book.microprice() - mid,
            book.imbalance(5),
            self.remaining / self.total_qty,
            max(0.0, 1.0 - self.step_i / self.n_steps),
        ]
        return np.asarray(feats, dtype=np.float32)

    def _info(self) -> Dict[str, Any]:
        return {"remaining": self.remaining,
                "filled": self.total_qty - self.remaining,
                "t": self.sim.t - self.t0}

    def _snap(self) -> None:
        bids, asks = self.sim.book.depth(10)
        self.snapshots.append({"t": self.sim.t - self.t0, "bids": bids, "asks": asks,
                               "mid": self.sim.book.mid(), "remaining": self.remaining})

    def report(self, label: str = "RL Agent (PPO)") -> ExecutionReport:
        return build_report(label, self.side, self.total_qty, self.fills, self.arrival,
                            self.sim.cfg.tick_size, self.n_children, self.sim.t - self.t0)
