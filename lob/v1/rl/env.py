"""Gymnasium-compatible execution research environment (M120).

``ExecutionEnv`` steps one execution mandate inside a simulated market (F4 classical / F5 neural via the v0.8
execution simulator with the corrected book, or F6 via the order-level ecology), one decision every 5 s:

* observation: the registered v0.9 RL state (time left, inventory left, spread, signed 5-level imbalance, log touch
  depth, recent volatility, behind-schedule gap), float32, shape (7,);
* actions (Discrete(4)): 0 wait, 1 rest the TWAP slice passively, 2 market the slice, 3 market twice the slice;
* reward: minus the bps cost (vs arrival mid, incl. registered fees) of the fills between this decision and the
  next; the terminal step adds the forced-completion residual, so the episode return equals minus the episode cost
  up to the simulator's residual-valuation convention (|difference| < 0.05 bps in the tests).

It runs the episode by driving a policy whose ``decide`` blocks on the agent's action (a generator coroutine), so the
simulator's semantics are untouched. Seeds come from ``reset(seed=...)``. Gymnasium is a base dependency.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from ...v08.execution.policies import Policy
from ...v09 import rl as R9

N_ACTIONS = R9.N_ACTIONS


class _Bridge(Policy):
    name = "env_bridge"
    defaults = {}

    def __init__(self, env) -> None:
        super().__init__()
        self.env = env

    def decide(self, obs):
        s = R9.state_of(obs)
        a = self.env._request(s, obs.t)
        return R9.act(obs, a)


class ExecutionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, generator, *, qty: int = 50, side: int = 1, horizon_s: float = 300.0) -> None:
        self.generator, self.qty, self.side, self.horizon_s = generator, int(qty), int(side), float(horizon_s)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(len(R9.STATE),), dtype=np.float32)
        self._trace = None

    # the episode runs eagerly with a pre-committed action plan supplied step by step
    def _request(self, s, t):
        self._states.append((np.asarray(s, np.float32), t))
        k = len(self._states) - 1
        return self._plan[k] if k < len(self._plan) else 0

    def _run(self, plan: list[int]):
        from ...v08.execution.simulator import Episode, Fees, Mandate
        from ...v08.worlds import World
        from ..generators.classical.book import MarketSimV1
        import importlib
        sim = importlib.import_module("lob.v08.execution.simulator")
        saved = sim.MarketSim
        sim.MarketSim = MarketSimV1
        try:
            self._plan, self._states = list(plan), []
            mandate = Mandate(side=self.side, qty=self.qty, horizon_s=self.horizon_s, decision_dt_s=5.0,
                              warmup_s=30.0, completion="hard", fees=Fees(taker_bps=5.5, maker_bps=2.0))
            ep = Episode(World("env", self.generator, self.generator.book), _Bridge(self), mandate, self._seed,
                         context={"seed": self._seed})
            rep = ep.run()
        finally:
            sim.MarketSim = saved
        arrival = rep["arrival_mid_ticks"]
        times = [t for _, t in self._states] + [np.inf]
        rewards = np.zeros(len(self._states))
        for f in ep.fills:
            k = max(0, int(np.searchsorted(times, f[0], side="right")) - 1)
            if k < len(rewards):
                rewards[k] -= self.side * (f[1] - arrival) * f[2] / (arrival * self.qty) * 1e4 + \
                    f[2] / self.qty * (2.0 if f[3] else 5.5)
        if len(rewards):
            rewards[-1] -= (rep.get("hypothetical_residual_cost_bps") or 0.0) + rep.get("completion_penalty_bps", 0.0)
        return [s for s, _ in self._states], rewards, rep

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._seed = int(seed if seed is not None else 0)
        self._actions: list[int] = []
        states, _, _ = self._run([])
        self._k = 0
        return states[0], {"steps": len(states)}

    def step(self, action: int):
        self._actions.append(int(action))
        states, rewards, rep = self._run(self._actions)
        k = len(self._actions) - 1
        done = k >= len(states) - 1
        r = float(rewards[k]) if not done else float(rewards[k:].sum())
        obs = states[min(k + 1, len(states) - 1)]
        info = {"episode_cost_bps": rep.get("optimization_cost_bps")} if done else {}
        return obs, r, done, False, info


__all__ = ["ExecutionEnv", "N_ACTIONS"]
