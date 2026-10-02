"""v0.7 transfer statistics (H12, H13) on constructed replay rows."""
from __future__ import annotations

import numpy as np

from lob.v07.transfer import study as ts

COST = ts.COST


def _rows(agent_means: dict, seeds=(1, 2, 3, 4), episodes=40, sd=1.0, seed=0, learned=True):
    rng = np.random.default_rng(seed)
    rows = []
    for agent, mean in agent_means.items():
        for s in (seeds if ":" in agent else (None,)):
            for e in range(episodes):
                for mode in ("conservative", "optimistic"):
                    rows.append({"agent": agent, "training_seed": s, "episode": e, "fill_mode": mode,
                                 COST: mean + (0.5 if mode == "conservative" else 0) + rng.normal(0, sd)})
    return rows


def _pred(agent_means: dict, seeds=(1, 2, 3, 4), n=64, seed=1):
    rng = np.random.default_rng(seed)
    return [{"agent": a, "training_seed": s, COST: m + rng.normal(0, 1.0)} for a, m in agent_means.items()
            for s in seeds for _ in range(n)]


def test_h12_posterior_gap_smaller() -> None:
    hist = _rows({"ppo:single": 10.0, "ppo:ensemble": 10.0, "dqn:single": 10.0, "dqn:ensemble": 10.0})
    pred = _pred({"ppo:single": 15.0, "ppo:ensemble": 10.5, "dqn:single": 15.0, "dqn:ensemble": 10.5})
    result = ts.h12(hist, pred, alpha=0.0125, samples=300, seed=2)
    assert result["status"] == "ESTABLISHED"
    assert all(m["status"] == "ESTABLISHED" for m in result["members"].values())
    worse = ts.h12(hist, _pred({"ppo:single": 10.2, "ppo:ensemble": 16.0, "dqn:single": 10.2, "dqn:ensemble": 16.0}),
                   alpha=0.0125, samples=300, seed=2)
    assert all(m["status"] == "FAILED" for m in worse["members"].values())


def test_h12_missing_rows_not_available() -> None:
    assert ts.h12([], [], alpha=0.0125, samples=10, seed=0)["status"] == "NOT_AVAILABLE"


def test_h13_survival_and_failure() -> None:
    rows = _rows({"twap": 5.0, "vwap": 8.0, "pov": 5.1}, learned=False)
    assert ts.h13(rows, {}, alpha=0.05, samples=200, seed=0)["status"] == "INCONCLUSIVE"
    ok = ts.h13(rows, {"twap|vwap": "ROBUSTLY_BETTER"}, alpha=0.05, samples=200, seed=0)
    assert ok["status"] == "ESTABLISHED" and ok["pairs"]["twap|vwap"]["status"] == "SURVIVES"
    bad = ts.h13(rows, {"twap|vwap": "ROBUSTLY_WORSE"}, alpha=0.05, samples=200, seed=0)
    assert bad["status"] == "FAILED"
    weak = ts.h13(rows, {"twap|pov": "ROBUSTLY_BETTER"}, alpha=0.05, samples=200, seed=0)
    assert weak["status"] == "NOT_ESTABLISHED"
