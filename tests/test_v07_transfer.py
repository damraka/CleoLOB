"""v0.7 transfer statistics (H12, H13) on constructed replay rows."""
from __future__ import annotations

import os

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


def test_worker_cap_is_resource_control_only(monkeypatch) -> None:
    monkeypatch.setenv("CLEOLOB_WORKERS", "3")
    assert ts._workers() == min(3, max(1, (os.cpu_count() or 2) - 2))
    monkeypatch.delenv("CLEOLOB_WORKERS")
    assert 1 <= ts._workers() <= 14


def test_compact_episodes_expand_to_the_reference_episodes(tmp_path) -> None:
    """Regression (ABORTED final-transfer attempt, low memory): the compact form must not change any episode."""
    from tests.v07_fixtures import write_tardis
    files = write_tardis(tmp_path, seconds=1500.0)
    kwargs = {"tick": 0.05, "lots_per_native": 0.1, "label": "fixture"}
    reference, info = ts.extract_episodes(files, **kwargs)
    compact, info_compact = ts.extract_episodes(files, compact=True, **kwargs)
    assert info == info_compact and len(reference) >= 1
    for ref, small in zip(reference, compact, strict=True):
        assert isinstance(small, ts.CompactEpisode)
        expanded = small.expand()
        assert expanded == ref
        assert [list(b) for _, b, _ in expanded.updates] == [list(b) for _, b, _ in ref.updates]   # key order
        assert all(type(k) is int and type(q) is float for _, b, a in expanded.updates for k, q in {**b, **a}.items())
