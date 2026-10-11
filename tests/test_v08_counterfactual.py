"""v0.8 counterfactual historical bounds (M29, section 12, section 38) and leakage negative controls (section 15)."""
from __future__ import annotations

import numpy as np
import pytest

from lob.v08 import counterfactual as cf, protocol as pr
from lob.v08.execution import policies as P
from lob.v08.execution.simulator import Mandate, Observation
from lob.v08.market.sim import MarketSim
from lob.v08.regimes import compare
from tests.v08_fixtures import tiny_world

M = Mandate(qty=20, horizon_s=30.0, warmup_s=10.0, settlement_s=5.0)


@pytest.fixture(scope="module")
def frame():
    w = tiny_world()
    sim = MarketSim(w.generator, w.book, 4)
    sim.run(100 + 350 + 60)
    return sim.frame()


def test_replay_is_labelled_hypothetical_and_never_an_actual_fill(frame) -> None:
    r = cf.replay(frame, P.make("twap"), M, cf.Assumption(), start_bin=100)
    assert r["label"] == "HYPOTHETICAL" and r["actual_filled_qty"] is None and r["hypothetical_filled_qty"] > 0


def test_bounds_are_ordered_and_assumption_tagged(frame) -> None:
    b = cf.bounds(frame, lambda: P.make("heuristic"), M, cf.grid((0.0, 0.2), 0.5), start_bin=100)
    assert b["status"] == "VALID" and b["label"] == "BOUNDED" and b["assumption_dependent"]
    for f in cf.BOUND_FIELDS:
        if b[f]["lower"] is not None:
            assert b[f]["lower"] <= b[f]["upper"]
    assert len(b["rows"]) == 3 * 2 * 2 * 3


def test_queue_rules_order_passive_fills(frame) -> None:
    class Passive(P.Policy):
        name = "passive_cf_test"

        def decide(self, obs):
            return [("limit", obs.remaining, 0, 0)] if obs.step == 0 else []

    fills = {q: cf.replay(frame, Passive(), M.replace(completion="none"), cf.Assumption(q), start_bin=100)
             ["hypothetical_filled_qty"] for q in cf.QUEUE_RULES}
    assert fills["optimistic"] >= fills["neutral"] >= fills["conservative"]


def test_persistent_impact_never_cheaper_than_no_impact(frame) -> None:
    none = cf.replay(frame, P.make("twap"), M, cf.Assumption(impact="none"), start_bin=100)
    pers = cf.replay(frame, P.make("twap"), M, cf.Assumption(impact="persistent", permanent_ticks_per_lot=0.05),
                     start_bin=100)
    assert pers["economic_cost_bps"] >= none["economic_cost_bps"] - 1e-9


def test_short_or_invalid_frames_are_invalid_not_dropped(frame) -> None:
    with pytest.raises(ValueError):
        cf.replay(frame.slice(0, 200), P.make("twap"), M, cf.Assumption(), start_bin=100)
    b = cf.bounds(frame.slice(0, 200), lambda: P.make("twap"), M, [cf.Assumption()], start_bin=100)
    assert b["status"] == "INVALID" and b["reason"]


def test_pair_bound_statuses() -> None:
    def fake(v):
        return {"status": "VALID", "rows": {"k1": {"economic_cost_bps": v}, "k2": {"economic_cost_bps": v + 0.1}}}

    a = [fake(1.0 + 0.01 * i) for i in range(10)]
    b = [fake(5.0 + 0.01 * i) for i in range(10)]
    assert cf.pair_bound(a, b)["status"] == "A_LOWER"
    assert cf.pair_bound(b, a)["status"] == "B_LOWER"
    same = cf.pair_bound(a, a)
    assert same["status"] == "INCONCLUSIVE"
    assert cf.pair_bound(a[:2], b[:2])["status"] == "NOT_EVALUABLE"


def test_synthetic_coverage_benchmark_runs() -> None:
    res = cf.synthetic_coverage(tiny_world(), lambda: P.make("twap"), M, [1, 2], cf.grid((0.0, 0.1), 0.0))
    assert res["summary"]["fill_frac"]["n"] == 2 and "SYNTHETIC" in res["label"]


# ----------------------------------------------------------------------------- leakage negative controls


def test_replay_decisions_never_read_future_rows(frame) -> None:
    """Perturbing every row after a decision must not change that decision (no look-ahead)."""
    seen = []

    class Spy(P.Policy):
        name = "spy_cf_test"

        def decide(self, obs: Observation):
            seen.append((obs.step, obs.ask, tuple(obs.ask_depth), len(obs.recent_mids), obs.volume_since_last))
            return []

    cf.replay(frame, Spy(), M, cf.Assumption(latency_s=0.2), start_bin=100)
    first = list(seen)
    altered = frame.slice(0, len(frame))
    altered.depth = altered.depth.copy()
    altered.ask_px = altered.ask_px.copy()
    cut = 100 + 50          # decision at step 1 observes row 150 - 2
    altered.depth[cut:] += 7
    altered.ask_px[cut:] += 3
    seen.clear()
    cf.replay(altered, Spy(), M, cf.Assumption(latency_s=0.2), start_bin=100)
    assert seen[:2] == first[:2] and seen[2:] != first[2:]


def test_shuffled_regime_labels_lose_predictive_value() -> None:
    from lob.v08.generators import RegimeSwitchingGenerator
    from lob.v08.regimes import block_features
    g = RegimeSwitchingGenerator({"rates": [[0.2] * 6, [2.0] * 6], "trans": [[0.999, 0.001], [0.001, 0.999]],
                                  "init": [0.5, 0.5]})
    res = compare(block_features(g.simulate(6000, 1), 100), block_features(g.simulate(6000, 2), 100), seed=4)
    t = res["threshold"]
    assert t["next_block_r2_activity"] > 0.2 and t["negative_control_shuffled_r2_activity"] < t["next_block_r2_activity"]


def test_unrelated_feature_is_not_predictive() -> None:
    from lob.v08.statistics import spearman
    rng = np.random.default_rng(0)
    regret = rng.normal(0, 1, 400)
    noise_feature = rng.normal(0, 1, 400)
    assert abs(spearman(noise_feature, regret)) < 0.15


def test_holdout_hash_access_audit() -> None:
    """Every accessed holdout in the repository ledger has a sealed reader that lists it."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    state = pr.replay_ledger(pr.read_ledger(repo / pr.LEDGER_PATH), pr.load_protocol(repo / pr.PROTOCOL_PATH))
    protocol = pr.load_protocol(repo / pr.PROTOCOL_PATH)
    for d in protocol["datasets"]:
        if d["role"] in pr.HOLDOUT_ROLES and d["id"] in state.first_access:
            readers = [n for n, x in state.designs.items() if d["id"] in x.get("reads", [])]
            assert readers, f"{d['id']} accessed without a sealed reader"
