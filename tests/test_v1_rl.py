"""v1 learning: environment, BC/CQL/IQL, Decision Transformer, OPE and gating (M120-M126)."""
import numpy as np
import pytest

from lob.v08.generators import PoissonGenerator
from lob.v1.generators.classical import book as B
from lob.v1.rl import env as EN, learning as LR

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def world():
    base = B.BookModelV1(refill_mean=6.0, limit_size_mean=3.0, market_size_mean=2.0, mid_ticks=4000,
                         improve_by_spread=((2, 0.6), (3, 0.6), (4, 0.6)))
    return PoissonGenerator({**PoissonGenerator.default_params(), "market": 1.0}, base)


@pytest.fixture(scope="module")
def episodes(world):
    return LR.collect(world, episodes=12, qty=30)


def test_env_gymnasium_contract_and_return_equals_minus_cost(world):
    e = EN.ExecutionEnv(world, qty=30, horizon_s=60)
    obs, info = e.reset(seed=3)
    assert e.observation_space.contains(obs.astype(np.float32)) and e.action_space.n == 4
    total, done = 0.0, False
    while not done:
        obs, r, done, trunc, inf = e.step(2)
        total += r
    # equal up to the simulator's residual-valuation convention (documented tolerance)
    assert total == pytest.approx(-inf["episode_cost_bps"], abs=0.05)


def test_bc_learns_behaviour_and_cql_iql_return_scores(episodes):
    L = LR.learners(episodes)
    s = np.asarray(episodes[0]["s"][0])
    for name in ("bc", "cql", "iql"):
        assert L[name](s).shape == (4,)


def test_decision_transformer_trains_saves_and_predicts(episodes, tmp_path):
    dt = LR.DecisionTransformer(steps=60, seed=1).fit(episodes)
    fn = dt.policy()
    s = np.asarray(episodes[0]["s"][3])
    assert fn(s).shape == (4,)
    dt.save(tmp_path / "dt")
    from lob.v1.generators.neural.base import NeuralModel
    dt2 = NeuralModel.load(tmp_path / "dt")
    assert np.allclose(dt2.policy()(s), fn(s), atol=1e-6)


def test_ope_requires_propensities_and_reports_support(episodes):
    L = LR.learners(episodes)
    out = LR.ope(episodes, L["bc"])
    assert {"pdis", "wpdis", "dr", "effective_sample_size", "ood_action_rate"} <= set(out)
    broken = [{**e, "p": None} for e in episodes]
    with pytest.raises(LR.PropensityMissing):
        LR.ope(broken, L["bc"])


def test_gating_refuses_without_support_or_lower_bound():
    rng = np.random.default_rng(0)
    good = rng.normal(2.0, 0.5, 200)
    base = rng.normal(0.0, 0.5, 200)
    assert LR.gate(good, base, support=0.95, ess=50)["deploy"]
    assert not LR.gate(good, base, support=0.5)["deploy"]                    # out-of-support states
    assert not LR.gate(good, base, support=0.95, ess=1.0)["deploy"]          # degenerate importance weights
    assert not LR.gate(rng.normal(0.0, 0.5, 200), base, support=0.95)["deploy"]
