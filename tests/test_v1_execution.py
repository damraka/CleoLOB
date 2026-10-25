"""v1 execution research: baselines, adaptive, MPC variants, meta-policy and stress (M110-M119)."""
import numpy as np
import pytest

from lob.v1.execution import episode as EP, meta as MP, policies as XP


@pytest.fixture(scope="module")
def runs():
    names = ["twap", "sweep", "passive_deadline", "completion_aware", "imbalance_adaptive", "mpc_stochastic",
             "mpc_chance", "mpc_cvar", "mpc_dro", "mpc_lexicographic"]
    return {n: EP.f6_episode(XP.make(n), qty=100, seed=2, horizon_s=60) for n in names}


def test_baselines_registered_and_complete(runs):
    cat = XP.catalogue()
    assert set(XP.BASELINES) <= set(cat["registered"]) and "completion_aware" in cat["registered"]
    assert runs["twap"]["status"] == "OK" and runs["twap"]["completion"] == 1.0
    assert runs["sweep"]["maker_fraction"] == 0.0


def test_completion_aware_completes_and_uses_passive_fills(runs):
    r = runs["completion_aware"]
    assert r["status"] == "OK" and r["filled"] >= 100 and r["maker_fraction"] > 0
    assert r["overfill"] <= 5                    # latency races can overfill slightly; reported, never hidden


def test_liquidity_imbalance_policy_runs(runs):
    assert runs["imbalance_adaptive"]["status"] == "OK" and runs["imbalance_adaptive"]["filled"] >= 100


def test_mpc_family_executes(runs):
    for n in ("mpc_stochastic", "mpc_cvar", "mpc_dro"):
        assert runs[n]["status"] == "OK" and runs[n]["filled"] >= 100 and runs[n]["overfill"] <= 5


def test_chance_and_lexicographic_objectives():
    lex = XP.make("mpc_lexicographic")
    costs = np.array([[1.0, 2.0, 30.0], [2.0, 2.5, 3.0], [0.5, 0.6, 0.7]])
    completes = np.array([[True, True, True], [True, True, True], [False, False, True]])
    obj = lex._objective(costs, completes)
    # plan 2 is cheapest but rarely completes; plan 0 has a fat tail; the lexicographic order picks plan 1
    assert int(np.argmin(obj)) == 1
    ch = XP.make("mpc_chance")
    assert int(np.argmin(ch._objective(costs, completes))) in (0, 1)


def test_meta_policy_selects_and_falls_back():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 7))
    costs = {"twap": 5 + X[:, 0], "passive_deadline": 5 - X[:, 0], "liquidity_pov": np.full(300, 5.2)}
    mp = MP.MetaPolicy(list(costs)).fit(X, costs)
    out = mp.choose(X[:50], completion_risk=np.r_[np.ones(5, bool), np.zeros(45, bool)])
    assert out["fallback_choice"][:5] == ["sweep"] * 5
    picks = np.asarray(out["mixture_choice"][5:])
    assert (picks[X[5:50, 0] > 1] == "passive_deadline").mean() > 0.7
    assert set(out["uncertainty_level"]) <= {"low", "moderate", "high", "completion_danger"}


def test_stress_failure_envelope():
    env = MP.stress(lambda: XP.make("passive_deadline"), qty=80, horizon_s=40, seeds=(0,),
                    grid={k: MP.STRESS_GRID[k] for k in ("baseline", "liquidity_withdrawal")},
                    compare_factory=lambda: XP.make("twap"))
    assert set(env["grid"]) == {"baseline", "liquidity_withdrawal"}
    assert env["envelope"]["worst_cost_bps"] is not None and env["envelope"]["min_completion"] <= 1.0
