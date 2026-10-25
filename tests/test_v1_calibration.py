"""v1 calibration and inference (M90-M99) with known-answer checks."""
import numpy as np
import pytest

from lob.v08.generators import PoissonGenerator
from lob.v08.market.sim import BookModel, simulate
from lob.v1.calibration import api as CA
from lob.v1.inference import methods as IN
from lob.v1.sdk import SDKError

BOOK = BookModel(refill_mean=6.0, limit_size_mean=3.0, market_size_mean=2.0, mid_ticks=4000)


def _frame(market, seed, bins=1500):
    g = PoissonGenerator({**PoissonGenerator.default_params(), "market": market}, BOOK)
    return simulate(g, BOOK, bins, seed=seed, warmup_bins=100)


def test_api_likelihood_calibration_recovers_parameter_and_separates_data():
    train = [_frame(1.6, s) for s in range(2)]
    test = [_frame(1.6, 9)]
    prior = IN.Prior.from_space({"market": (0.3, 4.0)})

    def model(theta):
        return PoissonGenerator({**PoissonGenerator.default_params(), "market": theta["market"]}, BOOK)

    def score(frames):
        return lambda m: float(np.nanmean(np.concatenate([m.loglik(f) for f in frames])))
    res = CA.calibrate(model, prior, "likelihood", scorer=score(train), train=("d1", "d2"), test=("d3",),
                       budget=12, refine=10, seed=1, test_scorer=score(test))
    assert abs(res.parameters["market"] - 1.6) < 0.35 and "test_objective" in res.diagnostics
    with pytest.raises(SDKError):
        CA.calibrate(model, prior, "likelihood", scorer=score(train), train=("d1",), test=("d1",), budget=2, refine=0)
    with pytest.raises(ValueError):
        CA.calibrate(model, prior, "vibes", scorer=score(train), train=("a",), test=("b",))


def test_realism_aware_distance_prefers_metrics_inside_bands():
    band = {"a": (0.0, 2.0), "b": (10.0, 20.0)}
    assert CA.realism_distance({"a": 1.0, "b": 15.0}, band) == 0.0
    assert CA.realism_distance({"a": 5.0, "b": 15.0}, band) > CA.realism_distance({"a": 2.5, "b": 15.0}, band)


def test_sbi_abc_smc_contracts_on_rate():
    obs = _frame(2.0, 5, bins=1200)
    from lob.v08.market.frame import summaries
    gen = PoissonGenerator(None, BOOK)
    prob = IN.Problem(gen, IN.Prior.from_space({"market": (0.3, 5.0)}), summaries(obs), n_bins=600, seed=3, pilot=8)
    post = IN.abc_smc(prob, particles=32, generations=3)
    lo, hi = IN.ABC.credible_interval(post, "market", 0.9)
    # strong contraction around the truth (32 particles: a single run's 90% interval is not a coverage guarantee)
    assert (hi - lo) < 1.5 and lo - 0.3 < 2.0 < hi + 0.3


def test_sbc_calibrated_for_exact_posterior_and_detects_bias():
    def prior(rng):
        return rng.gamma(2.0, 1.0)

    def sim(theta, rng):
        return rng.poisson(theta, 20)
    ok = IN.sbc(prior, sim, IN.conjugate_poisson_posterior, n=200, draws=49, seed=0)
    assert ok["calibrated"]

    def biased(data, draws, seed):
        return IN.conjugate_poisson_posterior(data, draws=draws, seed=seed) * 1.5
    bad = IN.sbc(prior, sim, biased, n=200, draws=49, seed=0)
    assert not bad["calibrated"] and bad["chi2"] > ok["chi2"]


def test_hierarchical_pooling_shrinks_noisy_days():
    est = {"BTC": {"d1": (1.0, 0.01), "d2": (1.2, 0.01), "d3": (5.0, 4.0)},
           "ETH": {"d1": (2.0, 0.01), "d2": (2.1, 0.01)}}
    out = CA.pool_parameters(est)
    btc = out["instruments"]["BTC"]["days"]
    assert btc["d3"] < 5.0 and abs(btc["d1"] - 1.0) < 0.2         # the noisy day moves most
    assert "mu" in out["global"]


def test_active_design_picks_most_informative_point():
    cands = np.array([[0.0], [1.0], [3.0]])
    out = IN.active_design(cands, lambda c: np.array([[c[0]]]), np.eye(1), np.eye(1))
    assert out["best"] == 2 and out["gains"][0] == pytest.approx(0.0)


def test_identifiability_fisher_rank_and_confusion():
    f = IN.fisher(lambda th: np.array([th[0] + th[1], th[0] + th[1]]), np.array([1.0, 2.0]), cov=np.eye(2))
    assert f["rank"] == 1                                        # only the sum is identified
    g = IN.fisher(lambda th: np.array([th[0], th[1]]), np.array([1.0, 2.0]), cov=np.eye(2))
    assert g["rank"] == 2

    def simulate_cls(lam, rng):
        return rng.poisson(lam, 200)

    def score(lam, data):
        return float((data * np.log(lam) - lam).sum())
    conf = IN.model_class_confusion({"low": 1.0, "high": 3.0}, simulate_cls, score, reps=10)
    assert conf["identifiable"]


def test_ppc_flags_misfit():
    rng = np.random.default_rng(0)
    rep = rng.normal(0, 1, (500, 2))
    out = IN.ppc(np.array([0.1, 6.0]), rep, ["fine", "off"])
    assert out["flagged"] == ["off"]


def test_half_life_recovery_and_decision_decomposition():
    hl = CA.half_life({0: [1.0, 1.1], 1: [1.5], 3: [2.2, 2.1], 5: [3.0]})
    assert hl["half_life_days"] == 3
    rec = CA.recovery([2.0, 2.2], [1.1, 1.0])
    assert rec["recovered_fraction"] > 0.4
    rng = np.random.default_rng(1)
    vals = {"A": {f"p{k}": list(rng.normal(0 + 0.1 * k, 0.5, 20)) for k in range(5)},
            "B": {f"p{k}": list(rng.normal(3 + 0.1 * k, 0.5, 20)) for k in range(5)}}
    dec = CA.decompose(vals)
    assert dec["shares"]["model_class"] > dec["shares"]["parameter"]
    assert sum(dec["shares"].values()) == pytest.approx(1.0, abs=0.02)
