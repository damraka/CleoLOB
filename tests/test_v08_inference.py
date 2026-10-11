"""v0.8 inference: priors, ABC rejection/SMC, synthetic likelihood, reweighting, hierarchical pooling,
adaptive calibration and identifiability (M12-M15), with small synthetic recovery checks."""
from __future__ import annotations

import numpy as np
import pytest

from lob.v08.generators import HawkesGenerator, PoissonGenerator
from lob.v08.inference import abc, hierarchical as hi, identifiability as idf
from lob.v08.inference.adaptive import SCHEMES, compare, concat
from lob.v08.market.frame import summaries

TRUTH = PoissonGenerator({"market": 1.0, "add": 3.0, "cancel": 2.7, "market_buy_share": 0.5, "add_bid_share": 0.5,
                          "cancel_bid_share": 0.5})


def test_prior_transforms_round_trip() -> None:
    prior = abc.Prior.from_space({"a": (0.1, 10.0), "b": (-2.0, 2.0)}, log={"a"})
    theta = prior.sample(50, np.random.default_rng(0))
    assert np.allclose(prior.from_unit(prior.to_unit(theta)), theta)
    assert np.allclose(prior.from_free(prior.to_free(theta)), theta, rtol=1e-6, atol=1e-8)
    assert np.all((theta[:, 0] >= 0.1) & (theta[:, 0] <= 10)) and np.all(np.abs(theta[:, 1]) <= 2)
    with pytest.raises(ValueError):
        abc.Prior(("a",), (1.0,), (0.5,), (False,))


@pytest.fixture(scope="module")
def problem():
    obs = summaries(TRUTH.simulate(400, seed=99))
    prior = abc.Prior.from_space({"market": (0.3, 3.0), "add": (1.0, 9.0)}, log={"market", "add"})
    return abc.Problem(PoissonGenerator(), prior, obs, n_bins=400, seed=1, pilot=6, summary_index=list(range(10)))


def test_abc_smc_contracts_towards_truth_and_is_deterministic(problem) -> None:
    post = abc.smc(problem, particles=20, generations=2)
    lo, hi = abc.credible_interval(post, "add", 0.9)
    assert lo < 3.0 < hi
    assert post["summary"]["parameters"]["add"]["contraction"] > 0.1
    again = abc.smc(abc.Problem(PoissonGenerator(), problem.prior, np.asarray(summaries(TRUTH.simulate(400, seed=99))),
                                n_bins=400, seed=1, pilot=6, summary_index=list(range(10))), particles=20, generations=2)
    assert post["particles"] == again["particles"] and post["weights"] == again["weights"]


def test_rejection_regression_adjustment_and_draws(problem) -> None:
    rej = abc.rejection(problem, draws=40, accept=0.3)
    assert len(rej["particles"]) >= 10 and rej["summaries"] is not None
    adj = abc.regression_adjust(rej, problem)
    assert adj["method"].endswith("regression_adjustment") and len(adj["particles"]) == len(rej["particles"])
    draws = abc.draw(rej, 5, seed=3)
    assert len(draws) == 5 and set(draws[0]) == {"market", "add"}


def test_synthetic_likelihood_prefers_truth_and_mcmc_runs(problem) -> None:
    near = abc.synthetic_loglik(problem, np.asarray([1.0, 3.0]), replicates=6, stage=60)
    far = abc.synthetic_loglik(problem, np.asarray([2.8, 8.5]), replicates=6, stage=60)
    assert near > far
    chain = abc.metropolis(problem, steps=8, replicates=3)
    assert chain["method"] == "synthetic_likelihood_mcmc" and len(chain["particles"]) == 6


def test_importance_reweighting_with_forgetting(problem) -> None:
    rej = abc.rejection(problem, draws=24, accept=0.5)
    new = abc.reweight(problem, rej, summaries(TRUTH.simulate(400, seed=7)), replicates=3, forgetting=0.5)
    assert abs(sum(new["weights"]) - 1) < 1e-9 and new["info"]["forgetting"] == 0.5


def test_count_modes_detects_bimodality() -> None:
    u = np.r_[np.full(50, 0.1), np.full(50, 0.9)]
    assert abc.count_modes(u, np.ones(100) / 100) == 2
    assert abc.count_modes(np.full(100, 0.5), np.ones(100) / 100) == 1


def test_dersimonian_laird_and_shrinkage() -> None:
    pooled = hi.dersimonian_laird(np.asarray([0.0, 0.0, 0.0]), np.asarray([1.0, 1.0, 1.0]))
    assert pooled["tau2"] == 0.0 and pooled["mu"] == pytest.approx(0.0)
    het = hi.dersimonian_laird(np.asarray([-3.0, 0.0, 3.0]), np.asarray([0.01, 0.01, 0.01]))
    assert het["tau2"] > 1.0
    m, v = hi.shrink([-3.0, 0.0, 3.0], [1.0, 1.0, 1.0], {"mu": 0.0, "tau2": 1.0})
    assert np.allclose(m, [-1.5, 0.0, 1.5]) and np.allclose(v, 0.5)


def test_pool_tree_structure_and_identification_flags() -> None:
    leaves = [{"instrument": i, "day": f"{i}{d}", "mean": {"p": float(d)}, "var": {"p": 0.1},
               "contraction": {"p": 0.9 if i == "ETH" else 0.1}} for i in ("ETH", "BTC") for d in range(3)]
    tree = hi.pool_tree(leaves, ("instrument", "day"))
    eth = tree["children"]["ETH"]
    assert eth["pooling"]["groups"]["ETH0"]["identification"]["p"] == "CONTRACTED"
    assert tree["children"]["BTC"]["pooling"]["groups"]["BTC1"]["identification"]["p"] == "NOT_IDENTIFIED"
    with pytest.raises(ValueError):
        hi.pool_tree(leaves, ("galaxy",))


def test_adaptive_schemes_track_a_rate_step() -> None:
    low = PoissonGenerator({**TRUTH.params, "market": 0.5, "add": 2.0, "cancel": 1.8})
    high = PoissonGenerator({**TRUTH.params, "market": 2.0, "add": 6.0, "cancel": 5.5})
    windows = [low.simulate(300, k) for k in range(3)] + [high.simulate(300, 10 + k) for k in range(4)]
    res = compare(PoissonGenerator(), windows, initial=2, window=2, schemes=("frozen", "rolling", "expanding",
                                                                             "forgetting", "change_triggered"))
    assert set(res["losses"]) <= set(SCHEMES)
    frozen, rolling = np.mean(res["losses"]["frozen"][-3:]), np.mean(res["losses"]["rolling"][-3:])
    assert rolling < frozen


def test_concat_marks_window_boundaries_invalid() -> None:
    a, b = TRUTH.simulate(50, 1), TRUTH.simulate(50, 2)
    c = concat([a, b])
    assert len(c) == 100 and not c.valid[50] and c.valid[49]


def test_identifiability_local_global_profile_and_assessment(problem) -> None:
    loc = idf.local(PoissonGenerator(), problem.prior, np.asarray([1.0, 3.0]), n_bins=200, replicates=3, seed=2,
                    summary_index=list(range(10)))
    assert loc["effective_rank"] >= 1 and len(loc["eigenvalues"]) == 2 and "local" in loc["validity"]
    rng = np.random.default_rng(0)
    th = problem.prior.sample(10, rng)
    s = np.asarray([abc.simulate_summary(PoissonGenerator().to_dict(), ["market", "add"], list(t), 200, k)
                    for k, t in enumerate(th)])
    glob = idf.global_indices(th, s, ["market", "add"], bins=4)
    assert set(glob["first_order"]) == {"market", "add"}
    post = abc.rejection(problem, draws=24, accept=0.5)
    prof = idf.profile(post, bins=5)
    verdict = idf.assess("poisson", loc, post, prof)
    assert set(verdict["parameters"]) == {"market", "add"}
    assert all(v["verdict"] in ("PRACTICALLY_IDENTIFIED", "WEAKLY_IDENTIFIED", "NOT_IDENTIFIED")
               for v in verdict["parameters"].values())
    assert idf.STRUCTURAL_FLAGS["hawkes"]


def test_misspecified_family_fits_worse_than_truth_family() -> None:
    data = HawkesGenerator().simulate(2000, 4)
    train, test = data.slice(0, 1000), data.slice(1000, 2000)
    assert np.nanmean(HawkesGenerator().fit(train).loglik(test)) > np.nanmean(PoissonGenerator().fit(train).loglik(test))
