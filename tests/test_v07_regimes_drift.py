"""v0.7 latent regimes, change points and calibration half-life on synthetic processes."""
from __future__ import annotations

import numpy as np
import pytest

from lob.v07.drift import analysis as da
from lob.v07.regimes import hmm


def test_hmm_recovers_synthetic_regimes_and_beats_one_state() -> None:
    trans = np.array([[0.97, 0.03], [0.05, 0.95]])
    x, s = hmm.sample([[0.0, 0.0], [3.0, 1.0]], [[1.0, 1.0], [0.6, 0.6]], trans, 800, seed=1)
    two = hmm.GaussianHMM(2, seed=2).fit(x)
    one = hmm.GaussianHMM(1, seed=2).fit(x)
    accuracy = np.mean(two.states(x) == s)
    assert max(accuracy, 1 - accuracy) > 0.95
    np.testing.assert_allclose(two.means, [[0, 0], [3, 1]], atol=0.25)
    assert abs(two.trans[0, 0] - 0.97) < 0.03 and two.bic(x) < one.bic(x)
    x_new, _ = hmm.sample([[0.0, 0.0], [3.0, 1.0]], [[1.0, 1.0], [0.6, 0.6]], trans, 400, seed=3)
    assert two.score(x_new) > one.score(x_new)


def test_change_points_find_known_shift_and_stay_quiet_under_the_null() -> None:
    hits = sum(bool((p := da.change_points(np.r_[np.random.default_rng(s).normal(0, 1, 60),
                                                np.random.default_rng(s + 99).normal(3, 1, 60)])["change_points"]))
               and any(57 <= q <= 63 for q in p) for s in range(30))
    assert hits >= 28
    false_alarms = sum(bool(da.change_points(np.random.default_rng(s).normal(0, 1, 200))["change_points"])
                       for s in range(30))
    assert false_alarms <= 3
    assert da.cusum(np.r_[np.zeros(30), np.full(30, 10.0)] + np.random.default_rng(0).normal(0, 1, 60))["change_points"]


def test_half_life_stable_versus_drifting() -> None:
    months = [0, 2, 3, 4, 5, 6]
    assert da.half_life(months, [2.0, 2.01, 1.99, 2.0, 2.02, 2.0])["status"] == "stable"
    tau = 2.0
    errors = [2.0 * (1 + 0.5 * (1 - 2 ** (-m / tau))) for m in months]
    out = da.half_life(months, errors, per_period_draws=[np.asarray([e * 0.99, e, e * 1.01]) for e in errors],
                       samples=200)
    assert out["half_life_months"] == pytest.approx(tau, rel=0.1) and out["long_run_degradation"] == pytest.approx(0.5, rel=0.1)
    assert out["half_life_ci"][0] <= tau <= out["half_life_ci"][1]
    assert da.half_life(months, [2.0, 1.5, 1.4, 1.3, 1.2, 1.1])["status"] == "INCONCLUSIVE"
    assert da.half_life([0, 1], [1.0, 2.0])["status"] == "NOT_AVAILABLE"


def test_discrete_baseline_is_a_valid_two_state_model() -> None:
    from lob.v07.drift.study import discrete_baseline
    rng = np.random.default_rng(1)
    x = np.column_stack([np.r_[rng.normal(-1, 0.2, 50), rng.normal(1, 0.2, 50)], rng.normal(0, 1, (100, 2))])
    model = discrete_baseline(x)
    assert np.allclose(model.trans.sum(1), 1) and model.means[1, 0] > model.means[0, 0]
    assert np.isfinite(model.score(x)) and model.bic(x) > 0
