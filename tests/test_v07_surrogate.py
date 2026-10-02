"""v0.7 surrogates (held-out emulator error) and active vs random calibration on a synthetic recovery problem."""
from __future__ import annotations

import numpy as np

from lob.v07.calibration import surrogate as sg


def _quadratic(x, truth=0.3):
    return np.sum((np.atleast_2d(x) - truth) ** 2 * np.array([4.0, 2.0, 1.0, 0.5, 0.0]), axis=1)


def test_gp_held_out_error_and_calibration() -> None:
    rng = np.random.default_rng(0)
    x = rng.random((160, 5))
    y = _quadratic(x) + rng.normal(0, 0.01, 160)
    report = sg.emulator_report(sg.GaussianProcess(seed=1), x[:120], y[:120], x[120:], y[120:])
    assert report["rmse"] < 0.35 * np.std(y[120:]) and report["rank_correlation"] > 0.9
    assert report["interval90_coverage"] > 0.7
    forest = sg.emulator_report(sg.RegressionForest(trees=16, seed=2), x[:120], y[:120], x[120:], y[120:])
    assert forest["rank_correlation"] > 0.5 and forest["rmse"] > report["rmse"]


def test_active_beats_random_at_equal_budget_on_average() -> None:
    gains = []
    for seed in range(3):
        a = sg.active_search(_quadratic, 48, 5, seed, initial=16, batch=8, pool=1000)
        r = sg.random_search(_quadratic, 48, 5, seed + 100)
        assert len(a["y"]) == len(r["y"]) == 48
        gains.append(r["best"] - a["best"])
    assert np.mean(gains) > 0
    assert np.linalg.norm(sg.active_search(_quadratic, 48, 5, 7)["best_x"][:3] - 0.3) < 0.3
