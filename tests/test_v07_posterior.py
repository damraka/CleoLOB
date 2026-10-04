"""v0.7 SMC-ABC mechanics on a cheap analytic distance (no simulator)."""
from __future__ import annotations

import numpy as np

from lob.v07.posterior import smc_abc


class Analytic:
    """Distance = |x - truth| (sup norm) + small noise; a stand-in for the simulator objective."""

    def __init__(self, truth, seed=0):
        self.truth, self.rng, self.evaluations = np.asarray(truth), np.random.default_rng(seed), 0

    def __call__(self, units, tag):
        self.evaluations += len(units)
        d = np.max(np.abs(units - self.truth), axis=1) + self.rng.normal(0, 0.005, len(units))
        return d, [{"objective": float(v)} for v in d]


def test_reflection_stays_in_box_and_is_identity_inside() -> None:
    x = np.asarray([-0.2, 0.3, 1.4, 2.6, -1.7])
    y = smc_abc.reflect(x)
    assert np.all((y >= 0) & (y <= 1)) and y[1] == 0.3
    np.testing.assert_allclose(y, [0.2, 0.3, 0.6, 0.6, 0.3])


def test_smc_abc_concentrates_on_truth_and_is_deterministic() -> None:
    truth = np.full(smc_abc.DIM, 0.3)
    a = smc_abc.run(Analytic(truth), particles=200, generations=8, seed=1)
    b = smc_abc.run(Analytic(truth), particles=200, generations=8, seed=1)
    np.testing.assert_array_equal(a["particles"], b["particles"])
    tolerances = [h["tolerance"] for h in a["history"][1:]]
    assert all(t2 <= t1 + 1e-12 for t1, t2 in zip(tolerances, tolerances[1:]))
    # The ABC posterior at tolerance eps is (roughly) the eps-box around the truth, truncated to [0, 1].
    assert np.max(np.abs(a["particles"].mean(0) - truth)) < a["tolerance"]
    assert np.max(np.abs(a["particles"] - truth)) <= a["tolerance"] + 0.02
    assert smc_abc.coverage(a["particles"], truth)["covered_fraction"] >= 0.8
    assert a["evaluations"] <= 200 * 8


def test_clusters_detect_two_modes() -> None:
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0.2, 0.02, (100, 3)), rng.normal(0.8, 0.02, (60, 3)), rng.normal(0.5, 0.01, (3, 3))])
    c = smc_abc.clusters(np.clip(x, 0, 1))
    assert c["major_components"] == 2 and c["components"] >= 3
    assert smc_abc.clusters(rng.normal(0.5, 0.02, (100, 3)))["major_components"] == 1


def test_systematic_resampling_respects_zero_weights() -> None:
    idx = smc_abc.systematic_resample(np.asarray([0, 1, 0, 3.0]), np.random.default_rng(0))
    assert set(idx) <= {1, 3} and np.sum(idx == 3) == 3
