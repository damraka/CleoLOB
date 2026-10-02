"""v0.7 identifiability v2 and observable design on synthetic models with known identifiability."""
from __future__ import annotations

import numpy as np

from lob.v07.identifiability import analysis as ia

NAMES = ["p0", "p1", "p2", "p3"]


def _linear(a):
    return lambda theta: a @ theta


def test_identifiable_model() -> None:
    a = np.eye(4) * np.array([3.0, 2.0, 1.0, 0.5])
    j, steps = ia.jacobian(_linear(a), np.full(4, 0.5))
    np.testing.assert_allclose(j, a, atol=1e-9)
    s = ia.sloppiness(j)
    assert s["effective_rank"] == 4 and s["log10_eigen_spread"] > 1


def test_structural_and_practical_non_identifiability() -> None:
    a = np.array([[1.0, 1.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])   # p0+p1 only; p3 inert
    j, _ = ia.jacobian(_linear(a), np.full(4, 0.5))
    assert ia.sloppiness(j)["effective_rank"] == 2
    rng = np.random.default_rng(0)
    s = rng.uniform(0.3, 0.7, 400)
    particles = np.column_stack([s, 1 - s, rng.normal(0.5, 0.02, 400), rng.uniform(0, 1, 400)])
    distances = np.abs(particles[:, 0] + particles[:, 1] - 1) + np.abs(particles[:, 2] - 0.5)
    post = ia.posterior_summary(particles, distances, NAMES)
    assert ("p0", "p1") == post["compensation_pairs"][0][:2] and post["compensation_pairs"][0][2] < -0.9
    table = ia.classify(j, post, NAMES)
    assert table["p3"]["status"] == "STRUCTURALLY_NOT_IDENTIFIED"
    assert table["p2"]["status"] == "IDENTIFIED"
    assert table["p0"]["status"] in {"PRACTICALLY_NOT_IDENTIFIED", "IDENTIFIED"}


def test_observable_design_known_influence() -> None:
    j = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1e-4], [0.1, 0.1, 0.9]])
    out = ia.observable_design(j, ["a", "a2", "b", "weak", "mix"], ["F1", "F1", "F2", "F3", "F4"], ["x", "y", "z"])
    assert out["weak_observables"] == ["weak"] and out["uninformative_families"] == ["F3"]
    assert ("a", "a2", 1.0) in [(p, q, round(c, 6)) for p, q, c in out["redundant_pairs"]]
    assert out["complementary_observables"]["b"]["parameter"] == "y"
    assert out["complementary_observables"]["mix"]["parameter"] == "z"
