"""v1 reconstruction/liquidity/regime/continual hub helpers (H14, H17, H27, H42 unit computations)."""
import math

import numpy as np
import pytest

from lob.v1.research.studies import reconstruction_hub as R


def test_reconstruction_hub_poisson_nll_matches_closed_form():
    y = np.array([0, 1, 3])
    lam = np.array([0.5, 2.0, 3.0])
    want = lam - y * np.log(lam) + np.array([math.lgamma(k + 1) for k in y])
    assert R._poisson_nll(y, lam) == pytest.approx(want)


def test_reconstruction_hub_wilson_interval_contains_proportion():
    lo, hi = R._wilson(30, 100)
    assert lo < 0.3 < hi and 0 <= lo and hi <= 1
    assert R._wilson(0, 0) == (0.0, 1.0)


def test_reconstruction_hub_logistic_beats_constant_on_informative_feature():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4000, 2))
    y = (rng.random(4000) < 1 / (1 + np.exp(-2 * X[:, 0]))).astype(float)
    m = R._logit_fit(X, y)
    assert R._logloss(m, X, y) < math.log(2) - 0.1
