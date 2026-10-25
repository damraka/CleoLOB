"""v1 queue survival / fill hazard and markout / adverse selection (M108, M109)."""
import numpy as np

from lob.v09.queue import survival as S
from lob.v1.queue import models as QM


def test_survival_fill_interval_orders_queue_bounds():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3000, 9))
    X[:, 0] = np.abs(X[:, 0]) * 2
    T = rng.exponential(np.exp(X[:, 0]) * 2.0)
    E = T < 30
    T = np.minimum(T, 30)
    m = S.WeibullAFT().fit(X, T, E)
    out = QM.fill_interval(m, X[:100], (np.full(100, 2.0), np.full(100, 50.0)), 5.0)
    assert np.all(out["p_fill_optimistic"] >= out["p_fill_pessimistic"] - 1e-9)
    assert "partially identified" in out["label"]


def test_markout_adverse_selection_classifier_learns_signal():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4000, 5))
    y = (rng.random(4000) < 1 / (1 + np.exp(-2 * X[:, 0]))).astype(float)
    clf = QM.AdverseSelection().fit(X[:3000], y[:3000])
    ev = clf.evaluate(X[3000:], y[3000:])
    assert ev["log_loss"] < ev["base_log_loss"] and 0.7 < ev["calibration_slope"] < 1.3
