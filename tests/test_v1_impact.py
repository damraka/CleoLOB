"""v1 impact kernels, state-dependent impact and cross impact (M105-M107)."""
import numpy as np

from lob.v1.impact import models as IMP


def test_kernels_parametric_decomposition_recovers_shape():
    lags = np.arange(31.0)
    cum = 0.5 + 1.5 * (1 - np.exp(-lags / 5.0))
    fit = IMP._param_fit(cum)
    assert abs(fit["temporary_bps"] - 0.5) < 0.05 and abs(fit["tau_s"] - 5.0) < 1.0
    assert abs(fit["persistent_bps"] - cum[-1]) < 1e-9 and fit["transient_bps"] >= 0


def test_state_impact_and_cross_labels_are_not_causal():
    assert "not a causal" in IMP.LABEL


def test_cross_impact_on_synthetic_caches():
    rng = np.random.default_rng(0)
    n = 36_000
    caches = []
    for _ in range(2):
        mid = 10_000 + np.cumsum(rng.standard_normal(n + 1) * 0.1)
        c = {"mid": mid, "valid": np.ones(n, bool), "trade_t": np.sort(rng.random(4000) * n / 10),
             "trade_q": rng.exponential(1.0, 4000), "trade_s": rng.choice([-1, 1], 4000),
             "depth": np.ones((n, 2, 10)), "bid_px": (mid[:-1] - 0.5).astype(int), "ask_px": (mid[:-1] + 0.5).astype(int),
             "tick": 1.0, "state": np.zeros((n, 6))}
        caches.append(c)
    out = IMP.cross(caches, caches, boot=50)
    assert out["days"] == 2 and "common information" in out["label"]
