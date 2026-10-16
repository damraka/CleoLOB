"""v0.9 model components on generated data (fast; no market data)."""
from __future__ import annotations

import numpy as np
import pytest

from lob.v09 import causal, twin
from lob.v09.generators import point_process as PP
from lob.v09.queue import survival as SV
from lob.v09.realism import path as RP
from lob.v09.studies import descriptive as DS
from lob.v09.uncertainty import conformal as CO


def _poisson_segments(rate: np.ndarray, seconds: float, seed: int):
    rng = np.random.default_rng(seed)
    n = rng.poisson(rate.sum() * seconds)
    t = np.sort(rng.uniform(0, seconds, n))
    m = rng.choice(len(rate), size=n, p=rate / rate.sum())
    return [(t, m)]


def test_marked_poisson_recovers_rates_and_scores_better_than_wrong_rates() -> None:
    rate = np.linspace(0.5, 2.0, PP.N_MARKS)
    segs = _poisson_segments(rate, 4000.0, 1)
    model = PP.MarkedPoisson().fit(segs)
    assert np.allclose(model.lam, rate, rtol=0.15)
    wrong = PP.MarkedPoisson()
    wrong.lam = rate[::-1].copy()
    assert model.nll(segs).mean() < wrong.nll(segs).mean()


def test_survival_models_rank_risk_and_beat_km_on_informative_covariates() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(800, 3))
    T = rng.exponential(np.exp(-0.8 * X[:, 0]) * 4.0)
    E = T < 30.0
    T = np.minimum(T, 30.0)
    km, cox = SV.KaplanMeier().fit(X, T, E), SV.CoxPH().fit(X, T, E)
    assert SV.integrated_brier(cox.predict(X), T, E) < SV.integrated_brier(km.predict(X), T, E)
    assert SV.c_index(X[:, 0], T, E) > 0.6


def test_path_signature_mmd_separates_different_dynamics() -> None:
    rng = np.random.default_rng(5)
    L = RP.LENGTH + 1

    def paths(scale, n):
        mid = np.cumsum(rng.choice([-1, 0, 0, 1], size=(n, L)) * scale, axis=1)
        return np.stack([mid - mid[:, :1], np.ones((n, L))], axis=-1).astype(float)

    a, b, c = paths(1, 40), paths(1, 40), paths(4, 40)
    sa, sb, sc = (RP.signature(x, 5.0) for x in (a, b, c))
    assert RP.mmd2(sa, sc) > RP.mmd2(sa, sb)


def test_conformal_split_and_aci_reach_nominal_coverage_on_exchangeable_data() -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(1200, 3))
    y = X @ np.asarray([0.5, -0.2, 0.1]) + rng.normal(scale=0.3, size=1200)
    coef = CO.fit_base(X[:600], y[:600])
    train_scores = np.abs(y[:600] - CO.predict(coef, X[:600]))
    test = {"X": X[600:], "y": y[600:], "t": np.arange(600, dtype=float) * 10.0}
    out = CO.run_day(coef, train_scores, X[:600, 1], test)
    for m in ("split", "aci"):
        assert abs(out[m]["coverage"] - (1 - CO.ALPHA)) < 0.05


def test_causal_registry_does_not_identify_observational_flow_impact() -> None:
    rep = causal.registry_report()
    assert rep["flow_impact"]["identification"]["status"] == "NOT_IDENTIFIED"
    assert rep["own_order_impact_bounded"]["identification"]["status"] == "ASSUMPTION_DEPENDENT"
    assert rep["fallback_effect_simulated"]["identification"]["status"] == "IDENTIFIED"


def test_twin_agreement_and_kendall_distance() -> None:
    assert twin.kendall_distance(["a", "b", "c"], ["a", "b", "c"]) == 0.0
    assert twin.kendall_distance(["a", "b", "c"], ["c", "b", "a"]) == 1.0
    agree, n = twin.agreement({"x": 1, "y": -1, "z": 1}, {"x": 1, "y": 1, "z": 0})
    assert n >= 1 and 0.0 <= agree <= 1.0


def test_binary_segmentation_finds_a_variance_shift() -> None:
    rng = np.random.default_rng(2)
    x = np.r_[rng.normal(0, 1, 300), rng.normal(3, 1, 300)]
    cps = DS._binary_segmentation((x - x.mean()) / x.std(), penalty=2 * np.log(600) * 2)
    assert cps and abs(cps[0] - 300) < 15
    assert DS._binary_segmentation(rng.normal(size=600), penalty=2 * np.log(600) * 2) == []


def test_hill_estimator_on_pareto_tails() -> None:
    rng = np.random.default_rng(4)
    x = rng.pareto(3.0, 200_000) + 1.0
    assert DS._hill(x) == pytest.approx(3.0, rel=0.15)


def test_sbc_positive_control_is_detected() -> None:
    out = DS.sbc_task(1, replicates=200)
    assert out["correct"]["status"] == "CALIBRATED" and out["positive_control_detected"]
