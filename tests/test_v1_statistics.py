"""v1 cluster-level inference (M130, M7)."""
import math

import numpy as np
import pytest

from lob.v1 import statistics as S

# reference values: Student t quantiles (standard tables) and incomplete beta identities
T975 = {1: 12.7062047362, 2: 4.3026527297, 3: 3.1824463053, 5: 2.5705818366, 10: 2.2281388520, 30: 2.0422724563}


@pytest.mark.parametrize("df,q", sorted(T975.items()))
def test_t_quantiles_match_tables(df, q):
    assert S.t_ppf(0.975, df) == pytest.approx(q, rel=1e-9)
    assert S.t_cdf(q, df) == pytest.approx(0.975, abs=1e-9)
    assert S.t_ppf(0.025, df) == pytest.approx(-q, rel=1e-9)


def test_betainc_identities():
    assert S.betainc(1.0, 1.0, 0.3) == pytest.approx(0.3)
    assert S.betainc(2.0, 3.0, 0.4) + S.betainc(3.0, 2.0, 0.6) == pytest.approx(1.0)
    assert S.norm_ppf(0.975) == pytest.approx(1.959963984540054, rel=1e-10)


def test_cluster_t_equals_cluster_means_t_for_balanced_clusters():
    rng = np.random.default_rng(1)
    dates = np.repeat(np.arange(6), 3)
    x = rng.normal(2.0, 1.0, 18) + np.repeat(rng.normal(0, 2.0, 6), 3)
    r = S.cluster_t(x, dates, threshold=1.0)
    means = np.array([x[dates == d].mean() for d in range(6)])
    se = means.std(ddof=1) / math.sqrt(6)
    assert r.clusters == 6 and r.df == 5
    assert r.se == pytest.approx(se, rel=1e-12)
    assert r.p_one_sided == pytest.approx(S.t_sf((means.mean() - 1.0) / se, 5), rel=1e-10)


def test_duplicated_units_do_not_shrink_cluster_se():
    x = np.array([1.0, 3.0, 2.0, 5.0])
    once = S.cluster_t(x, ["a", "b", "c", "d"])
    twice = S.cluster_t(np.repeat(x, 2), np.repeat(["a", "b", "c", "d"], 2))
    assert twice.se == pytest.approx(once.se)            # copying a unit to a second stream adds no information
    naive = np.repeat(x, 2).std(ddof=1) / math.sqrt(8)
    assert naive < twice.se


def test_sign_flip_floor_and_exactness():
    r = S.sign_flip([5.0, 6.0, 7.0, 8.0], ["d1", "d2", "d3", "d4"], threshold=1.0)
    assert r["exact"] and r["p_floor"] == 1 / 16 and r["p_one_sided"] == pytest.approx(1 / 16)
    assert S.min_clusters_for(0.05) == 5 and S.min_clusters_for(0.0125) == 7


def test_multiway_reduces_to_oneway_when_second_is_unique():
    rng = np.random.default_rng(3)
    x = rng.normal(size=12)
    a = np.repeat(np.arange(4), 3)
    b = np.arange(12)
    one = S.cluster_t(x, a)
    two = S.multiway_t(x, a, b)
    # V_b and V_intersection coincide (singleton clusters) up to their small-sample factors
    assert two.se == pytest.approx(one.se, rel=0.1)


def test_design_effect_detects_shared_date_dependence():
    rng = np.random.default_rng(5)
    date_effect = rng.normal(0, 3.0, 10)
    x = np.repeat(date_effect, 3) + rng.normal(0, 0.1, 30)
    d = S.design_effect(x, np.repeat(np.arange(10), 3))
    assert d["icc"] > 0.95 and d["effective_units"] == pytest.approx(10, rel=0.05)


def test_power_and_feasibility_are_monotone():
    p = [S.cluster_power(1.0, 2.0, g, alpha=0.05) for g in (5, 10, 20, 40)]
    assert p == sorted(p) and p[-1] > 0.9
    g = S.feasible_clusters(1.0, 2.0, alpha=0.05)
    assert g is not None and S.cluster_power(1.0, 2.0, g, alpha=0.05) >= 0.8
    assert S.cluster_power(1.0, 2.0, g - 1, alpha=0.05) < 0.8 or g == S.min_clusters_for(0.05)
    assert S.feasible_clusters(-1.0, 2.0, alpha=0.05) is None


def test_holm_adjust_monotone():
    adj = S.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        S.cluster_t([1.0, float("nan")], ["a", "b"])
    with pytest.raises(ValueError):
        S.cluster_t([1.0], ["a", "b"])
