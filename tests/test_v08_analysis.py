"""v0.8 analysis modules: data quality (M3), clocks (M4), regimes (M5), drift and half-life (M6, M7), predictive
checks (M16), model averaging (M17), uncertainty decomposition (M18), OOD (M19) and conformal coverage (M20)."""
from __future__ import annotations

import numpy as np
import pytest

from lob.v08 import clocks, conformal, drift, model_averaging as ma, ood, ppc, quality, regimes
from lob.v08.generators import HawkesGenerator, PoissonGenerator, RegimeSwitchingGenerator
from lob.v08.market.frame import summaries
from lob.v08.uncertainty import decompose, summarize_policy_costs


# ----------------------------------------------------------------------------- quality and clocks


def _table(rows):
    a = np.asarray(rows, float)
    return quality.EventTable(a[:, 0].astype(np.int8), a[:, 1].astype(np.int64), a[:, 2].astype(np.int64),
                              a[:, 3].astype(np.int8), a[:, 4], a[:, 5], np.zeros(len(a), bool))


def test_clean_generated_tape_passes_quality() -> None:
    report = quality.validate(quality.from_frame(PoissonGenerator().simulate(400, 1), tick=0.5), tick=0.5)
    assert report["status"] == "PASS" and report["score"] == pytest.approx(1.0) and report["repairs"].startswith("none")


@pytest.mark.parametrize("flag,rows", [
    ("crossed_book", [(0, 1, 1, 1, 101.0, 5), (0, 2, 2, -1, 100.0, 5)]),
    ("locked_book", [(0, 1, 1, 1, 100.0, 5), (0, 2, 2, -1, 100.0, 5)]),
    ("negative_size", [(0, 1, 1, 1, 100.0, -5)]),
    ("duplicate_event", [(0, 1, 1, 1, 100.0, 5), (0, 1, 1, 1, 100.0, 5)]),
    ("out_of_order", [(0, 5, 5, 1, 100.0, 5), (0, 6, 4, 1, 99.0, 5)]),
    ("timestamp_regression", [(0, 5, 5, 1, 100.0, 5), (0, 4, 6, 1, 99.0, 5)]),
    ("missing_timestamp", [(0, 0, 1, 1, 100.0, 5)]),
    ("zero_volume_trade", [(0, 1, 1, 1, 100.0, 5), (0, 1, 1, -1, 101.0, 5), (1, 2, 2, 1, 101.0, 0)]),
    ("price_grid", [(0, 1, 1, 1, 100.03, 5)]),
    ("long_gap", [(0, 1, 1, 1, 100.0, 5), (0, 10_000_000, 10_000_000, 1, 99.0, 5)]),
    ("impossible_transition", [(0, 1, 1, 1, 100.0, 0)]),
    ("trade_book_inconsistent", [(0, 1, 1, 1, 100.0, 5), (0, 1, 1, -1, 101.0, 5), (1, 2, 2, 1, 98.0, 1)]),
])
def test_quality_flags_are_detected_without_repairs(flag, rows) -> None:
    report = quality.validate(_table(rows), tick=0.5)
    assert report["flags"][flag]["count"] >= 1 and report["flags"][flag]["status"] != "PASS"


def test_clock_normalization_and_causal_order() -> None:
    assert clocks.normalize([1.5], "s").tolist() == [1_500_000_000]
    cs = clocks.ClockSet().add("exchange", [1, 2, 3], "ms").add("receive", [2, 3, 4], "ms")
    cs.add("send", [5, 6, 7], "ms").add("ack", [6, 5, 8], "ms")
    report = clocks.validate(cs)
    assert report["issues"]["ack_before_send"]["count"] == 1 and report["status"] == "FAIL"
    good = clocks.ClockSet().add("exchange", [1, 2], "us").add("receive", [3, 4], "us")
    assert clocks.validate(good)["status"] == "PASS"
    with pytest.raises(ValueError):
        clocks.ClockSet().add("wall", [1, 2], "ms").add("event", [1], "ms")
    assert clocks.volume_time([1, 1, 1, 1], 2).tolist() == [0, 1, 1, 2]
    grid, vals = clocks.resample_wall(np.asarray([0, 10, 25]), np.asarray([1, 2, 3]), 10)
    assert vals.tolist() == [1, 2, 2]


# ----------------------------------------------------------------------------- regimes and drift


def test_regime_methods_fit_on_development_and_negative_control() -> None:
    g = RegimeSwitchingGenerator({"rates": [[0.2] * 6, [1.6] * 6], "trans": [[0.998, 0.002], [0.002, 0.998]],
                                  "init": [0.5, 0.5]})
    dev, ev = regimes.block_features(g.simulate(4000, 1), 100), regimes.block_features(g.simulate(4000, 2), 100)
    res = regimes.compare(dev, ev, seed=1)
    assert set(res) == set(regimes.METHODS)
    t = res["threshold"]
    assert t["next_block_r2_activity"] > t["negative_control_shuffled_r2_activity"]
    assert t["labels_fitted_on"] == "development blocks only"


def test_binary_segmentation_finds_a_mean_shift() -> None:
    x = np.r_[np.zeros(50), np.full(50, 3.0)] + np.random.default_rng(0).normal(0, 0.3, 100)
    assert any(45 <= c <= 55 for c in regimes.binary_segmentation(x))


def test_detectors_find_change_and_rarely_alarm_without_one() -> None:
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(0, 1, 150), rng.normal(2.0, 1, 150)]
    for method in drift.DETECTORS:
        report = drift.evaluate(drift.detect(x, ref=75, method=method), [150], tolerance=40, n=len(x))
        if method != "support":
            assert report["detected"] == 1
    rates = [drift.evaluate(drift.cusum(np.random.default_rng(10 + k).normal(0, 1, 600), ref=200), [], tolerance=40,
                            n=600)["false_alarm_rate_per_100"] for k in range(10)]
    assert np.mean(rates) < 1.0          # average false-alarm rate without any change


def test_half_life_estimation_and_honest_failure() -> None:
    h = np.arange(1, 11)
    est = drift.half_life(h, 2.0 * (1 - np.exp(-h / 3.0)), 0.0)
    assert est["status"] == "ESTIMATED" and est["half_life"] == pytest.approx(3.0 * np.log(2), rel=0.1)
    assert drift.half_life(h, -np.ones(10), 0.0)["status"] == "NOT_ESTABLISHED"
    assert drift.half_life([1, 2], [1, 2], 0.0)["status"] == "NOT_EVALUABLE"


# ----------------------------------------------------------------------------- PPC and model averaging


def test_ppc_flags_a_wrong_model_more_than_the_right_one() -> None:
    truth = HawkesGenerator()
    observed = truth.simulate(1200, 5)
    right = ppc.check(observed, [truth.simulate(1200, 10 + k) for k in range(8)])
    wrong = ppc.check(observed, [PoissonGenerator({**PoissonGenerator.default_params(), "market": 4.0, "add": 9.0,
                                                   "cancel": 8.0}).simulate(1200, 20 + k) for k in range(8)])
    assert wrong["failure_fraction"] > right["failure_fraction"]
    assert right["horizons"]["60s"]["status"] == "NOT_EVALUABLE"      # 1200 bins < 5 x 600


def test_model_averaging_weights_and_dynamic_path() -> None:
    rng = np.random.default_rng(0)
    ll = np.column_stack([rng.normal(-1.0, 0.1, 500), rng.normal(-1.5, 0.1, 500)])
    w = ma.stacking_weights(ll)
    assert w[0] > 0.9
    pmp = ma.posterior_model_probabilities({"a": -100.0, "b": -110.0}, {"a": 3, "b": 3}, 500)
    assert pmp["a"] > 0.99 and ma.material(pmp)["material"] is False
    assert ma.material({"a": 0.6, "b": 0.4})["material"] is True
    switched = np.vstack([ll[:250], ll[250:, ::-1]])
    dyn = ma.dynamic(switched, alpha=0.95)
    assert dyn["path"][100, 0] > 0.9 and dyn["path"][-1, 1] > 0.9
    assert ma.mixture_score(ll, [0.5, 0.5]).shape == (500,)


# ----------------------------------------------------------------------------- uncertainty, OOD, conformal


def test_uncertainty_decomposition_recovers_designed_sources() -> None:
    rng = np.random.default_rng(1)
    families = np.repeat(["a", "b"], 10)
    queue = np.tile(["u", "f"], 10)
    means = np.where(families == "a", 0.0, 5.0)
    y = means[:, None] + rng.normal(0, 0.5, (20, 12))
    out = decompose(y, {"model_family": families.tolist(), "queue": queue.tolist()}, n_boot=100)
    assert out["share_model_family"] > 0.8 and out["share_queue"] < 0.05
    assert out["epistemic_share_ci"][0] > 0.5 and out["share_aleatoric"] < 0.2
    rows = [{"world_id": f"w{i}", "seed": s, "policy": p, "economic_cost_bps": float(means[i] + (p == "b")),
             "level_model_family": families[i]} for i in range(20) for s in range(3) for p in ("a", "b")]
    summary = summarize_policy_costs(rows, value="economic_cost_bps", policies=["a", "b"], sources=["model_family"],
                                     n_boot=20)
    assert "a|b" in summary["pairs"] and summary["pairs"]["a|b"]["epistemic_var"] == pytest.approx(0.0, abs=1e-9)


def test_ood_detectors_separate_known_shift_at_registered_fpr() -> None:
    ref = np.asarray([summaries(PoissonGenerator().simulate(400, k)) for k in range(30)])
    cal = np.asarray([summaries(PoissonGenerator().simulate(400, 100 + k)) for k in range(20)])
    shifted = np.asarray([summaries(PoissonGenerator({**PoissonGenerator.default_params(), "market": 4.0})
                                    .simulate(400, 200 + k)) for k in range(10)])
    r = ood.Reference(ref)
    thr = ood.threshold(r.mahalanobis(cal), 0.05)
    report = ood.evaluate(r.mahalanobis(cal), {"burst": r.mahalanobis(shifted)}, thr)
    assert report["false_positive_rate"] <= 0.1 and report["shifts"]["burst"]["detection_rate"] >= 0.8
    assert (r.conformal(shifted) > 0.9).mean() >= 0.8 and r.support(shifted).mean() > r.support(cal).mean()
    assert ood.ensemble_score(np.asarray([[1.0, 1.0], [0.0, 2.0]])).tolist() == [0.0, 1.0]
    with pytest.raises(ValueError):
        ood.Reference(ref[:3])


def test_split_conformal_coverage_under_exchangeability() -> None:
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(20):      # the guarantee is marginal over calibration draws: average over repetitions
        cal, test = rng.normal(0, 1, 400), rng.normal(0, 1, 2000)
        s = conformal.split(cal, np.zeros(400), np.zeros(2000), 0.1)
        covs.append(conformal.coverage(test, s["lower"], s["upper"], nominal=0.9)["coverage"])
    assert 0.89 <= np.mean(covs) <= 0.92
    shifted = conformal.coverage(test + 3, s["lower"], s["upper"], nominal=0.9)
    assert shifted["coverage"] < 0.5


def test_rolling_mondrian_and_classification_sets() -> None:
    rng = np.random.default_rng(2)
    roll = conformal.rolling(rng.normal(0, 1, 200), np.zeros(200), 0.1, window=50)
    hits = [x for x in roll["covered"] if x is not None]
    assert 0.8 <= np.mean(hits) <= 0.98 and roll["covered"][0] is None
    m = conformal.mondrian(rng.normal(0, 1, 50), np.zeros(50), ["a"] * 45 + ["b"] * 5, [0.0], ["b"], 0.1, min_group=20)
    assert m["fallback_groups"] == ["b"]
    sets = conformal.classification_sets([1, 0, 1, 1], [0.9, 0.2, 0.8, 0.7], [0.95, 0.05], 0.25)
    assert sets["sets"][0] == [1] and sets["sets"][1] == [0]
    assert conformal.quantile([], 0.1) == float("inf")
