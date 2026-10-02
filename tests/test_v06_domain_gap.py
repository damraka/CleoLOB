"""v0.6 domain gap (leakage guard, classifiers, AUC), OOD support, regime transitions and transfer rules."""
from __future__ import annotations

import math

import numpy as np
import pytest

from lob.v06 import domain_gap as dg
from lob.v06 import regimes as rg
from lob.v06 import transfer as tr
from tests.test_v06_realism import _tape


def test_window_features_are_whitelisted_and_scale_free() -> None:
    tape = _tape(6000, seed=1)
    features = dg.window_features(tape)
    assert features.shape == (10, len(dg.FEATURES)) and np.isfinite(features).all()
    shifted = _tape(6000, seed=1)
    shifted.bp += 100.0
    shifted.ap += 100.0
    # price level enters only through bps quantities, not as a feature
    assert np.allclose(dg.window_features(shifted)[:, dg.FEATURES.index("log1p_trades")],
                       features[:, dg.FEATURES.index("log1p_trades")])
    for name in ("timestamp", "dataset_id", "seed", "mid_price", "spread_mean_time"):
        with pytest.raises(ValueError, match="whitelist"):
            dg.feature_matrix({name: np.zeros(3)})
    assert dg.feature_matrix({"spread_mean": np.ones(3), "return_sd": np.zeros(3)}).shape == (3, 2)


def test_auc_and_balanced_accuracy() -> None:
    y = np.asarray([1, 1, 0, 0])
    assert dg.auc(np.asarray([0.9, 0.8, 0.2, 0.1]), y) == 1.0
    assert dg.auc(np.asarray([0.1, 0.2, 0.8, 0.9]), y) == 0.0
    assert dg.auc(np.asarray([0.5, 0.5, 0.5, 0.5]), y) == 0.5
    assert math.isnan(dg.auc(np.asarray([0.1, 0.2]), np.asarray([1, 1])))
    assert dg.balanced_accuracy(np.asarray([0.9, 0.1, 0.2, 0.1]), y) == 0.75


def _windows(n: int, shift: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 1, (n, len(dg.FEATURES))) + shift


def test_two_sample_test_separates_shift_and_not_identical() -> None:
    real = _windows(400, 0.0, 1)
    shifted = [_windows(100, 1.0, 10 + i) for i in range(8)]
    same = [_windows(180, 0.0, 20 + i) for i in range(16)]
    gap = dg.two_sample_test(real, shifted, seed=3, samples=200)
    null = dg.two_sample_test(real, same, seed=3, samples=200)
    assert gap["classifiers"]["logistic"]["auc"] > 0.9 and dg.gap_status(gap) == "ESTABLISHED"
    assert abs(null["classifiers"]["logistic"]["auc"] - 0.5) < 0.08
    for name in ("tree", "forest"):
        assert 0 <= gap["classifiers"][name]["auc"] <= 1
    assert gap["train"]["real"] == gap["train"]["simulated"]
    assert dg.two_sample_test(real[:10], shifted, seed=0)["status"] == "NOT_AVAILABLE"
    assert dg.gap_status({"status": "NOT_AVAILABLE"}) == "NOT_AVAILABLE"
    again = dg.two_sample_test(real, shifted, seed=3, samples=200)
    assert again["classifiers"]["forest"]["auc"] == gap["classifiers"]["forest"]["auc"]


def test_chronological_split_purges() -> None:
    train, test = dg.chronological_split(100, purge=10)
    assert train.max() == 49 and test.min() == 60 and not set(train) & set(test)


def test_logistic_recovers_signal_direction() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 3))
    y = (x[:, 0] + 0.1 * rng.normal(size=500) > 0).astype(float)
    model = dg.Logistic().fit(x, y)
    assert model.coefficients[0] > 1 and abs(model.coefficients[1]) < 0.5


def test_support_flags_out_of_support_history() -> None:
    sims = [_windows(200, 0.0, i) for i in range(4)]
    scaler = dg.Standardizer(np.vstack(sims))
    inside = dg.support(_windows(100, 0.0, 9), sims, scaler)
    outside = dg.support(_windows(100, 6.0, 9), sims, scaler)
    assert inside["out_of_support_fraction"] < 0.1 and inside["label"] == "SUPPORTED_ENOUGH"
    assert outside["out_of_support_fraction"] > 0.9 and outside["label"] == "NO_CLAIM"
    assert dg.support(_windows(5, 0, 1), sims[:1], scaler)["status"] == "NOT_AVAILABLE"


def test_transitions_and_dwell() -> None:
    labels = ["low", "low", "high", "high", "high", "low"]
    stats = rg.transitions([labels], ("low", "high"))
    assert stats["counts"] == [[1.0, 1.0], [1.0, 2.0]]
    assert stats["mean_dwell_blocks"] == {"low": 1.5, "high": 3.0}
    two = rg.transitions([["low", "high"], ["high", "low"]], ("low", "high"))
    assert np.asarray(two["counts"]).sum() == 2          # no transition across separate paths
    comparison = rg.transition_comparison(stats, stats)
    assert comparison["transition_row_tv_mean"] == 0 and comparison["occupancy_tv"] == 0


def test_regime_labels_use_v05_code_on_top5_view() -> None:
    tape = _tape(9000, seed=2)
    thresholds = {name: {"median": 0.0, "p90": 1e9} for name in ("volatility", "spread", "depth", "activity",
                                                                   "imbalance", "trend")}
    blocks = rg.labelled_blocks(tape, thresholds)
    assert len(blocks) == 3 and all(b["labels"]["stress"] == "normal" for b in blocks)
    view = rg.v05_view(tape)
    assert view.bq.shape[1] == 5


def test_fill_semantics_rules() -> None:
    def pairs(c: int | None, o: int | None) -> dict:
        out = {}
        for left, right in tr.PAIRS:
            out[f"conservative|{left}|{right}"] = {"direction": c}
            out[f"optimistic|{left}|{right}"] = {"direction": o}
        return out
    assert tr.fill_semantics(pairs(1, 1))["status"] == "ESTABLISHED"
    assert tr.fill_semantics(pairs(0, 0))["status"] == "ESTABLISHED"
    assert tr.fill_semantics(pairs(1, -1))["status"] == "FAILED"
    assert tr.fill_semantics(pairs(0, 1))["status"] == "NOT_ESTABLISHED"
    assert tr.fill_semantics(pairs(None, None))["status"] == "NOT_AVAILABLE"


@pytest.mark.parametrize("predicted,conservative,optimistic,label", [
    (1, 1, 1, "agrees"), (1, -1, -1, "reverses"), (1, 0, 1, "assumption_dependent"), (0, 1, 1, "indeterminate"),
    (1, 0, 0, "indeterminate"), (None, 1, 1, "not_evaluable"), (1, None, 1, "not_evaluable")])
def test_transfer_classification(predicted, conservative, optimistic, label) -> None:
    def interval(d):
        if d is None:
            return {"status": "WITHHELD"}
        return {"status": "AVAILABLE", "ci_low": {1: 1.0, -1: -2.0, 0: -1.0}[d], "ci_high": {1: 2.0, -1: -1.0, 0: 1.0}[d]}
    historical = {"conservative": {"direction": conservative}, "optimistic": {"direction": optimistic}}
    assert tr.classify(interval(predicted), historical) == label


def test_predicted_difference_two_stage_bootstrap() -> None:
    from lob.v06.execution import cells
    from tests.test_v06_execution import MARKETS, _means, _rows
    rows = _rows("a", _means()) + _rows("b", _means(), seed=1)
    cell_map = cells(rows, MARKETS)
    result = tr.predicted_difference(cell_map, ["a", "b"], "pov", "ac", alpha=0.05, samples=300, seed=1)
    assert result["status"] == "AVAILABLE" and result["ci_low"] > 0 and result["worlds"] == 2
    assert tr.predicted_difference(cell_map, ["a", "missing"], "pov", "ac", alpha=0.05, samples=10,
                                   seed=1)["status"] == "WITHHELD"
