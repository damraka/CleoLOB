"""v0.7 realism v2 metrics on synthetic feature windows (null and alternative behaviour)."""
from __future__ import annotations

import numpy as np
import pytest

from lob.v07.realism import metrics as rm

D = 20


def _windows(rng, n, shift=0.0, scale=1.0):
    return rng.normal(shift, scale, (n, D))


def test_auc_contrast_detects_closer_family_and_is_null_calibrated() -> None:
    rng = np.random.default_rng(0)
    real = _windows(rng, 400)
    near = [_windows(rng, 60, 0.05) for _ in range(8)]
    far = [_windows(rng, 60, 1.0) for _ in range(8)]
    better = rm.auc_contrast(real, near, far, alpha=0.05, samples=300, seed=1)
    assert better["auc_family"] < better["auc_baseline"] and better["ci_high"] < 0
    assert rm.h2_status(better)[0] == "ESTABLISHED"
    worse = rm.auc_contrast(real, far, near, alpha=0.05, samples=300, seed=1)
    assert rm.h2_status(worse)[0] == "FAILED"
    same = rm.auc_contrast(real, [_windows(rng, 60) for _ in range(8)], [_windows(rng, 60) for _ in range(8)],
                           alpha=0.05, samples=300, seed=2)
    assert same["ci_low"] < 0 < same["ci_high"]


def test_h2_ceiling_is_vacuous() -> None:
    result = {"status": "AVAILABLE", "auc_family_ci": [0.995, 1.0], "auc_baseline_ci": [0.999, 1.0],
              "ci_low": -0.004, "ci_high": -0.001}
    assert rm.h2_status(result) == ("NOT_ESTABLISHED", ["VACUOUS"])


def test_coverage_contrast() -> None:
    rng = np.random.default_rng(3)
    history = _windows(rng, 200)
    wide = [_windows(rng, 60, 0.0, 1.0) for _ in range(6)]
    off = [_windows(rng, 60, 3.0, 0.3) for _ in range(6)]
    result = rm.coverage_contrast(history, wide, off, alpha=0.05, samples=100, seed=4)
    assert result["coverage_family"] > 0.5 > result["coverage_baseline"]
    assert rm.h3_status(result)[0] == "ESTABLISHED"
    assert rm.h3_status(rm.coverage_contrast(history, off, wide, alpha=0.05, samples=100, seed=4))[0] == "FAILED"


def test_descriptive_metrics() -> None:
    rng = np.random.default_rng(5)
    a, b, c = _windows(rng, 300), _windows(rng, 300), _windows(rng, 300, 2.0)
    assert abs(rm.mmd_rbf(a, b)["mmd2"]) < rm.mmd_rbf(a, c)["mmd2"]
    pr_same, pr_far = rm.precision_recall(a, b), rm.precision_recall(a, c)
    assert pr_same["precision"] > pr_far["precision"] and pr_same["recall"] > pr_far["recall"]
    roc = rm.roc_curve(np.full(10, 0.9), np.full(10, 0.1))
    assert roc["tpr"][0] == 1.0 and roc["fpr"][-1] == 0.0
    cal = rm.calibration_curve(np.full(10, 0.9), np.full(10, 0.1))
    assert cal["brier"] == pytest.approx(0.01)
    assert rm.pr_curve(np.full(5, 0.9), np.full(5, 0.1))["precision"][-1] == 1.0


def test_scale_auc_and_aggregation_consistency(tmp_path) -> None:
    from lob.v07.data.tape import build_tape
    from tests.v07_fixtures import write_tardis
    files = write_tardis(tmp_path, seconds=900.0)
    tape, _ = build_tape(files["l2"], files["trades"], tick=0.05)
    scales = rm.multiscale_features(tape)
    assert set(scales) == {"10", "60", "300"}
    # Aggregation consistency: trade counts in 10 s windows sum to the 60 s window counts (log1p feature).
    from lob.v06.domain_gap import FEATURES
    i = FEATURES.index("log1p_trades")
    ten, sixty = np.expm1(scales["10"][:, i]), np.expm1(scales["60"][:, i])
    assert abs(ten[:6].sum() - sixty[0]) < 1e-6
    rng = np.random.default_rng(0)
    assert rm.scale_auc(_windows(rng, 100), [_windows(rng, 30, 3.0) for _ in range(4)], seed=1)["auc"] > 0.9
    assert rm.scale_auc(_windows(rng, 10), [], seed=1)["status"] == "NOT_AVAILABLE"
