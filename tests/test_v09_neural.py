"""v0.9 neural research models (CPU torch; skipped when the neural extra is not installed)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lob.v09.generators import point_process as PP  # noqa: E402
from lob.v09.representations import sequence as S  # noqa: E402


def test_sequence_classifier_learns_a_lagged_class_rule() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = np.zeros(n, np.int64)
    y[1:] = (X[:-1, 0] > 0).astype(np.int64)
    idx = np.arange(S.HISTORY, n)
    tr, te = idx[: len(idx) // 2], idx[len(idx) // 2:]
    model = S.SeqClassifier("gru", hidden=16, steps=300, seed=1, n_classes=2).fit(X, y, tr)
    assert model.nll(X, y, te).mean() < 0.5 * np.log(2) + 0.2


def test_rmtpp_fits_and_scores_finite_nll() -> None:
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.exponential(0.5, 3000))
    m = rng.integers(0, PP.N_MARKS, 3000)
    segs = [(t[:1500], m[:1500]), (t[1500:], m[1500:])]
    model = PP.RMTPP(hidden=8, steps=60, seed=0).fit(segs)
    nll = model.nll(segs)
    assert np.isfinite(nll).all() and len(nll) > 0
