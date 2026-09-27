"""M6 regimes and dataset registry. Inputs are synthetic or the committed registry."""
from __future__ import annotations

import numpy as np
import pytest

from lob.dataset_registry import REQUIRED, build, comparable
from lob.observables_v2 import SAMPLE_DT, Tape
from lob.regimes import DIMENSIONS, block_statistics, fit_thresholds, label, regime_counts


def _tape(n=12000, impact=0.2, seed=0, events=300):
    rng = np.random.default_rng(seed)
    mid = np.full(n, 100.0)
    trade_t = np.sort(rng.uniform(5, n * SAMPLE_DT - 80, events))
    signs = rng.choice([-1.0, 1.0], events)
    sizes = rng.uniform(1, 30, events)
    for t, s, q in zip(trade_t, signs, sizes):
        k = int(np.ceil(t / SAMPLE_DT))
        mid[k:] *= np.exp(s * impact * q * 1e-4)  # permanent signed impact proportional to size
    bp = mid[:, None] - 0.025 - 0.05 * np.arange(5)
    ap = mid[:, None] + 0.025 + 0.05 * np.arange(5)
    bq = np.full((n, 5), 20.0)
    aq = np.full((n, 5), 20.0)
    for t, s in zip(trade_t, signs):
        k = int(np.ceil(t / SAMPLE_DT))
        side = aq if s > 0 else bq
        side[k:k + 5, 0] = 2.0  # depleted, recovers after 0.5 s
    return Tape(np.arange(n) * SAMPLE_DT, bp, bq, ap, aq, trade_t, np.full(events, 100.0), sizes, signs)


def test_regimes_use_development_thresholds_only():
    dev_blocks = block_statistics(_tape(n=90000, seed=7, events=3000))
    assert len(dev_blocks) >= 20
    frozen = fit_thresholds(dev_blocks)
    assert set(frozen) == set(DIMENSIONS)
    holdout = block_statistics(_tape(n=36000, seed=8, events=1000))
    counts = regime_counts(holdout, frozen)
    assert set(counts) == set(DIMENSIONS) | {"stress"}
    assert label(holdout[0], frozen)["stress"] in {"stress", "normal"}
    with pytest.raises(ValueError):
        fit_thresholds(dev_blocks[:5])


def test_registry_has_all_required_fields_and_ledger_freshness():
    registry = build()
    for entry in registry.values():
        assert set(REQUIRED) <= set(entry)
    assert registry["deribit-eth-perp-2020-06-01"]["freshness"] == "consumed"
    assert registry["deribit-eth-perp-2020-06-01"]["consumed"] is True


def test_cross_venue_comparison_is_refused():
    registry = build()
    ok, reasons = comparable(registry["deribit-eth-perp-2020-04-01"], registry["bitstamp-btcusd-mbo-dev"])
    assert not ok and "different capability levels" in reasons
    ok, reasons = comparable(registry["deribit-eth-perp-2020-04-01"], registry["deribit-btc-perp-2020-07-01"],
                             pooled=True)
    assert not ok and reasons == ["pooling across instruments or tick sizes is unsupported"]
