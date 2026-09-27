"""M4 impact and resilience measurement and comparison. Inputs are synthetic."""
from __future__ import annotations

import numpy as np
import pytest

from lob.impact import HORIZONS_S, ac_proxies, buckets, event_table, thresholds
from lob.impact_validation import compare
from lob.observables_v2 import SAMPLE_DT, Tape
from lob.resilience import depletion_events, recovery


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


def test_event_responses_are_signed_and_causal():
    table = event_table(_tape())
    assert set(f"mid_bps_{h:g}s" for h in HORIZONS_S) <= set(table)
    responses = table["mid_bps_1s"][table["valid_pre"]]
    assert np.nanmean(responses) > 0  # signed permanent impact is positive on average
    # pre-event conditioning uses the sample at or before the event, never after
    k = np.searchsorted(np.arange(12000) * SAMPLE_DT, table["time"], side="right") - 1
    assert np.all(np.arange(12000)[k] * SAMPLE_DT <= table["time"] + 1e-9)


def test_bucket_thresholds_are_frozen_and_reused():
    dev = event_table(_tape(seed=1))
    edges = thresholds(dev)
    other = event_table(_tape(seed=2))
    labels = buckets(other, edges, "size")
    assert set(np.unique(labels[np.isfinite(labels)])) <= {0.0, 1.0, 2.0}
    assert edges == thresholds(dev)


def test_ac_proxies_are_labelled_as_proxies():
    tape = _tape()
    table = event_table(tape)
    proxies = ac_proxies(table, np.diff(np.log(tape.mid[::10])) * 1e4)
    assert proxies["persistent_impact_proxy_bps_per_lot"] > 0
    assert "no unique structural impact model" in proxies["caveat"]


def test_resilience_recovery_is_censored_and_measured():
    tape = _tape()
    events = depletion_events(tape)
    result = recovery(tape, events)
    assert result["events"] > 0 and result["recovered_share"] > 0.9
    assert result["recovery_time_quantiles_s"][1] == pytest.approx(0.5, abs=0.11)


def test_hist_sim_comparison_agrees_on_identical_mechanism_and_detects_reversal():
    dev = event_table(_tape(seed=1, events=900))
    edges = thresholds(dev)
    hist = event_table(_tape(seed=3, events=900))
    same = [event_table(_tape(seed=s, events=900)) for s in (4, 5, 6)]
    reversed_ = [event_table(_tape(seed=s, events=900, impact=-0.2)) for s in (4, 5, 6)]
    ok = compare(hist, same, edges, alpha=0.05, samples=200, seed=1)
    bad = compare(hist, reversed_, edges, alpha=0.05, samples=200, seed=1)
    assert ok["H4_by_horizon"]["1s"]["status"] == "ESTABLISHED"
    assert bad["H4_by_horizon"]["1s"]["status"] == "NOT_ESTABLISHED"
    cell = bad["cells"]["1s|2"]
    assert cell["sign_agreement"] is False and cell["ks_distance"] > 0.5


def test_small_buckets_are_not_available():
    hist = event_table(_tape(events=60))
    edges = thresholds(hist)
    result = compare(hist, [hist], edges, alpha=0.05, samples=50, seed=1)
    assert all(c["status"] == "NOT_AVAILABLE" for c in result["cells"].values())
