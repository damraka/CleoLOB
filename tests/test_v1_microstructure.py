"""v1 microstructure feature store and analytics (M50-M53, M57-M59)."""
import numpy as np
import pytest

from lob.v1.microstructure import analysis as AN, features as FT


def toy_cache(n=36_000, seed=0):
    rng = np.random.default_rng(seed)
    bid = 1000 + np.cumsum(rng.integers(-1, 2, n) * (rng.random(n) < 0.02))
    ask = bid + 1 + (rng.random(n) < 0.01)
    depth = rng.integers(1, 50, (n, 2, 10)).astype(float)
    counts = rng.poisson(0.5, (n, 6))
    vol = rng.exponential(1.0, (n, 2)) * (counts[:, :2] > 0)
    return {"bid_px": bid, "ask_px": ask, "depth": depth, "counts": counts, "volume": vol,
            "valid": np.ones(n, bool), "mid": np.r_[(bid + ask) / 2.0, (bid[-1] + ask[-1]) / 2.0], "dataset": "toy"}


def test_ofi_definition_on_hand_case():
    bid, ask = np.array([10, 10, 11, 11]), np.array([12, 12, 12, 13])
    d = np.zeros((4, 2, 10))
    d[:, 0, 0] = [5, 7, 3, 3]
    d[:, 1, 0] = [4, 2, 2, 6]
    # bin1: bid +2, ask -2 -> 4; bin2: bid up -> +3, ask same 0 -> 3; bin3: ask up -> -(-2)=+2 -> 2
    assert FT.ofi(bid, ask, d, 1).tolist() == [0.0, 4.0, 3.0, 2.0]


def test_features_store_roundtrip_and_fingerprint(tmp_path):
    c = toy_cache()
    store = FT.FeatureStore(tmp_path)
    f1 = store.get(c, interval_ms=1000)
    assert len(f1["spread"]) == 3600 and set(f1) >= {"ofi_l5", "imbalance_l10", "microprice_offset", "rv_60s"}
    f2 = store.get(c, interval_ms=1000)
    assert all(np.array_equal(f1[k], f2[k]) for k in f1)
    c2 = toy_cache(seed=1)
    c2["dataset"] = "toy"
    assert not np.array_equal(store.get(c2)["spread"], f1["spread"])     # a changed cache is recomputed


def test_ofi_resiliency_and_shock_detection():
    c = toy_cache()
    f = FT.compute(c)
    f["log_touch_depth"] = f["log_touch_depth"].copy()
    f["log_touch_depth"][1000:1005] = np.log(0.5)                          # inject a drought
    ev = AN.liquidity_events(f)
    assert any(abs(e["start_s"] - 1000) <= 1 for e in ev["droughts"]) and ev["shocks"]


def test_memory_hurst_and_acf_decay():
    rng = np.random.default_rng(3)
    white = rng.standard_normal(20_000)
    assert AN.hurst_dfa(white)["hurst"] == pytest.approx(0.5, abs=0.08)
    walk_increments = np.cumsum(rng.standard_normal(20_000)) * 0.01 + white   # persistent
    assert AN.hurst_dfa(walk_increments)["hurst"] > 0.7
    ar = np.zeros(5000)
    for t in range(1, 5000):
        ar[t] = 0.9 * ar[t - 1] + rng.standard_normal()
    assert AN.acf_decay(ar)["acf"][1] == pytest.approx(0.9, abs=0.05)


def test_seasonality_profile_and_fingerprint_similarity():
    days = [FT.compute(toy_cache(seed=s)) for s in range(3)]
    prof = AN.seasonality(days, "trade_count")
    assert len(prof["hour_mean"]) == 1 and prof["days"] == 3
    z = AN.deseasonalize(days[0]["trade_count"], prof)
    assert abs(z.mean()) < 0.5
    ref = {f"d{i}": AN.fingerprint(d) for i, d in enumerate(days)}
    out = AN.similarity(ref, ref["d1"])
    assert out["nearest"][0][0] == "d1" and out["nearest"][0][1] == pytest.approx(1.0)
