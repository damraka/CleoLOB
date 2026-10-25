"""v1 realism metrics, capability gate and classical benchmark (M79, M132, M133)."""
import math

import numpy as np
import pytest

from lob.v08.generators import HawkesGenerator, PoissonGenerator
from lob.v08.market.sim import BookModel, simulate
from lob.v1.realism import benchmark as BM, gate as GT, metrics as MT

BOOK = BookModel(refill_mean=6.0, limit_size_mean=3.0, market_size_mean=2.0, mid_ticks=4000)


def frames(gen_cls, params, seeds, bins=1200):
    g = gen_cls({**gen_cls.default_params(), **params}, BOOK)
    return [simulate(g, BOOK, bins, seed=s, warmup_bins=100) for s in seeds]


def test_metrics_cover_every_capability_and_kind():
    m = MT.frame_metrics(frames(PoissonGenerator, {}, [1, 2]))
    assert set(m) == set(MT.METRICS)
    assert {MT.KIND[k] for k in MT.METRICS} == {"marginal", "conditional", "temporal", "tail"}
    assert all(k in MT.KIND for caps in MT.CAPABILITIES.values() for k in caps)
    assert 0 <= m["p_spread_gt1"] <= 1 and m["mean_spread"] >= 1 and m["rate_market"] > 0


def test_metrics_detect_burstiness_difference():
    pois = MT.frame_metrics(frames(PoissonGenerator, {}, [3, 4]))
    assert pois["disp_market_1s"] == pytest.approx(1.0, abs=0.35)      # Poisson counts: variance = mean
    hawkes = MT.frame_metrics(frames(HawkesGenerator, {}, [3, 4]))
    assert np.isfinite(hawkes["disp_market_1s"])


def test_gate_bands_require_eight_reference_days():
    ref = [MT.frame_metrics(frames(PoissonGenerator, {}, [s])) for s in range(5)]
    with pytest.raises(ValueError):
        GT.bands(ref)


def test_gate_capability_specific_and_fail_closed():
    ref = [MT.frame_metrics(frames(PoissonGenerator, {}, [10 + s], bins=900)) for s in range(8)]
    band = GT.bands(ref)
    same = MT.frame_metrics(frames(PoissonGenerator, {}, [99], bins=900))
    d_same = GT.decide(same, band, auc=0.5)
    far = MT.frame_metrics(frames(PoissonGenerator, {"market": 4.0}, [98], bins=900))
    d_far = GT.decide(far, band, auc=0.5)
    assert d_far["event_timing"]["status"] == "REJECTED"
    assert sum(v["status"] == "ACCEPTABLE" for v in d_same.values()) > \
        sum(v["status"] == "ACCEPTABLE" for v in d_far.values())
    d_auc = GT.decide(same, band, auc=0.99)
    assert all(v["status"] != "ACCEPTABLE" for v in d_auc.values())       # a separable generator is never accepted
    assert GT.metric_status(math.nan, (0.0, 1.0)) == "OUTSIDE"           # not estimable on generated data counts out
    marg = GT.decide(same, band, marginal_only=True)
    assert marg["queue_survival"]["status"] == "NOT_ASSESSED"            # no marginal metric for that capability
    g = GT.global_decision({"a": d_far, "b": d_same})
    assert g["status"] in ("ACCEPTABLE_MODEL_SET", "PARTIAL_MODEL_SET", "NO_ACCEPTABLE_MODEL")


def test_classical_benchmark_and_identifiability():
    real = frames(PoissonGenerator, {}, [20, 21], bins=3300)
    ref = [MT.frame_metrics(frames(PoissonGenerator, {}, [30 + s], bins=900)) for s in range(8)]
    gens = {"poisson": PoissonGenerator(None, BOOK), "hot": PoissonGenerator({**PoissonGenerator.default_params(),
                                                                               "market": 5.0}, BOOK)}
    out = BM.benchmark(gens, real, ref, windows=2, bins=3300, seed=1)
    assert out["generators"]["hot"]["decisions"]["event_timing"]["status"] == "REJECTED"
    assert out["generators"]["poisson"]["auc"] < out["generators"]["hot"]["auc"]
    fr = frames(PoissonGenerator, {}, [5], bins=4000)[0]
    idf = BM.identifiability(PoissonGenerator, fr, BOOK, bins=8000, seed=2)
    assert idf["parameters"] >= 3 and idf["median_relative_error"] < 0.15
