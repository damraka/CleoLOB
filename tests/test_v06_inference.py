"""v0.6 inference: resampling, intervals, multiplicity, rank statistics, power and the realism bootstrap."""
from __future__ import annotations

import math

import numpy as np
import pytest

from lob.v06 import inference as inf
from lob.v06 import observables as ob
from lob.v06 import realism as rl
from tests.test_v06_realism import _tape


def test_resampling_weights_preserve_size_and_determinism() -> None:
    rng = np.random.default_rng(1)
    for weights in (inf.counts(17, rng), inf.moving_block_counts(17, 4, rng), inf.stationary_counts(17, 3.0, rng)):
        assert weights.sum() == 17 and (weights >= 0).all()
    assert np.array_equal(inf.counts(50, np.random.default_rng(5)), inf.counts(50, np.random.default_rng(5)))
    with pytest.raises(ValueError):
        inf.counts(0, rng)
    with pytest.raises(ValueError):
        inf.moving_block_counts(5, 6, rng)
    with pytest.raises(ValueError):
        inf.stationary_counts(5, 0.5, rng)


def test_moving_blocks_are_contiguous() -> None:
    assert np.array_equal(inf.moving_block_counts(100, 100, np.random.default_rng(0)), np.ones(100))


def test_intervals_and_pvalues() -> None:
    draws = np.arange(1, 1001, dtype=float)
    low, high = inf.interval(draws, 0.1)
    assert low == pytest.approx(50.95) and high == pytest.approx(950.05)
    assert inf.interval(draws, 0.1, sides="upper")[0] == -math.inf
    assert inf.bootstrap_pvalue(draws) == 0.0
    assert inf.bootstrap_pvalue(np.r_[-draws, draws]) == pytest.approx(1.0)
    assert math.isnan(inf.interval([math.nan], 0.05)[0])
    with pytest.raises(ValueError):
        inf.interval(draws, 1.5)


def test_mean_and_difference_intervals() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(1.0, 1.0, 400), rng.normal(0.0, 1.0, 400)
    paired = inf.difference_interval(a, b, alpha=0.05, samples=2000, seed=1, paired=True)
    independent = inf.difference_interval(a, b, alpha=0.05, samples=2000, seed=1, paired=False)
    for result in (paired, independent):
        assert result["status"] == "AVAILABLE" and result["ci_low"] > 0 and inf.direction(result) == 1
    assert inf.direction(inf.mean_interval(rng.normal(0, 1, 400), alpha=0.05, samples=2000, seed=2)) == 0
    assert inf.mean_interval([1.0], alpha=0.05, samples=100, seed=0)["status"] == "WITHHELD"
    assert inf.mean_interval([1.0, math.nan], alpha=0.05, samples=100, seed=0)["status"] == "WITHHELD"
    assert inf.direction({"status": "WITHHELD"}) is None
    with pytest.raises(ValueError):
        inf.difference_interval([1, 2], [1, 2, 3], alpha=0.05, samples=10, seed=0, paired=True)


def test_multiplicity() -> None:
    assert inf.family_alpha(0.05, 4) == 0.0125
    with pytest.raises(ValueError):
        inf.family_alpha(0.05, 0)
    assert inf.bonferroni([0.01, 0.04]) == [0.02, 0.08]
    assert inf.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert inf.holm([]) == []


def test_rank_statistics() -> None:
    assert inf.ranks([3, 1, 3, 2]).tolist() == [3.5, 1.0, 3.5, 2.0]
    assert inf.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert inf.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(inf.spearman([1, 2], [1, 2]))
    assert inf.kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert inf.kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    assert inf.kendall_tau(["a"], ["a"]) is None
    rng = np.random.default_rng(3)
    x = rng.normal(size=30)
    strong = inf.spearman_interval(x, x + rng.normal(0, 0.1, 30), alpha=0.05, samples=500, seed=1)
    assert strong["ci_low"] > 0.5 and strong["p_one_sided"] == 0.0
    assert inf.spearman_interval([1, 2, 3], [1, 2, 3], alpha=0.05, samples=10, seed=0)["status"] == "WITHHELD"


def test_power_helpers() -> None:
    assert inf._z(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert inf._z(0.001) == pytest.approx(-3.090232, abs=1e-5)
    mde = inf.minimum_detectable_effect(1.0, 100, alpha=0.05, power=0.8)
    assert mde == pytest.approx(0.2802, abs=1e-3)
    assert inf.required_units(1.0, mde, alpha=0.05, power=0.8) in (100, 101)
    assert inf.required_units(1.0, 0.0, alpha=0.05) is None
    assert math.isnan(inf.minimum_detectable_effect(1.0, 1, alpha=0.05))
    assert inf.expected_half_width(2.0, 400, alpha=0.05) == pytest.approx(0.196, abs=1e-3)
    with pytest.raises(ValueError):
        inf._z(1.0)


def test_realism_bootstrap_detects_a_worse_model_and_is_deterministic() -> None:
    dev = [_tape(18000, seed=s) for s in (1, 2)]
    raws = [ob.measure(b) for tape in dev for b in tape.blocks(600)]
    design = ob.build_design(raws, tick_bps=2.5)
    hist = [ob.sketch(r, design) for r in raws]
    good = [ob.pool([ob.sketch(ob.measure(b), design) for b in _tape(12000, seed=s).blocks(600)]) for s in (5, 6, 7)]
    bad = [ob.pool([ob.sketch(ob.measure(b), design) for b in _tape(12000, seed=s, trade_rate=3.0, walk=False)
                    .blocks(600)]) for s in (8, 9, 10)]
    margins = {f: 0.5 for f in ob.FAMILIES}
    kwargs = dict(samples=60, seed=3, alpha=0.05, contrasts=(("good", "bad"),), margins=margins, equivalence_alpha=0.05)
    result = rl.bootstrap_realism(hist, {"good": good, "bad": bad}, design, None, **kwargs)
    again = rl.bootstrap_realism(hist, {"good": good, "bad": bad}, design, None, **kwargs)
    assert result == again
    contrast = result["contrasts"]["good-bad"]
    assert contrast["ci_high"] < 0 and rl.improvement_status(contrast) == "ESTABLISHED"
    assert rl.improvement_status({"ci_low": 0.1, "ci_high": 0.2}) == "FAILED"
    assert rl.improvement_status({"ci_low": -0.1, "ci_high": 0.2}) == "NOT_ESTABLISHED"
    assert rl.improvement_status({"ci_low": None, "ci_high": 0.2}) == "INVALID"
    statuses = {f["equivalence"] for f in result["models"]["bad"]["families"].values()}
    assert statuses <= {"EQUIVALENT_WITHIN_MARGIN", "NOT_ESTABLISHED", "FAILED_MARGIN", "NOT_EVALUABLE"}
    with pytest.raises(ValueError):
        rl.bootstrap_realism(hist[:1], {"good": good}, design, None, samples=5, seed=0, alpha=0.05)
    with pytest.raises(ValueError):
        rl.bootstrap_realism(hist, {"good": good[:1]}, design, None, samples=5, seed=0, alpha=0.05)
