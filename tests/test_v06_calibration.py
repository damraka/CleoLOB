"""v0.6 calibration v3: parameter transforms, search determinism, selection rules and identifiability helpers."""
from __future__ import annotations

import math

import numpy as np
import pytest

from lob.v06 import calibration as cal
from lob.v06 import identifiability as ident

BASE = cal.base_config(3.79)
TABLES = {"market_size_quantiles": [1.0] * 10 + [2.0] * 11, "limit_size_quantiles": [1.0] * 10 + [5.0] * 11}


def test_parameter_transforms_roundtrip_and_bounds() -> None:
    for parameter in cal.PARAMETERS:
        assert parameter.value(-1) == pytest.approx(parameter.value(0)) == pytest.approx(parameter.low)
        assert parameter.value(2) == pytest.approx(parameter.high, rel=1e-3)
        if not parameter.integer:
            assert parameter.unit(parameter.value(0.37)) == pytest.approx(0.37)
    assert len(cal.PARAMETERS) == 14
    assert "limit_qty_mean" not in cal.NAMES and "market_qty_mean" not in cal.NAMES


def test_spec_from_unit_respects_simulator_constraints() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        u = rng.random(14)
        spec = cal.spec_from_unit(u, BASE, TABLES)
        assert spec.extensions["hawkes_alpha"] < spec.extensions["hawkes_decay"]
        assert isinstance(spec.config["target_level_vol"], int)
        assert np.max(np.abs(cal.unit_from_spec(spec) - u)) < 0.01
        spec.build(1).step(1.0)   # FlowExtensions validation passes


def test_size_tables_are_monotone_and_floored() -> None:
    raws = [{"trade_size": np.asarray([0.1, 0.5, 3.0, 7.0]), "add_size": np.asarray([2.0, 4.0])}]
    tables = cal.size_tables(raws)
    for values in tables.values():
        assert len(values) == 21 and min(values) >= 1.0 and values == sorted(values)


def _fake_runner(tasks, context):
    # deterministic objective: distance of the unit vector to 0.3, read back through the spec
    out = []
    for key, spec_dict, *_ in tasks:
        from lob.sim_v2 import SimulatorSpec
        u = cal.unit_from_spec(SimulatorSpec(spec_dict["config"], spec_dict["extensions"]))
        value = float(np.nansum((u - 0.3) ** 2))
        out.append({"key": key, "objective": value if not key.endswith("007") else None,
                    "families": [value] * 9, "seed_objectives": [value], "error": None})
    return out


def test_search_is_deterministic_and_improves() -> None:
    kwargs = dict(starts=2, global_draws=10, rounds=(0.2, 0.1), per_round=8, seed=5, base=BASE, tables=TABLES,
                  seeds=[1], seconds=10.0, context={}, target="t", runner=_fake_runner)
    a, b = cal.search(**kwargs), cal.search(**kwargs)
    assert [r["unit"] for r in a] == [r["unit"] for r in b]
    assert len(a) == 2 * (10 + 16)
    for start in (0, 1):
        globals_ = [r["objective"] for r in a if r["start"] == start and r["stage"] == "global" and r["objective"]]
        rounds = [r["objective"] for r in a if r["start"] == start and r["stage"] != "global" and r["objective"]]
        assert min(rounds) <= min(globals_)
    assert any(r["objective"] is None for r in a)   # failures are retained, not dropped


def test_near_optimal_and_distinct_rules() -> None:
    scores = {"a": 1.0, "b": 1.05, "c": 1.2, "d": 1.09}
    assert cal.near_optimal(scores, "a", 0.0) == ["a", "b", "d"]
    assert cal.near_optimal(scores, "a", 0.1) == ["a", "b", "d", "c"]
    units = {"a": np.zeros(3), "b": np.full(3, 0.1), "d": np.asarray([0.0, 0.0, 0.3]), "c": np.ones(3)}
    assert cal.distinct_set(["a", "b", "d", "c"], units) == ["a", "d", "c"]
    assert cal.distinct_set(["a", "b", "d", "c"], units, limit=2) == ["a", "d"]
    assert cal.distinct_set(["a", "b"], units) == ["a"]


def test_pareto_front() -> None:
    vectors = {"a": np.asarray([1.0, 2.0]), "b": np.asarray([2.0, 1.0]), "c": np.asarray([2.0, 2.0]),
               "d": np.asarray([1.0, math.nan])}
    assert cal.pareto_front(vectors) == ["a", "b"]


def test_near_optimal_structure() -> None:
    rng = np.random.default_rng(2)
    units = {str(i): rng.random(14) for i in range(6)}
    structure = ident.near_optimal_structure(units)
    assert structure["members"] == 6 and len(structure["principal_variances"]) == 14
    assert structure["principal_variances"] == sorted(structure["principal_variances"], reverse=True)
    assert ident.near_optimal_structure({})["members"] == 0
    assert ident.near_optimal_structure({"a": np.zeros(14)})["correlations"] is None


def test_sensitivity_and_profile_bookkeeping() -> None:
    selected = np.full(14, 0.5)
    tasks = ident.sensitivity_tasks(selected, step=0.1, base=BASE, tables=TABLES, seeds=[1], seconds=1.0, target="t")
    assert len(tasks) == 1 + 2 * 14
    records = {key: {"families": [float(i)] * 9} for i, (key, *_) in enumerate(tasks)}
    matrix = ident.sensitivity_matrix(records, selected, step=0.1)
    assert np.asarray(matrix["matrix"]).shape == (9, 14)
    assert matrix["matrix"][0][0] == pytest.approx((1.0 - 2.0) / 0.2)
    profile = ident.profile_tasks(selected, draws=3, step=0.1, seed=1, base=BASE, tables=TABLES, seeds=[1],
                                  seconds=1.0, target="t")
    assert len(profile) == 14 * 5 * 3
    records = {key: {"objective": 1.0 + (0.5 if key.startswith("prof-limit_rate-4") else 0.0)} for key, *_ in profile}
    curves = ident.profiles(records, draws=3, noise_se=0.1)
    assert curves["limit_rate"]["range"] == pytest.approx(0.5) and not curves["limit_rate"]["flat_within_noise"]
    assert curves["market_rate"]["flat_within_noise"]


def test_morris_layout_stays_in_unit_box() -> None:
    tasks, layout = ident.morris_tasks(trajectories=3, seed=4, base=BASE, tables=TABLES, seeds=[1], seconds=1.0,
                                       target="t")
    assert len(tasks) == 3 * 15 and all(len(path) == 15 for path in layout)
    records = {key: {"families": [float(len(key))] * 9} for key, *_ in tasks}
    effects = ident.morris_effects(records, layout)
    assert effects["status"] == "EXPLORATORY" and set(effects["effects"]) == set(cal.NAMES)
