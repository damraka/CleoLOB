from dataclasses import replace
import json
import math

import pandas as pd
import pytest

from lob.config import ResearchConfig
from lob.controls import estimate_ac_parameters, positive_control, residual_sensitivity
from lob.engine import SimConfig
from lob.runner import make_baseline


def test_ac_identification_has_physical_units_and_interval_scaling():
    cfg = SimConfig(record_events=False)
    first = estimate_ac_parameters(cfg, [43010, 43011], horizon=3, fractions=(.1, .25, .5))
    second = estimate_ac_parameters(cfg, [43010, 43011], horizon=3, execution_interval=2,
                                    fractions=(.1, .25, .5))
    assert first["temp_impact"] > 0
    assert first["sigma"] >= 0
    assert second["temp_impact"] == pytest.approx(first["temp_impact"] * 2)
    assert second["sigma"] == first["sigma"]
    assert first["median_top5_depth"] > 0
    assert first["probe_coverage"] == 1
    assert first["impact_units"] == "currency * seconds / quantity_unit"


def test_ac_fit_rejects_reused_component_streams():
    with pytest.raises(ValueError, match="component seed"):
        estimate_ac_parameters(replace(SimConfig(), market_seed=2), [1, 2])


def test_ac_strict_config_reaches_real_baseline():
    cfg = ResearchConfig(execution={"temp_impact": .002, "sigma": .4, "risk_aversion": .1,
                                   "warmup_seconds": 30.})
    params = cfg.runner_params(cfg.evaluation.seeds[0])
    ac = make_baseline("ac", params, cfg.market.engine_config(1), 5.)
    assert ac.kappa == pytest.approx(math.sqrt(.1 / .002) * .4)
    assert params["warmup_seconds"] == 30.
    with pytest.raises(ValueError):
        ResearchConfig(execution={"temp_impact": 0.})


def test_control_registration_precedes_data_and_confirmation_only_once(tmp_path, monkeypatch):
    path = tmp_path / "control"
    calls = []

    def fake_run(agent, params, **_kwargs):
        assert (path / "plan.json").exists()
        calls.append((agent, params["seed"], params["qty"]))
        # Diagnostic first cell passes, but confirmation fails. Do not search
        # further confirmation cells or silently select the second grid point.
        effect = 2. if params["seed"] < 20 else 0.
        return {"agent": agent, "seed": params["seed"], "status": "VALID",
                "effective_bps": 1 + (effect if agent == "random" else 0.)}

    monkeypatch.setattr("lob.controls.run_episode", fake_run)
    result = positive_control({"dt": 1., "sim": {}}, [10, 11], [20, 21], path,
                              median_top5_depth=100, grid=((2., 30.), (5., 60.)), n_boot=30)
    assert result["status"] == "FAIL"
    assert result["selected_params"]["qty"] == 200
    assert len(calls) == 18
    assert all(qty == 200 for _, seed, qty in calls if seed >= 20)
    assert json.loads((path / "result.json").read_text())["status"] == "FAIL"


def test_control_seed_blocks_cannot_overlap(tmp_path):
    with pytest.raises(ValueError, match="disjoint"):
        positive_control({}, [10, 11], [11, 12], tmp_path / "bad", median_top5_depth=100)


def _rows():
    rows = []
    for scenario in ("normal", "thin"):
        for seed in (1, 2, 3, 4):
            for agent in ("twap", "random"):
                rows.append({"scenario": scenario, "seed": seed, "agent": agent,
                             "label": agent, "status": "VALID", "effective_bps": 5. if agent == "twap" else 6.,
                             "arrival": 100., "fill_frac": 1., "gross_cost": 2., "total_fees": 1.,
                             "invalid_reasons": []})
    return rows


def test_invalid_sensitivity_preserves_raw_counts_and_unaffected_comparison():
    rows = _rows()
    rows[-1].update(status="INVALID", effective_bps=None, fill_frac=.5,
                    invalid_reasons=["insufficient_terminal_depth"])
    frame = pd.DataFrame(rows)
    result = residual_sensitivity(frame, quantity=100, haircuts_bps=(100.,), n_boot=40)
    raw, proxy = result["arms"]
    assert raw["comparisons"][0]["status"] == "AVAILABLE"
    assert raw["comparisons"][1]["status"] == "WITHHELD_FOR_THIS_COMPARISON"
    assert raw["comparisons"][0]["family_size"] == 2
    assert proxy["imputed_rows"] == 1
    assert proxy["comparisons"][1]["delta_mean_bps"] == pytest.approx((1 + 1 + 1 + 48) / 4)
    assert result["raw_status_counts"] == {"VALID": 15, "INVALID": 1}
    assert frame.iloc[-1].status == "INVALID"
    assert pd.isna(frame.iloc[-1].effective_bps)


def test_unsettled_orders_never_receive_synthetic_prices():
    rows = _rows()
    rows[-1].update(status="INVALID", effective_bps=None, fill_frac=.5,
                    invalid_reasons=["unsettled_strategy_orders"])
    result = residual_sensitivity(pd.DataFrame(rows), quantity=100, n_boot=40)
    assert all(arm["comparisons"][0]["status"] == "AVAILABLE" for arm in result["arms"])
    assert all(arm["comparisons"][1]["status"] == "WITHHELD_FOR_THIS_COMPARISON" for arm in result["arms"])
    assert all(arm["imputed_rows"] == 0 for arm in result["arms"])
