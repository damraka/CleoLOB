"""v0.7 purged folds, accounting v3, horizon semantics, schema versioning and configuration validation."""
from __future__ import annotations

import json
from pathlib import Path

from hypothesis import given, settings, strategies as st
import pytest

from lob.accounting import FeeConfig
from lob.engine import Side
from lob.sim_v2 import SimulatorSpec
from lob.v07.execution import accounting as acc
from lob.v07.execution import episode
from lob.v07.protocol import schemas
from lob.v07.realism.folds import purged_folds

ROOT = Path(__file__).resolve().parents[1]
VALID = {"venue": "deribit", "instrument": "ETH-PERPETUAL", "queue_model": "conservative",
         "fees": {"maker_bps": 0.0, "taker_bps": 1.0}, "horizon_s": 120.0, "decision_dt_s": 6.0, "latency_s": 0.01}


def test_purged_folds_never_leak() -> None:
    for train, test in purged_folds(200, 5, purge=10, embargo=7):
        assert not set(train) & set(range(test.min() - 10, test.max() + 8))
    with pytest.raises(ValueError):
        purged_folds(5, 4)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.tuples(st.floats(0, 200), st.floats(90, 110), st.integers(1, 20), st.booleans()), max_size=20),
       st.floats(-1.0, 0.0), st.sampled_from([Side.BUY, Side.SELL]))
def test_accounting_invariants(fills, maker_bps, side) -> None:
    target = max(1, sum(f[2] for f in fills))
    s = acc.statement(fills, side=side, target=target, arrival=100.0, mark=101.0, horizon_end=120.0,
                      fees=FeeConfig(maker_bps=maker_bps, taker_bps=1.0))
    assert s["within_horizon_filled"] + s["post_horizon_filled"] + s["residual_inventory"] == target
    assert s["fees"] >= 0 and s["rebates"] >= 0
    assert s["inventory"] == side.value * (target - s["residual_inventory"])
    assert s["pnl"] == pytest.approx(s["equity"], abs=1e-6)        # initial equity is zero


def test_execution_rows_keep_horizon_semantics() -> None:
    world = SimulatorSpec({"tick_size": 100 / 2640, "initial_mid_ticks": 2640, "target_level_vol": 26,
                           "limit_rate": 2.9, "market_rate": 0.14, "cancel_rate": 0.03, "offset_p": 0.05,
                           "resilience": 0.2, "max_events": 3_000_000}, {})
    row = episode.run(world, "twap", {**episode.MANDATE, "horizon_s": 30.0, "warmup_s": 10.0}, 3)
    acc.check_horizon_fields(row)
    with pytest.raises(ValueError):
        acc.check_horizon_fields({"mandate_within_horizon_filled_qty": 1})


def test_schema_registry_and_read_only_v06_migration() -> None:
    for name in ("protocol", "hypotheses", "statistical-families", "result-taxonomy", "power-design"):
        doc = json.loads((ROOT / f"configs/v07/{name}.json").read_text(encoding="utf-8"))
        assert schemas.schema_of(doc)
    with pytest.raises(schemas.ConfigError):
        schemas.schema_of({"schema": "unknown"})
    run = ROOT / "results/v06/m6/select"
    if run.is_dir():
        view = schemas.read_v06_run(run)
        assert view["source_schema"].startswith("cleolob-v06") and "read-only" in view["migration"]


@pytest.mark.parametrize("change,message", [
    ({"queue_model": "exact_fifo"}, "L2 + exact FIFO"),
    ({"venue": "binance", "instrument": "BTCUSDT"}, "unsupported venue rule"),
    ({"fees": None}, "missing fee schedule"),
    ({"latency_s": 500.0}, "impossible latency"),
    ({"decision_dt_s": 300.0}, "illegal horizon"),
    ({"queue_model": "teleport"}, "incompatible queue model"),
    ({"requires": ["hidden_order_quantity"]}, "impossible data capability"),
])
def test_invalid_config_suite(change, message) -> None:
    problems = schemas.validate_study_config({**VALID, **change})
    assert any(message in p for p in problems), problems
    with pytest.raises(schemas.ConfigError):
        schemas.require_valid({**VALID, **change})


def test_valid_config_passes() -> None:
    assert schemas.validate_study_config(VALID) == []
    assert schemas.validate_study_config({**VALID, "venue": "bitstamp", "instrument": "btcusd",
                                          "requires": ["order_identity"]}) == []
