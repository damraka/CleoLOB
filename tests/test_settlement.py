"""Post-decision exchange settlement reconciles late fills without new actions."""
import math

import pytest

from lob.accounting import Ledger
from lob.engine import ExchangeSimulator, OrderStatus, Side, SimConfig
from lob.execution import build_report
from lob.risk import ExecutionRisk
from lob.rl_env import LOBExecutionEnv
from lob.runner import BASELINES, run_episode
from lob.scenarios import scenario_params
from lob.settlement import settle_orders, validate_settlement


def quiet(**overrides):
    return SimConfig(**{"limit_rate": 0, "market_rate": 0, "cancel_rate": 0,
                        "resilience": 0, "latency_base": 0.3, "latency_jitter": 0,
                        **overrides})


def test_cancel_before_arrival_retries_until_passive_order_is_terminal():
    sim = ExchangeSimulator(quiet())
    ledger = Ledger(inventory=100, average_cost=100)
    risk = ExecutionRisk(100, Side.SELL)
    latencies = iter([0.2, 0.01, 0.01, 0.01, 0.01, 0.01])
    sim._latency = lambda: next(latencies)
    oid = risk.submit(sim, "TEST", 100, 0, ledger, price_ticks=10100)
    polls = []

    def poll():
        polls.append(sim.t)
        assert risk.available(sim, 0) in (0, 100)

    result = settle_orders(sim, risk, poll, timeout=0.5, poll_dt=0.05)
    assert result.complete and result.pending_order_ids == ()
    assert sim.orders[oid].status is OrderStatus.CANCELLED
    cancels = [event for event in sim.events if event["kind"] == "cancel"]
    assert any(not event["accepted"] for event in cancels)
    assert cancels[-1]["accepted"]
    assert risk.outstanding(sim) == 0 and risk.available(sim, 0) == 100
    assert polls[-1] == result.ended_at and result.duration <= 0.5


def test_rl_late_taker_fill_is_accounted_once_and_reward_matches_final_report():
    env = LOBExecutionEnv(total_qty=100, horizon=0.1, decision_dt=0.1, warmup=0,
                          cfg=quiet(), fees={"taker_bps": 2, "per_share": 0.01}, record=True)
    env.reset(seed=1)
    latencies = iter([0.3, 0.5])
    env.sim._latency = lambda: next(latencies)
    _, reward, terminated, truncated, info = env.step(4)
    report = env.report()
    assert terminated and not truncated and info["settlement_complete"]
    assert report.late_filled_qty == report.filled_qty == 100
    assert report.late_fees == pytest.approx(report.total_fees)
    assert report.hypothetical_liquidation_cost == report.hypothetical_liquidation_fees == 0
    assert reward == pytest.approx(-report.optimization_cost_bps)
    assert info["reward_terms"]["late_fills"] != 0
    assert info["reward_terms"]["late_fees"] < 0
    assert info["reward_terms"]["fills"] == info["reward_terms"]["fees"] == 0
    assert report.duration == report.decision_duration == pytest.approx(0.1)
    assert report.total_duration == pytest.approx(report.duration + report.settlement_duration)
    assert report.settlement_duration > 0
    assert env.ledger.inventory == 0 and report.outstanding_qty == 0
    assert len(env.decisions) == 1 and env.decisions[0]["t"] == 0
    env.sim.step(2)
    fills, _ = env.sim.fills_for(env.OWNER, 0)
    assert sum(tr.qty for tr in fills) == 100
    assert all(env.sim.orders[oid].is_terminal for oid in env.risk.order_ids)


def test_rl_late_partial_maker_fill_rebate_then_cancellation_values_only_remainder():
    env = LOBExecutionEnv(total_qty=100, horizon=0.1, decision_dt=0.1, warmup=0,
                          cfg=quiet(), fees={"maker_bps": -1, "taker_bps": 2})
    env.reset(seed=1)
    latencies = iter([0.2, 0.15, 0.5])
    env.sim._latency = lambda: next(latencies)
    # External liquidity takes the improved child after the decision boundary.
    env.sim.submit(Side.BUY, 40, "EXTERNAL", price_ticks=10000)
    _, reward, terminated, truncated, _ = env.step(2)
    report = env.report()
    assert terminated and not truncated and report.settlement_complete
    assert report.filled_qty == report.late_filled_qty == 40
    assert report.leftover_qty == report.fillable_leftover_qty == 60
    assert report.late_fees == report.total_fees < 0
    assert report.hypothetical_liquidation_fees > 0
    assert report.outstanding_qty == 0 and env.ledger.inventory == 60
    assert reward == pytest.approx(-report.optimization_cost_bps)


def test_timeout_truncates_rl_and_withholds_final_economic_metrics():
    env = LOBExecutionEnv(total_qty=100, horizon=0.1, decision_dt=0.1, warmup=0,
                          cfg=quiet(latency_base=2), settlement_timeout=0.2,
                          settlement_poll_dt=0.03)
    env.reset(seed=1)
    _, reward, terminated, truncated, info = env.step(4)
    report = env.report()
    assert not terminated and truncated and not info["settlement_complete"]
    assert report.status == "INVALID" and "unsettled_orders" in report.invalid_reasons
    assert report.effective_shortfall_bps is report.net_effective_bps is report.gross_effective_bps is None
    assert report.settlement_duration == pytest.approx(0.2)
    assert report.outstanding_qty == report.unpriced_leftover_qty == 100
    assert report.settlement_pending_order_ids
    assert report.hypothetical_liquidation_cost == report.hypothetical_liquidation_fees == 0
    assert math.isfinite(reward) and reward == pytest.approx(-report.optimization_cost_bps)
    with pytest.raises(RuntimeError, match="episode ended"):
        env.step(4)


def test_zero_timeout_still_accepts_zero_latency_cancel():
    sim = ExchangeSimulator(quiet(latency_base=0))
    risk = ExecutionRisk(100, Side.SELL)
    risk.submit(sim, "TEST", 100, 0, Ledger(inventory=100, average_cost=100), price_ticks=10100)
    sim.step(0)
    result = settle_orders(sim, risk, lambda: None, timeout=0)
    assert result.complete and result.duration == 0 and risk.outstanding(sim) == 0


def test_report_never_presents_outstanding_orders_as_final_liquidation():
    report = build_report("test", Side.SELL, 100, [], 10000, 0.01, 1, 0.1,
                          9999, leftover_fillable_qty=100, outstanding_qty=100)
    assert report.status == "INVALID" and "unsettled_orders" in report.invalid_reasons
    assert report.net_effective_bps is None and report.hypothetical_liquidation_cost == 0


def test_fills_are_polled_even_if_exchange_fails_after_a_processed_prefix(monkeypatch):
    sim = ExchangeSimulator(quiet())
    risk = ExecutionRisk(100, Side.SELL)
    risk.submit(sim, "TEST", 100, 0, Ledger(inventory=100, average_cost=100))
    polled = []
    original = sim.step

    def failed_step(dt):
        original(dt)
        raise RuntimeError("exchange event budget")

    monkeypatch.setattr(sim, "step", failed_step)
    with pytest.raises(RuntimeError, match="event budget"):
        settle_orders(sim, risk, lambda: polled.append(sim.t))
    assert len(polled) == 2  # Initial reconciliation and the failed step's prefix.


@pytest.mark.parametrize("agent", BASELINES + ("heuristic", "random"))
def test_runner_has_no_working_strategy_orders_after_successful_settlement(agent):
    params = scenario_params("calm", seed=4, qty=100, horizon=0.35, dt=0.1,
                             record_audit=True, sim=quiet().__dict__,
                             fees={"taker_bps": 2})
    row = run_episode(agent, params)
    assert row["settlement_complete"] and row["outstanding_qty"] == 0
    terminal = {"filled", "cancelled", "rejected", "expired"}
    assert all(order["status"].lower() in terminal for order in row["audit"]["orders"])
    assert row["total_duration"] == pytest.approx(row["duration"] + row["settlement_duration"])
    assert row["decision_duration"] <= row["decision_horizon"] + 1e-12
    assert all(order["submitted_at"] < row["decision_horizon"] for order in row["audit"]["orders"])


def test_runner_baseline_timeout_preserves_invalid_outcome():
    row = run_episode("twap", scenario_params("calm", seed=1, qty=100, horizon=0.1, dt=0.1,
                      sim=quiet(latency_base=2).__dict__, settlement_timeout=0,
                      record_audit=True))
    assert row["status"] == "INVALID" and row["effective_bps"] is None
    assert "unsettled_orders" in row["invalid_reasons"]
    assert row["settlement_pending_order_ids"] and row["outstanding_qty"] > 0


def test_policy_series_uses_actual_post_settlement_clock():
    from lob.runner import run_policy
    result = run_policy(scenario_params("calm", seed=4, qty=100, horizon=0.35, dt=0.1,
                         sim=quiet(latency_base=.2).__dict__), policy="heuristic", record=True)
    assert result["series"]["t"][-1] == pytest.approx(result["raw"]["total_duration"], abs=.00051)


@pytest.mark.parametrize("timeout,poll_dt", [(-1, 0.1), (1, 0), (math.inf, 0.1),
                                             (1, math.nan), (True, 0.1), (1, False),
                                             (5, 1e-10), ("5", 0.1), (1, None)])
def test_invalid_or_unbounded_settlement_options_fail(timeout, poll_dt):
    with pytest.raises(ValueError, match="settlement"):
        validate_settlement(timeout, poll_dt)
