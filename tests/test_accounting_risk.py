"""Accounting identities and latency-race execution-budget regressions."""
import math

import pytest

from lob.accounting import FeeConfig, Ledger
from lob.engine import ExchangeSimulator, Side, SimConfig, Trade
from lob.execution import AlmgrenChrissAgent, TWAPAgent, VWAPAgent, build_report
from lob.risk import ExecutionRisk, RiskConfig


def quiet_config(**kwargs):
    return SimConfig(**dict(dict(limit_rate=0, market_rate=0, cancel_rate=0,
                                  resilience=0, latency_base=0.2, latency_jitter=0), **kwargs))


def test_roundtrip_cash_inventory_realized_unrealized_and_fee_identity():
    ledger = Ledger(cash=10_000, fees=FeeConfig(taker_bps=10, per_share=0.01))
    ledger.apply_fill(Side.BUY, 100, 10)
    ledger.apply_fill(Side.BUY, 120, 10)
    assert ledger.average_cost == 110
    ledger.apply_fill(Side.SELL, 130, 15)
    mark = ledger.snapshot(125)
    assert mark.inventory == 5
    assert mark.gross_realized_pnl == 300
    assert mark.unrealized_pnl == 75
    assert mark.total_fees == pytest.approx(4.5)
    assert mark.total_pnl == pytest.approx(370.5)
    assert mark.equity == pytest.approx(10_370.5)
    assert mark.total_pnl == pytest.approx(mark.realized_pnl + mark.unrealized_pnl)


def test_short_position_crossing_and_initial_endowment_are_not_profit():
    ledger = Ledger(inventory=10, average_cost=100)
    assert ledger.snapshot(100).total_pnl == 0
    ledger.apply_fill(Side.SELL, 120, 15)
    assert ledger.inventory == -5 and ledger.average_cost == 120
    assert ledger.snapshot(110).total_pnl == 250
    ledger.apply_fill(Side.BUY, 100, 8)
    state = ledger.snapshot(105)
    assert state.inventory == 3 and state.average_cost == 100
    assert state.gross_realized_pnl == 300 and state.unrealized_pnl == 15


def test_owner_side_and_maker_rebate_are_inferred_from_trade():
    ledger = Ledger(fees=FeeConfig(maker_bps=-1, taker_bps=2))
    trade = Trade(0, 10_000, 10, Side.SELL, "seller", "buyer")
    ledger.apply_trade(trade, "buyer", 0.01)
    assert ledger.inventory == 10
    assert ledger.total_fees == pytest.approx(-0.1)
    assert ledger.cash == pytest.approx(-999.9)
    with pytest.raises(ValueError):
        ledger.apply_trade(trade, "unrelated", 0.01)


@pytest.mark.parametrize("config", [lambda: FeeConfig(maker_bps=math.nan),
                                     lambda: FeeConfig(per_share=-1),
                                     lambda: RiskConfig(max_order_qty=1.5),
                                     lambda: RiskConfig(max_loss=-1),
                                     lambda: RiskConfig(kill_switch="false")])
def test_invalid_fee_and_risk_configs_fail(config):
    with pytest.raises(ValueError):
        config()


def test_multiple_due_schedule_slices_reserve_inflight_quantity():
    sim = ExchangeSimulator(quiet_config())
    agent = TWAPAgent(1000, 1, n_slices=20)
    sim.step(0.95)
    agent.on_step(sim, "agent")  # all 20 release times due; no fills yet
    assert agent.filled == 0
    assert agent.risk.outstanding(sim) == 1000
    assert agent.available(sim) == 0
    sim.step(0.3)
    agent.poll_fills(sim, "agent")
    assert agent.filled == 1000
    assert agent.risk.outstanding(sim) == 0


def test_cancel_does_not_release_parent_reservation_until_acknowledgement():
    sim = ExchangeSimulator(quiet_config())
    ledger = Ledger(inventory=100, average_cost=100)
    risk = ExecutionRisk(100, Side.SELL)
    oid = risk.submit(sim, "agent", 100, 0, ledger, price_ticks=20_000)
    sim.step(0.2)
    risk.cancel_all(sim)
    assert risk.outstanding(sim) == 100
    assert risk.submit(sim, "agent", 100, 0, ledger) is None
    sim.step(0.2)
    assert sim.orders[oid].is_terminal
    assert risk.outstanding(sim) == 0
    assert risk.submit(sim, "agent", 100, 0, ledger) is not None


def test_market_ioc_unfilled_leaves_release_reservation():
    sim = ExchangeSimulator(quiet_config(target_level_vol=1))
    ledger = Ledger(inventory=100, average_cost=100)
    risk = ExecutionRisk(100, Side.SELL)
    risk.submit(sim, "agent", 100, 0, ledger)
    sim.step(0.3)
    fills, _ = sim.fills_for("agent", 0)
    filled = sum(t.qty for t in fills)
    assert filled == 20
    assert risk.outstanding(sim) == 0 and risk.available(sim, filled) == 80


def test_risk_order_position_and_notional_caps_include_working_orders():
    sim = ExchangeSimulator(quiet_config())
    ledger = Ledger()
    risk = ExecutionRisk(100, Side.BUY, RiskConfig(max_order_qty=30, max_position=40,
                                                 max_order_notional=2050))
    one = risk.submit(sim, "agent", 100, 0, ledger, price_ticks=10_000)
    two = risk.submit(sim, "agent", 100, 0, ledger, price_ticks=10_000)
    assert sim.orders[one].qty == 20 and sim.orders[two].qty == 20
    assert risk.submit(sim, "agent", 100, 0, ledger, price_ticks=10_000) is None


def test_kill_switch_requests_cancel_and_retains_inflight_reservation():
    sim = ExchangeSimulator(quiet_config())
    ledger = Ledger(inventory=100, average_cost=100)
    risk = ExecutionRisk(100, Side.SELL, RiskConfig(max_loss=10))
    risk.submit(sim, "agent", 50, 0, ledger, price_ticks=20_000)
    ledger.cash -= 20  # an exogenous loss must be represented consistently
    ledger.gross_realized_pnl -= 20
    assert risk.submit(sim, "agent", 50, 0, ledger) is None
    assert risk.halted
    assert risk.outstanding(sim) == 50
    assert any(event["event"] == "kill_switch" for event in risk.events)


def test_partial_terminal_depth_is_unpriced_not_extrapolated():
    report = build_report("test", Side.SELL, 100, [], 10_000, 0.01, 0, 1,
                          9900, leftover_fillable_qty=20, strict_terminal=True,
                          fees=FeeConfig(taker_bps=1), terminal_penalty_bps=25)
    assert report.total_cost == 0 and report.total_fees == 0
    assert report.hypothetical_liquidation_cost == 20
    assert report.hypothetical_liquidation_fees == pytest.approx(0.198)
    assert report.unpriced_leftover_qty == 80
    assert report.status == "INVALID"
    assert report.effective_shortfall_bps is None and report.net_effective_bps is None
    assert report.optimization_cost_bps == pytest.approx(45.198)


def test_actual_costs_and_terminal_estimates_are_separate_and_side_symmetric():
    for side, price, quote in [(Side.SELL, 9990, 9980), (Side.BUY, 10_010, 10_020)]:
        fill = Trade(0, price, 50, side, "agent", "ZI")
        report = build_report("test", side, 100, [fill], 10_000, 0.01, 1, 1,
                              quote, leftover_fillable_qty=50, strict_terminal=True,
                              total_fees=1, fees=FeeConfig(), terminal_penalty_bps=25)
        assert report.gross_cost == pytest.approx(5)
        assert report.total_cost == pytest.approx(6)
        assert report.hypothetical_liquidation_cost == pytest.approx(10)
        assert report.gross_effective_bps == pytest.approx(15)
        assert report.net_effective_bps == pytest.approx(16)
        assert report.optimization_cost_bps == pytest.approx(28.5)


def test_ac_large_kappa_stays_finite_and_frontloads():
    agent = AlmgrenChrissAgent(1000, 60, risk_aversion=1e10)
    assert agent.targets[0] == 1000 and agent.targets[1:] == [0] * 20


@pytest.mark.parametrize("profile", [[-1, 2], [0, 0], [math.nan, 1], [math.inf, 1]])
def test_invalid_vwap_profile_rejected(profile):
    with pytest.raises(ValueError):
        VWAPAgent(100, 1, n_slices=2, volume_profile=profile)
