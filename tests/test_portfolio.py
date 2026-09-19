"""Portfolio fill reconciliation, latency reservations, FX and empirical tail risk."""
from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
import json

import pytest

from lob.portfolio import (
    DAY_NS, InstrumentSpec, MarketDataError, Portfolio, PortfolioAction, PortfolioLimits,
    PortfolioScenario, load_portfolio_config, portfolio_config, run_portfolio_demo, scenario_report,
)


D = Decimal


def funded(*, limits=None, positions=None, cash=None, multiplier='1', margin='0.25', dual=False):
    instruments = [InstrumentSpec('A', 'USD', multiplier=multiplier, margin_rate=margin)]
    if dual:
        instruments.append(InstrumentSpec('B', 'EUR', margin_rate=margin))
    portfolio = Portfolio(base_currency='USD', instruments=instruments, limits=limits,
                          initial_cash={'USD': '10000'} if cash is None else cash,
                          initial_positions=positions)
    portfolio.mark('A', '100', 0)
    if dual:
        portfolio.mark('B', '50', 0)
        portfolio.fx('EUR', '2', 0)
    portfolio.snapshot(0)
    return portfolio


def submit(portfolio, oid, side, qty, *, symbol='A', price='100', fee='0', now=0, reduce_only=False):
    return portfolio.submit(oid, symbol, side, qty, price, fee, now, reduce_only=reduce_only)


def scenario(name, a, *, b=None, eur=None):
    symbols = (('A', a),) if b is None else (('A', a), ('B', b))
    fx = () if eur is None else (('EUR', eur),)
    return PortfolioScenario(name, symbols, fx)


def test_decimal_fills_preserve_exact_cash_and_fees_even_with_low_ambient_precision():
    with localcontext() as context:
        context.prec = 5
        p = funded(multiplier='0.001')
        assert submit(p, 'b', 'buy', 3, fee='0.001')['accepted']
        p.fill('f1', 'b', 1, '99.99', '0.0001', 1)
        p.fill('f2', 'b', 2, '99.98', '0.0002', 2)
        assert p.cash['USD'] == D('9999.69975')
        state = p.snapshot(2)
        assert state['fees_by_currency']['USD'] == '0.0003'
        assert D(state['equity']) == D('9999.99975')
        assert D(state['total_pnl']) == D('-0.00025')
        assert p.positions['A'] == 3
        assert p.orders['b'].status == 'filled'
        assert p.orders['b'].fee_remaining == 0
        p.reconcile()


def test_signed_short_crossing_and_foreign_cash_reconcile():
    p = funded(positions={'A': -2, 'B': 2}, cash={'USD': '10000', 'EUR': '5'}, dual=True)
    assert submit(p, 'cover', 'buy', 3, fee='1')['accepted']
    p.fill('cover-fill', 'cover', 3, '90', '1', 0)
    assert p.positions['A'] == 1 and p.cash['USD'] == D('9729')
    assert submit(p, 'foreign', 'sell', 4, symbol='B', price='50', fee='0.2')['accepted']
    p.fill('foreign-fill', 'foreign', 4, '51', '0.2', 0)
    assert p.positions['B'] == -2 and p.cash['EUR'] == D('208.8')
    snap = p.snapshot(0)
    assert D(snap['equity']) == D('10046.6')
    assert D(snap['total_pnl']) == D('36.6')
    assert snap['fees_by_currency'] == {'USD': '1', 'EUR': '0.2'}


def test_exact_cash_reconciliation_across_maximum_allowed_price_multiplier_scales():
    p = Portfolio(base_currency='USD', instruments=[
        InstrumentSpec('LARGE', 'USD', multiplier='1e18', margin_rate=0),
        InstrumentSpec('SMALL', 'USD', multiplier='1e-18', tick_size='1e-18', margin_rate=0)],
        initial_cash={'USD': 0}, initial_positions={'LARGE': 10**9})
    p.mark('LARGE', '1e18', 0)
    p.mark('SMALL', '1e-18', 0)
    p.snapshot(0)
    assert p.submit('reduce', 'LARGE', 'sell', 10**9, '1e18', 0, 0, reduce_only=True)['accepted']
    p.fill('large-sale', 'reduce', 10**9, '1e18', 0, 0)
    assert p.submit('tiny', 'SMALL', 'buy', 1, '1e-18', 0, 0)['accepted']
    p.fill('tiny-buy', 'tiny', 1, '1e-18', 0, 0)
    with localcontext() as context:
        context.prec = 192
        assert p.cash['USD'] == D('1e45') - D('1e-36')
    p.reconcile()


def test_cancel_pending_keeps_leaves_and_fee_reservations_and_accepts_inflight_fill():
    p = funded(limits=PortfolioLimits(max_gross='1000'))
    assert submit(p, 'old', 'buy', 10, fee='5')['accepted']
    p.cancel_request('old', 1)
    assert p.orders['old'].remaining == 10
    assert not submit(p, 'new', 'buy', 1, now=1)['accepted']
    p.fill('late', 'old', 4, '100', '1', 2)
    assert p.orders['old'].status == 'cancel_pending'
    assert p.orders['old'].remaining == 6
    assert p.orders['old'].fee_remaining == 4
    assert D(p.snapshot(2)['worst_gross']) == 1000
    p.acknowledge_terminal('old', 'cancelled', 3)
    assert p.orders['old'].remaining == 0 and p.orders['old'].fee_remaining == 0
    assert submit(p, 'after', 'buy', 6, now=3)['accepted']
    with pytest.raises(ValueError, match='active order'):
        p.fill('after-ack', 'old', 1, '100', '0', 4)


def test_same_symbol_opposite_orders_reserve_independent_position_interval():
    p = funded(limits=PortfolioLimits(max_gross='1000', max_abs_net='1000'))
    assert submit(p, 'b', 'buy', 10)['accepted']
    assert submit(p, 's', 'sell', 10)['accepted']
    state = p.snapshot(0)
    assert state['position_intervals']['A'] == [-10, 10]
    assert D(state['worst_gross']) == 1000 and D(state['worst_abs_net']) == 1000
    assert not submit(p, 'b2', 'buy', 1)['accepted']


def test_opposite_instruments_cannot_net_away_gross_or_reservation_net_risk():
    p = funded(dual=True, limits=PortfolioLimits(max_gross='1500'))
    assert submit(p, 'a', 'buy', 10)['accepted']
    result = submit(p, 'b', 'sell', 10, symbol='B', price='50')
    assert not result['accepted'] and 'gross' in result['reasons']
    q = funded(dual=True, limits=PortfolioLimits(max_abs_net='900'))
    result = submit(q, 'a', 'buy', 10)
    assert not result['accepted'] and 'net' in result['reasons']


def test_margin_uses_fx_multiplier_and_conservative_fee_slippage_equity():
    p = funded(cash={'USD': '100'}, multiplier='2', margin='0.5')
    result = submit(p, 'margin', 'buy', 1, price='101', fee='1')
    assert not result['accepted'] and 'margin' in result['reasons']
    q = funded(dual=True, cash={'USD': '100'}, margin='1')
    result = submit(q, 'fxmargin', 'buy', 2, symbol='B', price='50')
    assert not result['accepted'] and 'margin' in result['reasons']


def test_leverage_uses_equity_after_reserved_adverse_slippage_and_fees():
    p = funded(cash={'USD': '100'}, limits=PortfolioLimits(max_leverage='1'))
    result = submit(p, 'expensive', 'buy', 1, price='110', fee='1')
    assert not result['accepted'] and 'leverage' in result['reasons']
    assert submit(p, 'bounded', 'buy', 1)['accepted']


def test_net_exposure_signed_positions_and_local_to_base_conversion():
    p = funded(positions={'A': 10, 'B': -10}, dual=True)
    state = p.snapshot(0)
    assert D(state['gross']) == 2000 and D(state['net']) == 0
    assert state['exposures'] == {'A': '1000', 'B': '-1000'}
    assert D(state['worst_margin']) == 500


def test_concentration_allows_held_hedge_but_not_assumed_pending_hedge():
    limits = PortfolioLimits(max_concentration='0.6')
    p = funded(positions={'A': 10, 'B': -10}, dual=True, limits=limits)
    assert submit(p, 'ok', 'buy', 5)['accepted']
    result = submit(p, 'too_big', 'buy', 1)
    assert not result['accepted'] and 'concentration' in result['reasons']
    q = funded(dual=True, limits=limits)
    result = submit(q, 'no_guaranteed_hedge', 'buy', 1)
    assert not result['accepted'] and 'concentration' in result['reasons']


def test_instrument_notional_and_order_limits_include_multiplier_fx():
    p = funded(dual=True, limits=PortfolioLimits(max_symbol_notional='150', max_order_notional='120'))
    assert submit(p, 'b1', 'buy', 1, symbol='B', price='50')['accepted']
    result = submit(p, 'b2', 'buy', 1, symbol='B', price='50')
    assert 'symbol_notional' in result['reasons']
    result = submit(p, 'b3', 'buy', 2, symbol='B', price='50')
    assert 'order_notional' in result['reasons']


def test_loss_kill_latches_after_recovery_and_across_day_reset():
    p = funded(positions={'A': 10}, limits=PortfolioLimits(max_daily_loss='50', max_drawdown='80'))
    p.mark('A', '94', 1)
    state = p.snapshot(1)
    assert state['halted'] and state['halt_reasons'] == ['daily_loss']
    p.mark('A', '90', 2)
    assert p.snapshot(2)['halt_reasons'] == ['daily_loss', 'drawdown']
    p.mark('A', '100', DAY_NS)
    state = p.snapshot(DAY_NS)
    assert state['halted'] and D(state['daily_loss']) == 0
    assert not submit(p, 'increase', 'buy', 1, now=DAY_NS)['accepted']
    assert submit(p, 'reduce', 'sell', 5, now=DAY_NS, reduce_only=True)['accepted']


def test_reduce_only_cannot_overshoot_position_with_pending_same_side_cancels():
    p = funded(positions={'A': 10})
    assert submit(p, 's', 'sell', 8, reduce_only=True)['accepted']
    p.cancel_request('s', 0)
    result = submit(p, 'overshoot', 'sell', 3, reduce_only=True)
    assert not result['accepted'] and 'not_robustly_reducing' in result['reasons']
    assert submit(p, 'safe', 'sell', 2, reduce_only=True)['accepted']
    p.fill('s-fill', 's', 8, '100', '0', 0)
    p.fill('safe-fill', 'safe', 2, '100', '0', 0)
    assert p.positions['A'] == 0


def test_reduce_only_short_cover_accounts_for_prior_buy_reservations():
    p = funded(positions={'A': -10})
    assert submit(p, 'b', 'buy', 7, reduce_only=True)['accepted']
    assert not submit(p, 'bad', 'buy', 4, reduce_only=True)['accepted']
    assert submit(p, 'good', 'buy', 3, reduce_only=True)['accepted']


def test_risk_reduction_can_preserve_existing_gross_breach_but_cannot_remove_required_net_hedge():
    p = funded(positions={'A': 20}, limits=PortfolioLimits(max_gross='1000'))
    assert 'gross' in p.snapshot(0)['limit_violations']
    assert submit(p, 'reduce', 'sell', 10, reduce_only=True)['accepted']
    q = funded(positions={'A': 10, 'B': -10}, dual=True, limits=PortfolioLimits(max_abs_net='100'))
    result = submit(q, 'unsafe_unhedge', 'sell', 5, reduce_only=True)
    assert not result['accepted'] and 'net' in result['reasons']


def test_nonpositive_equity_blocks_orders_and_never_serializes_infinity():
    p = funded(cash={'USD': '0'})
    state = p.snapshot(0)
    assert state['worst_leverage'] is None
    assert state['limit_violations'] == ['nonpositive_equity']
    assert not submit(p, 'bad', 'buy', 1)['accepted']
    json.dumps(state, allow_nan=False)


def test_stale_marks_and_fx_fail_closed_without_mutating_order_reservations():
    limits = PortfolioLimits(max_mark_age_ns=5, max_fx_age_ns=3)
    p = funded(dual=True, limits=limits)
    with pytest.raises(MarketDataError, match='EUR'):
        submit(p, 'stale', 'buy', 1, now=4)
    assert 'stale' not in p.orders
    p.fx('EUR', '2', 4)
    with pytest.raises(MarketDataError, match='A'):
        p.snapshot(6)
    p.mark('A', '100', 6)
    p.mark('B', '50', 6)
    assert submit(p, 'fresh', 'buy', 1, now=6)['accepted']


def test_missing_quotes_future_clock_and_epoch_nanoseconds_validation():
    p = Portfolio(base_currency='USD', instruments=[InstrumentSpec('A', 'USD')], initial_cash={'USD': 1000})
    with pytest.raises(MarketDataError, match='missing'):
        p.snapshot(0)
    epoch = 1_790_000_000_000_000_000
    p.mark('A', 100, epoch)
    with pytest.raises(ValueError, match='monotone'):
        p.snapshot(epoch - 1)
    assert p.snapshot(epoch)['timestamp_ns'] == epoch


@pytest.mark.parametrize('mutation', ['overfill', 'price', 'fee', 'tick', 'duplicate'])
def test_invalid_fill_rejects_atomically(mutation):
    p = funded()
    submit(p, 'o', 'buy', 2, fee='1')
    p.fill('first', 'o', 1, '100', '0.5', 0)
    before = p.snapshot(0)
    args = dict(fill_id='next', order_id='o', quantity=1, price='100', fee='0.5', now_ns=0)
    args.update({'overfill': {'quantity': 2}, 'price': {'price': '101'}, 'fee': {'fee': '0.6'},
                 'tick': {'price': '99.999'}, 'duplicate': {'fill_id': 'first'}}[mutation])
    with pytest.raises(ValueError):
        p.fill(**args)
    assert p.snapshot(0) == before


def test_sell_floor_rejection_and_rebates_do_not_inflate_future_fee_capacity():
    p = funded(positions={'A': 2})
    submit(p, 's', 'sell', 2, fee='1')
    with pytest.raises(ValueError, match='price bound'):
        p.fill('bad', 's', 1, '99', '0', 0)
    p.fill('rebate', 's', 1, '101', '-0.5', 0)
    assert p.cash['USD'] == D('10101.5')
    assert p.orders['s'].fee_remaining == D('1')
    with pytest.raises(ValueError, match='fee'):
        p.fill('badfee', 's', 1, '101', '1.5', 0)


def test_terminal_acknowledgements_require_consistent_lifecycle():
    p = funded()
    submit(p, 'o', 'buy', 2)
    p.fill('f', 'o', 1, '100', '0', 0)
    with pytest.raises(ValueError, match='partially filled'):
        p.acknowledge_terminal('o', 'rejected', 0)
    with pytest.raises(ValueError, match='terminal acknowledgement'):
        p.acknowledge_terminal('o', 'filled', 0)
    p.acknowledge_terminal('o', 'expired', 0)
    with pytest.raises(ValueError, match='active'):
        p.cancel_request('o', 0)
    with pytest.raises(ValueError, match='reused'):
        submit(p, 'o', 'buy', 1)


def test_scenario_pnl_includes_signed_multiplier_positions_fx_cash_and_cross_terms():
    p = funded(positions={'A': 2, 'B': -3}, cash={'USD': '1000', 'EUR': '10'}, dual=True, multiplier='2')
    state = p.snapshot(0)
    assert D(state['equity']) == 1120
    report = scenario_report(p, [scenario('joint', '.1', b='-.2', eur='.25')], now_ns=0)
    # USD cash 1000 + EUR cash 25 + A 440 - B 300 = 1165.
    assert D(report['scenarios'][0]['equity']) == 1165
    assert D(report['scenarios'][0]['pnl']) == 45
    assert report['historical_risk'] is None


def test_pending_orders_are_explicitly_excluded_from_scenario_pnl():
    p = funded()
    assert submit(p, 'pending', 'buy', 10)['accepted']
    report = scenario_report(p, [scenario('down', '-.5')], now_ns=0)
    assert D(report['scenarios'][0]['pnl']) == 0
    assert report['pending_order_count'] == 1
    assert report['pending_orders_in_scenario_pnl'] is False


def test_inverse_empirical_var_and_fractionally_weighted_expected_shortfall():
    p = funded(positions={'A': 1})
    rows = [scenario(str(i), ret) for i, ret in enumerate(['.1', '0', '-.1', '-.2'])]
    report = scenario_report(p, rows, now_ns=0, confidence='.625', historical=True)
    risk = report['historical_risk']
    assert D(risk['var']) == 10
    assert D(risk['tail_observation_mass']) == D('1.5')
    with localcontext() as context:
        context.prec = 192
        assert D(risk['expected_shortfall']) == D(50) / D(3)
    assert risk['statistical_guarantee'] is False
    assert any('five effective' in warning for warning in report['warnings'])


def test_tail_mass_less_than_one_uses_maximum_loss_and_gains_floor_at_zero():
    p = funded(positions={'A': 1})
    rows = [scenario('up', '.1'), scenario('moreup', '.2')]
    risk = scenario_report(p, rows, now_ns=0, confidence='.99', historical=True)['historical_risk']
    assert D(risk['var']) == D(risk['expected_shortfall']) == 0
    assert D(risk['raw_loss_quantile']) == D(risk['raw_tail_mean']) == -10


@pytest.mark.parametrize('rows', [[], [scenario('one', '0')],
                                  [scenario('same', '0'), scenario('same', '.1')]])
def test_historical_tail_inputs_reject_missing_small_or_duplicate_rows(rows):
    with pytest.raises(ValueError):
        scenario_report(funded(), rows, now_ns=0, historical=True)


def test_scenarios_require_exact_currency_and_instrument_alignment():
    p = funded(dual=True)
    for rows in [[scenario('missing', '0')], [scenario('nofx', '0', b='0')],
                 [PortfolioScenario('extra', (('A', 0), ('B', 0), ('Z', 0)), (('EUR', 0),))]]:
        with pytest.raises(ValueError, match='align exactly'):
            scenario_report(p, rows, now_ns=0)


@pytest.mark.parametrize('bad', ['NaN', 'Infinity', '-Infinity', True, '1e100', '-1.01', '11'])
def test_scenario_returns_are_finite_and_bounded(bad):
    with pytest.raises(ValueError):
        scenario('bad', bad)


@pytest.mark.parametrize('bad', ['0', '1', '-.2', 'NaN', True])
def test_tail_confidence_is_validated(bad):
    with pytest.raises(ValueError):
        scenario_report(funded(), [scenario('a', 0)], now_ns=0, confidence=bad)


@pytest.mark.parametrize('factory', [
    lambda: InstrumentSpec('A', 'USD', multiplier='nan'),
    lambda: InstrumentSpec('A', 'USD', lot_size=True),
    lambda: InstrumentSpec('A', 'USD', tick_size=0),
    lambda: InstrumentSpec('A', 'USD', margin_rate='1.1'),
    lambda: PortfolioLimits(max_leverage=0),
    lambda: PortfolioLimits(max_concentration='1.01'),
    lambda: PortfolioLimits(max_fx_age_ns=-1),
    lambda: PortfolioLimits(max_orders=True),
    lambda: PortfolioScenario('zero-fx', (('A', 0),), (('EUR', -1),)),
    lambda: PortfolioScenario('duplicates', (('A', 0), ('A', 0))),
])
def test_strict_frozen_settings_reject_invalid_values(factory):
    with pytest.raises(ValueError):
        factory()


def test_limits_quotes_and_orders_are_not_mutable_through_public_views():
    p = funded()
    with pytest.raises(FrozenInstanceError):
        p.limits.max_gross = 1
    with pytest.raises(TypeError):
        p.positions['A'] = 100
    with pytest.raises(TypeError):
        p.cash['USD'] = 0
    submit(p, 'o', 'buy', 1)
    with pytest.raises(FrozenInstanceError):
        p.orders['o'].remaining = 0
    audit = p.audit
    audit[0]['price'] = '999'
    assert p.audit[0]['price'] != '999'


def test_lot_and_tick_admission_and_collection_bounds():
    p = Portfolio(base_currency='USD', instruments=[InstrumentSpec('A', 'USD', lot_size=2, tick_size='.05')],
                  initial_cash={'USD': 1000}, limits=PortfolioLimits(max_orders=1))
    with pytest.raises(ValueError, match='tick'):
        p.mark('A', '100.01', 0)
    p.mark('A', '100', 0)
    with pytest.raises(ValueError, match='lot size'):
        submit(p, 'bad', 'buy', 1)
    assert submit(p, 'first', 'buy', 2)['accepted']
    with pytest.raises(ValueError, match='order limit'):
        submit(p, 'overflow', 'buy', 2)


def test_audit_cap_is_atomic_for_mark_and_fill():
    p = Portfolio(base_currency='USD', instruments=[InstrumentSpec('A', 'USD')],
                  initial_cash={'USD': 1000}, limits=PortfolioLimits(max_audit_events=2))
    p.mark('A', '100', 0)
    submit(p, 'o', 'buy', 1)
    with pytest.raises(ValueError, match='audit event limit'):
        p.fill('f', 'o', 1, '100', 0, 0)
    assert p.positions['A'] == 0 and p.cash['USD'] == 1000 and p.orders['o'].remaining == 1


def test_config_unknown_duplicate_and_nonfinite_fields_rejected(tmp_path):
    path = tmp_path / 'portfolio.json'
    for content in ['{"base_currency":"USD","base_currency":"EUR"}',
                    '{"base_currency":"USD","instruments":[],"extra":1}',
                    '{"base_currency":"USD","instruments":[],"x":NaN}']:
        path.write_text(content, encoding='utf-8')
        with pytest.raises(ValueError):
            load_portfolio_config(path)
    with pytest.raises(ValueError, match='argument'):
        PortfolioAction('snapshot', (('now_ns', 0), ('unknown', 1)))


def test_usable_demo_exercises_real_reservations_rejections_loss_kill_and_stress():
    report = run_portfolio_demo()
    assert report['status'] == 'COMPLETED'
    state = report['snapshot']
    assert state['reconciled'] and state['halted']
    assert set(state['halt_reasons']) == {'daily_loss', 'drawdown'}
    assert state['positions'] == {'ALPHA': 75, 'BETA': -100}
    assert state['orders']['buy']['status'] == 'cancelled'
    assert state['orders']['reduce']['remaining'] == 15
    assert state['orders']['excess']['status'] == 'rejected'
    assert state['orders']['halted-buy']['status'] == 'rejected'
    assert report['scenario_report']['kind'] == 'stress'
    assert report['scenario_report']['pending_order_count'] == 1
    json.dumps(report, allow_nan=False)
    # Frozen configuration parsing and execution produce a deterministic audit.
    assert report == run_portfolio_demo()


def test_mapping_config_is_strict_and_freezes_all_nested_scalar_state():
    config = portfolio_config({'base_currency': 'USD', 'instruments': [{'symbol': 'A', 'currency': 'USD'}],
                               'initial_cash': {'USD': '1000'},
                               'actions': [{'kind': 'mark', 'symbol': 'A', 'price': '100', 'timestamp_ns': 0}]})
    with pytest.raises(FrozenInstanceError):
        config.base_currency = 'EUR'
    assert run_portfolio_demo(config)['snapshot']['reconciled']
    with pytest.raises(ValueError):
        portfolio_config({'base_currency': 'USD', 'instruments': [{'symbol': 'A', 'currency': 'USD', 'unknown': 1}]})
