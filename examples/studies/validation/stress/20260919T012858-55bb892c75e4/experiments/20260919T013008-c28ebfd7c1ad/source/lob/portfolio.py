"""Offline portfolio accounting and reservation risk for linear instruments.

All prices are local quote-currency prices; FX is base-currency units per one
quote-currency unit. Positions are signed whole lots and cash is actually debited
or credited on fills. This is a funded linear-instrument model, not a futures,
options, borrow, liquidation, or variation-margin engine. ``margin_rate`` is an
additional exposure-capacity overlay. Marks and FX must be supplied explicitly.

Outstanding buys and sells form independent position intervals: they are never
netted against each other. Reservations survive cancel requests until an exchange
terminal acknowledgement. Limits are conservative at the supplied current marks,
not guarantees against later price moves. Decimal inputs/outputs preserve cash
reconciliation; a bounded 192-digit arithmetic context is used throughout.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
from functools import wraps
from math import ceil
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


ZERO = Decimal(0)
ONE = Decimal(1)
DAY_NS = 86_400_000_000_000
MAX_INSTRUMENTS = 100
MAX_SCENARIOS = 10_000


def _decimal(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{name} must be a decimal number")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a decimal number") from exc
    if (not number.is_finite() or abs(number) > Decimal('1e18') or
            len(number.as_tuple().digits) > 18 or abs(number.as_tuple().exponent) > 18):
        raise ValueError(f"{name} must be finite with at most 18 digits and bounded magnitude/exponent")
    if positive and number <= 0 or nonnegative and number < 0:
        raise ValueError(f"{name} is outside its allowed range")
    return number


def _integer(value: Any, name: str, *, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 80 or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty identifier of at most 80 characters")
    return value


def _precise(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        with localcontext() as context:
            context.prec = 192
            return method(*args, **kwargs)
    return wrapped


def _serial(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serial(item) for item in value]
    return value


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    currency: str
    multiplier: Decimal = ONE
    lot_size: int = 1
    tick_size: Decimal = Decimal('0.01')
    margin_rate: Decimal = ONE

    def __post_init__(self):
        _identifier(self.symbol, 'symbol')
        _identifier(self.currency, 'currency')
        _integer(self.lot_size, 'lot_size', minimum=1, maximum=10**9)
        for name in ('multiplier', 'tick_size'):
            object.__setattr__(self, name, _decimal(getattr(self, name), name, positive=True))
        rate = _decimal(self.margin_rate, 'margin_rate', nonnegative=True)
        if rate > ONE:
            raise ValueError('margin_rate must not exceed one')
        object.__setattr__(self, 'margin_rate', rate)


@dataclass(frozen=True)
class PortfolioLimits:
    max_gross: Decimal = Decimal('1000000')
    max_abs_net: Decimal = Decimal('1000000')
    max_leverage: Decimal = Decimal('2')
    max_symbol_notional: Decimal | None = None
    max_concentration: Decimal | None = None
    max_order_notional: Decimal | None = None
    max_drawdown: Decimal | None = None
    max_daily_loss: Decimal | None = None
    max_mark_age_ns: int = 60_000_000_000
    max_fx_age_ns: int = 60_000_000_000
    max_orders: int = 10_000
    max_audit_events: int = 100_000

    def __post_init__(self):
        if any(getattr(self, name) is None for name in ('max_gross', 'max_abs_net', 'max_leverage')):
            raise ValueError('gross, net and leverage limits are required')
        for name in ('max_gross', 'max_abs_net', 'max_leverage', 'max_symbol_notional',
                     'max_concentration', 'max_order_notional', 'max_drawdown', 'max_daily_loss'):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _decimal(value, name, positive=True))
        if self.max_concentration is not None and self.max_concentration > ONE:
            raise ValueError('max_concentration must not exceed one')
        for name in ('max_mark_age_ns', 'max_fx_age_ns'):
            _integer(getattr(self, name), name)
        _integer(self.max_orders, 'max_orders', minimum=1, maximum=100_000)
        _integer(self.max_audit_events, 'max_audit_events', minimum=1, maximum=1_000_000)


@dataclass(frozen=True)
class Quote:
    value: Decimal
    timestamp_ns: int

    def __post_init__(self):
        object.__setattr__(self, 'value', _decimal(self.value, 'quote', positive=True))
        _integer(self.timestamp_ns, 'timestamp_ns')


@dataclass(frozen=True)
class Reservation:
    order_id: str
    symbol: str
    side: str
    quantity: int
    remaining: int
    worst_price: Decimal
    fee_reserve: Decimal
    fee_remaining: Decimal
    status: str = 'submitted'

    @property
    def terminal(self) -> bool:
        return self.status in ('filled', 'cancelled', 'rejected', 'expired')


class MarketDataError(ValueError):
    """A current mark or FX quote is missing, future-dated, or stale."""


class Portfolio:
    def __init__(self, *, base_currency: str, instruments: Sequence[InstrumentSpec],
                 limits: PortfolioLimits | None = None, initial_cash: Mapping[str, Any] | None = None,
                 initial_positions: Mapping[str, int] | None = None):
        self.base_currency = _identifier(base_currency, 'base_currency')
        if not 1 <= len(instruments) <= MAX_INSTRUMENTS or any(not isinstance(i, InstrumentSpec) for i in instruments):
            raise ValueError(f'instruments must contain 1..{MAX_INSTRUMENTS} InstrumentSpec entries')
        specs = {item.symbol: item for item in instruments}
        if len(specs) != len(instruments):
            raise ValueError('instrument symbols must be unique')
        self.instruments = MappingProxyType(specs)
        if limits is not None and not isinstance(limits, PortfolioLimits):
            raise ValueError('limits must be PortfolioLimits')
        self.limits = limits or PortfolioLimits()
        cash = dict(initial_cash or {})
        if len(cash) > MAX_INSTRUMENTS:
            raise ValueError('too many cash currencies')
        self._cash = {_identifier(key, 'cash currency'): _decimal(value, 'initial cash') for key, value in cash.items()}
        positions = dict(initial_positions or {})
        if set(positions) - set(specs):
            raise ValueError('unknown initial position symbol')
        self._positions = {symbol: self._quantity(symbol, positions.get(symbol, 0), signed=True) for symbol in specs}
        self._initial_cash = self._cash.copy()
        self._initial_positions = self._positions.copy()
        self._marks: dict[str, Quote] = {}
        self._fx: dict[str, Quote] = {}
        self._orders: dict[str, Reservation] = {}
        self._fees: dict[str, Decimal] = {}
        self._fill_cash: dict[str, Decimal] = {}
        self._fill_notional: dict[str, Decimal] = {}
        self._fill_positions = {symbol: 0 for symbol in specs}
        self._fill_ids: set[str] = set()
        self._audit: list[dict[str, Any]] = []
        self._clock = 0
        self._initial_equity: Decimal | None = None
        self._peak: Decimal | None = None
        self._day: int | None = None
        self._day_equity: Decimal | None = None
        self.halted = False
        self.halt_reasons: list[str] = []

    @property
    def positions(self):
        return MappingProxyType(self._positions.copy())

    @property
    def cash(self):
        return MappingProxyType(self._cash.copy())

    @property
    def orders(self):
        return MappingProxyType(self._orders.copy())

    @property
    def audit(self):
        return tuple(_serial(event) for event in self._audit)

    def _quantity(self, symbol: str, value: Any, *, signed: bool = False) -> int:
        if symbol not in self.instruments:
            raise ValueError(f'unknown instrument {symbol!r}')
        qty = _integer(value, 'quantity', minimum=-10**9 if signed else 1, maximum=10**9)
        if qty % self.instruments[symbol].lot_size:
            raise ValueError('quantity must be an exact multiple of the instrument lot size')
        return qty

    def _time(self, now_ns: int) -> None:
        _integer(now_ns, 'now_ns')
        if now_ns < self._clock:
            raise ValueError('portfolio time must be monotone')

    def _event(self, now_ns: int, event: str, **details) -> None:
        if len(self._audit) >= self.limits.max_audit_events:
            raise ValueError('portfolio audit event limit reached')
        self._clock = now_ns
        self._audit.append({'timestamp_ns': now_ns, 'event': event, **details})

    @_precise
    def mark(self, symbol: str, price: Any, timestamp_ns: int) -> None:
        self._time(timestamp_ns)
        if symbol not in self.instruments:
            raise ValueError('unknown instrument')
        quote = Quote(price, timestamp_ns)
        if quote.value % self.instruments[symbol].tick_size:
            raise ValueError('mark must align with tick_size')
        self._event(timestamp_ns, 'mark', symbol=symbol, price=quote.value)
        self._marks[symbol] = quote

    def fx(self, currency: str, rate: Any, timestamp_ns: int) -> None:
        self._time(timestamp_ns)
        currency = _identifier(currency, 'currency')
        if currency not in self.currencies or currency == self.base_currency:
            raise ValueError('FX is supplied only for configured non-base currencies')
        quote = Quote(rate, timestamp_ns)
        self._event(timestamp_ns, 'fx', currency=currency, base_per_quote=quote.value)
        self._fx[currency] = quote

    @property
    def currencies(self) -> set[str]:
        return {self.base_currency, *self._cash, *(i.currency for i in self.instruments.values())}

    def _prices(self, now_ns: int) -> tuple[dict[str, Decimal], dict[str, Decimal]]:
        self._time(now_ns)
        prices: dict[str, Decimal] = {}
        rates = {self.base_currency: ONE}
        for key, quotes, target, age in (
                (set(self.instruments), self._marks, prices, self.limits.max_mark_age_ns),
                (self.currencies - {self.base_currency}, self._fx, rates, self.limits.max_fx_age_ns)):
            for name in sorted(key):
                quote = quotes.get(name)
                if quote is None:
                    raise MarketDataError(f'missing quote for {name}')
                if not 0 <= now_ns - quote.timestamp_ns <= age:
                    raise MarketDataError(f'stale or future quote for {name}')
                target[name] = quote.value
        return prices, rates

    def _metrics(self, prices: Mapping[str, Decimal], rates: Mapping[str, Decimal],
                 candidate: Reservation | None = None) -> dict[str, Any]:
        equity = sum((value * rates[currency] for currency, value in self._cash.items()), ZERO)
        net = gross = margin = ZERO
        bounds: dict[str, tuple[int, int]] = {}
        exposures: dict[str, Decimal] = {}
        maxima: dict[str, Decimal] = {}
        minima: dict[str, Decimal] = {}
        lower_net = upper_net = reserved_cost = ZERO
        pending = [order for order in self._orders.values() if not order.terminal]
        if candidate is not None:
            pending.append(candidate)
        for symbol, spec in self.instruments.items():
            qty = self._positions[symbol]
            unit = prices[symbol] * spec.multiplier * rates[spec.currency]
            exposures[symbol] = qty * unit
            equity += exposures[symbol]
            net += exposures[symbol]
            gross += abs(exposures[symbol])
            buys = sells = 0
            for order in pending:
                if order.symbol != symbol:
                    continue
                if order.side == 'buy':
                    buys += order.remaining
                    adverse = max(ZERO, order.worst_price - prices[symbol])
                else:
                    sells += order.remaining
                    adverse = max(ZERO, prices[symbol] - order.worst_price)
                reserved_cost += (adverse * order.remaining * spec.multiplier + order.fee_remaining) * rates[spec.currency]
            low, high = qty - sells, qty + buys
            bounds[symbol] = (low, high)
            maxima[symbol] = max(abs(low), abs(high)) * unit
            minima[symbol] = (0 if low <= 0 <= high else min(abs(low), abs(high))) * unit
            lower_net += low * unit
            upper_net += high * unit
            margin += maxima[symbol] * spec.margin_rate
        worst_gross = sum(maxima.values(), ZERO)
        concentration = max((maximum / (maximum + sum((v for k, v in minima.items() if k != symbol), ZERO))
                             if maximum else ZERO for symbol, maximum in maxima.items()), default=ZERO)
        equity_floor = equity - reserved_cost
        return {'equity': equity, 'gross': gross, 'net': net, 'exposures': exposures,
                'position_intervals': bounds, 'worst_gross': worst_gross,
                'worst_abs_net': max(abs(lower_net), abs(upper_net)), 'worst_margin': margin,
                'worst_symbol_notional': max(maxima.values(), default=ZERO),
                'worst_concentration': concentration, 'reserved_cost': reserved_cost,
                'equity_floor': equity_floor,
                'worst_leverage': worst_gross / equity_floor if equity_floor > ZERO else None}

    def _limit_values(self, metrics: Mapping[str, Any]) -> dict[str, tuple[Decimal, Decimal]]:
        values = {'gross': (metrics['worst_gross'], self.limits.max_gross),
                  'net': (metrics['worst_abs_net'], self.limits.max_abs_net),
                  'margin': (metrics['worst_margin'], metrics['equity_floor'])}
        if metrics['worst_leverage'] is not None:
            values['leverage'] = (metrics['worst_leverage'], self.limits.max_leverage)
        for setting, key in (('max_symbol_notional', 'worst_symbol_notional'),
                             ('max_concentration', 'worst_concentration')):
            cap = getattr(self.limits, setting)
            if cap is not None:
                values[setting.removeprefix('max_')] = (metrics[key], cap)
        return values

    def _violations(self, metrics: Mapping[str, Any], previous: Mapping[str, Any] | None = None) -> list[str]:
        issues = [] if metrics['equity_floor'] > ZERO else ['nonpositive_equity']
        before = self._limit_values(previous) if previous is not None else {}
        for name, (value, cap) in self._limit_values(metrics).items():
            # A reduce-only order may preserve an already breached envelope, but
            # never worsen that breach; margin compares capacity shortfalls.
            if value > cap and (name not in before or value - cap > max(ZERO, before[name][0] - before[name][1])):
                issues.append(name)
        return issues

    def _refresh_limits(self, metrics: Mapping[str, Any], now_ns: int) -> None:
        equity = metrics['equity']
        day = now_ns // DAY_NS
        initial = equity if self._initial_equity is None else self._initial_equity
        peak = max(self._peak if self._peak is not None else equity, equity)
        daily = equity if self._day != day else self._day_equity
        reasons = []
        if self.limits.max_drawdown is not None and peak - equity >= self.limits.max_drawdown:
            reasons.append('drawdown')
        if self.limits.max_daily_loss is not None and daily - equity >= self.limits.max_daily_loss:
            reasons.append('daily_loss')
        new_reasons = [reason for reason in reasons if reason not in self.halt_reasons]
        if new_reasons:
            self._event(now_ns, 'kill_latched', reasons=new_reasons, equity=equity)
            self.halted = True
            self.halt_reasons.extend(new_reasons)
        self._initial_equity, self._peak = initial, peak
        self._day, self._day_equity = day, daily
        self._clock = now_ns

    @_precise
    def snapshot(self, now_ns: int) -> dict[str, Any]:
        prices, rates = self._prices(now_ns)
        metrics = self._metrics(prices, rates)
        self._refresh_limits(metrics, now_ns)
        self.reconcile()
        return _serial({'base_currency': self.base_currency, 'timestamp_ns': now_ns,
                        **metrics, 'cash': self._cash, 'positions': self._positions,
                        'fees_by_currency': self._fees, 'initial_equity': self._initial_equity,
                        'total_pnl': metrics['equity'] - self._initial_equity,
                        'drawdown': self._peak - metrics['equity'],
                        'daily_loss': self._day_equity - metrics['equity'],
                        'daily_baseline': 'first valuation observed in UTC calendar day',
                        'halted': self.halted, 'halt_reasons': list(self.halt_reasons),
                        'limit_violations': self._violations(metrics),
                        'orders': {key: asdict(value) for key, value in self._orders.items()},
                        'reconciled': True})

    @_precise
    def submit(self, order_id: str, symbol: str, side: str, quantity: int, worst_price: Any,
               fee_reserve: Any, now_ns: int, *, reduce_only: bool = False) -> dict[str, Any]:
        """Reserve an order; worst_price is buy ceiling or sell floor.

        ``fee_reserve`` is a total nonnegative local-currency fee budget. Fills
        outside either promised bound are rejected before accounting mutation.
        Callers must not use an uncapped market order with this interface.
        """
        self._time(now_ns)
        _identifier(order_id, 'order_id')
        if order_id in self._orders:
            raise ValueError('order IDs cannot be reused')
        if len(self._orders) >= self.limits.max_orders:
            raise ValueError('portfolio order limit reached')
        if side not in ('buy', 'sell') or not isinstance(reduce_only, bool):
            raise ValueError('side must be buy/sell and reduce_only must be boolean')
        qty = self._quantity(symbol, quantity)
        price = _decimal(worst_price, 'worst_price', positive=True)
        fee = _decimal(fee_reserve, 'fee_reserve', nonnegative=True)
        if price % self.instruments[symbol].tick_size:
            raise ValueError('worst_price must align with tick_size')
        candidate = Reservation(order_id, symbol, side, qty, qty, price, fee, fee)
        prices, rates = self._prices(now_ns)
        before = self._metrics(prices, rates)
        self._refresh_limits(before, now_ns)
        projected = self._metrics(prices, rates, candidate)
        low, high = before['position_intervals'][symbol]
        reducing = (side == 'sell' and low > 0 and qty <= low or side == 'buy' and high < 0 and qty <= -high)
        reasons = []
        if reduce_only and not reducing:
            reasons.append('not_robustly_reducing')
        if self.halted and not (reduce_only and reducing):
            reasons.append('kill_switch')
        spec = self.instruments[symbol]
        order_notional = qty * max(price, prices[symbol]) * spec.multiplier * rates[spec.currency]
        if self.limits.max_order_notional is not None and order_notional > self.limits.max_order_notional:
            reasons.append('order_notional')
        reasons.extend(self._violations(projected, before if reduce_only and reducing else None))
        accepted = not reasons
        self._event(now_ns, 'submitted' if accepted else 'rejected', order_id=order_id,
                    symbol=symbol, side=side, quantity=qty, reduce_only=reduce_only,
                    reasons=reasons, projected=projected)
        self._orders[order_id] = candidate if accepted else Reservation(
            order_id, symbol, side, qty, 0, price, fee, ZERO, 'rejected')
        return {'accepted': accepted, 'order_id': order_id, 'reasons': reasons}

    def _active(self, order_id: str) -> Reservation:
        order = self._orders.get(order_id)
        if order is None or order.terminal:
            raise ValueError('an active order is required')
        return order

    def cancel_request(self, order_id: str, now_ns: int) -> None:
        self._time(now_ns)
        order = self._active(order_id)
        if order.status == 'cancel_pending':
            raise ValueError('cancellation is already pending')
        self._event(now_ns, 'cancel_requested', order_id=order_id, remaining=order.remaining)
        self._orders[order_id] = Reservation(**{**asdict(order), 'status': 'cancel_pending'})

    def acknowledge_terminal(self, order_id: str, status: str, now_ns: int) -> None:
        self._time(now_ns)
        order = self._active(order_id)
        if status not in ('cancelled', 'rejected', 'expired'):
            raise ValueError('terminal acknowledgement must be cancelled, rejected or expired')
        if status == 'rejected' and order.remaining != order.quantity:
            raise ValueError('a partially filled order cannot be rejected')
        self._event(now_ns, status, order_id=order_id, released_quantity=order.remaining,
                    released_fee=order.fee_remaining)
        self._orders[order_id] = Reservation(**{**asdict(order), 'remaining': 0,
                                              'fee_remaining': ZERO, 'status': status})

    @_precise
    def fill(self, fill_id: str, order_id: str, quantity: int, price: Any, fee: Any, now_ns: int) -> None:
        """Apply one exchange-ordered fill exactly, including during cancel latency."""
        self._time(now_ns)
        _identifier(fill_id, 'fill_id')
        if fill_id in self._fill_ids:
            raise ValueError('duplicate fill ID')
        order = self._active(order_id)
        qty = self._quantity(order.symbol, quantity)
        execution_price = _decimal(price, 'fill price', positive=True)
        execution_fee = _decimal(fee, 'fill fee')
        spec = self.instruments[order.symbol]
        if execution_price % spec.tick_size:
            raise ValueError('fill price must align with tick_size')
        if qty > order.remaining:
            raise ValueError('fill exceeds reserved leaves')
        if (order.side == 'buy' and execution_price > order.worst_price or
                order.side == 'sell' and execution_price < order.worst_price):
            raise ValueError('fill violates the reserved execution price bound')
        if execution_fee > order.fee_remaining:
            raise ValueError('fill fee exceeds remaining reserved fee budget')
        signed_qty = qty if order.side == 'buy' else -qty
        cash_delta = -signed_qty * execution_price * spec.multiplier - execution_fee
        remaining = order.remaining - qty
        # Rebates credit cash, but do not create fresh capacity for later fees.
        fee_remaining = max(ZERO, order.fee_remaining - max(ZERO, execution_fee)) if remaining else ZERO
        status = ('filled' if remaining == 0 else 'cancel_pending'
                  if order.status == 'cancel_pending' else 'partially_filled')
        self._event(now_ns, 'fill', fill_id=fill_id, order_id=order_id, symbol=order.symbol,
                    signed_quantity=signed_qty, price=execution_price, fee=execution_fee,
                    currency=spec.currency, cash_delta=cash_delta, remaining=remaining)
        self._positions[order.symbol] += signed_qty
        self._cash[spec.currency] = self._cash.get(spec.currency, ZERO) + cash_delta
        self._fees[spec.currency] = self._fees.get(spec.currency, ZERO) + execution_fee
        self._fill_cash[spec.currency] = self._fill_cash.get(spec.currency, ZERO) + cash_delta
        self._fill_notional[spec.currency] = self._fill_notional.get(spec.currency, ZERO) + signed_qty * execution_price * spec.multiplier
        self._fill_positions[order.symbol] += signed_qty
        self._fill_ids.add(fill_id)
        self._orders[order_id] = Reservation(**{**asdict(order), 'remaining': remaining,
                                              'fee_remaining': fee_remaining, 'status': status})
        self.reconcile()

    @_precise
    def reconcile(self) -> None:
        for currency in set(self._cash) | set(self._initial_cash) | set(self._fill_cash):
            if self._cash.get(currency, ZERO) != self._initial_cash.get(currency, ZERO) + self._fill_cash.get(currency, ZERO):
                raise ArithmeticError('cash does not reconcile with fills and fees')
            if self._cash.get(currency, ZERO) != (self._initial_cash.get(currency, ZERO) -
                                                self._fill_notional.get(currency, ZERO) - self._fees.get(currency, ZERO)):
                raise ArithmeticError('cash does not reconcile independently with signed notionals and fees')
        for symbol in self.instruments:
            if self._positions[symbol] != self._initial_positions[symbol] + self._fill_positions[symbol]:
                raise ArithmeticError('position does not reconcile with fills')


@dataclass(frozen=True)
class PortfolioScenario:
    name: str
    instrument_returns: tuple[tuple[str, Decimal], ...]
    fx_returns: tuple[tuple[str, Decimal], ...] = ()

    def __post_init__(self):
        _identifier(self.name, 'scenario name')
        for field_name in ('instrument_returns', 'fx_returns'):
            source = getattr(self, field_name)
            if not isinstance(source, tuple) or len(source) > MAX_INSTRUMENTS:
                raise ValueError('scenario returns must be bounded immutable tuples')
            result = []
            for key, value in source:
                _identifier(key, 'return key')
                number = _decimal(value, 'return')
                if number < -ONE or number > Decimal(10) or field_name == 'fx_returns' and number == -ONE:
                    raise ValueError('simple instrument returns must be in [-1,10], FX returns in (-1,10]')
                result.append((key, number))
            if len({key for key, _ in result}) != len(result):
                raise ValueError('duplicate scenario return key')
            object.__setattr__(self, field_name, tuple(result))


@_precise
def scenario_report(portfolio: Portfolio, scenarios: Sequence[PortfolioScenario], *, now_ns: int,
                    confidence: Any = '0.95', historical: bool = False) -> dict[str, Any]:
    """Revalue current filled holdings using aligned *local-price* and FX returns.

    Includes cash FX PnL and price/FX cross terms. Pending orders are excluded and
    prominently counted. Historical VaR uses the inverse empirical CDF; ES uses
    exact fractional probability mass in the worst (1-confidence) tail. Reported
    risk measures floor at zero; unfloored values are also exposed. No normality,
    independence, annualization or predictive-coverage assumption is made.
    """
    alpha = _decimal(confidence, 'confidence', positive=True)
    if alpha >= ONE or not isinstance(historical, bool):
        raise ValueError('confidence must be below one and historical must be boolean')
    if not isinstance(scenarios, (list, tuple)) or not 1 <= len(scenarios) <= MAX_SCENARIOS:
        raise ValueError(f'scenarios must contain 1..{MAX_SCENARIOS} aligned rows')
    if any(not isinstance(row, PortfolioScenario) for row in scenarios):
        raise ValueError('scenarios must contain PortfolioScenario entries')
    if len({row.name for row in scenarios}) != len(scenarios):
        raise ValueError('scenario names must be unique')
    if historical and len(scenarios) < 2:
        raise ValueError('historical risk requires at least two observations')
    prices, rates = portfolio._prices(now_ns)
    snapshot = portfolio.snapshot(now_ns)
    equity = Decimal(snapshot['equity'])
    currencies = portfolio.currencies - {portfolio.base_currency}
    rows = []
    for scenario in scenarios:
        local, fx = dict(scenario.instrument_returns), dict(scenario.fx_returns)
        if set(local) != set(portfolio.instruments) or set(fx) != currencies:
            raise ValueError('every scenario must align exactly to configured instruments and non-base currencies')
        new_rates = {currency: rate * (ONE + fx.get(currency, ZERO)) for currency, rate in rates.items()}
        stressed = sum((cash * new_rates[currency] for currency, cash in portfolio.cash.items()), ZERO)
        for symbol, spec in portfolio.instruments.items():
            stressed += portfolio.positions[symbol] * spec.multiplier * prices[symbol] * (ONE + local[symbol]) * new_rates[spec.currency]
        rows.append({'name': scenario.name, 'equity': stressed, 'pnl': stressed - equity, 'loss': equity - stressed})
    losses = sorted(row['loss'] for row in rows)
    tail_mass = Decimal(len(losses)) * (ONE - alpha)
    risk = None
    warnings = ['Scenario PnL covers filled holdings only; order reservations are reported separately.',
                'Historical tail estimates describe supplied observations and do not guarantee future coverage.']
    if historical:
        index = ceil(alpha * len(losses)) - 1
        quantile = losses[index]
        full = int(tail_mass)
        descending = list(reversed(losses))
        tail_total = sum(descending[:full], ZERO)
        if full < len(descending):
            tail_total += (tail_mass - full) * descending[full]
        expected_shortfall = tail_total / tail_mass
        risk = {'confidence': alpha, 'observations': len(losses), 'tail_observation_mass': tail_mass,
                'var': max(ZERO, quantile), 'expected_shortfall': max(ZERO, expected_shortfall),
                'raw_loss_quantile': quantile, 'raw_tail_mean': expected_shortfall,
                'quantile_convention': 'inverse empirical CDF: order statistic ceil(confidence * n)',
                'es_convention': 'mean of exactly n*(1-confidence) worst observations with fractional boundary weight',
                'horizon': 'one supplied aligned observation; no rescaling',
                'statistical_guarantee': False}
        if tail_mass < 5:
            warnings.append('Fewer than five effective tail observations: tail estimate is weakly supported.')
    return _serial({'base_currency': portfolio.base_currency, 'timestamp_ns': now_ns,
                    'kind': 'historical' if historical else 'stress', 'equity': equity,
                    'scenarios': rows, 'worst_loss': max(ZERO, losses[-1]), 'historical_risk': risk,
                    'pending_order_count': sum(not order.terminal for order in portfolio.orders.values()),
                    'pending_orders_in_scenario_pnl': False, 'warnings': warnings})


@dataclass(frozen=True)
class PortfolioAction:
    kind: str
    arguments: tuple[tuple[str, Any], ...]

    def __post_init__(self):
        fields = {
            'mark': ({'symbol', 'price', 'timestamp_ns'}, set()),
            'fx': ({'currency', 'rate', 'timestamp_ns'}, set()),
            'submit': ({'order_id', 'symbol', 'side', 'quantity', 'worst_price', 'fee_reserve', 'now_ns'}, {'reduce_only'}),
            'fill': ({'fill_id', 'order_id', 'quantity', 'price', 'fee', 'now_ns'}, set()),
            'cancel_request': ({'order_id', 'now_ns'}, set()),
            'acknowledge_terminal': ({'order_id', 'status', 'now_ns'}, set()),
            'snapshot': ({'now_ns'}, set()),
        }
        if self.kind not in fields or not isinstance(self.arguments, tuple):
            raise ValueError('unsupported portfolio action')
        if len(dict(self.arguments)) != len(self.arguments):
            raise ValueError('duplicate action arguments')
        if any(not isinstance(key, str) or not isinstance(value, (str, int, float, Decimal, bool))
               for key, value in self.arguments):
            raise ValueError('action arguments must be scalar')
        required, optional = fields[self.kind]
        if not required <= dict(self.arguments).keys() or dict(self.arguments).keys() - required - optional:
            raise ValueError('missing or unknown portfolio action argument')


@dataclass(frozen=True)
class PortfolioConfig:
    base_currency: str
    instruments: tuple[InstrumentSpec, ...]
    limits: PortfolioLimits = field(default_factory=PortfolioLimits)
    initial_cash: tuple[tuple[str, Decimal], ...] = ()
    initial_positions: tuple[tuple[str, int], ...] = ()
    actions: tuple[PortfolioAction, ...] = ()
    scenarios: tuple[PortfolioScenario, ...] = ()
    confidence: Decimal = Decimal('0.95')
    historical: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the complete resolved immutable configuration as JSON-safe data."""
        return _serial(asdict(self))

    def __post_init__(self):
        for name in ('instruments', 'initial_cash', 'initial_positions', 'actions', 'scenarios'):
            if not isinstance(getattr(self, name), tuple):
                raise ValueError(f'{name} must be an immutable tuple')
        if len(dict(self.initial_cash)) != len(self.initial_cash) or len(dict(self.initial_positions)) != len(self.initial_positions):
            raise ValueError('duplicate initial cash or position key')
        portfolio = Portfolio(base_currency=self.base_currency, instruments=self.instruments,
                              limits=self.limits, initial_cash=dict(self.initial_cash),
                              initial_positions=dict(self.initial_positions))
        object.__setattr__(self, 'initial_cash', tuple(portfolio.cash.items()))
        if len(self.actions) > 100_000 or any(not isinstance(a, PortfolioAction) for a in self.actions):
            raise ValueError('actions must contain at most 100000 PortfolioAction entries')
        if len(self.scenarios) > MAX_SCENARIOS or any(not isinstance(s, PortfolioScenario) for s in self.scenarios):
            raise ValueError('scenarios must contain bounded PortfolioScenario entries')
        alpha = _decimal(self.confidence, 'confidence', positive=True)
        if alpha >= ONE or not isinstance(self.historical, bool):
            raise ValueError('invalid confidence or historical flag')
        object.__setattr__(self, 'confidence', alpha)


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key {key}')
        result[key] = value
    return result


def portfolio_config(value: Mapping[str, Any]) -> PortfolioConfig:
    allowed = {'base_currency', 'instruments', 'limits', 'initial_cash', 'initial_positions',
               'actions', 'scenarios', 'confidence', 'historical'}
    if not isinstance(value, Mapping) or set(value) - allowed or not {'base_currency', 'instruments'} <= set(value):
        raise ValueError('invalid portfolio configuration fields')
    try:
        instruments = value['instruments']
        actions = value.get('actions', [])
        scenarios = value.get('scenarios', [])
        if (not isinstance(instruments, (list, tuple)) or not 1 <= len(instruments) <= MAX_INSTRUMENTS or
                not isinstance(actions, (list, tuple)) or len(actions) > 100_000 or
                not isinstance(scenarios, (list, tuple)) or len(scenarios) > MAX_SCENARIOS):
            raise ValueError('portfolio configuration collections exceed bounds or have invalid types')
        parsed_scenarios = []
        for row in scenarios:
            if set(row) - {'name', 'instrument_returns', 'fx_returns'}:
                raise ValueError('unknown scenario field')
            parsed_scenarios.append(PortfolioScenario(row['name'], tuple(row['instrument_returns'].items()),
                                                       tuple(row.get('fx_returns', {}).items())))
        return PortfolioConfig(
            base_currency=value['base_currency'], instruments=tuple(InstrumentSpec(**item) for item in instruments),
            limits=PortfolioLimits(**value.get('limits', {})),
            initial_cash=tuple(value.get('initial_cash', {}).items()),
            initial_positions=tuple(value.get('initial_positions', {}).items()),
            actions=tuple(PortfolioAction(item['kind'], tuple((k, v) for k, v in item.items() if k != 'kind')) for item in actions),
            scenarios=tuple(parsed_scenarios), confidence=value.get('confidence', '0.95'),
            historical=value.get('historical', False))
    except (TypeError, KeyError, AttributeError) as exc:
        raise ValueError(f'invalid portfolio configuration: {exc}') from exc


def load_portfolio_config(path: str | Path) -> PortfolioConfig:
    source = Path(path)
    if source.stat().st_size > 2_000_000:
        raise ValueError('portfolio config exceeds 2 MB')
    with source.open(encoding='utf-8') as handle:
        value = json.load(handle, object_pairs_hook=_object_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'invalid JSON number {value}')))
    return portfolio_config(value)


def run_portfolio_demo(config: PortfolioConfig | str | Path | None = None) -> dict[str, Any]:
    """Run an explicit offline action script and return reconciled risk/audit output."""
    if config is None:
        config = Path(__file__).resolve().parents[1] / 'configs' / 'portfolio_example.json'
    if isinstance(config, (str, Path)):
        config = load_portfolio_config(config)
    if not isinstance(config, PortfolioConfig):
        raise ValueError('config must be PortfolioConfig or a JSON path')
    portfolio = Portfolio(base_currency=config.base_currency, instruments=config.instruments, limits=config.limits,
                          initial_cash=dict(config.initial_cash), initial_positions=dict(config.initial_positions))
    outcomes = []
    for action in config.actions:
        try:
            result = getattr(portfolio, action.kind)(**dict(action.arguments))
        except TypeError as exc:
            raise ValueError(f'invalid {action.kind} action: {exc}') from exc
        if result is not None:
            outcomes.append({'action': action.kind, 'result': result})
    snapshot = portfolio.snapshot(portfolio._clock)
    scenarios = scenario_report(portfolio, config.scenarios, now_ns=portfolio._clock,
                                confidence=config.confidence, historical=config.historical) if config.scenarios else None
    return {'schema_version': 1, 'mode': 'offline_portfolio_risk', 'status': 'COMPLETED',
            'config': _serial(asdict(config)), 'snapshot': snapshot, 'outcomes': outcomes,
            'scenario_report': scenarios, 'audit': list(portfolio.audit),
            'limitations': ['Linear funded instruments only; no derivatives settlement, borrow or forced liquidation.',
                            'Limits use current supplied marks and FX; future price moves can breach limits.',
                            'Daily baseline is the first observed valuation of each UTC day.',
                            'Scenario returns must be synchronized observations supplied by the caller.',
                            'No external orders are submitted.']}
