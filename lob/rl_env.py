"""Gymnasium environment: optimal execution on the simulated LOB."""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .accounting import FeeConfig, Ledger, fee_config
from .completion import CompletionConstraint, completion_constraint, urgency_order
from .engine import ExchangeSimulator, Side, SimConfig
from .execution import ExecutionReport, build_report
from .observations import FEATURES, OBSERVATION_CONTRACT, ObservationNormalization, normalization
from .risk import ExecutionRisk, RiskConfig, risk_config
from .settlement import (DEFAULT_SETTLEMENT_POLL_DT, DEFAULT_SETTLEMENT_TIMEOUT,
                         SettlementResult, settle_orders, validate_settlement)


class LOBExecutionEnv(gym.Env):
    """Liquidate ``total_qty`` within ``horizon`` seconds, minimising implementation shortfall.

    Observation (24,):
        top-5 bids  : (price − mid)/10 ticks, volume/500        -> 10 features
        top-5 asks  : same                                       -> 10
        micro-price − mid (ticks), book imbalance (top-5),
        remaining inventory fraction, remaining time fraction    -> 4

    Actions (Discrete 5):
        0  wait (keep nothing working)
        1  passive  : limit joining own-side best quote
        2  improve  : limit one tick inside the spread
        3  cross    : marketable limit at the opposite touch
        4  aggress  : market order (1.5× child size)

    Reward: per-fill implementation shortfall in bps (sign-adjusted, share-weighted),
    plus a terminal penalty pricing leftovers at their walk-the-book liquidation VWAP
    and an extra fixed penalty per leftover share.
    """

    metadata: Dict[str, List[str]] = {"render_modes": []}
    OWNER = "RL"

    def __init__(self, total_qty: int = 10_000, horizon: float = 60.0,
                 decision_dt: float = 0.5, side: Side = Side.SELL,
                 warmup: float = 5.0, cfg: Optional[SimConfig] = None,
                 record: bool = False, *, fees: FeeConfig | Mapping[str, Any] | None = None,
                 risk: RiskConfig | Mapping[str, Any] | None = None,
                 terminal_penalty_bps: float = 25.0,
                 settlement_timeout: float = DEFAULT_SETTLEMENT_TIMEOUT,
                 settlement_poll_dt: float = DEFAULT_SETTLEMENT_POLL_DT,
                 seed_range: Optional[Tuple[int, int]] = None,
                 completion: CompletionConstraint | Mapping[str, Any] | None = None,
                 observation_version: str = "v03",
                 observation_normalization: ObservationNormalization | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if isinstance(total_qty, bool) or not isinstance(total_qty, int) or total_qty <= 0:
            raise ValueError("total_qty must be a positive integer")
        if not all(math.isfinite(v) for v in (horizon, decision_dt, warmup, terminal_penalty_bps)) or horizon <= 0 or decision_dt <= 0 or warmup < 0 or terminal_penalty_bps < 0:
            raise ValueError("horizon/dt must be positive; warmup/terminal penalty nonnegative, all finite")
        if not isinstance(side, Side):
            raise ValueError("side must be Side.BUY or Side.SELL")
        self.total_qty = total_qty
        self.horizon = horizon
        self.decision_dt = decision_dt
        self.side = side
        self.warmup = warmup
        self.base_cfg = cfg or SimConfig()
        self.record = record
        self.fees = fee_config(fees)
        self.risk_config = risk_config(risk)
        self.terminal_penalty_bps = terminal_penalty_bps
        self.completion = completion_constraint(completion)
        if observation_version not in {"v03", "v04"}:
            raise ValueError("observation_version must be v03 or v04")
        self.observation_version = observation_version
        self.observation_capabilities = OBSERVATION_CONTRACT
        self.normalization = normalization(observation_normalization)
        if self.normalization is not None and observation_version != "v04":
            raise ValueError("fitted normalization requires v04 observations")
        validate_settlement(settlement_timeout, settlement_poll_dt)
        self.settlement_timeout = settlement_timeout
        self.settlement_poll_dt = settlement_poll_dt
        if seed_range is not None and (len(seed_range) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in seed_range)
                                       or not 0 <= seed_range[0] < seed_range[1] <= 2**31):
            raise ValueError("seed_range must be integer bounds 0 <= low < high <= 2**31")
        self.seed_range = seed_range
        self.n_steps = max(1, int(math.ceil(horizon / decision_dt - 1e-12)))
        self.child = max(1, total_qty // max(1, self.n_steps // 2))
        bound = self.normalization.clip if self.normalization is not None else np.inf
        self.observation_space = spaces.Box(-bound, bound, shape=(24 if observation_version == "v03" else len(FEATURES),), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        low, high = self.seed_range or (0, 2**31 - 1)
        # A supplied seed initializes Gym's RNG. With a restricted domain, draw
        # the actual market seed inside that domain on every reset, including first.
        sim_seed = int(seed) if seed is not None and self.seed_range is None else int(self.np_random.integers(low, high))
        self.market_seed = sim_seed
        cfg = SimConfig(**{**self.base_cfg.__dict__, "seed": sim_seed})
        if self.total_qty % getattr(cfg, "lot_size", 1):
            raise ValueError("total_qty must be a multiple of lot_size")
        self.child = max(getattr(cfg, "lot_size", 1), self.child)
        self.sim = ExchangeSimulator(cfg)
        while self.sim.t < self.warmup - 1e-12:
            self.sim.step(min(0.1, self.warmup - self.sim.t))
        self.t0 = self.sim.t
        self.arrival = self.sim.book.mid()
        self.remaining = self.total_qty
        self.fills: List[Any] = []
        self._cursor = len(self.sim.book.trades)
        self.first_trade = self._cursor
        self._live_order: Optional[int] = None
        self.n_children = 0
        self.step_i = 0
        self._done = False
        self.settlement: SettlementResult | None = None
        self._decision_duration: float | None = None
        self.late_filled_qty = 0
        self.late_fees = 0.0
        self.total_reward = 0.0
        self.reward_terms = {"fills": 0.0, "fees": 0.0, "terminal_shortfall": 0.0,
                             "terminal_fees": 0.0, "completion_penalty": 0.0,
                             "late_fills": 0.0, "late_fees": 0.0}
        self.risk = ExecutionRisk(self.total_qty, self.side, self.risk_config)
        self.ledger = Ledger(inventory=self.total_qty if self.side is Side.SELL else 0,
                             average_cost=self.arrival * cfg.tick_size, fees=self.fees)
        self.decisions: List[Dict[str, Any]] = []
        self.snapshots: List[Dict[str, Any]] = []
        if self.record:
            self._snap()
        return self._obs(), self._info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("step called after episode ended; call reset")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action!r}")
        sim, book = self.sim, self.sim.book
        decision_at = sim.t - self.t0
        decision_observation = self._obs().tolist() if self.record else None
        self.risk.check(sim, self.ledger)
        # Cancellation does not release the reservation. Every old child remains
        # covered while a replacement races its cancellation acknowledgement.
        self.risk.cancel_all(sim)
        self._live_order = None
        qty = min(self.child, self.risk.available(sim, self.total_qty - self.remaining))
        bb, ba = book.best_bid(), book.best_ask()
        price = None
        submitted = None
        forced = self.completion.active(decision_at, self.horizon, self.decision_dt)
        can_quote = bb is not None and ba is not None
        can_cross = (bb if self.side is Side.SELL else ba) is not None
        if forced:
            submitted = urgency_order(sim, self.risk, self.ledger, self.OWNER,
                                      self.total_qty - self.remaining, elapsed=decision_at,
                                      horizon=self.horizon, decision_dt=self.decision_dt)
            if submitted is not None:
                self._live_order = submitted
                self.n_children += 1
        elif qty > 0 and (can_quote or (action == 4 and can_cross)):
            if action == 1:                                   # passive join
                price = ba if self.side is Side.SELL else bb
            elif action == 2:                                 # improve inside spread
                price = (ba - 1) if self.side is Side.SELL else (bb + 1)
                if not ((self.side is Side.SELL and price > bb) or
                        (self.side is Side.BUY and price < ba)):
                    price = ba if self.side is Side.SELL else bb
            elif action == 3:                                 # marketable limit at touch
                price = bb if self.side is Side.SELL else ba
            elif action == 4:                                 # aggressive market
                qty = min(self.risk.available(sim, self.total_qty - self.remaining), int(1.5 * self.child))
            if action != 0:
                submitted = self.risk.submit(sim, self.OWNER, qty, self.total_qty - self.remaining,
                                             self.ledger, price_ticks=price)
            if submitted is not None:
                self._live_order = submitted
                self.n_children += 1

        sim.step(min(self.decision_dt, max(0.0, self.horizon - (sim.t - self.t0))))
        self.step_i += 1

        self.reward_terms = dict.fromkeys(self.reward_terms, 0.0)
        self._poll_fills()
        decisions_finished = self.remaining <= 0 or self.step_i >= self.n_steps
        terminated, truncated = False, False
        if decisions_finished:
            self._decision_duration = sim.t - self.t0
            self.settlement = settle_orders(sim, self.risk, lambda: self._poll_fills(late=True),
                                            timeout=self.settlement_timeout,
                                            poll_dt=self.settlement_poll_dt,
                                            on_step=self._snap if self.record else None)
            # Actual late fills and fees enter this final reward before valuing
            # the remaining quantity. A timeout withholds final economic IS;
            # its finite reward is a provisional objective, explicitly truncated.
            terminal_report = self.report()
            denominator = self.total_qty * self.arrival * sim.cfg.tick_size
            self.reward_terms["terminal_shortfall"] = -terminal_report.hypothetical_liquidation_cost / denominator * 1e4
            self.reward_terms["terminal_fees"] = -terminal_report.hypothetical_liquidation_fees / denominator * 1e4
            self.reward_terms["completion_penalty"] = -terminal_report.completion_penalty_bps
            terminated = self.settlement.complete
            truncated = not self.settlement.complete
        reward = sum(self.reward_terms.values())
        self._done = terminated or truncated
        self.total_reward += float(reward)
        if self.record:
            self.decisions.append({"step": self.step_i, "t": decision_at, "result_time": sim.t - self.t0, "action": int(action),
                                   "completion_override": forced,
                                   "order_id": submitted, "reward": float(reward),
                                   "reward_terms": dict(self.reward_terms),
                                   "remaining": self.remaining, "reserved_qty": self.risk.outstanding(sim),
                                   "observation": decision_observation, "next_observation": self._obs().tolist()})
            self._snap()
        return self._obs(), float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------ helpers
    def _poll_fills(self, *, late: bool = False) -> None:
        new, self._cursor = self.sim.fills_for(self.OWNER, self._cursor)
        sign = 1.0 if self.side is Side.SELL else -1.0
        denominator = self.total_qty * self.arrival * self.sim.cfg.tick_size
        for tr in new:
            fee = self.ledger.apply_trade(tr, self.OWNER, self.sim.cfg.tick_size)
            self.fills.append(tr)
            self.remaining -= tr.qty
            if self.remaining < 0:
                raise ArithmeticError("parent execution overfilled")
            self.reward_terms["late_fills" if late else "fills"] += (
                sign * (tr.price - self.arrival) / self.arrival * 1e4 * tr.qty / self.total_qty)
            self.reward_terms["late_fees" if late else "fees"] -= fee / denominator * 1e4
            if late:
                self.late_filled_qty += tr.qty
                self.late_fees += fee
        self.risk.available(self.sim, self.total_qty - self.remaining)
        self.risk.check(self.sim, self.ledger)

    def _obs(self) -> np.ndarray:
        book = self.sim.book
        mid = book.mid()
        bids, asks = book.depth(5)
        feats: List[float] = []
        for i in range(5):
            p, v = bids[i] if i < len(bids) else (mid - (i + 1), 0)
            feats += [(p - mid) / 10.0, v / 500.0]
        for i in range(5):
            p, v = asks[i] if i < len(asks) else (mid + (i + 1), 0)
            feats += [(p - mid) / 10.0, v / 500.0]
        feats += [
            book.microprice() - mid,
            book.imbalance(5),
            self.remaining / self.total_qty,
            max(0.0, 1.0 - (self.sim.t - self.t0) / self.horizon),
        ]
        if self.observation_version == "v04":
            # Restore physical units before applying the train-only scaler.
            for i in range(0, 20, 2):
                feats[i] *= 10.0
                feats[i + 1] *= 500.0
            own = [self.sim.orders[oid] for oid in self.risk.order_ids if not self.sim.orders[oid].is_terminal]
            resting = {"RESTING", "PARTIALLY_FILLED"}
            priced = [order for order in own if order.price is not None]
            priced_qty = sum(order.remaining for order in priced)
            feats += [self.risk.outstanding(self.sim) / self.total_qty,
                      self.risk.available(self.sim, self.total_qty - self.remaining) / self.total_qty,
                      sum(o.remaining for o in own if o.status.value in resting) / self.total_qty,
                      sum(o.remaining for o in own if o.status.value == "CANCEL_PENDING") / self.total_qty,
                      sum((o.price - mid) * o.remaining for o in priced) / priced_qty if priced_qty else 0.0,
                      max((self.sim.t - o.ts_submit for o in own), default=0.0) / self.horizon,
                      float(self.risk.halted),
                      float((book.best_bid() if self.side is Side.SELL else book.best_ask()) is not None)]
            raw = np.asarray(feats, dtype=np.float64)
            if not np.isfinite(raw).all():
                raise ValueError("nonfinite current-state observation")
            return self.normalization.apply(raw) if self.normalization is not None else raw.astype(np.float32)
        return np.asarray(feats, dtype=np.float32)

    def _info(self) -> Dict[str, Any]:
        return {"remaining": self.remaining,
                "market_seed": self.market_seed, "reward_terms": dict(self.reward_terms),
                "filled": self.total_qty - self.remaining,
                "t": self._decision_duration if self._decision_duration is not None else self.sim.t - self.t0,
                "exchange_time": self.sim.t - self.t0,
                "decision_horizon": self.horizon,
                "settlement_duration": self.settlement.duration if self.settlement is not None else 0.0,
                "settlement_complete": self.settlement.complete if self.settlement is not None else None,
                "late_filled_qty": self.late_filled_qty, "late_fees": self.late_fees,
                "reserved_qty": self.risk.outstanding(self.sim),
                "total_fees": self.ledger.total_fees,
                "cash": self.ledger.cash, "inventory": self.ledger.inventory,
                "risk_halted": self.risk.halted,
                **({"status": self.report().status,
                    "unpriced_leftover_qty": self.report().unpriced_leftover_qty,
                    "optimization_cost_bps": self.report().optimization_cost_bps} if self._done else {})}

    def _snap(self) -> None:
        bids, asks = self.sim.book.depth(10)
        self.snapshots.append({"t": self.sim.t - self.t0, "bids": bids, "asks": asks,
                               "mid": self.sim.book.mid(), "remaining": self.remaining})

    def report(self, label: str = "RL Agent (PPO)") -> ExecutionReport:
        liq, fillable = self.sim.book.walk_cost(self.side, self.remaining) if self.remaining else (None, 0)
        return build_report(label, self.side, self.total_qty, self.fills, self.arrival,
                            self.sim.cfg.tick_size, self.n_children,
                            self._decision_duration if self._decision_duration is not None else self.sim.t - self.t0, liq,
                            leftover_fillable_qty=fillable, fees=self.fees,
                            total_fees=self.ledger.total_fees, terminal_penalty_bps=self.terminal_penalty_bps,
                            strict_terminal=True, outstanding_qty=self.risk.outstanding(self.sim),
                            accounting=asdict(self.ledger.snapshot(self.sim.book.mid() * self.sim.cfg.tick_size)),
                            risk_rejections=sum(e["event"] == "rejected" for e in self.risk.events),
                            decision_horizon=self.horizon,
                            settlement_duration=self.settlement.duration if self.settlement is not None else 0.0,
                            settlement_complete=self.settlement.complete if self.settlement is not None else None,
                            late_filled_qty=self.late_filled_qty, late_fees=self.late_fees,
                            settlement_pending_order_ids=self.settlement.pending_order_ids if self.settlement is not None else ())
