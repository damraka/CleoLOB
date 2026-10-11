"""v0.8 execution: completion accounting, fees, latency, stale books, cancel races, policies, MPC."""
from __future__ import annotations

import math

import numpy as np
import pytest

from lob.v07.exchange.latency import LatencyModel
from lob.v08.execution import mpc  # noqa: F401  (registers MPC)
from lob.v08.execution import policies as P
from lob.v08.execution.simulator import (SEPARATE_FIELDS, Episode, Fees, LatencyProfile, Mandate, cancel_all, limit,
                                         market, run_episode)
from lob.v08.generators import PoissonGenerator
from lob.v08.market.book import BUY
from lob.v08.market.sim import BookModel
from lob.v08.worlds import World

BOOK = BookModel(refill_mean=8.0)


def world(market=1.0, add=4.0, cancel=3.8, buy_share=0.5, book=BOOK, latency=None) -> World:
    g = PoissonGenerator({"market": market, "add": add, "cancel": cancel, "market_buy_share": buy_share,
                          "add_bid_share": 0.5, "cancel_bid_share": 0.5}, book)
    return World("w", g, book, latency)


STATIC = world(1e-9, 1e-9, 1e-9)          # no background flow: the book never changes by itself


class Fixed(P.Policy):
    """Test policy: replays a fixed list of actions per decision step."""

    name, label = "fixed_test", "fixed (test)"

    def __init__(self, plan=None):
        super().__init__()
        self.plan = plan or {}

    def decide(self, obs):
        return self.plan.get(obs.step, [])


def run(w, policy, mandate=None, seed=1, latency=None, context=None):
    return Episode(w, policy, mandate or Mandate(qty=20), seed, latency=latency, context=context).run()


def test_full_completion_has_no_residual_terms() -> None:
    r = run(world(), P.make("twap"))
    assert r["actual_completion"] and r["terminal_inventory"] == 0 and r["fill_frac"] == 1.0
    assert r["hypothetical_residual_cost_bps"] == 0.0 and r["completion_penalty_bps"] == 0.0
    assert r["optimization_cost_bps"] == pytest.approx(r["economic_cost_bps"])
    assert r["economic_cost_bps"] == pytest.approx(r["gross_cost_bps"] + r["fee_bps"])


def test_zero_completion_keeps_actual_fill_and_hypothetical_liquidation_apart() -> None:
    r = run(world(), Fixed())
    assert r["actual_filled_qty"] == 0 and not r["actual_completion"] and r["terminal_inventory"] == 20
    assert r["implementation_shortfall_bps"] is None and r["gross_cost_bps"] == 0.0 and r["fee_bps"] == 0.0
    # the hypothetical liquidation is priced (it can be negative after a favourable drift) but is not a fill
    assert math.isfinite(r["hypothetical_residual_cost_bps"]) and r["completion_penalty_bps"] == pytest.approx(25.0)
    assert r["optimization_cost_bps"] == pytest.approx(
        r["economic_cost_bps"] + r["hypothetical_residual_cost_bps"] + r["completion_penalty_bps"])


def test_partial_completion_penalty_is_proportional() -> None:
    r = run(STATIC, Fixed({0: [market(5)]}), Mandate(qty=20, penalty_bps=40.0))
    assert r["actual_filled_qty"] == 5 and r["terminal_inventory"] == 15
    assert r["completion_penalty_bps"] == pytest.approx(40.0 * 15 / 20)
    assert r["fill_frac"] == pytest.approx(0.25)


def test_market_order_cost_equals_manual_book_walk_and_taker_fee() -> None:
    fees = Fees(taker_bps=2.0, maker_bps=0.0)
    e = Episode(STATIC, Fixed({0: [market(12)]}), Mandate(qty=12, fees=fees), seed=3)
    r = e.run()
    fills = e.fills
    arrival = r["arrival_mid_ticks"]
    notional = sum(f[1] * f[2] for f in fills)
    assert r["gross_cost_bps"] == pytest.approx((notional - arrival * 12) / (arrival * 12) * 1e4)
    assert r["fee_bps"] == pytest.approx(notional * 2e-4 / (arrival * 12) * 1e4)
    assert r["taker_filled_qty"] == 12 and r["maker_filled_qty"] == 0 and r["rebate_bps"] == 0.0


def test_maker_rebate_is_reported_as_rebate_not_netted_silently() -> None:
    sells = world(market=3.0, add=1e-9, cancel=1e-9, buy_share=0.0)       # only market sells hit our bid
    fees = Fees(taker_bps=1.0, maker_bps=-0.5)
    r = run(sells, Fixed({0: [limit(10)]}), Mandate(qty=10, fees=fees, horizon_s=30.0, decision_dt_s=30.0))
    assert r["maker_filled_qty"] > 0 and r["taker_filled_qty"] == 0
    assert r["rebate_bps"] > 0 and r["fee_bps"] == pytest.approx(-r["rebate_bps"])


def test_per_lot_fee_is_added() -> None:
    a = run(STATIC, Fixed({0: [market(10)]}), Mandate(qty=10, fees=Fees(0.0, 0.0, 0.0)))
    b = run(STATIC, Fixed({0: [market(10)]}), Mandate(qty=10, fees=Fees(0.0, 0.0, 1.0)))
    assert b["fee_bps"] - a["fee_bps"] == pytest.approx(10 * 1.0 / (a["arrival_mid_ticks"] * 10) * 1e4)
    assert a["gross_cost_bps"] == pytest.approx(b["gross_cost_bps"])


def test_unpriceable_residual_is_flagged_not_invented() -> None:
    r = run(STATIC, Fixed(), Mandate(qty=100_000))
    assert r["residual_unpriceable"] and r["hypothetical_residual_cost_bps"] is None
    assert r["optimization_cost_bps"] is None and r["completion_penalty_bps"] == pytest.approx(25.0)


def test_separate_fields_present() -> None:
    r = run(world(), P.make("heuristic"))
    assert set(SEPARATE_FIELDS) <= set(r)


def test_feed_latency_means_decisions_see_old_snapshots() -> None:
    seen = []

    class Spy(P.Policy):
        name, label = "spy_test", "spy"

        def decide(self, obs):
            seen.append((obs.t, obs.observed_at))
            return []

    slow = LatencyProfile(feed=LatencyModel("fixed", 1.0, 0.0))
    run(world(), Spy(), latency=slow)
    assert all(t - at >= 1.0 - 1e-9 for t, at in seen)
    seen.clear()
    run(world(), Spy())
    assert all(abs(t - at) < 1e-9 for t, at in seen)


def test_stale_book_execution_costs_show_in_latency_proxy() -> None:
    w = world(market=2.0)
    fast = [run(w, P.make("twap"), seed=s)["latency_cost_proxy_bps"] for s in range(4)]
    slow_profile = LatencyProfile(feed=LatencyModel("fixed", 2.0, 0.0), send=LatencyModel("fixed", 0.5, 0.0))
    slow = [run(w, P.make("twap"), seed=s, latency=slow_profile)["latency_cost_proxy_bps"] for s in range(4)]
    assert np.mean(np.abs(slow)) > np.mean(np.abs(fast))


def test_cancel_races_can_be_lost_to_fills() -> None:
    sells = world(market=8.0, add=1e-9, cancel=1e-9, buy_share=0.0)
    slow_cancel = LatencyProfile(cancel=LatencyModel("fixed", 2.0, 0.0))
    r = run(sells, Fixed({0: [limit(10), cancel_all()]}), Mandate(qty=10, horizon_s=10.0, decision_dt_s=5.0),
            latency=slow_cancel)
    assert r["cancel_races_lost"] >= 1 and r["actual_filled_qty"] == 10


def test_hard_completion_buys_the_remainder_after_cancels_confirm() -> None:
    w = world(market=0.2)
    lag = LatencyProfile(send=LatencyModel("fixed", 0.05, 0.0), ack=LatencyModel("fixed", 0.05, 0.0),
                         cancel=LatencyModel("fixed", 0.05, 0.0))
    passive = Fixed({0: [limit(20)]})
    soft = run(w, passive, Mandate(qty=20, completion="none"), latency=lag)
    hard = run(w, Fixed({0: [limit(20)]}), Mandate(qty=20, completion="hard"), latency=lag)
    assert not soft["actual_completion"]
    assert hard["actual_completion"] and hard["post_horizon_filled_qty"] > 0 and hard["overfill_qty"] == 0


def test_urgency_completion_switches_to_marketable_orders() -> None:
    r = run(world(market=0.2), Fixed({0: [limit(20)]}), Mandate(qty=20, completion="urgency", urgency_fraction=0.5))
    assert r["actual_completion"] and r["within_horizon_completion"]


def test_participation_cap_limits_market_children() -> None:
    capped = run(world(), P.make("twap"), Mandate(qty=200, participation_cap=0.05))
    free = run(world(), P.make("twap"), Mandate(qty=200))
    assert capped["actual_filled_qty"] < free["actual_filled_qty"]


@pytest.mark.parametrize("name", sorted(P.REGISTRY))
def test_every_registered_policy_runs_deterministically(name) -> None:
    w = world()
    a = run_episode(w, P.make(name), Mandate(qty=40), 5)
    b = run_episode(w, P.make(name), Mandate(qty=40), 5)
    assert a["status"] == "VALID" and a == b
    assert a["overfill_qty"] == 0
    assert P.make(name).describe()["requirements"]["completion_behaviour"] in P.COMPLETION_BEHAVIOURS


def test_capability_gate_fails_closed() -> None:
    class NeedsQueue(P.Policy):
        name = "needs_queue_test"
        requirements = P.PolicyRequirements(queue_identity=True)

    with pytest.raises(P.CapabilityError):
        P.check_capabilities(NeedsQueue(), {"aggregate_depth", "trade_prints"})
    P.check_capabilities(NeedsQueue(), {"aggregate_depth", "order_identity"})

    class Undelayed(P.Policy):
        name = "undelayed_test"
        requirements = P.PolicyRequirements(delayed_observations=False)

    with pytest.raises(P.CapabilityError):
        P.check_capabilities(Undelayed(), {"aggregate_depth"}, delayed=True)
    with pytest.raises(P.CapabilityError):
        P.check_capabilities(P.make("pov"), {"aggregate_depth"})


def test_unknown_policy_parameters_are_refused() -> None:
    with pytest.raises(ValueError):
        P.make("twap", speed=3)


def test_ac_fraction_shapes() -> None:
    assert P.ac_fraction(0.5, 0.0) == pytest.approx(0.5)
    assert P.ac_fraction(0.5, 3.0) > 0.5
    assert P.ac_fraction(1.0, 3.0) == pytest.approx(1.0)


def test_invalid_rows_are_retained() -> None:
    class Broken(P.Policy):
        name = "broken_test"

        def decide(self, obs):
            raise ValueError("boom")

    row = run_episode(world(), Broken(), Mandate(qty=10), 1)
    assert row["status"] == "INVALID" and "boom" in row["error"] and row["policy"] == "broken_test"


def test_mandate_validation() -> None:
    with pytest.raises(ValueError):
        Mandate(completion="sometimes")
    with pytest.raises(ValueError):
        Mandate(qty=0)
    assert Mandate(horizon_s=120, decision_dt_s=5).steps == 24
    assert math.isclose(Mandate().replace(qty=7).qty, 7)
    assert BUY == Mandate().side
