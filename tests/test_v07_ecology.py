"""v0.7 multi-agent ecology, strategic interaction, market-maker metrics, cross-impact and portfolio execution."""
from __future__ import annotations

import numpy as np
import pytest

from lob.engine import Side
from lob.sim_v2 import SimulatorSpec
from lob.v07.execution import ecology as ec
from lob.v07.execution import portfolio as pf

CONFIG = {"tick_size": 100 / 2640, "initial_mid_ticks": 2640, "target_level_vol": 26, "limit_rate": 2.9,
          "market_rate": 0.14, "cancel_rate": 0.03, "offset_p": 0.05, "limit_qty_mean": 35, "market_qty_mean": 40,
          "resilience": 0.2, "max_events": 3_000_000}
WORLD = SimulatorSpec(CONFIG, {})


def _agents():
    return [ec.LiquidityProvider("LP", 1), ec.Taker("TK", 2), ec.NoiseTrader("NT", 3), ec.InventoryMaker("MM", 4),
            ec.MetaOrderTrader("META", 5)]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ecology_conserves_inventory_and_cash_and_stays_valid(seed) -> None:
    result = ec.Ecology(WORLD, _agents()).run(seed, 60.0)
    accounts = result["accounts"]
    assert sum(v[0] for v in accounts.values()) == 0
    assert sum(v[1] for v in accounts.values()) == pytest.approx(0.0)
    assert {"LP", "TK", "MM", "META"} & set(accounts)


def test_market_maker_metrics_and_reactive_interaction() -> None:
    result = ec.Ecology(WORLD, _agents()).run(4, 60.0)
    m = ec.maker_metrics(result, "MM")
    assert m["net_after_penalty"] == pytest.approx(m["mark_to_market_ticks"] - m["inventory_penalty"])
    costs = {}
    for name, maker in (("passive", lambda: ec.LiquidityProvider("MK", 7)), ("reactive", lambda: ec.ReactiveMaker("MK", 7))):
        values = []
        for s in range(4):
            r = ec.Ecology(WORLD, [maker(), ec.MetaOrderTrader("META", 5, quantity=200, slices=10)]).run(s, 20.0)
            values.append(r["accounts"].get("META", (0, 0.0)))
        costs[name] = values
    assert len(costs["passive"]) == len(costs["reactive"]) == 4


def test_zero_cross_impact_baseline_and_synthetic_coupling() -> None:
    def run(coupling, trade):
        m = pf.CoupledMarkets({"A": WORLD, "B": WORLD}, coupling, seed=11)
        m.step(10.0)
        if trade:
            for _ in range(5):
                m.submit("A", Side.BUY, 30, "S")
                m.step(1.0)
        else:
            m.step(5.0)
        return m.sims["B"].book.mid(), sum(t.taker_owner == "COUPLED" for t in m.sims["B"].book.trades)
    zero = np.zeros((2, 2))
    assert run(zero, True) == run(zero, False)                     # B unaffected without coupling
    coupled = run(np.array([[0, 1.0], [1.0, 0]]), True)
    assert coupled[1] > 0


def test_portfolio_accounting_and_limits() -> None:
    m = pf.CoupledMarkets({"A": WORLD, "B": WORLD}, np.zeros((2, 2)), seed=3)
    m.step(10.0)
    run = pf.PortfolioExecution(m, [pf.Parent("A", Side.BUY, 40), pf.Parent("B", Side.SELL, 40)], horizon=20.0,
                                slices=10, risk_limit=6).run()
    assert run["identity_error"] < 1e-6
    assert run["portfolio_shortfall_ticks"] == pytest.approx(sum(v["shortfall_ticks"] for v in run["per_instrument"].values()))
    m2 = pf.CoupledMarkets({"A": WORLD, "B": WORLD}, np.zeros((2, 2)), seed=3)
    m2.step(10.0)
    capped = pf.PortfolioExecution(m2, [pf.Parent("A", Side.BUY, 40), pf.Parent("B", Side.BUY, 40)], horizon=20.0,
                                   slices=10, capital_limit=3 * 2640).run()
    assert capped["limit_binding_steps"]["capital"] > 0
