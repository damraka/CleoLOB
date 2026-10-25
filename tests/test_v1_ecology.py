"""v1 multi-agent ecology on the order-level engine (M100-M104)."""
import numpy as np

from lob.v1.ecology import agents as AG
from lob.v1.ecology.runtime import Ecology
from lob.v1.realism import metrics as MT


def test_runtime_produces_valid_frames_and_trades():
    out = Ecology(AG.default_ecology(), seed=1).run(120.0)
    fr = out["frame"]
    assert fr.valid.mean() > 0.95 and out["engine"]["fills"] > 50
    assert set(MT.frame_metrics([fr])) == set(MT.METRICS)
    a = Ecology(AG.default_ecology(), seed=1).run(60.0)["frame"]
    b = Ecology(AG.default_ecology(), seed=1).run(60.0)["frame"]
    assert np.array_equal(a.bid_px, b.bid_px) and np.array_equal(a.counts, b.counts)   # seeded and deterministic


def test_market_maker_inventory_skew_and_toxicity_widening():
    mm = AG.MarketMaker("mm", 1.0, {"half_spread": 1, "gamma": 0.5, "size": 5})
    view = {"bid": 99, "ask": 101, "bids": [(99, 5)], "asks": [(101, 5)]}
    long = mm.act(0.0, view, AG.AgentContext({}, 10, [], []), np.random.default_rng(0))
    short = mm.act(0.0, view, AG.AgentContext({}, -10, [], []), np.random.default_rng(0))

    def px(cmds, s):
        return [c[1]["price"] for c in cmds if c[0] == "submit" and c[1]["side"] == s][0]
    assert px(long, 1) < px(short, 1) and px(long, -1) < px(short, -1)          # long inventory skews quotes down
    toxic = [(0.0, 101, 50, 1)] * 5
    wide = mm.act(1.0, view, AG.AgentContext({}, 0, toxic, []), np.random.default_rng(0))
    calm = mm.act(1.0, view, AG.AgentContext({}, 0, [], []), np.random.default_rng(0))
    assert px(wide, -1) - px(wide, 1) > px(calm, -1) - px(calm, 1)


def test_proxy_agents_follow_rules():
    mom = AG.Momentum("m", 1.0, {"lookback": 5.0, "threshold": 1.0, "size": 2})
    ctx = AG.AgentContext({}, 0, [], [(0.0, 100.0)])
    up = {"bid": 102, "ask": 104, "bids": [], "asks": []}
    assert mom.act(10.0, up, ctx, np.random.default_rng(0))[0][1]["side"] == 1
    mr = AG.MeanReversion("r", 1.0, {"lookback": 5.0, "threshold": 1.0, "size": 2})
    assert mr.act(10.0, up, ctx, np.random.default_rng(0))[0][1]["side"] == -1


def test_strategic_cancel_and_feedback_to_strategy_trades():
    lp = AG.LiquidityProvider("lp", 1.0, {"adverse": 0.3, "window": 3.0})
    view = {"bid": 99, "ask": 101, "bids": [(99, 20)], "asks": [(101, 20)]}
    sells = [(9.0, 99, 30, -1)] * 3
    cmds = lp.act(10.0, view, AG.AgentContext({7: (1, 99, 5)}, 0, sells, []), np.random.default_rng(0))
    assert ("cancel", 7) in cmds                                               # adverse flow cancels the resting bid
    calm = Ecology(AG.default_ecology(), seed=4).run(90.0)
    pushed = Ecology(AG.default_ecology(), seed=4)

    def buyer(t, view, ctx):
        return [("submit", {"side": 1, "qty": 40, "kind": "market"})] if 20 <= t < 40 else []
    hot = pushed.run(90.0, strategy=buyer, decision_s=1.0)
    assert np.nanmean(hot["frame"].mid[-100:]) > np.nanmean(calm["frame"].mid[-100:])   # self-impact moves the market
