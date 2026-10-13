"""Strategic multi-agent ecology with endogenous response (F6, M50; H11, H12).

Ten behavioural-proxy agents contribute event rates to the six marks. They are *model classes*, not claims about
the identity of real participants:

* ``market_maker`` - quotes at the touch, more when the spread is wide; pulls quotes on the side the imbalance leans
  against;
* ``inventory_maker`` - like a maker but skews quoting against its inventory (inventory accumulates from the market
  flow it absorbs);
* ``liquidity_taker`` - symmetric market orders;
* ``noise`` - small symmetric adds, cancels and market orders;
* ``momentum`` - market orders in the direction of the recent signed flow (EMA, half-life ~3 s);
* ``mean_reversion`` - market orders against the recent mid drift;
* ``large_execution`` - a persistent one-sided metaorder (sign and intensity seeded per episode);
* ``liquidity_seeker`` - market orders that grow when the opposite touch is deep;
* ``toxicity_proxy`` - informed-flow proxy: market orders in the direction of a latent drift that also moves the
  queue-cancel rates of makers on the exposed side;
* ``latency_sensitive`` - reacts within the next bin to *own* (strategy) aggressive flow by cancelling quotes on the
  hit side and adding behind it (endogenous response).

Endogenous response: the strategy's own market flow (passed by the simulator as ``own``) enters the momentum signal,
the inventory of the makers, the toxicity drift and the latency-sensitive reaction, so a large execution changes
future liquidity (self-impact). ``static`` ignores own flow (exogenous control for H11/H12).

Fit: method of moments - one scale for market flow and one for limit flow so the mean rates match the training
frame (the reaction strengths are registered constants, not tuned to outcomes).
"""
from __future__ import annotations

import math

import numpy as np

from ...v08.generators.base import K, MarketGenerator
from ...v08.market.frame import EventFrame

AGENTS = ("market_maker", "inventory_maker", "liquidity_taker", "noise", "momentum", "mean_reversion",
          "large_execution", "liquidity_seeker", "toxicity_proxy", "latency_sensitive")
REACTION = {"momentum_gain": 0.6, "inventory_skew": 0.5, "toxicity_gain": 0.5, "latency_cancel": 1.5,
            "flow_halflife_bins": 30.0, "drift_persistence": 0.995, "drift_volatility": 0.03}


class StrategicEcology(MarketGenerator):
    name = "strategic_ecology"
    version = "1"
    capabilities = frozenset({"count_process", "agent_ecology", "endogenous_response", "state_dependent"})

    @classmethod
    def default_params(cls) -> dict:
        return {"agents": {"market_maker": 1.4, "inventory_maker": 0.6, "liquidity_taker": 0.4, "noise": 0.8,
                           "momentum": 0.15, "mean_reversion": 0.1, "large_execution": 0.08,
                           "liquidity_seeker": 0.1, "toxicity_proxy": 0.1, "latency_sensitive": 0.6},
                "reaction": dict(REACTION), "static": False, "market_scale": 1.0, "limit_scale": 1.0}

    def initial_state(self, simulate: bool = False):
        return {"flow": 0.0, "own": 0.0, "inventory": 0.0, "drift": 0.0, "meta_sign": 1.0, "simulate": simulate,
                "mid_drift": 0.0, "react": np.zeros(2)}

    def _rates(self, s, x) -> np.ndarray:
        p, r = self.params["agents"], self.params["reaction"]
        spread, imbalance, _, bq, aq, ret10 = (float(v) for v in x)
        out = np.zeros(K)
        mm = p["market_maker"] * (1.0 + 0.8 * max(spread - 1.0, 0.0))
        out[2] += mm
        out[3] += mm
        out[4] += p["market_maker"] * 0.9 * (1.0 + max(-imbalance, 0.0))
        out[5] += p["market_maker"] * 0.9 * (1.0 + max(imbalance, 0.0))
        skew = float(np.tanh(r["inventory_skew"] * s["inventory"]))
        out[2] += p["inventory_maker"] * (1.0 - skew)
        out[3] += p["inventory_maker"] * (1.0 + skew)
        out[4] += p["inventory_maker"] * 0.8 * (1.0 + skew)
        out[5] += p["inventory_maker"] * 0.8 * (1.0 - skew)
        out[0] += 0.5 * p["liquidity_taker"]
        out[1] += 0.5 * p["liquidity_taker"]
        out += p["noise"] * np.asarray([0.1, 0.1, 0.45, 0.45, 0.4, 0.4])
        mom = float(np.tanh(r["momentum_gain"] * s["flow"]))
        out[0] += p["momentum"] * max(mom, 0.0)
        out[1] += p["momentum"] * max(-mom, 0.0)
        rev = float(np.tanh(s["mid_drift"]))
        out[0] += p["mean_reversion"] * max(-rev, 0.0)
        out[1] += p["mean_reversion"] * max(rev, 0.0)
        out[0 if s["meta_sign"] > 0 else 1] += p["large_execution"]
        depth_ratio = math.log((aq + 1.0) / (bq + 1.0))
        out[0] += p["liquidity_seeker"] * max(depth_ratio, 0.0)
        out[1] += p["liquidity_seeker"] * max(-depth_ratio, 0.0)
        tox = float(np.tanh(s["drift"] + r["toxicity_gain"] * s["own"]))
        out[0] += p["toxicity_proxy"] * max(tox, 0.0)
        out[1] += p["toxicity_proxy"] * max(-tox, 0.0)
        out[5] += p["toxicity_proxy"] * max(tox, 0.0)          # exposed asks are pulled when flow is toxic upward
        out[4] += p["toxicity_proxy"] * max(-tox, 0.0)
        ls = p["latency_sensitive"]
        out[2] += ls * 0.5
        out[3] += ls * 0.5
        out[4] += ls * (0.45 + r["latency_cancel"] * s["react"][1])     # own selling hits bids -> pull bids
        out[5] += ls * (0.45 + r["latency_cancel"] * s["react"][0])     # own buying lifts asks -> pull asks
        scale = np.asarray([self.params["market_scale"]] * 2 + [self.params["limit_scale"]] * 4)
        return np.clip(out * scale, 0.0, None)

    def expected(self, state, x) -> np.ndarray:
        return self._rates(state, np.asarray(x, float))

    def draw(self, state, x, rng) -> np.ndarray:
        r = self.params["reaction"]
        if state["simulate"]:
            state["drift"] = float(np.clip(r["drift_persistence"] * state["drift"]
                                           + r["drift_volatility"] * rng.standard_normal(), -3, 3))
            if "meta_drawn" not in state:
                state["meta_sign"] = 1.0 if rng.random() < 0.5 else -1.0
                state["meta_drawn"] = True
        return rng.poisson(np.clip(self.expected(state, x), 0.0, 60.0))

    def update(self, state, realized, x, own=None):
        r = self.params["reaction"]
        a = 0.5 ** (1.0 / r["flow_halflife_bins"])
        realized = np.asarray(realized, float)
        own = np.zeros(2) if own is None or self.params["static"] else np.asarray(own, float)
        scale = max(float(x[3] + x[4]) / 2.0, 1.0)                     # touch depth scale
        own_net = (own[0] - own[1]) / scale
        net = float(realized[0] - realized[1])
        state["flow"] = a * state["flow"] + (1 - a) * (net + 5.0 * own_net)
        state["own"] = a * state["own"] + (1 - a) * 5.0 * own_net
        state["inventory"] = 0.999 * state["inventory"] - 0.05 * (net + 5.0 * own_net)
        state["mid_drift"] = 0.95 * state["mid_drift"] + 0.05 * float(np.sign(net))
        state["react"] = np.clip(own / scale, 0.0, 5.0)
        return state

    def fit(self, frame: EventFrame) -> StrategicEcology:
        v = frame.valid
        target = frame.counts[v].mean(0)
        sample = frame.state[v][:: max(1, int(v.sum() // 2000))]
        base = StrategicEcology({**self.params, "market_scale": 1.0, "limit_scale": 1.0}, self.book)
        model = np.mean([base.expected(base.initial_state(), x) for x in sample], 0)
        ratio = lambda a, b: float(a / b) if b > 0 else 1.0  # noqa: E731
        out = StrategicEcology({**self.params, "market_scale": ratio(target[:2].sum(), model[:2].sum()),
                                "limit_scale": ratio(target[2:].sum(), model[2:].sum())}, self.book)
        out.fit_info = {"method": "method of moments (market and limit-flow scales); reactions registered"}
        return out

    def exogenous(self) -> StrategicEcology:
        out = StrategicEcology({**self.params, "static": True}, self.book)
        out.fit_info = dict(self.fit_info)
        return out


def describe_agents() -> dict:
    return {a: "behavioural proxy / model class (not a real participant identity)" for a in AGENTS}
