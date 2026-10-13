"""Deterministic builder of the frozen v0.9 configuration (M1, M10, M11, M12, M2 candidates).

``build(root)`` writes, once and before the freeze:

* ``configs/v09/protocol.json`` - research question, vocabularies, pools and datasets, holdout and acquisition rules,
  seed namespaces, promotion and power rules, statistics, mandates, policies, the execution conclusions carried
  through the reality ladder, the capability-specific realism gate, counterfactual and causal assumptions, resource
  ceilings, failure semantics and evidence schemas;
* ``configs/v09/hypotheses-candidates.json`` - H1-H36 with preregistered endpoints and practical thresholds;
* ``configs/v09/statistical-families.json``, ``dataset-registry.json`` (registry v4), ``capability-matrix.json``,
  ``model-capabilities.json``, ``compute-budget.json``, ``claim-registry.json``.

Day lists are drawn from each pool's calendar range by a registered ``SeedSequence`` permutation before any data are
read; slots beyond the registered count are ordered backups. Re-running the builder reproduces the same bytes.
"""
from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np

from ..preregistration import ProtocolError, document_sha256
from . import protocol as pr

PROBE_DATE = "2026-10-13"
BYBIT_L2 = "https://quote-saver.bycsi.com/orderbook/{segment}/{sym}/{day}_{sym}_ob{depth}.data.zip"
BYBIT_TRADES = {"linear": "https://public.bybit.com/trading/{sym}/{sym}{day}.csv.gz",
                "spot": "https://public.bybit.com/spot/{sym}/{sym}_{day}.csv.gz"}
BINANCE = "https://data.binance.vision/data/futures/um/daily/{kind}/{sym}/{sym}-{kind}-{day}.zip"

# venue/segment -> expected instrument metadata (tick, lot, multiplier); the quality engine verifies the observed
# price/size grid at access and records any difference as a known anomaly (normalization uses the observed grid)
INSTRUMENTS = {
    ("bybit", "linear", "BTCUSDT"): {"tick": 0.1, "lot": 0.001, "multiplier": 1.0, "type": "linear_perpetual"},
    ("bybit", "linear", "ETHUSDT"): {"tick": 0.01, "lot": 0.01, "multiplier": 1.0, "type": "linear_perpetual"},
    ("bybit", "linear", "SOLUSDT"): {"tick": 0.01, "lot": 0.1, "multiplier": 1.0, "type": "linear_perpetual"},
    ("bybit", "spot", "BTCUSDT"): {"tick": 0.01, "lot": 0.000001, "multiplier": 1.0, "type": "spot"},
    ("binance", "um", "BTCUSDT"): {"tick": 0.1, "lot": 0.001, "multiplier": 1.0, "type": "linear_perpetual"},
    ("binance", "um", "ETHUSDT"): {"tick": 0.01, "lot": 0.001, "multiplier": 1.0, "type": "linear_perpetual"},
}

# pool -> role, calendar range, streams, registered count and backups (fixed before any access)
POOLS = {
    "dev": {"role": "development", "start": "2023-07-15", "end": "2023-07-15", "count": 1, "backups": 0,
            "streams": [("bybit", "linear", "BTCUSDT")]},
    "explore": {"role": "explore", "start": "2023-08-01", "end": "2023-10-31", "count": 8, "backups": 2,
                "streams": [("bybit", "linear", "BTCUSDT"), ("bybit", "linear", "ETHUSDT"),
                            ("binance", "um", "BTCUSDT"), ("binance", "um", "ETHUSDT")]},
    "confirm": {"role": "confirm", "start": "2023-11-01", "end": "2024-03-28", "count": 40, "backups": 6,
                "streams": [("bybit", "linear", "BTCUSDT"), ("bybit", "linear", "ETHUSDT"),
                            ("binance", "um", "BTCUSDT"), ("binance", "um", "ETHUSDT")]},
    "confirm_late": {"role": "confirm_late", "start": "2024-06-01", "end": "2024-12-31", "count": 20, "backups": 4,
                     "streams": [("bybit", "linear", "BTCUSDT"), ("bybit", "linear", "ETHUSDT")]},
    "final": {"role": "final", "start": "2025-05-01", "end": "2025-08-28", "count": 12, "backups": 3,
              "streams": [("bybit", "linear", "SOLUSDT"), ("bybit", "linear", "BTCUSDT"),
                          ("bybit", "spot", "BTCUSDT")]},
}
POOL_SEED = 909_013


def pool_days(name: str) -> list[str]:
    """Registered ordered day list of a pool: count slots then backups (SeedSequence permutation)."""
    p = POOLS[name]
    start, end = date.fromisoformat(p["start"]), date.fromisoformat(p["end"])
    days = [(start + timedelta(k)).isoformat() for k in range((end - start).days + 1)]
    tag = int(document_sha256(name)[:12], 16)
    order = np.random.default_rng(np.random.SeedSequence([POOL_SEED, tag])).permutation(len(days))
    return [days[i] for i in order[: p["count"] + p["backups"]]]


def _sources(venue: str, segment: str, sym: str, day: str) -> dict:
    if venue == "bybit":
        depth = 500 if segment == "linear" else 200
        return {"l2": {"url": BYBIT_L2.format(segment=segment, sym=sym, day=day, depth=depth), "format": f"bybit-ob{depth}"},
                "trades": {"url": BYBIT_TRADES[segment].format(sym=sym, day=day), "format": f"bybit-trades-{segment}"}}
    return {"l1": {"url": BINANCE.format(kind="bookTicker", sym=sym, day=day), "format": "binance-bookticker",
                   "checksum_url": BINANCE.format(kind="bookTicker", sym=sym, day=day) + ".CHECKSUM"},
            "trades": {"url": BINANCE.format(kind="trades", sym=sym, day=day), "format": "binance-trades",
                       "checksum_url": BINANCE.format(kind="trades", sym=sym, day=day) + ".CHECKSUM"}}


def _capabilities(venue: str, segment: str) -> dict:
    l2 = venue == "bybit"
    return {"exchange_timestamps": True, "receive_timestamps": False, "timestamp_resolution": "1ms",
            "sequence_numbers": True, "order_ids": False, "trade_ids": True, "aggressor_side": True,
            "snapshot_support": l2, "incremental_l2": l2, "l1_quotes": True, "market_by_price": l2,
            "market_by_order": False, "cancel_visibility": "inferred_from_depth_minus_trades" if l2 else "none",
            "modify_visibility": "none", "hidden_liquidity_observability": "none",
            "depth_levels": (500 if segment == "linear" else 200) if l2 else 1,
            "queue_clearance_observable": l2}


def datasets() -> list[dict]:
    out = []
    for name, p in POOLS.items():
        days = pool_days(name)
        for venue, segment, sym in p["streams"]:
            meta = INSTRUMENTS[(venue, segment, sym)]
            for slot, day in enumerate(days):
                out.append({
                    "id": f"{venue}-{segment}-{sym.lower()}-{day}", "provider": f"{venue} public data archive",
                    "venue": f"{venue}-{segment}", "instrument": sym, "instrument_type": meta["type"], "date": day,
                    "data_types": (["l2_snapshot_delta", "trades"] if venue == "bybit" else ["l1_bookticker", "trades"]),
                    "role": p["role"], "pool": name, "slot": slot,
                    "slot_kind": "primary" if slot < p["count"] else "backup", "freshness_at_freeze": "fresh",
                    "sources": _sources(venue, segment, sym, day), "tick_size": meta["tick"], "lot_size": meta["lot"],
                    "contract_multiplier": meta["multiplier"], "capabilities": _capabilities(venue, segment)})
    return out


# ============================================================================= hypotheses


def _h(title, question, domain, endpoint, estimator, unit, direction, threshold, units, family, fidelity, caps,
       level, roles, pool, max_units, pilot, *, test="superiority", headline=False, **extra) -> dict:
    rule = {"superiority": "ESTABLISHED iff the Holm-adjusted one-sided lower confidence bound of the oriented effect "
                           "is > 0 and the point estimate >= the practical threshold; FAILED iff the upper bound is < 0 "
                           "(the opposite direction is determinate); otherwise NOT_ESTABLISHED",
            "noninferiority": "ESTABLISHED iff the Holm-adjusted lower confidence bound of the oriented effect is > "
                              "-margin; FAILED iff the upper bound is < -margin; otherwise NOT_ESTABLISHED",
            "agreement": "ESTABLISHED iff the Holm-adjusted lower bound of (agreement - agreement_level) is > 0 and "
                         "the point estimate >= agreement_level + threshold; FAILED iff the upper bound is < 0",
            "transfer": "ESTABLISHED iff at least one carried conclusion meets its own registered rule on the "
                        "target pool after Holm over the carried conclusions; FAILED iff every carried conclusion is "
                        "determinate in the opposite direction; NOT_AVAILABLE iff no conclusion was carried or the "
                        "target data are unavailable; otherwise NOT_ESTABLISHED"}[test]
    return {"title": title, "question": question, "domain": domain, "endpoint": endpoint, "estimator": estimator,
            "unit": unit, "direction": direction, "test": test, "threshold": threshold, "threshold_units": units,
            "family": family, "fidelity": fidelity, "required_capabilities": caps, "evidence_level_target": level,
            "data_roles": roles, "pool": pool, "max_units": max_units, "pilot": pilot, "decision_rule": rule,
            "headline": headline, **extra}


DAY = "instrument-day (Bybit linear BTCUSDT and ETHUSDT days of the pool; windows nested in days)"
WORLD = "world cluster (one registered synthetic world with its seed block)"
HYB = "backbone instrument-day of the hybrid twin (registered windows nested in days)"
R = ["explore", "confirm"]
L2 = ["incremental_l2", "trades_with_side"]


def candidates() -> dict:
    H = {
        "H1": _h("Hybrid replay vs fully generative response realism",
                 "Does the hybrid twin reproduce the observed market response to large aggressive prints better than "
                 "fully generative simulators?", "realism",
                 "relative reduction of post-print response prediction error (mid move and touch-depth recovery at "
                 "1 s, 5 s, 30 s; standardized) of the hybrid twin vs the best fully generative family",
                 "per unit: (err_generative - err_hybrid) / err_generative over registered large-print events",
                 DAY, "increase", 0.10, "relative error reduction", "realism", ["F1", "F3", "F4", "F5"], L2, "E4", R,
                 "confirm", 80, {"study": "hybrid-twin-v09", "key": "h1"}),
        "H2": _h("Neural temporal point process vs classical point processes",
                 "Do neural TPPs improve future predictive realism of event timing and marks?", "realism",
                 "held-out negative log-likelihood per event (inter-arrival time + six-way mark) of the best classical "
                 "point process minus the neural TPP", "per unit: mean NLL difference over the day's events",
                 DAY, "increase", 0.02, "nats per event", "realism", ["F1", "F4", "F5"], L2 + ["neural_backend"],
                 "E4", R, "confirm", 80, {"study": "neural-point-process-v09", "key": "h2"}),
        "H3": _h("Sequence models vs classical conditional event models",
                 "Do sequence models improve conditional event realism P(mark | state, history)?", "realism",
                 "held-out conditional mark NLL of the classical state-conditioned model minus the best sequence "
                 "model (GRU/LSTM/causal Transformer)", "per unit: mean NLL difference", DAY, "increase", 0.01,
                 "nats per event", "realism", ["F1", "F4", "F5"], L2 + ["neural_backend"], "E4", R, "confirm", 80,
                 {"study": "sequence-generator-v09", "key": "h3"}),
        "H4": _h("Latent dynamics vs classical path realism",
                 "Do latent-dynamics generators improve path-level realism?", "realism",
                 "relative reduction of the signature-MMD between generated and held-out real 60 s mid/spread paths "
                 "(latent state-space or Neural SDE vs the best classical generator)",
                 "per unit: (mmd_classical - mmd_latent) / mmd_classical", DAY, "increase", 0.10,
                 "relative MMD reduction", "realism", ["F1", "F4", "F5"], L2 + ["neural_backend"], "E4", R, "confirm",
                 80, {"study": "latent-dynamics-v09", "key": "h4"}),
        "H5": _h("Multi-fidelity model averaging",
                 "Does dynamic model averaging improve future predictive performance over the best single model?",
                 "realism", "held-out log score per event of the dynamic model average minus the single model "
                 "selected on the preceding calibration data", "per unit: mean log-score difference", DAY,
                 "increase", 0.005, "nats per event", "realism", ["F1", "F4", "F5"], L2, "E4", R, "confirm", 80,
                 {"study": "model-averaging-v09", "key": "h5"}),
        "H6": _h("Execution-aware calibration and fill-probability calibration",
                 "Does execution-aware calibration improve fill-probability calibration?", "fills",
                 "Brier score of P(queue clearance at the touch within 5 s) predicted by the generator calibrated on "
                 "the execution-aware objective minus the generic-summary calibration (positive = better)",
                 "per unit: Brier(generic) - Brier(execution-aware) over registered probe times", DAY, "increase",
                 0.005, "Brier score", "fills", ["F1", "F4"], L2 + ["queue_clearance_observable"], "E4", R, "confirm",
                 80, {"study": "fill-study-v09", "key": "h6"}),
        "H7": _h("Queue-survival models vs Kaplan-Meier",
                 "Do queue-survival models improve time-to-fill (time-to-clearance) prediction?", "fills",
                 "integrated Brier score over 0-30 s of time to queue clearance: Kaplan-Meier minus the best "
                 "registered survival model (Cox, AFT, boosted, neural)", "per unit: IBS difference", DAY, "increase",
                 0.005, "integrated Brier score", "fills", ["F1"], L2 + ["queue_clearance_observable"], "E4", R,
                 "confirm", 80, {"study": "survival-study-v09", "key": "h7"}),
        "H8": _h("Fill calibration and execution regret",
                 "Does better fill calibration reduce execution decision regret?", "fills",
                 "mean regret (bps) of the passive-first policy driven by the uncalibrated (KM) fill model minus "
                 "the calibrated survival model, on the hybrid twin over real backbones",
                 "per unit: mean over registered windows", HYB, "increase", 0.25, "bps of arrival price", "fills",
                 ["F3"], L2 + ["queue_clearance_observable"], "E3", R, "confirm", 80,
                 {"study": "fill-study-v09", "key": "h8"}),
        "H9": _h("Adverse-selection-aware passive execution",
                 "Does toxicity-proxy-aware passive execution improve post-fill markout within the completion "
                 "constraint?", "fills",
                 "5 s post-fill markout (bps, positive = favourable) of hypothetical passive fills: toxicity-aware "
                 "minus naive passive policy, required under both the conservative and the optimistic queue bound; "
                 "completion non-inferior within 0.02", "per unit: mean over registered windows (bounded replay)",
                 DAY, "increase", 0.5, "bps markout", "fills", ["F2"], L2, "E5", R, "confirm", 80,
                 {"study": "markout-study-v09", "key": "h9"}, completion_margin=0.02),
        "H10": _h("State-dependent impact vs square-root impact",
                  "Do state-dependent impact models improve out-of-sample fit of observed post-burst price moves?",
                  "impact", "relative reduction of out-of-sample MSE of the 10 s post-burst mid move (state-dependent "
                  "nonlinear kernel vs square-root baseline); observational, ASSUMPTION_DEPENDENT",
                  "per unit: (mse_sqrt - mse_state) / mse_sqrt", DAY, "increase", 0.02, "relative MSE reduction",
                  "impact", ["F1"], L2, "E4", R, "confirm", 80, {"study": "impact-study-v09", "key": "h10"}),
        "H11": _h("Self-impact and large-mandate rankings",
                  "Does endogenous self-impact materially change large-mandate policy rankings?", "impact",
                  "normalized Kendall distance between registered policy rankings under the exogenous and the "
                  "self-impact (endogenous ecology) simulator for the large mandate",
                  "per unit: Kendall distance of mean-cost rankings", WORLD, "increase", 0.10,
                  "normalized Kendall distance", "impact", ["F6"], [], "E2", ["synthetic"], None, 200,
                  {"study": "strategic-response-v09", "key": "h11"}),
        "H12": _h("Strategic response vs static ecology realism",
                  "Does strategic-response simulation improve registered realism dimensions over a static ecology?",
                  "realism", "relative reduction of the H1 response-prediction error (strategic vs static ecology)",
                  "per unit: (err_static - err_strategic) / err_static", DAY, "increase", 0.05,
                  "relative error reduction", "realism", ["F1", "F6"], L2, "E4", R, "confirm", 80,
                  {"study": "ecology-study-v09", "key": "h12"}),
        "H13": _h("Active calibration efficiency",
                  "Does active calibration reach the registered predictive quality with lower simulation cost?",
                  "adaptation", "1 - (simulations to reach the registered posterior-predictive loss with active "
                  "acquisition / with random acquisition) on registered recovery tasks",
                  "per unit: relative simulation saving", "calibration task (registered synthetic recovery task with "
                  "known parameters)", "increase", 0.30, "relative simulation saving", "adaptation", ["F4"], [],
                  "E1", ["synthetic"], None, 200, {"study": "active-calibration-v09", "key": "h13"}),
        "H14": _h("Sequential posterior updating after drift",
                  "Does sequential posterior updating improve post-drift predictive recovery?", "adaptation",
                  "next-day held-out log score per event: sequentially updated minus frozen calibration",
                  "per unit: mean log-score difference", DAY, "increase", 0.005, "nats per event", "adaptation",
                  ["F1", "F4"], L2, "E6", R, "confirm", 80, {"study": "inference-v09", "key": "h14"}),
        "H15": _h("Few-shot / meta adaptation vs full refit",
                  "Does few-shot adaptation approach full-refit performance with at most 10% of the adaptation data?",
                  "adaptation", "held-out NLL per event of the few-shot (meta/hierarchical) adaptation minus the "
                  "full-day refit, oriented so that positive favours few-shot", "per unit: mean NLL difference",
                  DAY, "increase", 0.01, "nats per event (non-inferiority margin)", "adaptation", ["F1", "F4", "F5"],
                  L2, "E6", R, "confirm", 80, {"study": "meta-adaptation-v09", "key": "h15"},
                  test="noninferiority"),
        "H16": _h("Continual calibration and forgetting",
                  "Does continual calibration (replay / regularization) reduce forgetting relative to naive "
                  "fine-tuning?", "adaptation", "forgetting (NLL increase on earlier-day validation data after "
                  "sequential training) of naive fine-tuning minus the best continual method",
                  "per unit: forgetting difference", "sequence of registered days (one unit per instrument-day "
                  "sequence position)", "increase", 0.005, "nats per event", "adaptation", ["F1", "F5"],
                  L2 + ["neural_backend"], "E4", R, "confirm", 80, {"study": "continual-calibration-v09", "key": "h16"}),
        "H17": _h("Adaptive conformal coverage under drift",
                  "Does adaptive conformal prediction preserve the registered 90% coverage under moderate drift?",
                  "adaptation", "empirical coverage of 90% adaptive-conformal intervals for 60 s-ahead realized "
                  "volatility minus 0.90", "per unit: day coverage", DAY, "increase", 0.02,
                  "coverage (non-inferiority margin)", "adaptation", ["F1"], L2, "E6", R, "confirm", 80,
                  {"study": "conformal-study-v09", "key": "h17"}, test="noninferiority"),
        "H18": _h("Disagreement predicts regret",
                  "Does model/policy disagreement predict future execution regret?", "decisions",
                  "within-day Spearman correlation between pre-decision disagreement and realized regret on the hybrid "
                  "twin", "per unit: Spearman rho over registered windows", HYB, "increase", 0.10, "Spearman rho",
                  "decisions", ["F3"], L2, "E3", R, "confirm", 80, {"study": "abstention-study-v09", "key": "h18"}),
        "H19": _h("Abstention v3 and incorrect selection",
                  "Does improved abstention reduce incorrect policy-selection decisions?", "decisions",
                  "incorrect-selection rate of always-select minus abstention v3 (abstain = registered conservative "
                  "choice)", "per unit: rate difference over registered windows", HYB, "increase", 0.05,
                  "rate (proportion)", "decisions", ["F3"], L2, "E3", R, "confirm", 80,
                  {"study": "abstention-study-v09", "key": "h19"}),
        "H20": _h("Contextual fallback and catastrophic degradation",
                  "Does the registered contextual fallback ladder reduce catastrophic execution degradation?",
                  "decisions", "catastrophic-degradation rate (cost above the registered catastrophe threshold) of "
                  "no fallback minus the contextual fallback ladder, over registered shifted worlds",
                  "per unit: rate difference", WORLD, "increase", 0.02, "rate (proportion)", "decisions",
                  ["F4", "F6"], [], "E2", ["synthetic"], None, 200, {"study": "abstention-study-v09", "key": "h20"}),
        "H21": _h("Distributionally robust execution and worst-case regret",
                  "Does DRO execution reduce registered worst-case regret?", "decisions",
                  "worst-case regret over the registered ambiguity set: nominal minus DRO schedule",
                  "per unit: worst-case regret difference", WORLD, "increase", 0.5, "bps", "decisions",
                  ["F4", "F5", "F6"], [], "E2", ["synthetic"], None, 200, {"study": "dro-study-v09", "key": "h21"}),
        "H22": _h("Chance-constrained completion reliability",
                  "Does chance-constrained execution improve completion reliability within the cost margin?",
                  "decisions", "completion probability within the horizon: chance-constrained minus nominal MPC, "
                  "with mean cost non-inferior within 1 bps", "per unit: completion-rate difference", HYB,
                  "increase", 0.05, "completion probability", "decisions", ["F3"], L2, "E3", R, "confirm", 80,
                  {"study": "completion-study-v09", "key": "h22"}, cost_margin_bps=1.0),
        "H23": _h("Robust MPC and tail execution cost",
                  "Does robust (CVaR) MPC reduce tail execution cost relative to TWAP?", "decisions",
                  "CVaR90 of implementation shortfall: TWAP minus CVaR-MPC, medium mandate, bounded historical replay "
                  "(both cost bounds) and hybrid twin", "per unit: CVaR difference over registered windows", DAY,
                  "increase", 1.0, "bps of arrival price", "decisions", ["F2", "F3"], L2, "E5", R, "confirm", 80,
                  {"study": "mpc-study-v09", "key": "h23"}, headline=True),
        "H24": _h("Adaptive MPC after regime drift",
                  "Does adaptive MPC improve execution after regime drift?", "decisions",
                  "post-drift mean implementation shortfall: non-adaptive MPC minus adaptive MPC on registered "
                  "regime-switch worlds", "per unit: mean difference", WORLD, "increase", 0.5, "bps", "decisions",
                  ["F4", "F6"], [], "E2", ["synthetic"], None, 200, {"study": "mpc-study-v09", "key": "h24"}),
        "H25": _h("Meta-policy vs best global policy",
                  "Does a meta-policy / mixture of experts beat the preregistered best global policy?", "decisions",
                  "mean implementation shortfall: best global policy (selected on explore days) minus meta-policy, "
                  "hybrid twin", "per unit: mean difference over windows", HYB, "increase", 0.25, "bps", "decisions",
                  ["F3"], L2, "E3", R, "confirm", 80, {"study": "meta-policy-v09", "key": "h25"}),
        "H26": _h("Offline RL safe improvement",
                  "Does offline RL achieve safe improvement over registered classical baselines?", "decisions",
                  "mean implementation shortfall: best classical baseline minus the gated offline-RL policy on "
                  "registered worlds with logged behaviour propensities", "per unit: mean difference", WORLD,
                  "increase", 0.25, "bps", "decisions", ["F4"], ["propensities"], "E1", ["synthetic"], None, 200,
                  {"study": "offline-rl-v09", "key": "h26"}),
        "H27": _h("Conservative gating and harmful deployments",
                  "Does conservative policy-improvement gating reduce harmful learned-policy deployments?",
                  "decisions", "harmful-deployment rate (deployed policy worse than the baseline by more than "
                  "0.5 bps) of ungated minus gated deployment", "per unit: rate difference", WORLD, "increase", 0.10,
                  "rate (proportion)", "decisions", ["F4"], ["propensities"], "E1", ["synthetic"], None, 200,
                  {"study": "safe-improvement-v09", "key": "h27"}),
        "H28": _h("Rare-event-aware optimization",
                  "Does rare-event-aware optimization reduce catastrophic execution-cost probability?", "decisions",
                  "catastrophic-cost probability (registered threshold) of nominal minus rare-event-aware "
                  "optimization in registered stress worlds, mean cost non-inferior within 1 bps",
                  "per unit: probability difference", WORLD, "increase", 0.01, "probability", "decisions",
                  ["F4", "F6"], [], "E2", ["synthetic"], None, 200, {"study": "rare-event-study-v09", "key": "h28"},
                  cost_margin_bps=1.0),
        "H29": _h("Synthetic ranking predicts hybrid ranking",
                  "Does the synthetic policy ranking predict the hybrid-replay ranking?", "fidelity",
                  "pairwise sign agreement between synthetic (F4-F6 consensus) and hybrid-twin directions over the "
                  "registered policy pairs", "per unit: agreement over determinate pairs", HYB, "increase", 0.15,
                  "agreement above chance level", "fidelity", ["F3", "F4", "F5", "F6"], L2, "E3", R, "confirm", 80,
                  {"study": "multifidelity-agreement-v09", "key": "h29"}, test="agreement", agreement_level=0.5),
        "H30": _h("Hybrid ranking predicts bounded-historical ranking",
                  "Does the hybrid ranking predict the bounded-historical ranking on identified pairs?", "fidelity",
                  "pairwise sign agreement between hybrid-twin and historically identified (bounded) directions",
                  "per unit: agreement over identified pairs", DAY, "increase", 0.15, "agreement above chance level",
                  "fidelity", ["F2", "F3"], L2, "E5", R, "confirm", 80,
                  {"study": "multifidelity-agreement-v09", "key": "h30"}, test="agreement", agreement_level=0.5),
        "H31": _h("Multi-fidelity consensus vs synthetic-only",
                  "Does multi-fidelity consensus predict historical policy direction better than synthetic-only?",
                  "fidelity", "agreement with historically identified directions: consensus minus synthetic-only",
                  "per unit: agreement difference over identified pairs", DAY, "increase", 0.10,
                  "agreement difference", "fidelity", ["F2", "F3", "F4", "F5", "F6"], L2, "E5", R, "confirm", 80,
                  {"study": "multifidelity-agreement-v09", "key": "h31"}),
        "H32": _h("Transfer across time", "Does at least one promoted execution conclusion transfer across time?",
                  "transfer", "carried conclusions re-tested on the confirm_late pool (2024-06 to 2024-12)",
                  "per conclusion: its own registered estimator", DAY, "increase", 0.0, "conclusion-specific",
                  "transfer", ["F2", "F3"], L2, "E6", ["confirm_late"], "confirm_late", 40,
                  {"study": "transfer-study-v09", "key": "h32"}, test="transfer"),
        "H33": _h("Transfer across instruments",
                  "Does at least one promoted execution conclusion transfer across instruments?", "transfer",
                  "carried conclusions (selected on BTCUSDT) re-tested on ETHUSDT confirm days",
                  "per conclusion: its own registered estimator", DAY, "increase", 0.0, "conclusion-specific",
                  "transfer", ["F2", "F3"], L2, "E7", ["confirm"], "confirm", 40,
                  {"study": "transfer-study-v09", "key": "h33"}, test="transfer"),
        "H34": _h("Transfer across venues",
                  "Does at least one promoted L1-identifiable execution conclusion transfer to another venue?",
                  "transfer", "carried L1-identifiable conclusions re-tested on Binance USD-M L1 + trades confirm days",
                  "per conclusion: its own registered estimator", DAY, "increase", 0.0, "conclusion-specific",
                  "transfer", ["F2"], ["l1_quotes", "trades_with_side"], "E8", ["confirm"], "confirm", 40,
                  {"study": "transfer-study-v09", "key": "h34"}, test="transfer"),
        "H35": _h("Learned representations and transfer",
                  "Do learned market representations improve registered transfer metrics over hand-engineered "
                  "features?", "realism", "cross-instrument linear-probe log loss of the 10 s-ahead mid-move direction "
                  "(trained on BTCUSDT, evaluated on ETHUSDT days): hand features minus learned embedding",
                  "per unit: log-loss difference", DAY, "increase", 0.005, "nats", "realism", ["F1"],
                  L2 + ["neural_backend"], "E7", R, "confirm", 80, {"study": "representation-transfer-v09", "key": "h35"}),
        "H36": _h("Survival of the full reality ladder and the final vault",
                  "Does at least one execution conclusion survive the entire available reality ladder and the final "
                  "untouched multi-axis holdout?", "transfer",
                  "carried conclusions tested once on the final vault (2025: SOLUSDT linear, BTCUSDT linear, BTCUSDT "
                  "spot) after surviving synthetic, multi-model, hybrid, bounded-historical, time, instrument and "
                  "stress rungs", "per conclusion: its own registered estimator on vault days", DAY, "increase", 0.0,
                  "conclusion-specific", "transfer", ["F2", "F3"], L2, "E9", ["final"], "final", 36,
                  {"study": "final-analysis-v09", "key": "h36"}, test="transfer", headline=True),
    }
    return {"schema": "cleolob-v09-candidates-1", "status": "CANDIDATE",
            "rule": "every hypothesis is a CANDIDATE until hypothesis-promotion-v09 applies the registered gate; "
                    "endpoints, units, directions and practical thresholds are fixed here before any data access",
            "hypotheses": H}


FAMILIES = {
    "realism": ["H1", "H2", "H3", "H4", "H5", "H12", "H35"],
    "fills": ["H6", "H7", "H8", "H9"],
    "impact": ["H10", "H11"],
    "adaptation": ["H13", "H14", "H15", "H16", "H17"],
    "decisions": ["H18", "H19", "H20", "H21", "H22", "H23", "H24", "H25", "H26", "H27", "H28"],
    "fidelity": ["H29", "H30", "H31"],
    "transfer": ["H32", "H33", "H34", "H36"],
}


def families() -> dict:
    return {"schema": "cleolob-v09-families-1",
            "families": {k: {"members": v, "alpha": 0.05, "correction": "holm", "confirmatory": True,
                             "sided": "one-sided in the registered direction"} for k, v in FAMILIES.items()},
            "exploratory": {"correction": "benjamini_hochberg", "q": 0.10,
                            "note": "exploratory discovery only; never mixed with confirmatory families"}}


# ============================================================================= protocol


CONCLUSIONS = {
    "C1": {"treatment": "mpc_cvar", "control": "twap", "mandate": "medium", "endpoint": "cvar90_is_bps",
           "direction": "decrease", "threshold_bps": 1.0, "identifiability": "L2"},
    "C2": {"treatment": "twap", "control": "sweep", "mandate": "large", "endpoint": "mean_is_bps",
           "direction": "decrease", "threshold_bps": 1.0, "identifiability": "L2"},
    "C3": {"treatment": "passive_deadline", "control": "twap", "mandate": "small", "endpoint": "mean_is_bps",
           "direction": "decrease", "threshold_bps": 0.25, "identifiability": "L2"},
    "C4": {"treatment": "liquidity_pov", "control": "twap", "mandate": "medium", "endpoint": "mean_is_bps",
           "direction": "decrease", "threshold_bps": 0.25, "identifiability": "L2"},
    "C5": {"treatment": "toxicity_passive", "control": "passive_naive", "mandate": "small",
           "endpoint": "markout5s_bps", "direction": "increase", "threshold_bps": 0.5, "identifiability": "L2"},
    "C6": {"treatment": "twap", "control": "sweep", "mandate": "small", "endpoint": "mean_is_bps",
           "direction": "decrease", "threshold_bps": 0.25, "identifiability": "L1"},
}

MANDATES = {
    "small": {"notional_usd": {"BTCUSDT": 100_000, "ETHUSDT": 100_000, "SOLUSDT": 50_000}},
    "medium": {"notional_usd": {"BTCUSDT": 1_000_000, "ETHUSDT": 1_000_000, "SOLUSDT": 500_000}},
    "large": {"notional_usd": {"BTCUSDT": 5_000_000, "ETHUSDT": 5_000_000, "SOLUSDT": 2_500_000}},
}


def protocol_document(root: Path, base_commit: str) -> dict:
    history = pr.history_state(root)
    return {
        "schema": pr.SCHEMA, "protocol_id": "cleolob-v09", "frozen_at_utc": None, "base_commit": base_commit,
        "research_question": (
            "Which execution conclusions remain scientifically defensible when market reconstruction uncertainty, "
            "latent liquidity, self-impact, strategic response, causal ambiguity, multi-fidelity simulation, "
            "distribution shift, tail events, model uncertainty, online adaptation and real historical validation "
            "are considered simultaneously?"),
        "statuses": {"engineering": list(pr.ENGINEERING), "data": list(pr.DATA_STATUSES),
                     "lifecycle": list(pr.LIFECYCLE), "terminal": list(pr.TERMINAL),
                     "promotion": list(pr.PROMOTION_OUTCOMES), "realism": list(pr.REALISM),
                     "historical_identification": list(pr.HISTORICAL_ID),
                     "causal_identification": list(pr.CAUSAL_ID)},
        "evidence_levels": list(pr.EVIDENCE_LEVELS),
        "evidence_level_meaning": {
            "E0": "engineering verification only", "E1": "synthetic", "E2": "multi-model synthetic",
            "E3": "hybrid historical", "E4": "bounded historical", "E5": "historically direction-identified",
            "E6": "cross-time real support", "E7": "cross-instrument real support", "E8": "cross-venue real support",
            "E9": "untouched final holdout support", "E10": "prospective/shadow support (not attempted in v0.9)"},
        "history": {k: history[k] for k in ("v08_protocol_sha256", "v08_ledger_entries", "v08_ledger_head_sha256")},
        "pools": {k: {"role": v["role"], "start": v["start"], "end": v["end"], "count": v["count"],
                      "backups": v["backups"], "streams": [f"{a}-{b}:{c}" for a, b, c in v["streams"]],
                      "days": pool_days(k)} for k, v in POOLS.items()},
        "pool_rule": f"days = SeedSequence([{POOL_SEED}, sha256(pool)[:12]]) permutation of the calendar range; the "
                     "first `count` are primary slots, the next `backups` are ordered backups",
        "datasets": datasets(),
        "holdout_rules": {
            "explore": "explore and development days are readable after the freeze by any design and are never "
                       "used for confirmatory decisions",
            "confirm": "confirm and confirm_late days are read only after the promotion freeze, only by sealed designs "
                       "listing them; promotion fixes how many primary slots (a prefix) each hypothesis reads",
            "final": "final-vault days are read once, by the single sealed final design, after close_confirmatory; "
                     "no outcome statistic is inspected before the final analysis",
            "posthoc": "any later reader of guarded data is posthoc and capped at EXPLORATORY"},
        "acquisition_rules": {
            "probe": f"header-only availability probe before the freeze ({PROBE_DATE}); see "
                     "docs/v09-implementation-plan.md section 3",
            "backup": "a failed slot (HTTP error, missing archive, QUALITY_REJECTED) is replaced by the next unused "
                      "backup slot of the same pool in registered order; nothing else may substitute",
            "integrity": "raw bytes hashed at download; Binance archives verified against the published .CHECKSUM; "
                         "source hashes enter the ledger and never change",
            "exhausted": "if every registered option fails the dependent unit is NOT_AVAILABLE",
            "licensing": "public archives used for non-commercial research under the providers' terms; raw data are "
                         "never committed or bundled"},
        "seed_namespaces": {"names": list(pr.NAMESPACES), "ids": pr.NAMESPACE_IDS,
                            "rule": "SeedSequence([9, namespace id, sha256(study)[:12], index]); confirm, holdout and "
                                    "final require a sealed design that opened them"},
        "promotion_rules": {"min_reliability": 0.5, "min_pilot_units": 6, "min_power": 0.80, "shrinkage_se": 1.0,
                            "min_confirm_units": 8, "compute_hours_ceiling": 40.0,
                            "reliability": "split-half (odd/even windows) Spearman-Brown reliability of the unit-level "
                                           "endpoint on pilot units",
                            "order": ["data/capability", "measurement", "effect", "power", "compute"],
                            "conclusions": "an execution conclusion is carried when its explore-day pilot passes the "
                                           "same gate (its own threshold, unit sd, the decisions family alpha divided "
                                           "by the number of conclusions)",
                            "transfer_candidates": "the pilot record of H32, H33, H34 and H36 is that of the carried "
                                                   "conclusion with the largest planning effect relative to its own "
                                                   "threshold, measured on the explore streams most like the target "
                                                   "(BTCUSDT for H32, ETHUSDT for H33, Binance for H34, all for H36); "
                                                   "no carried conclusion -> NOT_PROMOTED_EFFECT_TOO_SMALL"},
        "power_rules": {"method": "normal approximation of the registered decision rule; planning alpha = family "
                                  "alpha / family size; planning effect = pilot estimate - 1 pilot SE",
                        "target": 0.80, "headline_preference": 0.90},
        "statistics": {"alpha": 0.05, "correction": "holm", "interval": "paired block bootstrap over units (BCa not "
                       "used; percentile)", "n_boot": 4000, "hierarchical": "days -> windows two-stage bootstrap where "
                       "windows matter", "proportions": "Wilson", "equivalence": "TOST with registered margins",
                       "seed_rule": "bootstrap seeds from the analysis namespace of the design"},
        "practical_thresholds": {hid: {"threshold": h["threshold"], "units": h["threshold_units"]}
                                 for hid, h in candidates()["hypotheses"].items()},
        "completion_constraints": {"horizon_s": 300.0, "decision_dt_s": 5.0, "warmup_s": 30.0,
                                   "lexicographic": ["complete within the horizon", "hard risk constraints",
                                                     "minimize tail cost (CVaR90)", "minimize mean cost"],
                                   "residual": "unfinished quantity is valued by walking the end-of-horizon book plus "
                                               "the registered 25 bps penalty; reported separately",
                                   "catastrophe_bps": 25.0},
        "risk_measures": {"cvar_level": 0.90, "markout_horizons_s": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
                          "survival_horizons_s": [0.05, 0.1, 0.5, 1.0, 5.0, 30.0]},
        "mandates": {"classes": MANDATES, "horizon_s": 300.0, "decision_dt_s": 5.0,
                     "side_rule": "buy on even registered windows, sell on odd windows",
                     "windows": "24 per day starting at HH:10:00 UTC (HH = 0..23), each with a 30 s warmup",
                     "quantity_rule": "notional / arrival mid, rounded down to the lot size",
                     "fees": {"taker_bps": 5.5, "maker_bps": 2.0, "note": "registered retail-tier fees; costs are "
                              "reported gross and net of fees separately"}},
        "policies": {
            "classical": ["twap", "vwap", "pov", "ac", "sweep"],
            "adaptive": ["adaptive_twap", "adaptive_pov", "liquidity_pov", "imbalance_passive", "toxicity_passive",
                         "passive_naive", "passive_deadline"],
            "optimization": ["static_opt", "mpc_stochastic", "mpc_cvar", "mpc_minmax", "mpc_dro", "mpc_chance",
                             "mpc_adaptive"],
            "learning": ["bc", "cql", "iql", "td3bc", "decision_transformer"],
            "meta": ["regime_selector", "mixture_of_experts", "policy_blender", "contextual_fallback"],
            "safe_baseline": "twap",
            "fallback_ladder": {"low_uncertainty": "selected optimized policy", "moderate": "liquidity_pov",
                                "high": "twap", "completion_danger": "completion-safe aggressive controller"}},
        "conclusions": {"set": CONCLUSIONS,
                        "rule": "pilot effects on explore days decide which conclusions are carried (promoted); each "
                                "carried conclusion is decided by both bounded-replay cost bounds (historically "
                                "identified only when the bounds agree) with Holm over the carried set",
                        "ladder": ["synthetic", "multi-model synthetic", "hybrid twin", "bounded historical",
                                   "cross-time", "cross-instrument", "cross-venue (L1-identifiable only)",
                                   "adversarial stress (reported)", "final multi-axis vault"]},
        "realism_gate": {
            "capabilities": ["spread_dynamics", "depth_dynamics", "event_timing", "queue_survival", "fill_prediction",
                             "impact", "tail_behavior", "strategic_response"],
            "reference": "day-to-day real variation: the 95th percentile of the same metric between pairs of held-out "
                         "real days",
            "acceptable": "every metric of the capability within 1.0x the real-real reference and classifier AUC "
                          "<= 0.60",
            "partial": "every metric within 2.0x the reference",
            "rejected": "otherwise",
            "global": {"ACCEPTABLE_MODEL_SET": "some generator ACCEPTABLE on every capability a study needs",
                       "PARTIAL_MODEL_SET": "otherwise, some generator at least PARTIALLY_ACCEPTABLE on each",
                       "NO_ACCEPTABLE_MODEL": "otherwise"},
            "fail_closed": "a generator REJECTED for a capability never supports a claim that needs it"},
        "counterfactual_assumptions": {
            "queue": ["optimistic", "neutral", "conservative"], "latency_s": [0.0, 0.05, 0.25],
            "hidden_fraction": [0.0, 0.5], "impact": ["none", "transient", "persistent"],
            "transient_half_life_s": 10.0,
            "identification": "a pairwise direction is IDENTIFIED when the cost intervals separate on the unit; "
                              "INDETERMINATE otherwise; nothing is an exact fill"},
        "causal_assumptions": {
            "rule": "every causal estimand registers treatment, outcome, confounders, DAG, identification assumptions, "
                    "estimator and sensitivity analysis; unidentified effects are NOT_IDENTIFIED and assumption-heavy "
                    "ones ASSUMPTION_DEPENDENT",
            "observational_impact": "ASSUMPTION_DEPENDENT (no randomized design exists in public data)"},
        "resource_ceilings": {"ram_mb_per_study": 12_000, "workers_max": 8, "workers_default": 6,
                              "concurrent_heavy_families": 1, "on_exceed": "RESOURCE_EXHAUSTED; budget increases "
                              "require an amendment before outcome inspection"},
        "failure_semantics": {
            "engineering_invalid": "retain the attempt, ledger it, fix the root cause with a regression test, rerun in "
                                   "a new directory on the same design and seeds",
            "scientific_failure": "reported, never fixed", "data_failure": "registered backups only, then "
                                                                          "NOT_AVAILABLE",
            "resource_failure": "resume via checkpoint or registered resource-safe retry"},
        "evidence_schemas": {"run": "cleolob-v09 run schema v3 (artifacts.py)", "checkpoint": "cleolob-v09-checkpoint-3",
                             "binding": pr.BINDING_SCHEMA, "ledger": pr.LEDGER_SCHEMA},
        "fidelities": {"F0": "raw historical observations", "F1": "historical replay", "F2": "bounded historical "
                       "counterfactual replay", "F3": "hybrid historical market twin", "F4": "calibrated classical "
                       "generative market", "F5": "neural generative market", "F6": "strategic multi-agent market"},
        "amendment_policy": "amendments are ledgered before any affected guarded data are read; the confirmatory "
                            "registry is amendable only before any confirmatory access",
        "referenced_configs": [f"{pr.CONFIG_DIR}/{n}" for n in pr.REQUIRED_CONFIGS],
        "exclusions": {"prior_versions": "every v0.5-v0.8 dataset and calendar period is imported as consumed",
                       "claims": "no live alpha, profitability, production/HFT or exact historical fill claim"},
    }


def dataset_registry(protocol: dict) -> dict:
    """Dataset registry v4 (M10) with source-trust metadata (M12)."""
    trust = {"bybit": {"source_trust": "official exchange public archive; no published checksum; hashed at download",
                       "known_anomalies": "occasional snapshot re-sends inside a day; handled as new snapshots"},
             "binance": {"source_trust": "official exchange public archive with published SHA-256 .CHECKSUM",
                         "known_anomalies": "bookTicker archive discontinued after 2024-03-30"}}
    entries = []
    for d in protocol["datasets"]:
        venue = d["venue"].split("-")[0]
        entries.append({"dataset_id": d["id"], "provider": d["provider"], "venue": d["venue"],
                        "instrument": d["instrument"], "instrument_type": d["instrument_type"],
                        "period": d["date"], "role": d["role"], "pool": d["pool"], "slot": d["slot"],
                        "license_use": "public research download; provider terms apply; raw data never redistributed",
                        "access": {k: v["url"] for k, v in d["sources"].items()}, "raw_hashes": "recorded at access",
                        **d["capabilities"], "tick_size": d["tick_size"], "lot_size": d["lot_size"],
                        "contract_multiplier": d["contract_multiplier"],
                        "fee_metadata": "registered mandate fees (protocol.mandates.fees)",
                        "funding_metadata": "not used (horizons are 300 s)", **trust[venue],
                        "quality_status": "PENDING (validated at access)",
                        "eligible_experiment_classes": _eligible(d)})
    return {"schema": "cleolob-v09-dataset-registry-4", "datasets": entries}


def _eligible(d: dict) -> list[str]:
    c = d["capabilities"]
    out = ["observational_statistics", "trade_flow"]
    if c["incremental_l2"]:
        out += ["l2_reconstruction", "queue_clearance", "bounded_counterfactual_l2", "hybrid_twin_backbone",
                "generator_calibration", "realism_reference"]
    else:
        out += ["bounded_counterfactual_l1"]
    return out


def capability_matrix() -> dict:
    """Machine-readable capability graph (M11): capability -> experiment classes and hypotheses that need it."""
    caps = {"incremental_l2": "full-depth snapshot+delta book", "trades_with_side": "prints with aggressor side",
            "l1_quotes": "best bid/ask updates", "update_sequence": "update ids / cross sequence",
            "queue_clearance_observable": "price-level clearance by trades is observable",
            "order_ids": "market-by-order identities", "hidden_liquidity": "hidden/iceberg observability",
            "neural_backend": "PyTorch available", "propensities": "logged behaviour-policy propensities"}
    needs: dict[str, list[str]] = {k: [] for k in caps}
    for hid, h in candidates()["hypotheses"].items():
        for c in h["required_capabilities"]:
            needs.setdefault(c, []).append(hid)
    return {"schema": "cleolob-v09-capability-matrix-1", "capabilities": caps, "required_by": needs,
            "provided_by": {"bybit-linear": ["incremental_l2", "trades_with_side", "l1_quotes", "update_sequence",
                                             "queue_clearance_observable"],
                            "bybit-spot": ["incremental_l2", "trades_with_side", "l1_quotes", "update_sequence",
                                           "queue_clearance_observable"],
                            "binance-um": ["l1_quotes", "trades_with_side", "update_sequence"],
                            "simulation": ["propensities", "order_ids", "hidden_liquidity"]},
            "never_provided": {"order_ids": "no public archive in v0.9 exposes order identities",
                               "hidden_liquidity": "not observable in any v0.9 source; bounded by assumption only"}}


def model_capabilities() -> dict:
    families_ = {
        "classical": ["poisson", "marked_poisson", "hawkes", "marked_hawkes", "queue_reactive", "regime_switching",
                      "semi_markov", "mixture"],
        "neural_tpp": ["rmtpp", "neural_hawkes", "transformer_tpp", "marked_neural_tpp"],
        "sequence": ["gru", "lstm", "causal_transformer", "state_conditioned"],
        "latent": ["linear_gaussian_ssm", "switching_ssm", "neural_sde"],
        "density": ["conditional_flow"], "hybrid": ["historical_backbone_twin"], "strategic": ["ecology_v9"]}
    return {"schema": "cleolob-v09-model-capabilities-1", "families": families_,
            "capabilities": ["spread_dynamics", "depth_dynamics", "event_timing", "queue_survival", "fill_prediction",
                             "impact", "tail_behavior", "strategic_response"],
            "rule": "capability decisions are produced by realism-gate-v09 from sealed realism-benchmark runs; this "
                    "file registers which families are evaluated, never their outcomes"}


def compute_budget() -> dict:
    return {"schema": "cleolob-v09-compute-budget-1",
            "host": "16 logical CPUs, 32 GB RAM; GPU optional (not required)",
            "ceilings": {"explore_hours": 60, "confirm_hours": 120, "final_hours": 30, "download_gb": 40},
            "per_study_default": {"ram_mb": 12_000, "workers_default": 6, "workers_max": 8, "checkpoint_every": 1},
            "rule": "exceeding a ceiling yields RESOURCE_EXHAUSTED; increases need an amendment before inspection"}


def claim_registry() -> dict:
    claims = {f"H{k}": {"hypothesis": f"H{k}", "source_analysis": "final-analysis-v09",
                        "pointer": f"/hypotheses/H{k}/status", "statement": candidates()["hypotheses"][f"H{k}"]["title"]}
              for k in range(1, 37)}
    claims.update({
        "DATA": {"source_analysis": "final-analysis-v09", "pointer": "/data_availability",
                 "statement": "registered data availability and consumption"},
        "GATE": {"source_analysis": "final-analysis-v09", "pointer": "/realism_gate/global",
                 "statement": "capability-specific realism gate outcome"},
        "PROMOTION": {"source_analysis": "final-analysis-v09", "pointer": "/promotion/promoted",
                      "statement": "candidates promoted by the registered gate"},
        "LADDER": {"source_analysis": "final-analysis-v09", "pointer": "/ladder",
                   "statement": "evidence level reached by each carried execution conclusion"},
        "GAP": {"source_analysis": "final-analysis-v09", "pointer": "/reality_gap",
                "statement": "reality-gap decomposition of the generators"},
        "INCIDENTS": {"source_analysis": "final-analysis-v09", "pointer": "/incidents",
                      "statement": "scientific incident record"}})
    return {"schema": "cleolob-v09-claim-registry-1", "marker_pattern": "[C09-<id>]",
            "rule": "documents cite claims by marker; claim-audit-v09 resolves each marker to the final analysis value "
                    "and fails closed on any mismatch, missing qualifier or failed source",
            "claims": claims}


def build(root: Path, base_commit: str, *, force: bool = False) -> dict:
    """Write every frozen config once (refuses to overwrite after the freeze)."""
    root = Path(root)
    if (root / pr.LEDGER_PATH).exists():
        raise ProtocolError("the v0.9 ledger exists; frozen configs are changed only by amendment")
    cfg = root / pr.CONFIG_DIR
    cfg.mkdir(parents=True, exist_ok=True)
    protocol = protocol_document(root, base_commit)
    files = {"protocol.json": protocol, "hypotheses-candidates.json": candidates(),
             "statistical-families.json": families(), "dataset-registry.json": dataset_registry(protocol),
             "capability-matrix.json": capability_matrix(), "model-capabilities.json": model_capabilities(),
             "compute-budget.json": compute_budget(), "claim-registry.json": claim_registry()}
    for name, doc in files.items():
        target = cfg / name
        if target.exists() and not force:
            raise ProtocolError(f"{target} exists; pass force before the freeze only")
        target.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    pr.validate_protocol(protocol, root)
    return {"written": sorted(files), "datasets": len(protocol["datasets"]),
            "protocol_sha256": pr.protocol_sha256(protocol)}
