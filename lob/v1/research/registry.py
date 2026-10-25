"""v1 research registry: protocol, pools, datasets, H1-H48 candidates, families and budgets (M2, M3, M6, M7).

``build(root)`` writes the configuration files under ``configs/v1``, deterministically, before any fresh data are
read; the protocol is then frozen in the ledger (``cleo v1 protocol freeze``). Everything a later decision depends
on is fixed here: date pools and their registered slot order (calendar dates are the independent clusters; every
stream of a pool uses the same date order), stream sets, the hypothesis endpoints and practical thresholds, the
statistical decision rules, the promotion rule (cluster-level power, no split-half screen), compute budgets and
the replication carry-overs from v0.9 (assumption grid, mandates, conclusions C1-C3, cost endpoints).
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np

from ..data import firewall as FW
from ..protocol import document as D

ROOT_POOL_SEED = 1_000_000_011

# ----------------------------------------------------------------------------------------------- streams and pools
L2_STREAMS = {
    "bybit-linear:BTCUSDT": ("bybit", "linear", "BTCUSDT"),
    "bybit-linear:ETHUSDT": ("bybit", "linear", "ETHUSDT"),
    "bybit-linear:SOLUSDT": ("bybit", "linear", "SOLUSDT"),
    "bybit-spot:BTCUSDT": ("bybit", "spot", "BTCUSDT"),
    "bybit-spot:ETHUSDT": ("bybit", "spot", "ETHUSDT"),
}
REPLICATION_STREAMS = ["bybit-linear:BTCUSDT", "bybit-linear:ETHUSDT"]       # the v0.9 confirmatory streams

POOLS = {
    "explore": {"role": "explore", "start": "2025-09-01", "end": "2025-11-30", "primaries": 16, "backups": 6,
                "streams": list(L2_STREAMS)},
    "calibrate": {"role": "calibrate", "start": "2025-12-01", "end": "2025-12-31", "primaries": 8, "backups": 4,
                  "streams": list(L2_STREAMS)},
    "confirm": {"role": "confirm", "start": "2026-01-01", "end": "2026-05-31", "primaries": 40, "backups": 12,
                "streams": list(L2_STREAMS)},
    "replicate": {"role": "replicate", "start": "2026-06-01", "end": "2026-08-31", "primaries": 24, "backups": 8,
                  "streams": list(L2_STREAMS)},
    "vault": {"role": "vault", "start": "2026-09-02", "end": "2026-10-24", "primaries": 20, "backups": 8,
              "streams": list(L2_STREAMS)},
}
MBO_POOLS = {
    "rule": ("complete UTC capture days of the prospective Bitstamp capture (>= 95% of seconds covered by COMPLETE "
             "segments, at least two censuses), in calendar order from 2026-10-26: days 1-5 explore, days 6 onward "
             "confirm (registered maximum 16 confirm days); no capture day is ever reassigned"),
    "streams": ["bitstamp-exchange:btcusd", "bitstamp-exchange:ethusd"],
    "explore_days": 5, "confirm_days_max": 16, "first_day": "2026-10-26", "min_coverage": 0.95,
}


def _days(start: str, end: str) -> list[str]:
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    return [(a + timedelta(k)).isoformat() for k in range((b - a).days + 1)]


def pool_dates(name: str) -> list[str]:
    """Registered slot order of a pool: eligible dates permuted by a registered seed; primaries then backups."""
    p = POOLS[name]
    days = [d for d in _days(p["start"], p["end"]) if FW.date_fresh(d)[0]]
    tag = int(hashlib.sha256(f"v1-pool-{name}".encode()).hexdigest()[:12], 16)
    order = np.random.default_rng(np.random.SeedSequence([ROOT_POOL_SEED, tag])).permutation(len(days))
    return [days[i] for i in order[: p["primaries"] + p["backups"]]]


def dataset_registry() -> dict:
    from ..data.connectors import base as CB
    datasets = []
    used: dict[str, str] = {}
    for name, p in POOLS.items():
        dates = pool_dates(name)
        for d in dates:
            if d in used:
                raise ValueError(f"date {d} in pools {used[d]} and {name}")
            used[d] = name
        for stream in p["streams"]:
            conn, seg, sym = L2_STREAMS[stream]
            c = CB.get(conn)
            for slot, day in enumerate(dates):
                datasets.append({
                    "id": f"{conn}-{seg}-{sym.lower()}-{day}", "connector": conn, "segment": seg, "symbol": sym,
                    "stream": stream, "date": day, "role": p["role"], "pool": name, "slot": slot,
                    "slot_kind": "primary" if slot < p["primaries"] else "backup",
                    "sources": [s.__dict__ for s in c.sources(seg, sym, day)],
                    "capability": c.capability(seg, sym, day).to_dict()})
    return {"schema": "cleolob-v1-dataset-registry-1", "pools": {k: {**v, "dates": pool_dates(k)}
                                                                 for k, v in POOLS.items()},
            "mbo_pools": MBO_POOLS, "datasets": datasets,
            "rule": "per pool and stream, primaries in registered order; a NOT_AVAILABLE or QUALITY_REJECTED primary "
                    "is replaced by the next registered backup date of the same pool (all streams share the order)"}


# ----------------------------------------------------------------------------------------------- hypotheses
DATE = "calendar date (cluster); stream values of one date are averaged into one cluster value"
FAMILIES = {
    "replication_prediction": {"members": ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]},
    "replication_execution": {"members": ["H8", "H10", "H11", "H12"]},
    "replication_fidelity": {"members": ["H9"]},
    "reconstruction_realism": {"members": ["H13", "H14", "H15", "H16", "H17", "H18", "H19", "H20"]},
    "generators": {"members": ["H21", "H22", "H23", "H24", "H25", "H26", "H27", "H28"]},
    "twins_agreement": {"members": ["H29", "H30", "H31", "H32", "H33", "H34", "H35", "H36"]},
    "execution_learning": {"members": ["H37", "H38", "H39", "H40", "H41", "H42", "H43", "H44"]},
    "external_validation": {"members": ["H45", "H46", "H47", "H48"]},
}
CONCLUSIONS = {   # the v0.9 carried conclusions (frozen definitions, copied unchanged)
    "C1": {"treatment": "mpc_cvar", "control": "twap", "mandate": "medium", "endpoint": "cvar90_is_bps",
           "direction": "decrease", "threshold_bps": 1.0},
    "C2": {"treatment": "twap", "control": "sweep", "mandate": "large", "endpoint": "mean_is_bps",
           "direction": "decrease", "threshold_bps": 1.0},
    "C3": {"treatment": "passive_deadline", "control": "twap", "mandate": "small", "endpoint": "mean_is_bps",
           "direction": "decrease", "threshold_bps": 0.25},
}


def _h(hid, title, family, study, endpoint, unit_value, test, threshold, units, *, streams, pools, fidelity,
       evidence, capability, promotion="pilot", confirm_clusters=None, margin=None, level=None, v09=None,
       conclusions=None, notes=""):
    return {"id": hid, "title": title, "family": family, "study": study, "endpoint": endpoint,
            "unit_value": unit_value, "test": test, "threshold": threshold, "threshold_units": units,
            "margin": margin, "level": level, "cluster": DATE, "streams": streams, "pools": pools,
            "fidelity": fidelity, "evidence_level_target": evidence, "capability": capability,
            "promotion": promotion, "confirm_clusters": confirm_clusters, "v09_reference": v09,
            "conclusions": conclusions, "notes": notes}


REP = "direct_replication"
R12 = 12
L2C = ["l2", "trade_side"]


def candidates() -> list[dict]:
    rs, allL2 = REPLICATION_STREAMS, list(L2_STREAMS)
    return [
        # ------------------------------------------------------------------ independent v0.9 replication
        _h("H1", "Replicate neural TPP vs classical point processes", "replication_prediction", "tpp",
           "held-out NLL per aggressive order", "NLL(selected classical) - NLL(selected neural), nats/event, models "
           "fitted on the calibrate pool with the v0.9 selection rule", "superiority", 0.02, "nats per event",
           streams=rs, pools=["calibrate", "confirm"], fidelity=["F0"], evidence="E4", capability=L2C,
           promotion=REP, confirm_clusters=R12, v09="H2"),
        _h("H2", "Replicate dynamic model averaging vs best single model", "replication_prediction", "tpp",
           "held-out NLL per aggressive order", "NLL(single model selected by inner validation) - NLL(fixed-share "
           "DMA, share 0.01)", "superiority", 0.005, "nats per event", streams=rs, pools=["calibrate", "confirm"],
           fidelity=["F0"], evidence="E4", capability=L2C, promotion=REP, confirm_clusters=R12, v09="H5"),
        _h("H3", "Replicate execution-aware fill-probability calibration", "replication_prediction", "fills",
           "Brier score of back-of-queue touch clearance within 5 s", "Brier(generic) - Brier(execution-aware), "
           "v0.9 estimator, calibrated on the calibrate pool; also reported against the v1 corrected book",
           "superiority", 0.005, "Brier score", streams=rs, pools=["calibrate", "confirm"], fidelity=["F0", "F4"],
           evidence="E4", capability=L2C + ["queue_clearance"], promotion=REP, confirm_clusters=R12, v09="H6",
           notes="audit: the v0.9 generic baseline was degenerate on ETHUSDT; per-stream values are reported"),
        _h("H4", "Replicate queue-survival models vs Kaplan-Meier", "replication_prediction", "survival",
           "integrated Brier score of touch-clearance time", "IBS(Kaplan-Meier) - IBS(best registered survival "
           "model), trained on the calibrate pool", "superiority", 0.005, "integrated Brier score", streams=rs,
           pools=["calibrate", "confirm"], fidelity=["F0"], evidence="E4", capability=L2C + ["queue_clearance"],
           promotion=REP, confirm_clusters=R12, v09="H7"),
        _h("H5", "Replicate fill calibration to execution regret on the hybrid twin", "replication_prediction",
           "fills", "regret of passive-mandate decisions inside the F3 twin", "regret(generic fills) - "
           "regret(calibrated fills), bps", "superiority", 0.25, "bps of arrival price", streams=rs,
           pools=["calibrate", "confirm"], fidelity=["F3"], evidence="E3", capability=L2C, promotion=REP,
           confirm_clusters=R12, v09="H8"),
        _h("H6", "Replicate continual calibration against forgetting", "replication_prediction", "continual",
           "NLL per event on earlier days after sequential updates", "NLL(naive fine-tuning) - NLL(replay-based "
           "continual calibration) on every earlier day of the sequence", "superiority", 0.005, "nats per event",
           streams=rs, pools=["confirm"], fidelity=["F0"], evidence="E4", capability=L2C, promotion=REP,
           confirm_clusters=R12, v09="H16"),
        _h("H7", "Replicate adaptive conformal coverage under drift (non-inferiority)", "replication_prediction",
           "conformal", "coverage of 90% adaptive conformal intervals of 10 s mid moves", "coverage - 0.90",
           "noninferiority", 0.0, "coverage", margin=0.02, streams=rs, pools=["calibrate", "confirm"],
           fidelity=["F0"], evidence="E4", capability=["l1"], promotion=REP, confirm_clusters=R12, v09="H17"),
        _h("H8", "Replicate CVaR-MPC vs TWAP tail cost in bounded replay", "replication_execution", "bounds",
           "conservative-bound CVaR90 implementation shortfall difference (C1)", "conservative bound over the "
           "registered assumption grid of CVaR90(TWAP) - CVaR90(MPC-CVaR), medium mandate", "superiority", 1.0,
           "bps of arrival price", streams=rs, pools=["confirm"], fidelity=["F2"], evidence="E5",
           capability=L2C, promotion=REP, confirm_clusters=R12, v09="H23", conclusions=["C1"]),
        _h("H9", "Replicate synthetic-to-hybrid ranking agreement under new conditions", "replication_fidelity",
           "multifidelity", "pairwise policy-ranking sign agreement between F4 and F3 on the same day",
           "agreement - 0.5", "agreement", 0.15, "agreement", level=0.5, streams=rs,
           pools=["calibrate", "confirm"], fidelity=["F3", "F4"], evidence="E3", capability=L2C, promotion=REP,
           confirm_clusters=R12, v09="H29", notes="new conditions: generators use the corrected v1 book model"),
        _h("H10", "Replicate temporal transfer of the carried execution conclusions", "replication_execution",
           "bounds", "conservative-bound C1-C3 effects on a later period", "per conclusion, the registered "
           "conservative-bound unit value on the replicate pool (BTCUSDT linear)", "transfer", None,
           "per conclusion (bps)", streams=["bybit-linear:BTCUSDT"], pools=["replicate"], fidelity=["F2"],
           evidence="E6", capability=L2C, promotion=REP, confirm_clusters=R12, v09="H32",
           conclusions=["C1", "C2", "C3"]),
        _h("H11", "Replicate cross-instrument transfer independently", "replication_execution", "bounds",
           "conservative-bound C1-C3 effects on an instrument never used in v0.9 confirmation", "per conclusion, "
           "the conservative-bound unit value on SOLUSDT linear, confirm pool", "transfer", None,
           "per conclusion (bps)", streams=["bybit-linear:SOLUSDT"], pools=["confirm"], fidelity=["F2"],
           evidence="E7", capability=L2C, promotion=REP, confirm_clusters=R12, v09="H33",
           conclusions=["C1", "C2", "C3"]),
        _h("H12", "Reevaluate v0.9 H36 with date clustering, new data and the full fidelity ladder",
           "replication_execution", "bounds", "C1-C3 on the untouched v1 vault, all L2 streams, date clusters",
           "per conclusion, the date-cluster mean of the conservative bound over the vault streams; the ladder "
           "status (F2, F3, F4) of each conclusion is reported and F3 must not contradict (sign) for ESTABLISHED",
           "transfer", None, "per conclusion (bps)", streams=allL2, pools=["vault"], fidelity=["F2", "F3", "F4"],
           evidence="E9", capability=L2C, promotion=REP, confirm_clusters=R12, v09="H36",
           conclusions=["C1", "C2", "C3"]),
        # ------------------------------------------------------------------ reconstruction and realism
        _h("H13", "Corrected spread-improvement estimation materially reduces spread mismatch",
           "reconstruction_realism", "realism", "absolute error of the simulated mean spread against the day",
           "log(|spread_v09 book - real| / |spread_v1 book - real|), generators fitted on the calibrate pool, "
           "same counts and seeds", "superiority", 0.6931, "log error ratio (log 2 = halving)", streams=allL2,
           pools=["explore", "confirm"], fidelity=["F4"], evidence="E4", capability=L2C),
        _h("H14", "Event-level reconstruction improves prediction over coarser aggregation",
           "reconstruction_realism", "reconstruction", "log loss of next-1 s mid-move sign classifier",
           "logloss(1 s aggregated order-flow features) - logloss(100 ms event-level features)", "superiority",
           0.005, "nats", streams=allL2, pools=["explore", "calibrate", "confirm"], fidelity=["F0"],
           evidence="E4", capability=L2C),
        _h("H15", "Genuine MBO information narrows queue-position uncertainty versus L2", "reconstruction_realism",
           "mbo", "relative width of the L2 size-ahead interval of hypothetical back-of-queue touch orders 30 s "
           "after joining, where MBO identifies the exact position", "mean (ahead_pessimistic - ahead_optimistic) "
           "/ initial ahead under L2 aggregation of the same MBO day", "superiority", 0.1, "fraction of queue",
           streams=MBO_POOLS["streams"], pools=["mbo_explore", "mbo_confirm"], fidelity=["F0"], evidence="E4",
           capability=["l3", "queue_position"]),
        _h("H16", "State-dependent cancellation improves conditional market realism", "reconstruction_realism",
           "realism", "held-out conditional log-likelihood of per-bin cancel counts", "NLL(constant-rate cancel "
           "model) - NLL(state-dependent Poisson GLM on queue size, imbalance, spread)", "superiority", 0.005,
           "nats per bin", streams=allL2, pools=["explore", "calibrate", "confirm"], fidelity=["F0"],
           evidence="E4", capability=L2C),
        _h("H17", "Latent-liquidity uncertainty improves calibrated fill-interval coverage",
           "reconstruction_realism", "liquidity", "coverage of nominal 90% clearance-probability intervals",
           "|coverage(point model, hidden = 0) - 0.9| - |coverage(evidence-based hidden interval) - 0.9|",
           "superiority", 0.02, "coverage", streams=allL2, pools=["explore", "calibrate", "confirm"],
           fidelity=["F0"], evidence="E4", capability=L2C + ["queue_clearance"]),
        _h("H18", "Better market reconstruction reduces historical counterfactual ambiguity",
           "reconstruction_realism", "bounds", "width of the F2 interval of C1-C3 unit effects", "width(v0.9 "
           "hidden-fraction grid {0, 0.5}) - width(evidence-based hidden interval from M27), mean over C1-C3",
           "superiority", 0.1, "bps", streams=rs, pools=["explore", "confirm"], fidelity=["F2"], evidence="E5",
           capability=L2C),
        _h("H19", "Conditional metrics detect failures missed by marginal distribution scores",
           "reconstruction_realism", "realism", "share of generator x capability cells passing marginal tests "
           "but failing conditional tests", "fraction of registered cells (generators x capabilities) accepted by "
           "the marginal test and rejected by the conditional/temporal test, per date", "superiority", 0.1,
           "proportion", streams=allL2, pools=["explore", "confirm"], fidelity=["F4", "F5"], evidence="E4",
           capability=L2C),
        _h("H20", "Execution-aware realism gates predict ranking reliability better than distribution-only gates",
           "reconstruction_realism", "multifidelity", "Kendall agreement of F4 and F2 policy rankings",
           "agreement(generators accepted by the execution-aware gate) - agreement(generators accepted by the "
           "distribution-only gate)", "superiority", 0.1, "Kendall tau", streams=rs,
           pools=["explore", "calibrate", "confirm"], fidelity=["F2", "F4"], evidence="E4", capability=L2C),
        # ------------------------------------------------------------------ generative models
        _h("H21", "Marked Hawkes improves conditional event realism over unmarked baselines", "generators", "tpp",
           "held-out NLL per aggressive order including the size mark", "NLL(unmarked Hawkes + independent "
           "marks) - NLL(marked Hawkes with mark-dependent excitation)", "superiority", 0.005, "nats per event",
           streams=allL2, pools=["explore", "calibrate", "confirm"], fidelity=["F0"], evidence="E4",
           capability=L2C),
        _h("H22", "Neural point processes improve held-out event likelihood on new instruments", "generators",
           "tpp", "held-out NLL per aggressive order", "NLL(best classical incl. v1 marked Hawkes) - NLL(best "
           "neural), SOLUSDT linear and BTC/ETH spot", "superiority", 0.02, "nats per event",
           streams=["bybit-linear:SOLUSDT", "bybit-spot:BTCUSDT", "bybit-spot:ETHUSDT"],
           pools=["explore", "calibrate", "confirm"], fidelity=["F0"], evidence="E4", capability=L2C),
        _h("H23", "Autoregressive sequence generators improve multiscale event consistency", "generators",
           "realism", "error of trade-count dispersion at 1, 10 and 60 s", "log(mean relative dispersion error of "
           "the best classical generator / that of the autoregressive neural generator)", "superiority", 0.1,
           "log error ratio", streams=allL2, pools=["explore", "calibrate", "confirm"], fidelity=["F4", "F5"],
           evidence="E4", capability=L2C),
        _h("H24", "Latent stochastic models improve path realism", "generators", "realism",
           "signature MMD of 5 min mid paths against the day", "log(MMD(best classical) / MMD(latent SDE))",
           "superiority", 0.1, "log MMD ratio", streams=allL2, pools=["explore", "calibrate", "confirm"],
           fidelity=["F4", "F5"], evidence="E4", capability=L2C),
        _h("H25", "State-conditioned neural generators improve spread and depth dynamics", "generators",
           "realism", "max normalized error of spread/depth capability metrics", "log(error(corrected classical) "
           "/ error(state-conditioned neural))", "superiority", 0.1, "log error ratio", streams=allL2,
           pools=["explore", "calibrate", "confirm"], fidelity=["F4", "F5"], evidence="E4", capability=L2C),
        _h("H26", "Model averaging improves out-of-time predictive performance", "generators", "tpp",
           "held-out NLL per aggressive order, out of time", "NLL(best single model selected on calibrate) - "
           "NLL(fixed-share DMA), replicate-pool days", "superiority", 0.005, "nats per event", streams=rs,
           pools=["explore", "calibrate", "replicate"], fidelity=["F0"], evidence="E6", capability=L2C),
        _h("H27", "Multi-regime calibration improves cross-regime generalization", "generators", "regimes",
           "held-out NLL of per-bin event counts on days of a regime other than the training majority",
           "NLL(pooled calibration) - NLL(regime-conditioned calibration)", "superiority", 0.005, "nats per bin",
           streams=allL2, pools=["explore", "calibrate", "confirm"], fidelity=["F0", "F4"], evidence="E4",
           capability=L2C),
        _h("H28", "At least one generator satisfies pre-frozen execution-relevant realism requirements on "
           "independent holdout data", "generators", "realism", "per date, whether the generator pre-selected on "
           "the calibrate pool is ACCEPTABLE on every execution capability", "pass indicator - 0.5", "agreement",
           0.0, "pass rate", level=0.5, streams=allL2, pools=["calibrate", "confirm"], fidelity=["F4", "F5"],
           evidence="E4", capability=L2C, notes="the gate thresholds are frozen here and never moved"),
        # ------------------------------------------------------------------ twins and decision agreement
        _h("H29", "Hybrid replay reduces the reality gap against fully synthetic replay", "twins_agreement",
           "multifidelity", "absolute gap to the F2 effect of C1-C3", "mean over C1-C3 of |F4 - F2| - |F3 - F2|",
           "superiority", 0.5, "bps", streams=rs, pools=["explore", "calibrate", "confirm"],
           fidelity=["F2", "F3", "F4"], evidence="E3", capability=L2C),
        _h("H30", "Realism improvements increase synthetic-to-historical policy sign agreement",
           "twins_agreement", "multifidelity", "pairwise ranking sign agreement with F2", "agreement(F4, corrected "
           "v1 book) - agreement(F4, v0.9 book)", "superiority", 0.1, "agreement", streams=rs,
           pools=["explore", "calibrate", "confirm"], fidelity=["F2", "F4"], evidence="E4", capability=L2C),
        _h("H31", "Multi-fidelity consensus predicts the bounded historical direction better than synthetic-only",
           "twins_agreement", "multifidelity", "accuracy of predicted F2 pairwise directions",
           "accuracy(F3+F4 consensus) - accuracy(F4 only)", "superiority", 0.1, "accuracy", streams=rs,
           pools=["explore", "calibrate", "confirm"], fidelity=["F2", "F3", "F4"], evidence="E3", capability=L2C),
        _h("H32", "Realism-gated policy selection has lower regret than ungated selection", "twins_agreement",
           "multifidelity", "F2 regret of the selected policy", "regret(policy selected by all generators) - "
           "regret(policy selected by gate-accepted generators only)", "superiority", 0.25, "bps",
           streams=rs, pools=["explore", "calibrate", "confirm"], fidelity=["F2", "F4"], evidence="E4",
           capability=L2C),
        _h("H33", "Historical-synthetic disagreement is partially explained by queue or impact assumptions",
           "twins_agreement", "multifidelity", "|F4 - F2| effect gap of C1-C3", "gap(F4 default) - gap(F4 with "
           "F2-matched conservative queue and transient impact)", "superiority", 0.25, "bps", streams=rs,
           pools=["explore", "calibrate", "confirm"], fidelity=["F2", "F4"], evidence="E4", capability=L2C),
        _h("H34", "Model-class uncertainty predicts policy-ranking instability", "twins_agreement",
           "multifidelity", "Spearman correlation between ensemble disagreement and F2 sign flips across policy "
           "pairs", "rho(disagreement, flip indicator)", "superiority", 0.1, "Spearman rho", streams=rs,
           pools=["explore", "calibrate", "confirm"], fidelity=["F2", "F4"], evidence="E4", capability=L2C),
        _h("H35", "Capability-specific acceptance reduces invalid downstream claims", "twins_agreement",
           "multifidelity", "rate of synthetic pairwise claims contradicted by F2", "contradiction rate(all "
           "generators) - contradiction rate(capability-accepted generators)", "superiority", 0.1, "rate",
           streams=rs, pools=["explore", "calibrate", "confirm"], fidelity=["F2", "F4"], evidence="E4",
           capability=L2C),
        _h("H36", "Decision-equivalent generator classes give stable execution choices", "twins_agreement",
           "multifidelity", "F2 regret difference of the policies chosen by two generator classes with different "
           "predictive fit", "|regret(class A choice) - regret(class B choice)|", "equivalence", 0.5, "bps",
           margin=0.5, streams=rs, pools=["explore", "calibrate", "confirm"], fidelity=["F2", "F4", "F5"],
           evidence="E4", capability=L2C),
        # ------------------------------------------------------------------ execution and learning
        _h("H37", "Completion-constrained policies improve completion without violating cost margins",
           "execution_learning", "bounds", "completion fraction under soft completion with a 1 bps cost margin",
           "completion(completion-aware adaptive) - completion(TWAP), with mean-cost non-inferiority (1 bps) "
           "required on the same units", "superiority", 0.02, "completion fraction", streams=rs,
           pools=["explore", "confirm"], fidelity=["F2"], evidence="E5", capability=L2C),
        _h("H38", "Robust MPC improves out-of-regime tail cost versus TWAP", "execution_learning", "bounds",
           "conservative CVaR90 difference on high-volatility dates", "CVaR90(TWAP) - CVaR90(DRO-MPC), dates whose "
           "realized volatility exceeds the calibrate-pool 75th percentile", "superiority", 1.0, "bps",
           streams=rs, pools=["explore", "calibrate", "confirm"], fidelity=["F2"], evidence="E5",
           capability=L2C),
        _h("H39", "Chance constraints improve completion reliability", "execution_learning", "bounds",
           "probability of completing a soft-completion mandate", "P(complete | chance-constrained MPC) - "
           "P(complete | stochastic MPC)", "superiority", 0.02, "probability", streams=rs,
           pools=["explore", "confirm"], fidelity=["F2"], evidence="E5", capability=L2C),
        _h("H40", "Contextual fallback improves on the failed v0.8/v0.9 abstention behaviour",
           "execution_learning", "learning", "F2 regret against the per-window best registered policy",
           "regret(best global policy on calibrate) - regret(contextual fallback)", "superiority", 0.25, "bps",
           streams=rs, pools=["explore", "calibrate", "confirm"], fidelity=["F2"], evidence="E5", capability=L2C),
        _h("H41", "Safe offline policy selection reduces harmful changes versus ungated learned policies",
           "execution_learning", "learning", "rate of deployments worse than TWAP by more than 1 bps",
           "harm rate(ungated offline RL) - harm rate(conservatively gated)", "superiority", 0.1, "rate",
           streams=rs, pools=["explore", "calibrate", "confirm"], fidelity=["F2", "F4"], evidence="E5",
           capability=L2C),
        _h("H42", "Continual adaptation improves post-drift recovery without unacceptable forgetting",
           "execution_learning", "continual", "NLL on post-drift days with a forgetting margin on pre-drift days",
           "NLL(frozen calibrate model) - NLL(continually adapted), with forgetting non-inferiority 0.01",
           "superiority", 0.005, "nats per event", streams=rs, pools=["calibrate", "confirm"], fidelity=["F0"],
           evidence="E4", capability=L2C),
        _h("H43", "Decision-focused calibration improves decision utility over likelihood-only calibration",
           "execution_learning", "decision_calibration", "F2 regret of the policy chosen inside the calibrated F4",
           "regret(likelihood-calibrated) - regret(decision-focused calibrated)", "superiority", 0.25, "bps",
           streams=rs, pools=["explore", "calibrate", "confirm"], fidelity=["F2", "F4"], evidence="E4",
           capability=L2C),
        _h("H44", "Explicit endogenous self-impact changes execution conclusions at registered mandate scales",
           "execution_learning", "multifidelity", "|C2 effect with self-impact feedback - without| in F3",
           "|effect(F3 with fitted impact) - effect(F3 without impact)|, large mandate", "superiority", 1.0,
           "bps", streams=rs, pools=["explore", "calibrate", "confirm"], fidelity=["F3"], evidence="E3",
           capability=L2C, conclusions=["C2"]),
        # ------------------------------------------------------------------ external validation
        _h("H45", "At least one execution effect replicates across independently sourced time periods",
           "external_validation", "bounds", "C1-C3 on the replicate pool, all L2 linear streams",
           "per conclusion, the date-cluster conservative bound; ESTABLISHED when at least one conclusion meets "
           "its rule after Holm", "transfer_any", None, "per conclusion (bps)",
           streams=["bybit-linear:BTCUSDT", "bybit-linear:ETHUSDT", "bybit-linear:SOLUSDT"], pools=["replicate"],
           fidelity=["F2"], evidence="E6", capability=L2C, promotion=REP, confirm_clusters=R12,
           conclusions=["C1", "C2", "C3"]),
        _h("H46", "At least one effect replicates across instruments and venue segments", "external_validation",
           "bounds", "C1-C3 on spot BTCUSDT and ETHUSDT (a different venue segment and order book)",
           "per conclusion, the date-cluster conservative bound on spot streams", "transfer_any", None,
           "per conclusion (bps)", streams=["bybit-spot:BTCUSDT", "bybit-spot:ETHUSDT"], pools=["replicate"],
           fidelity=["F2"], evidence="E8", capability=L2C, promotion=REP, confirm_clusters=R12,
           conclusions=["C1", "C2", "C3"],
           notes="Binance offers no fresh order book (bookTicker ended 2024-03-30); cross-venue is cross-segment"),
        _h("H47", "At least one realism-accepted generator preserves a preregistered execution ranking across "
           "fidelities", "external_validation", "multifidelity", "pairwise ranking sign agreement of the "
           "H28-accepted generator with F2", "agreement - 0.5 (NOT_EVALUABLE when no generator is accepted)",
           "agreement", 0.15, "agreement", level=0.5, streams=rs, pools=["calibrate", "replicate"],
           fidelity=["F2", "F4"], evidence="E6", capability=L2C),
        _h("H48", "At least one previously established execution conclusion survives the untouched vault under "
           "correct clustering", "external_validation", "bounds", "C1-C3 on the v1 vault, date clusters",
           "per conclusion, the date-cluster conservative bound over all vault L2 streams; ESTABLISHED when at "
           "least one meets its rule after Holm", "transfer_any", None, "per conclusion (bps)", streams=allL2,
           pools=["vault"], fidelity=["F2"], evidence="E9", capability=L2C, promotion=REP, confirm_clusters=R12,
           conclusions=["C1", "C2", "C3"], notes="aspirational headline; a negative result is reported as such"),
    ]


STATISTICS = {
    "independent_unit": "calendar date; values of the streams of one date form one cluster",
    "primary_test": "cluster_t: one-sided test of H0 mean <= threshold with the CR1 cluster-robust SE, Student t "
                    "on G - 1 df; Holm within the registered family at alpha 0.05",
    "secondary_test": "exact cluster sign-flip of (value - threshold); ESTABLISHED requires both the primary "
                      "Holm rejection and the sign-flip p <= the same Holm-adjusted alpha",
    "interval": "two-sided 95% cluster-t interval of the mean (reported with every estimate)",
    "min_clusters": 8, "alpha": 0.05, "strictest_family_alpha": 0.00625, "correction": "holm",
    "noninferiority": "value + margin tested against 0", "agreement": "value - level tested against threshold",
    "equivalence": "TOST with the registered margin on cluster-t intervals",
    "transfer": "every carried conclusion meets its threshold (Holm over the conclusions)",
    "transfer_any": "at least one carried conclusion meets its threshold (Holm over the conclusions)",
    "terminal_rule": {"ESTABLISHED": "registered rejection (above)",
                      "FAILED": "the upper 95% cluster-t bound lies below the threshold (effect below practical size)",
                      "INCONCLUSIVE": "tested, neither established nor failed",
                      "NOT_ESTABLISHED": "not promoted (never tested on guarded data)",
                      "NOT_AVAILABLE": "registered data unavailable after every backup",
                      "NOT_EVALUABLE": "a registered precondition is unmet (e.g. H47 without an accepted generator)"},
    "heterogeneity": "per-stream means and a date x stream two-way cluster interval are reported, not decided",
}
PROMOTION = {
    "direct_replication": "H1-H12, H45, H46, H48 are promoted without pilot-dependent selection; their confirm "
                          "cluster count is fixed here (12 dates); pilots inform only feasibility reporting",
    "pilot": "every other candidate is promoted iff (a) its required capabilities exist in the registered pools, "
             "(b) an exploratory pilot on >= 6 explore date clusters exists, (c) the planning effect (pilot cluster "
             "mean - 1 cluster SE, oriented) exceeds the practical threshold, (d) cluster-t power at the planning "
             "effect is >= 0.8 with G <= the pool's primaries and G >= min_clusters, and (e) expected compute is "
             "within the ceiling",
    "min_power": 0.8, "planning_effect": "pilot cluster mean - 1 cluster SE", "min_pilot_clusters": 6,
    "compute_ceiling_hours": 60.0, "max_confirm_clusters": 40,
    "no_split_half_screen": "v0.9 amendment 3 is not carried; neither split-half statistic is used",
    "frozen_before": "any v1 pilot outcome is read",
}
REALISM_GATE = {
    "capabilities": ["spread_dynamics", "depth_dynamics", "event_timing", "queue_survival", "fill_prediction",
                     "impact", "tail_behavior", "conditional_response"],
    "execution_capabilities": ["spread_dynamics", "depth_dynamics", "queue_survival", "fill_prediction", "impact"],
    "reference": "per metric, the 5-95% range of the same metric over the calibrate-pool days (>= 8 real days), "
                 "widened by the metric's bootstrap SE on the test day; the 4-day pair reference of v0.9 is retired",
    "acceptable": "the generator's metric lies inside the reference band for every metric of the capability and the "
                  "real-vs-generated classifier AUC <= 0.65",
    "partial": "every metric within twice the band half-width of the band centre",
    "rejected": "otherwise", "fail_closed": "a REJECTED capability never supports a claim that needs it",
    "frozen": "thresholds are never moved to create an ACCEPTABLE generator",
}


def protocol_document(base_commit: str) -> dict:
    from ...v09.registry import MANDATES as M9
    p9 = json.loads((Path(__file__).resolve().parents[3] / "configs/v09/protocol.json").read_text())
    return {
        "schema": D.SCHEMA, "protocol_id": "cleolob-v1.0-market-reality-lab", "base_commit": base_commit,
        "research_question": "Do the v0.9 conclusions replicate independently on genuinely fresh market data under "
                             "correct date clustering, and can repaired and new generator families reproduce observed "
                             "dynamics and execution-decision rankings under frozen capability-specific criteria?",
        "statuses": list(D.STATUSES), "evidence_levels": dict(D.EVIDENCE_LEVELS),
        "roles": ["consumed", "development", "explore", "calibrate", "confirm", "replicate", "vault"],
        "namespaces": ["dev", "explore", "calibrate", "confirm", "replicate", "vault"],
        "pools": {k: {**v, "dates": pool_dates(k)} for k, v in POOLS.items()},
        "mbo_pools": MBO_POOLS,
        "firewall": {"consumed_through": FW.CONSUMED_THROUGH, "excluded_dates": FW.EXCLUDED_DATES,
                     "rule": "dates on or before consumed_through and the excluded dates are never fresh; one date "
                             "belongs to one v1 pool across every stream"},
        "promotion_rules": PROMOTION, "statistics": STATISTICS, "realism_gate": REALISM_GATE,
        "fidelities": {"F0": "observational evidence", "F1": "exact historical replay", "F2": "bounded "
                       "counterfactual replay", "F3": "hybrid historical twin", "F4": "classical generative market",
                       "F5": "neural generative market", "F6": "strategic multi-agent market"},
        "carried_from_v09": {"counterfactual_assumptions": p9["counterfactual_assumptions"],
                             "mandates": p9["mandates"], "mandate_notionals": M9, "conclusions": CONCLUSIONS,
                             "cost_endpoints": p9["cost_endpoints"],
                             "completion_constraints": p9["completion_constraints"],
                             "risk_measures": p9["risk_measures"], "grid_normalization": p9["grid_normalization"],
                             "data_quality_thresholds": p9["data_quality"]["thresholds"]},
        "data_quality": {"thresholds": "lob.v1.data.daycache.QUALITY (v0.9 thresholds plus "
                                       "max_unreliable_message_rate 0.001)",
                         "gap_rule": "a sequence gap invalidates the book until the next snapshot"},
        "resources": {"concurrent_heavy_families": 1, "workers_default": 2, "min_free_gb": 8.0,
                      "ram_mb_per_study": 12000, "gpu_memory_fraction": 0.8,
                      "profiles": ["smoke", "small", "medium", "full", "replication"],
                      "on_exceed": "RESOURCE_EXHAUSTED; checkpoints kept; budget changes need an amendment before "
                                   "outcome inspection"},
        "amendment_policy": "numbered, hashed amendments; after any guarded access every amendment is posthoc, "
                            "disclosed as an incident with both analyses reported",
        "referenced_configs": ["hypotheses-candidates.json", "statistical-families.json", "dataset-registry.json",
                               "consumed-inventory.json"],
    }


def build(root: Path, base_commit: str, *, force: bool = False) -> dict:
    cfg = Path(root) / "configs/v1"
    out = {"hypotheses-candidates.json": {"schema": "cleolob-v1-candidates-1", "hypotheses": candidates()},
           "statistical-families.json": {"schema": "cleolob-v1-families-1", "families": {
               k: {**v, "alpha": 0.05, "correction": "holm"} for k, v in FAMILIES.items()}},
           "dataset-registry.json": dataset_registry()}
    written = {}
    for name, doc in out.items():
        p = cfg / name
        if p.exists() and not force:
            raise FileExistsError(f"{p} exists (configs are written once before the freeze)")
        D.write_json(p, doc)
        written[name] = D.file_sha256(p)
    doc = protocol_document(base_commit)
    issues = D.validate(doc)
    if issues:
        raise D.ProtocolError("; ".join(issues))
    return {"configs": written, "protocol": doc}


def check() -> list[str]:
    issues = []
    hs = candidates()
    ids = [h["id"] for h in hs]
    if ids != [f"H{i}" for i in range(1, 49)]:
        issues.append("candidates must be H1..H48 in order")
    fam = {m: f for f, v in FAMILIES.items() for m in v["members"]}
    for h in hs:
        if fam.get(h["id"]) != h["family"]:
            issues.append(f"{h['id']}: family mismatch")
        if h["test"] in ("superiority", "agreement") and h["threshold"] is None:
            issues.append(f"{h['id']}: missing threshold")
        if h["test"].startswith("transfer") and not h["conclusions"]:
            issues.append(f"{h['id']}: transfer without conclusions")
        if h["promotion"] == REP and not h["confirm_clusters"]:
            issues.append(f"{h['id']}: direct replication without a fixed cluster count")
        if h["confirm_clusters"] and h["confirm_clusters"] < STATISTICS["min_clusters"]:
            issues.append(f"{h['id']}: fewer clusters than min_clusters")
    all_dates = [d for k in POOLS for d in pool_dates(k)]
    if len(all_dates) != len(set(all_dates)):
        issues.append("a date appears in two pools")
    if any(not FW.date_fresh(d)[0] for d in all_dates):
        issues.append("a non-fresh date is in a pool")
    return issues


__all__ = ["POOLS", "MBO_POOLS", "L2_STREAMS", "FAMILIES", "CONCLUSIONS", "candidates", "pool_dates",
           "dataset_registry", "protocol_document", "build", "check", "STATISTICS", "PROMOTION", "REALISM_GATE"]
