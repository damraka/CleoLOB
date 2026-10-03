"""M11 regime, drift and calibration half-life study (workstreams 20-23). Development/validation/retrospective data.

Dated sequence (months after the development day): ETH 2020-04-01 (0), 06-01 (2), 07-01 (3), 08-01 (4),
09-01 (5), 10-01 (6). The selection day (05-01) is excluded: its role permits only selection.

* observable drift — v0.6 real-vs-real distance (objective and family errors) of each day to the development day;
* support drift — fraction of each day's 60 s windows outside the development-day window support;
* change points — binary segmentation of 300 s log realized volatility within each day;
* latent regimes — Gaussian HMMs (k = 1, 2, 3) fitted on development 300 s blocks, scored on later days, against
  the frozen v0.6 discrete volatility-threshold regime model (same features, fixed labels);
* performance decay and half-life — objective (generic and ES) of every banked model per day, with
  block-bootstrap draws per day; half-life per model and metric (exploratory, not preregistered);
* posterior drift — NOT_AVAILABLE: re-running SMC-ABC per day would cost 6 x 1,536 simulations (outside the
  registered budget); parameter drift is reported through the G1 refits of the transfer matrix (M12).
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from ...experiments.registry import PROJECT_ROOT
from ...v06.domain_gap import FEATURES, Standardizer, support, window_features
from ...v06.history import block_raws
from ...v06.identifiability import clean
from ...v06.inference import counts
from ...v06.observables import stack, weighted
from ...v06.realism import compare
from ..data import access
from ..evidence.runs import finalize, new_run
from ..generators.study import v06_inputs
from ..realism.contrast import component_scales, es_objective
from ..realism.holdout import load_bank
from ..regimes.hmm import GaussianHMM
from ..v06_compat import pool, sketch
from . import analysis

SEQUENCE = (("deribit-eth-perp-2020-04-01", "develop", 0), ("deribit-eth-perp-2020-06-01", "validate", 2),
            ("deribit-eth-perp-2020-07-01", "retrospective", 3), ("deribit-eth-perp-2020-08-01", "retrospective", 4),
            ("deribit-eth-perp-2020-09-01", "retrospective", 5), ("deribit-eth-perp-2020-10-01", "retrospective", 6))
DESIGN = "m11-regime-drift"
DRAWS = 200
SEED = 79601
BLOCK_FEATURES = ("return_sd", "spread_mean", "log1p_trades")


def _block_features(tape) -> np.ndarray:
    w = window_features(tape, 300.0)
    idx = [FEATURES.index(f) for f in BLOCK_FEATURES]
    x = w[:, idx].copy()
    x[:, 0] = np.log(np.maximum(x[:, 0], 1e-3))
    x[:, 1] = np.log(np.maximum(x[:, 1], 1e-3))
    return x


def discrete_baseline(dev_x: np.ndarray) -> GaussianHMM:
    """Two-state model whose states are fixed by a volatility threshold (median of development log-vol)."""
    labels = (dev_x[:, 0] > np.median(dev_x[:, 0])).astype(int)
    model = GaussianHMM(2)
    model.means = np.asarray([dev_x[labels == k].mean(0) for k in (0, 1)])
    model.vars = np.asarray([np.maximum(dev_x[labels == k].var(0), 1e-3) for k in (0, 1)])
    switches = np.zeros((2, 2))
    for a, b in zip(labels, labels[1:]):
        switches[a, b] += 1
    model.trans = (switches + 0.5) / (switches + 0.5).sum(1, keepdims=True)
    model.start = np.bincount(labels, minlength=2) / len(labels)
    return model


def run(out: str | Path, *, bank_run: str, root: Path = PROJECT_ROOT) -> dict:
    inputs = v06_inputs(root)
    frozen = inputs["frozen"]
    banked = load_bank(root / bank_run)
    models = {k: pool(v) for k, v in banked["sketches"].items()}
    es_scales = component_scales(frozen)
    out = new_run(root / out)
    rng = np.random.default_rng(SEED)
    per_day, dev_hist, dev_windows, dev_x, hmms, base_hmm = {}, None, None, None, {}, None
    for dataset, use, month in SEQUENCE:
        tape, _ = access.load_tape(dataset, use=use, design=DESIGN, purpose="regime/drift chronology", root=root,
                                   scale=frozen["scale_native_per_lot"])
        raws, _ = block_raws(tape)
        blocks = [sketch(r, frozen["design"]) for r in raws]
        hist = pool(blocks)
        windows = window_features(tape)
        x = _block_features(tape)
        vol = x[:, 0]
        if month == 0:
            dev_hist, dev_windows, dev_x = hist, windows, x
            hmms = {k: GaussianHMM(k, seed=SEED).fit(x) for k in (1, 2, 3)}
            base_hmm = discrete_baseline(x)
        drift = compare(dev_hist, hist, frozen["design"], frozen["objective_scales"]) if month else None
        chunks = [dev_windows[i::8] for i in range(8)]
        supp = support(windows, chunks, Standardizer(dev_windows)) if month else None
        stacked = stack(blocks)
        decay = {}
        for name, sim in models.items():
            point = compare(hist, sim, frozen["design"], frozen["objective_scales"])
            draws_g, draws_e = [], []
            for _ in range(DRAWS):
                h = weighted(stacked, counts(len(blocks), rng))
                c = compare(h, sim, frozen["design"], frozen["objective_scales"])
                draws_g.append(c["objective"])
                draws_e.append(es_objective(c, es_scales))
            decay[name] = {"objective": point["objective"], "es_objective": es_objective(point, es_scales),
                           "families": {f: v["error"] for f, v in point["families"].items()},
                           "draws_objective": draws_g, "draws_es": draws_e}
        per_day[dataset] = {
            "month": month, "role_use": use, "blocks": len(blocks),
            "observable_drift": None if drift is None else {"objective": drift["objective"],
                                                            "families": {f: v["error"] for f, v in drift["families"].items()}},
            "support_drift": None if supp is None else {k: supp[k] for k in ("out_of_support_fraction", "label")},
            "change_points": analysis.change_points(vol),
            "hmm_loglik_per_block": {str(k): m.score(x) / len(x) for k, m in hmms.items()},
            "discrete_baseline_loglik_per_block": base_hmm.score(x) / len(x),
            "decay": decay}
        del tape, raws, blocks, stacked
        gc.collect()
    months = [d["month"] for d in per_day.values()]
    half = {}
    for name in models:
        for metric, draws_key in (("objective", "draws_objective"), ("es_objective", "draws_es")):
            values = [per_day[d]["decay"][name][metric] for d in per_day]
            draws = [np.asarray(per_day[d]["decay"][name][draws_key], float) for d in per_day]
            half[f"{name}|{metric}"] = analysis.half_life(months, values, per_period_draws=draws, seed=SEED)
        for family in per_day[SEQUENCE[0][0]]["decay"][name]["families"]:
            values = [per_day[d]["decay"][name]["families"][family] for d in per_day]
            if all(v is not None for v in values):
                half[f"{name}|family:{family}"] = analysis.half_life(months, values)
    for day in per_day.values():
        for entry in day["decay"].values():
            entry.pop("draws_objective")
            entry.pop("draws_es")
    hmm_summary = {str(k): {"parameters": m.parameters, "development_bic": m.bic(dev_x),
                            "transition": m.trans.round(4).tolist(), "means": m.means.round(4).tolist()}
                   for k, m in hmms.items()}
    hmm_summary["discrete_baseline"] = {"development_bic": base_hmm.bic(dev_x), "parameters": base_hmm.parameters}
    result = {"sequence": [{"dataset": d, "month": m, "use": u} for d, u, m in SEQUENCE], "days": per_day,
              "half_life": half, "hmm": hmm_summary,
              "posterior_drift": "NOT_AVAILABLE: SMC-ABC per day exceeds the registered budget; see M12 G1 refits",
              "continuous_vs_discrete": "reported from the fresh holdout evaluation (G1/G3/G4 continuous state "
                                        "conditioning vs G2 discrete regimes vs G0 unconditioned) and here per day",
              "label": "RETROSPECTIVE/DEVELOPMENT; half-life is EXPLORATORY (not a registered hypothesis)",
              "interpretation": "latent states are statistical clusters of observables, not participant identities"}
    finalize(out, analysis="m11-regime-drift", dataset_ids=[d for d, _, _ in SEQUENCE],
             config={"bank_run": bank_run, "draws": DRAWS, "seed": SEED}, result=clean(json.loads(json.dumps(
                 result, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))), root=root,
             seeds={"bootstrap": SEED})
    return result
