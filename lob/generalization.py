"""Registered chronological observable calibration; no execution/fill validity claim.

Development data and later external data are loaded separately. Candidate choice
uses only development train/validation folds. A hash seal is written before
internal or external observations are scored. Existing historical holdouts are
consumed evidence and are not used by the redistributable synthetic smoke study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import FEATURES, _frame_hash, _interval, _validate_frame
from .experiments.registry import PROJECT_ROOT, runtime_metadata, sha256_file, write_json


FAMILIES = ("iid_joint", "spread_markov")
WINDOWS = ("expanding", "rolling")
OBSERVABLES = {
    "spread_bps": ("(best ask - best bid) / mid * 10000", "basis points", "valid sample"),
    "bid_depth5": ("sum of visible top-five bid quantities", "native quantity", "valid sample"),
    "ask_depth5": ("sum of visible top-five ask quantities", "native quantity", "valid sample"),
    "imbalance5": ("(bid depth - ask depth) / total depth", "dimensionless", "valid sample"),
    "log_return": ("log(mid[t] / mid[t-1])", "log ratio", "two consecutive valid samples"),
    "rms_return_10": ("sqrt(mean(last 10 squared log returns))", "log ratio",
                      "10 consecutive finite returns; trailing only"),
    "spread_change_bps": ("spread[t] - spread[t-1]", "basis points",
                          "two consecutive valid samples"),
    "relative_depth_change": ("(total depth[t] - total depth[t-1]) / total depth[t-1]",
                              "dimensionless", "two consecutive valid samples"),
}
GATE = {
    "distance_warning_above": .25, "distance_fail_above": .50,
    "coverage_warning_below": .90, "coverage_fail_below": .75,
    "valid_warning_below": .99, "valid_fail_below": .90,
    "minimum_finite_samples": 32,
}
LIMITATIONS = [
    "Observable generators are not execution simulators or validated historical fill models.",
    "Visible depth changes are net changes, not individual replenishment or cancellations.",
    "No event inter-arrival, trade/cancellation intensity or true OFI is inferred from sampled L2.",
    "The two-state spread model captures only first-order spread-regime persistence.",
    "No intraday seasonality is estimated from the short synthetic fixture.",
    "Diagnostic thresholds are fixed heuristics, not independent-observation hypothesis tests.",
    "Synthetic shifted-holdout evidence does not replace consumed May 2020 or September 2026 failures.",
]


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _integer(value, name: str, lower: int = 1, upper: int = 1_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def _section(frame: pd.DataFrame, start: int, stop: int) -> pd.DataFrame:
    result = frame.iloc[start:stop].copy().reset_index(drop=True)
    if not result.empty:
        result.loc[0, "log_return"] = np.nan
    return result


def observe(frame: pd.DataFrame, interval_us: int) -> pd.DataFrame:
    """Derive only causal observables; never bridge invalid samples, gaps or splits."""
    valid = _validate_frame(frame)
    _interval(frame, {"sample_interval_us": interval_us})
    times = frame.timestamp_us.to_numpy()
    consecutive = np.r_[False, valid[1:] & valid[:-1] & (np.diff(times) == interval_us)]
    expected = np.full(len(frame), np.nan)
    mids = frame.mid_price.to_numpy()
    indices = np.flatnonzero(consecutive)
    expected[indices] = np.log(mids[indices] / mids[indices - 1])
    actual = frame.log_return.to_numpy()
    if not np.allclose(actual[consecutive], expected[consecutive], rtol=1e-6, atol=1e-12,
                       equal_nan=False):
        raise ValueError("log_return must match consecutive mid prices")
    depth = frame.bid_depth5 + frame.ask_depth5
    imbalance = (frame.bid_depth5 - frame.ask_depth5) / depth
    if not np.allclose(frame.imbalance5.to_numpy()[valid], imbalance.to_numpy()[valid], atol=1e-12):
        raise ValueError("imbalance5 must match visible side depths")
    result = frame[list(FEATURES)].copy()
    # Reset rolling windows on every missing return, including a grid gap.
    result["rms_return_10"] = np.sqrt(result.log_return.pow(2).rolling(10, min_periods=10).mean())
    result["spread_change_bps"] = frame.spread_bps.diff().where(consecutive)
    result["relative_depth_change"] = (depth.diff() / depth.shift()).where(consecutive)
    return result


def chronological_folds(frame: pd.DataFrame, *, initial_train_rows: int,
                        validation_rows: int, folds: int, window: str,
                        embargo_us: int) -> list[dict]:
    """Fixed nonoverlapping validation blocks and elapsed-time train embargoes."""
    _validate_frame(frame)
    _integer(initial_train_rows, "initial_train_rows", 64)
    _integer(validation_rows, "validation_rows", 64)
    _integer(folds, "folds", 2, 20)
    _integer(embargo_us, "embargo_us", 0, 86_400_000_000)
    if window not in WINDOWS:
        raise ValueError("window must be expanding or rolling")
    if initial_train_rows + folds * validation_rows != len(frame):
        raise ValueError("declared folds must cover the entire selection period")
    output = []
    times = frame.timestamp_us.to_numpy()
    for index in range(folds):
        boundary = initial_train_rows + index * validation_rows
        stop = int(np.searchsorted(times, int(times[boundary]) - embargo_us, side="left"))
        start = max(0, stop - initial_train_rows) if window == "rolling" else 0
        if stop - start < 64:
            raise ValueError("embargo removes too much fold training data")
        output.append({"fold": index, "train": _section(frame, start, stop),
                       "validation": _section(frame, boundary, boundary + validation_rows)})
    return output


def fit_observable_model(frame: pd.DataFrame, *, family: str, interval_us: int,
                         seed: int, max_pool_rows: int = 256) -> dict:
    """Fit bounded train-only joint pools and, optionally, spread transition rates."""
    if family not in FAMILIES:
        raise ValueError("unknown observable model family")
    _integer(seed, "seed", 0, 2**63 - 1)
    _integer(max_pool_rows, "max_pool_rows", 32, 4096)
    observables = observe(frame, interval_us)
    complete = np.isfinite(frame[list(FEATURES)].to_numpy()).all(axis=1)
    pool = frame.loc[complete, list(FEATURES)].to_numpy()
    if len(pool) < 64:
        raise ValueError("model requires 64 complete training observations")
    threshold = float(np.median(pool[:, 0]))
    states = (frame.spread_bps.to_numpy() > threshold).astype(int)
    consecutive = np.r_[False, complete[1:] & complete[:-1]
                        & (np.diff(frame.timestamp_us.to_numpy()) == interval_us)]
    counts = np.ones((2, 2))  # Fixed Laplace prior; never chosen from validation.
    for index in np.flatnonzero(consecutive):
        counts[states[index - 1], states[index]] += 1
    transitions = counts / counts.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    pools = []
    for state in range(2 if family == "spread_markov" else 1):
        values = pool[(pool[:, 0] > threshold) == state] if family == "spread_markov" else pool
        if len(values) < 16:
            raise ValueError("spread regime has fewer than 16 training observations")
        if len(values) > max_pool_rows:
            values = values[np.sort(rng.choice(len(values), max_pool_rows, replace=False))]
        pools.append(values.tolist())
    scales, train_quantiles = {}, {}
    for name in OBSERVABLES:
        values = observables[name].dropna().to_numpy()
        if len(values) < GATE["minimum_finite_samples"]:
            raise ValueError(f"insufficient training observations: {name}")
        scales[name] = float(max(np.quantile(values, .75) - np.quantile(values, .25),
                                 values.std(), abs(values.mean()) * .01, 1e-12))
        train_quantiles[name] = np.quantile(values, np.linspace(.01, .99, 51)).tolist()
    model = {"family": family, "interval_us": interval_us, "pools": pools,
             "threshold_spread_bps": threshold, "transition_probabilities": transitions.tolist(),
             "initial_high_probability": float(np.mean(pool[:, 0] > threshold)),
             "initial_mid_price": float(frame.loc[frame.valid, "mid_price"].iloc[-1]),
             "train_start_us": int(frame.timestamp_us.iloc[0]),
             "train_end_us": int(frame.timestamp_us.iloc[-1]), "train_rows": len(frame),
             "train_valid_fraction": float(frame.valid.mean()),
             "train_complete_rows": int(complete.sum()),
             "train_frame_sha256": _frame_hash(frame), "scales": scales,
             "train_quantiles": train_quantiles,
             "parameter_vector": [threshold, float(pool[:, 1].mean()), float(pool[:, 2].mean()),
                                  float(pool[:, 4].std()), float(transitions[0, 1]),
                                  float(transitions[1, 0])],
             "fit_seed": seed, "max_pool_rows_per_state": max_pool_rows}
    model["model_sha256"] = _digest(model)
    return model


def _check_model(model: dict) -> None:
    if model.get("model_sha256") != _digest({k: v for k, v in model.items() if k != "model_sha256"}):
        raise ValueError("frozen model hash mismatch")


def generate_observables(model: dict, n: int, seed: int) -> pd.DataFrame:
    """Generate a fresh autonomous path without reading any target states or rows."""
    _check_model(model)
    _integer(n, "simulation_rows", 64, 100_000)
    _integer(seed, "simulation_seed", 0, 2**63 - 1)
    rng = np.random.default_rng(seed)
    pools = [np.asarray(values) for values in model["pools"]]
    state = int(rng.random() < model["initial_high_probability"])
    rows = []
    for _ in range(n):
        pool = pools[state] if model["family"] == "spread_markov" else pools[0]
        rows.append(pool[rng.integers(len(pool))])
        if model["family"] == "spread_markov":
            state = int(rng.random() >= model["transition_probabilities"][state][0])
    frame = pd.DataFrame(rows, columns=FEATURES)
    prices = math.log(model["initial_mid_price"]) + np.cumsum(frame.log_return.to_numpy())
    if (np.abs(prices) > 700).any():
        raise ValueError("generated path exceeds numeric price bounds")
    frame["mid_price"] = np.exp(prices)
    frame["timestamp_us"] = np.arange(n, dtype=np.int64) * model["interval_us"]
    frame["valid"] = True
    frame.loc[0, "log_return"] = np.nan
    return observe(frame, model["interval_us"])


def _status(distance: float, coverage: float, gate: dict) -> str:
    if distance > gate["distance_fail_above"] or coverage < gate["coverage_fail_below"]:
        return "FAIL"
    if distance > gate["distance_warning_above"] or coverage < gate["coverage_warning_below"]:
        return "WARNING"
    return "PASS"


def evaluate_observable_model(model: dict, frame: pd.DataFrame, *, seeds: list[int],
                              simulation_rows: int, gate: dict | None = None) -> dict:
    """Diagnostic quantile distances, central-95% coverage and train-relative drift."""
    _check_model(model)
    gate = dict(GATE if gate is None else gate)
    if frame.timestamp_us.iloc[0] <= model["train_end_us"]:
        raise ValueError("evaluation must be strictly later than training")
    if not seeds or len(set(seeds)) != len(seeds) or model["fit_seed"] in seeds:
        raise ValueError("evaluation seeds must be distinct and disjoint from fitting")
    actual = observe(frame, model["interval_us"])
    generated = pd.concat([generate_observables(model, simulation_rows, seed) for seed in seeds])
    metrics, attribution, losses, statuses = {}, [], [], []
    for name in OBSERVABLES:
        target = actual[name].dropna().to_numpy()
        prediction = generated[name].dropna().to_numpy()
        if len(target) < gate["minimum_finite_samples"]:
            metrics[name] = {"status": "FAIL", "finite_samples": len(target),
                             "reason": "insufficient_observable_coverage"}
            attribution.append(f"{name}: insufficient_observable_coverage")
            losses.append(1_000_000.0)
            statuses.append("FAIL")
            continue
        probabilities = np.linspace(.01, .99, 51)
        observed_q, predicted_q = np.quantile(target, probabilities), np.quantile(prediction, probabilities)
        scale = model["scales"][name]
        distance = float(np.mean(np.abs(observed_q - predicted_q)) / scale)
        lower, upper = np.quantile(prediction, [.025, .975])
        coverage = float(np.mean((target >= lower) & (target <= upper)))
        drift = float(np.mean(np.abs(observed_q - model["train_quantiles"][name])) / scale)
        status = _status(distance, coverage, gate)
        metrics[name] = {"status": status, "finite_samples": len(target),
                         "normalized_quantile_distance": distance,
                         "central_95pct_coverage": coverage, "coverage_error": float(abs(.95 - coverage)),
                         "train_relative_quantile_drift": drift, "train_normalization_scale": scale}
        losses.append(distance + max(0, .95 - coverage))
        statuses.append(status)
        if status != "PASS":
            attribution.append(f"{name}: " + ("regime_shift_or_support_failure" if drift > .5
                                               else "within_regime_model_misspecification_or_sampling_error"))
    valid_fraction = float(frame.valid.mean())
    data_status = ("FAIL" if valid_fraction < gate["valid_fail_below"] else
                   "WARNING" if valid_fraction < gate["valid_warning_below"] else "PASS")
    statuses.append(data_status)
    if data_status != "PASS":
        attribution.append("input: invalid_or_stale_observations")
    return {"status": "FAIL" if "FAIL" in statuses else "WARNING" if "WARNING" in statuses else "PASS",
            "loss": float(np.mean(losses)), "metrics": metrics, "valid_fraction": valid_fraction,
            "failure_attribution": attribution, "model_sha256": model["model_sha256"],
            "frame_sha256": _frame_hash(frame), "start_us": int(frame.timestamp_us.iloc[0]),
            "end_us": int(frame.timestamp_us.iloc[-1]), "rows": len(frame), "simulation_seeds": seeds,
            "simulation_rows_per_seed": simulation_rows, "model_unchanged": True}


def synthetic_fixture(descriptor: dict, interval_us: int) -> pd.DataFrame:
    """Redistributable two-state synthetic observations; explicitly not exchange data."""
    n = _integer(descriptor["rows"], "rows", 64)
    rng = np.random.default_rng(descriptor["seed"])
    state = 0
    states = []
    for _ in range(n):
        if rng.random() < .04:
            state = 1 - state
        states.append(state)
    states = np.asarray(states)
    shift = descriptor.get("shift", 1.0)
    spread = (1 + 2 * states + rng.uniform(.1, .5, n)) * shift
    depth = (900 - 350 * states) * rng.lognormal(0, .12, n) / shift
    imbalance = rng.uniform(-.4, .4, n)
    returns = rng.normal(0, (1 + 2 * states) * 1e-4 * shift, n)
    frame = pd.DataFrame({"timestamp_us": descriptor["start_us"] + np.arange(n) * interval_us,
                          "valid": True, "mid_price": 100 * np.exp(np.cumsum(returns)),
                          "spread_bps": spread, "bid_depth5": depth * (1 + imbalance) / 2,
                          "ask_depth5": depth * (1 - imbalance) / 2,
                          "imbalance5": imbalance, "log_return": returns})
    frame.loc[0, "log_return"] = np.nan
    return frame


def _validate_config(config: dict) -> None:
    if config.get("schema_version") != 1 or config.get("kind") not in {"synthetic_smoke", "feature_csv"}:
        raise ValueError("unsupported registered study kind/schema")
    for key, lower, upper in (("interval_us", 1, 86_400_000_000), ("embargo_us", 0, 86_400_000_000),
                              ("initial_train_rows", 64, 100_000), ("validation_rows", 64, 100_000),
                              ("internal_rows", 64, 100_000), ("folds", 2, 20),
                              ("simulation_rows", 64, 100_000), ("max_pool_rows", 32, 4096)):
        _integer(config[key], key, lower, upper)
    seeds = [config["fit_seed"]]
    for key in ("validation_seeds", "internal_seeds", "external_seeds"):
        if not isinstance(config[key], list) or not 1 <= len(config[key]) <= 16:
            raise ValueError("one to 16 declared simulation seeds required per phase")
        seeds.extend(config[key])
    for key in ("development", "external"):
        descriptor = config[key]
        _integer(descriptor["rows"], "rows", 64)
        _integer(descriptor["start_us"], "start_us", 0, 2**63 - 1)
        if descriptor["start_us"] + descriptor["rows"] * config["interval_us"] >= 2**63:
            raise ValueError("dataset timestamp overflow")
        if config["kind"] == "synthetic_smoke":
            seeds.append(descriptor["seed"])
            shift = descriptor.get("shift", 1.0)
            if isinstance(shift, bool) or not isinstance(shift, (int, float)) or not .1 <= shift <= 10:
                raise ValueError("synthetic shift must be in [.1, 10]")
        else:
            digest = descriptor["sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("feature CSV SHA-256 is required before registration")
            path = Path(descriptor["path"])
            if path.is_absolute() or ".." in path.parts or "\\" in descriptor["path"] or ":" in descriptor["path"]:
                raise ValueError("feature CSV paths must be portable repository-relative paths")
    for seed in seeds:
        _integer(seed, "seed", 0, 2**63 - 1)
    if len(seeds) != len(set(seeds)):
        raise ValueError("fit, data and all evaluation phase seeds must be disjoint")
    development = config["development"]
    if development["rows"] != (config["initial_train_rows"] + config["folds"] * config["validation_rows"]
                                + config["internal_rows"]):
        raise ValueError("development size must equal declared train, folds and internal rows")
    development_end = development["start_us"] + (development["rows"] - 1) * config["interval_us"]
    if config["external"]["start_us"] <= development_end + config["embargo_us"]:
        raise ValueError("external must be later than development plus embargo")
    for key in ("exchange", "symbol"):
        if not isinstance(config.get(key), str) or not config[key] or len(config[key]) > 128:
            raise ValueError("exchange and symbol must be declared")


def _load(config: dict, phase: str) -> pd.DataFrame:
    descriptor = config[phase]
    if config["kind"] == "synthetic_smoke":
        frame = synthetic_fixture(descriptor, config["interval_us"])
    else:
        path = PROJECT_ROOT / descriptor["path"]
        if sha256_file(path) != descriptor["sha256"]:
            raise ValueError(f"{phase} source hash differs from registration")
        frame = pd.read_csv(path)
        if sha256_file(path) != descriptor["sha256"]:
            raise ValueError(f"{phase} source changed while reading")
    observe(frame, config["interval_us"])
    expected = descriptor["start_us"] + np.arange(descriptor["rows"], dtype=np.int64) * config["interval_us"]
    if len(frame) != descriptor["rows"] or not np.array_equal(frame.timestamp_us.to_numpy(), expected):
        raise ValueError(f"{phase} period/rows differ from registration")
    return frame


def _provenance() -> dict:
    runtime = runtime_metadata()
    runtime.pop("executable", None)  # Portable artifacts never expose personal paths.
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                         text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                                            text=True, stderr=subprocess.DEVNULL, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty, "runtime": runtime,
            "source_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in
                              ("lob/generalization.py", "lob/calibration.py", "lob/experiments/registry.py")}}


def _check_seal(root: Path) -> None:
    seal = json.loads((root / "selection-seal.json").read_text())
    for name, expected in seal["files"].items():
        if sha256_file(root / name) != expected:
            raise ValueError("selection seal mismatch before evaluation")


def run_study(config: dict, out: str | Path) -> dict:
    """Register, select on train/validation, seal, then score internal and external.

    Output is write-once. A failed run retains its plan and any completed results;
    rerunning requires a new output directory, never an overwritten holdout.
    """
    config = json.loads(json.dumps(config, allow_nan=False))
    _validate_config(config)
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    plan = {"config": config, "gate": GATE, "candidates": [f"{f}/{w}" for f in FAMILIES for w in WINDOWS],
            "selection_rule": "minimum mean validation loss; lexical candidate tie-break",
            "loss": "equal observable mean of normalized quantile distance + max(0, .95 - coverage)",
            "final_refit": "selected family/window on train+validation only; embargo before internal",
            "outcome_horizon_us": 0, "lookback": "causal, reset separately at every partition",
            "observables": {name: {"definition": values[0], "units": values[1], "estimator": values[2],
                                   "sampling_interval_us": config["interval_us"],
                                   "missing": "NaN; no interpolation; invalid rows stay in input-validity denominator; "
                                              "interval coverage is conditional on finite measurements",
                                   "loss_weight": 1 / len(OBSERVABLES)} for name, values in OBSERVABLES.items()},
            "limitations": LIMITATIONS}
    write_json(root / "plan.json", plan)
    write_json(root / "provenance.json", _provenance())
    development = _load(config, "development")
    selection_stop = len(development) - config["internal_rows"]
    selection = _section(development, 0, selection_stop)
    candidates = []
    for family in FAMILIES:
        for window in WINDOWS:
            records, parameters = [], []
            for split in chronological_folds(selection, initial_train_rows=config["initial_train_rows"],
                                             validation_rows=config["validation_rows"], folds=config["folds"],
                                             window=window, embargo_us=config["embargo_us"]):
                try:
                    model = fit_observable_model(split["train"], family=family,
                                                interval_us=config["interval_us"], seed=config["fit_seed"],
                                                max_pool_rows=config["max_pool_rows"])
                    score = evaluate_observable_model(model, split["validation"],
                                                      seeds=config["validation_seeds"],
                                                      simulation_rows=config["simulation_rows"])
                    parameters.append(model["parameter_vector"])
                    records.append({"fold": split["fold"], "status": score["status"], "loss": score["loss"],
                                    "train_start_us": model["train_start_us"], "train_end_us": model["train_end_us"],
                                    "validation_start_us": score["start_us"], "validation_end_us": score["end_us"],
                                    "model_sha256": model["model_sha256"],
                                    "validation_frame_sha256": score["frame_sha256"],
                                    "train_valid_fraction": model["train_valid_fraction"],
                                    "train_complete_rows": model["train_complete_rows"],
                                    "valid_fraction": score["valid_fraction"], "metrics": score["metrics"],
                                    "failure_attribution": score["failure_attribution"], "error": None})
                except (ValueError, RuntimeError) as exc:
                    records.append({"fold": split["fold"], "status": "FAIL", "loss": None,
                                    "error": f"{type(exc).__name__}: {exc}"})
            eligible = all(record["error"] is None for record in records)
            vectors = np.asarray(parameters)
            stability = ((np.std(vectors, axis=0) / np.maximum(np.mean(np.abs(vectors), axis=0), 1e-12)).tolist()
                         if parameters else [])
            losses = [r["loss"] for r in records if r["loss"] is not None]
            candidates.append({"candidate": f"{family}/{window}", "family": family, "window": window,
                               "eligible": eligible, "folds": records,
                               "mean_validation_loss": float(np.mean(losses)) if eligible else None,
                               "fold_loss_variance": float(np.var(losses)) if eligible else None,
                               "parameter_relative_std": stability if parameters else [],
                               "parameter_names": ["median_spread", "mean_bid_depth", "mean_ask_depth",
                                                   "return_std", "p_low_to_high", "p_high_to_low"]})
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    write_json(root / "validation.json", {"candidates": candidates})
    if not eligible:
        raise ValueError("all registered calibration candidates failed; results retained")
    selected = min(eligible, key=lambda row: (row["mean_validation_loss"], row["candidate"]))
    internal = _section(development, selection_stop, len(development))
    stop = int(np.searchsorted(selection.timestamp_us, int(internal.timestamp_us.iloc[0]) - config["embargo_us"]))
    start = max(0, stop - config["initial_train_rows"]) if selected["window"] == "rolling" else 0
    model = fit_observable_model(_section(selection, start, stop), family=selected["family"],
                                interval_us=config["interval_us"], seed=config["fit_seed"],
                                max_pool_rows=config["max_pool_rows"])
    write_json(root / "model.json", model)
    write_json(root / "selection.json", {"candidate": selected["candidate"],
                                        "mean_validation_loss": selected["mean_validation_loss"],
                                        "development_frame_sha256": _frame_hash(development),
                                        "model_sha256": model["model_sha256"]})
    write_json(root / "selection-seal.json", {"sealed_before_internal_and_external_evaluation": True,
                                             "files": {name: sha256_file(root / name) for name in
                                                       ("plan.json", "provenance.json", "validation.json",
                                                        "model.json", "selection.json")}})
    _check_seal(root)
    internal_score = evaluate_observable_model(model, internal, seeds=config["internal_seeds"],
                                               simulation_rows=config["simulation_rows"])
    write_json(root / "internal.json", internal_score)
    _check_seal(root)
    external = _load(config, "external")  # First read/generation occurs only after immutable selection.
    _check_seal(root)
    external_score = evaluate_observable_model(model, external, seeds=config["external_seeds"],
                                               simulation_rows=config["simulation_rows"])
    write_json(root / "external.json", external_score)
    _check_seal(root)
    result = {"schema_version": 1, "kind": config["kind"], "exchange": config["exchange"],
              "symbol": config["symbol"], "selected_candidate": selected["candidate"],
              "internal_status": internal_score["status"], "external_status": external_score["status"],
              "status": "FAIL" if "FAIL" in (internal_score["status"], external_score["status"]) else
                        "WARNING" if "WARNING" in (internal_score["status"], external_score["status"]) else "PASS",
              "model_sha256": model["model_sha256"], "model_unchanged": True,
              "external_consumed": True, "external_used_for_selection": False,
              "real_market_generalization": "NOT_ESTABLISHED", "limitations": LIMITATIONS}
    write_json(root / "result.json", result)
    write_json(root / "manifest.json", {"algorithm": "sha256", "files": {
        path.name: sha256_file(path) for path in sorted(root.iterdir()) if path.is_file()}})
    return result


def verify_study(out: str | Path) -> dict:
    root = Path(out).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    files = manifest.get("files", {})
    issues = []
    if manifest.get("algorithm") != "sha256" or not isinstance(files, dict):
        raise ValueError("invalid study manifest")
    actual = {path.name for path in root.iterdir() if path.is_file() and path.name != "manifest.json"}
    if actual != set(files):
        issues.append("study file set changed")
    for name, expected in files.items():
        path = root / name
        if path.name != name or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            issues.append(f"unsafe/missing artifact: {name}")
        elif sha256_file(path) != expected:
            issues.append(f"checksum mismatch: {name}")
    required = {"plan.json", "provenance.json", "model.json", "selection.json", "selection-seal.json",
                "validation.json", "internal.json", "external.json", "result.json"}
    if not required <= set(files):
        issues.append("missing required artifacts")
    if not issues:
        try:
            _check_seal(root)
            _check_model(json.loads((root / "model.json").read_text()))
        except ValueError as exc:
            issues.append(str(exc))
    return {"valid": not issues, "issues": issues}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v03-calibration.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    result = verify_study(args.out) if args.verify else run_study(
        json.loads(Path(args.config).read_text()), args.out)
    print(json.dumps(result, sort_keys=True))
    if args.verify and not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
