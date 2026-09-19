"""Causal L2 observations and a frozen empirical observable-market baseline.

Calibration concerns observable spreads, depth, imbalance and returns only. L2
does not identify individual arrivals, cancellations, hidden liquidity, queue
priority or counterfactual fills. The generator samples IID joint observations;
it is deliberately not an execution simulator or a model of temporal dynamics.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from decimal import localcontext
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .replay.l2 import L2Replay, L2State


FEATURES = ("spread_bps", "bid_depth5", "ask_depth5", "imbalance5", "log_return")
UNITS = {
    "mid_price": "dataset native price", "spread_bps": "basis points",
    "bid_depth5": "dataset native amount (no contract/base/share conversion)",
    "ask_depth5": "dataset native amount (no contract/base/share conversion)",
    "imbalance5": "dimensionless", "log_return": "log price ratio per sample interval",
}
LIMITATIONS = (
    "L2 cannot identify order-level arrivals, cancellations, FIFO, hidden liquidity or fills.",
    "IID joint bootstrap preserves sampled contemporaneous relationships, not serial dynamics.",
    "Empirical support does not extrapolate unseen tails or establish strategy profitability.",
    "Capture-clock sampling is causal to capture time; it does not model feed or order latency.",
    "Depth remains in dataset-native amount units; no shares, ETH or monetary conversion is implied.",
    "Distribution thresholds are diagnostic heuristics, not statistical significance tests.",
)


def _positive_int(value: int, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _duration_us(value: float, name: str, *, zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite seconds")
    micros = value * 1_000_000
    if (value < 0 if zero else value <= 0) or micros > 86_400_000_000 or not float(micros).is_integer():
        raise ValueError(f"{name} must have whole-microsecond precision and be at most one day")
    return int(micros)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample(state: L2State, timestamp: int, stale_us: int) -> dict:
    age = timestamp - state.local_timestamp_us
    if age < 0:
        raise ValueError("sampling cannot use a future state")
    reason = ("stale" if age > stale_us else "empty" if state.empty else
              "one_sided" if state.one_sided else "crossed_or_locked" if state.crossed else "")
    row = {
        "timestamp_us": timestamp, "source_timestamp_us": state.local_timestamp_us,
        "exchange_timestamp_us": state.timestamp_us, "age_us": age,
        "valid": not reason, "invalid_reason": reason,
        "mid_price": math.nan, **{name: math.nan for name in FEATURES},
    }
    if reason:
        return row
    with localcontext() as context:
        context.prec = 80
        bid, ask = state.bids[0][0], state.asks[0][0]
        mid = (bid + ask) / 2
        row["mid_price"] = float(mid)
        row["spread_bps"] = float((ask - bid) / mid * 10_000)
        bid_depth = float(sum(q for _, q in state.bids))
        ask_depth = float(sum(q for _, q in state.asks))
    row.update(bid_depth5=bid_depth, ask_depth5=ask_depth,
               imbalance5=(bid_depth - ask_depth) / (bid_depth + ask_depth))
    return row


def extract_features(
    updates_path: str | Path, *, sample_interval_seconds: float = 1.0,
    max_staleness_seconds: float = 5.0, max_samples: int = 1_000_000,
    replay_options: Mapping | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Replay once and sample last completed groups on a UTC capture-clock grid.

    Invalid and stale samples stay in the table with NaN measurements. Returns
    require consecutive valid grid points, and never bridge an invalid sample.
    The first/last observed capture times bound output: no extrapolation at EOF.
    Limits reject the source rather than quietly publishing a truncated dataset.
    """
    interval = _duration_us(sample_interval_seconds, "sample_interval_seconds")
    stale = _duration_us(max_staleness_seconds, "max_staleness_seconds", zero=True)
    _positive_int(max_samples, "max_samples", 5_000_000)
    path = Path(updates_path)
    options = dict(replay_options or {})
    if "depth" in options and options["depth"] != 5:
        raise ValueError("feature extraction requires depth=5")
    options["depth"] = 5
    replay = L2Replay(path, **options)
    if path.stat().st_size > replay.max_file_bytes:
        raise ValueError("L2 source exceeds max_file_bytes")
    before_hash = _sha256(path)
    rows: list[dict] = []
    previous = None
    next_us = None

    def append_sample() -> None:
        nonlocal next_us
        if len(rows) >= max_samples:
            raise ValueError("feature extraction exceeds max_samples; no partial dataset returned")
        row = _sample(previous, next_us, stale)
        if rows and row["valid"] and rows[-1]["valid"]:
            row["log_return"] = math.log(row["mid_price"] / rows[-1]["mid_price"])
        rows.append(row)
        next_us += interval

    for state in replay:
        if next_us is None:
            next_us = ((state.local_timestamp_us + interval - 1) // interval) * interval
        while previous is not None and next_us < state.local_timestamp_us:
            append_sample()
        previous = state
    if previous is not None and next_us == previous.local_timestamp_us:
        append_sample()
    if not replay.stats["complete"] or not rows:
        raise ValueError("source has no complete sampled observations")
    if _sha256(path) != before_hash:
        raise ValueError("source changed during feature extraction")
    frame = pd.DataFrame(rows)
    metadata = {
        "schema_version": 1, "source": str(path.resolve()), "source_sha256": before_hash,
        "exchange": replay.stats["exchange"], "symbol": replay.stats["symbol"],
        "sample_interval_us": interval, "max_staleness_us": stale,
        "first_timestamp_us": int(frame.timestamp_us.iloc[0]),
        "last_timestamp_us": int(frame.timestamp_us.iloc[-1]), "samples": len(frame),
        "valid_samples": int(frame.valid.sum()),
        "invalid_reasons": frame.loc[~frame.valid, "invalid_reason"].value_counts().to_dict(),
        "units": dict(UNITS), "sampling": "previous completed capture group on fixed UTC grid",
        "source_complete": True, "replay": replay.stats, "limitations": list(LIMITATIONS),
    }
    frame.attrs["calibration_metadata"] = metadata
    return frame, metadata


def _validate_frame(frame: pd.DataFrame) -> np.ndarray:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("features must be a nonempty pandas DataFrame")
    if len(frame) > 5_000_000:
        raise ValueError("feature table exceeds 5,000,000 samples")
    required = {"timestamp_us", "valid", "mid_price", *FEATURES}
    if not required.issubset(frame.columns) or not frame.columns.is_unique:
        raise ValueError(f"feature table requires unique columns including {sorted(required)}")
    times = frame["timestamp_us"]
    if not pd.api.types.is_integer_dtype(times.dtype) or pd.api.types.is_bool_dtype(times.dtype):
        raise ValueError("timestamp_us must be exact integer microseconds")
    if times.isna().any() or (times < 0).any() or (times > 2**63 - 1).any():
        raise ValueError("timestamp_us must be nonnegative signed-64-bit integers")
    if len(times) > 1 and not np.all(times.to_numpy()[1:] > times.to_numpy()[:-1]):
        raise ValueError("feature timestamps must strictly increase without duplicates")
    if not pd.api.types.is_bool_dtype(frame.valid.dtype) or frame.valid.isna().any():
        raise ValueError("valid must contain booleans")
    valid = frame.valid.to_numpy(dtype=bool)
    try:
        values = frame[["mid_price", *FEATURES]].to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("feature measurements must be numeric") from exc
    if not np.all(np.isfinite(values[valid, :-1])):
        raise ValueError("valid book measurements must be finite")
    if np.isinf(values).any():
        raise ValueError("feature measurements must not contain infinity")
    selected = values[valid]
    if len(selected) and ((selected[:, :4] <= 0).any() or (selected[:, 1] >= 20_000).any()
                          or (np.abs(selected[:, 4]) > 1).any()):
        raise ValueError("price, spread, depth or imbalance is outside its physical domain")
    if np.isfinite(values[~valid]).any():
        raise ValueError("invalid samples must have NaN measurements")
    # Return validity is determined by the represented sample grid, including
    # filtered tables that contain time gaps. Do not silently bridge a split.
    returns = values[:, -1]
    if np.isfinite(returns[0]):
        raise ValueError("first feature row must have NaN log_return; reset at split boundaries")
    return valid


def _metadata(frame: pd.DataFrame, metadata: Mapping | None) -> dict:
    result = dict(metadata if metadata is not None else frame.attrs.get("calibration_metadata", {}))
    if result.get("source_complete") is False:
        raise ValueError("cannot calibrate or evaluate an incomplete source")
    return result


def _identity(metadata: Mapping) -> tuple[str, str]:
    values = tuple(metadata.get(key, "unspecified") for key in ("exchange", "symbol"))
    if any(not isinstance(value, str) or not value or len(value) > 128 for value in values):
        raise ValueError("exchange and symbol must be bounded strings")
    return values


def _source_hashes(metadata: Mapping) -> tuple[str, ...]:
    hashes = metadata.get("source_sha256", ())
    hashes = (hashes,) if isinstance(hashes, str) else tuple(hashes)
    if any(not isinstance(value, str) or len(value) != 64 or
           any(char not in "0123456789abcdef" for char in value) for value in hashes):
        raise ValueError("source_sha256 must contain SHA-256 lowercase hex digests")
    return hashes


def _frame_hash(frame: pd.DataFrame) -> str:
    content = frame[["timestamp_us", "valid", "mid_price", *FEATURES]].to_csv(
        index=False, lineterminator="\n", float_format="%.17g", na_rep="NaN"
    ).encode()
    return hashlib.sha256(content).hexdigest()


def _interval(frame: pd.DataFrame, metadata: Mapping) -> int:
    explicit = metadata.get("sample_interval_us")
    differences = np.diff(frame.timestamp_us.to_numpy())
    if explicit is None:
        if not len(differences):
            raise ValueError("sampling interval is unavailable")
        explicit = int(differences.min())
    interval = _positive_int(explicit, "sample_interval_us", 86_400_000_000)
    if len(differences) and (differences % interval != 0).any():
        raise ValueError("timestamps are not on the stated sampling grid")
    invalid_return = np.r_[True, (differences != interval) |
                           ~frame.valid.to_numpy()[:-1] | ~frame.valid.to_numpy()[1:]]
    if np.isfinite(frame.log_return.to_numpy(dtype=float)[invalid_return]).any():
        raise ValueError("returns must not bridge invalid samples or sampling gaps")
    return interval


@dataclass(frozen=True)
class EmpiricalCalibrationModel:
    schema_version: int
    model_type: str
    features: tuple[str, ...]
    exchange: str
    symbol: str
    sample_interval_us: int
    train_start_us: int
    train_end_us: int
    train_rows: int
    train_valid_rows: int
    train_joint_rows: int
    train_source_sha256: tuple[str, ...]
    train_frame_sha256: str
    initial_mid_price: float
    probabilities: tuple[float, ...]
    quantiles: tuple[tuple[float, ...], ...]
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]
    pool: tuple[tuple[float, ...], ...]
    fit_seed: int
    limitations: tuple[str, ...] = LIMITATIONS

    def __post_init__(self) -> None:
        for name in ("features", "train_source_sha256", "probabilities", "quantiles", "means",
                     "standard_deviations", "pool", "limitations"):
            if not isinstance(getattr(self, name), tuple):
                raise ValueError("calibration parameters must be immutable tuples")
        if any(not isinstance(row, tuple) for row in (*self.quantiles, *self.pool)):
            raise ValueError("calibration matrix rows must be immutable tuples")
        if self.schema_version != 1 or self.model_type != "empirical_joint_iid_observables":
            raise ValueError("unsupported calibration schema/model")
        if self.features != FEATURES or not 32 <= len(self.pool) <= 100_000:
            raise ValueError("invalid calibration feature schema or pool size")
        arrays = [np.asarray(value, dtype=float) for value in (
            self.pool, self.quantiles, self.means, self.standard_deviations)]
        if any(not np.isfinite(value).all() for value in arrays):
            raise ValueError("calibration parameters must be finite")
        if (arrays[0].shape != (len(self.pool), len(FEATURES)) or
                arrays[1].shape != (len(FEATURES), 101) or
                tuple(self.probabilities) != tuple(np.linspace(0, 1, 101)) or
                arrays[2].shape != (len(FEATURES),) or arrays[3].shape != (len(FEATURES),)):
            raise ValueError("invalid calibration dimensions")
        if (np.diff(arrays[1], axis=1) < 0).any() or (arrays[3] < 0).any():
            raise ValueError("calibration quantiles/deviations outside valid domain")
        pool = arrays[0]
        if (pool[:, :3] <= 0).any() or (pool[:, 0] >= 20_000).any() or (np.abs(pool[:, 3]) > 1).any():
            raise ValueError("calibration bootstrap pool outside physical domain")
        if not math.isfinite(self.initial_mid_price) or self.initial_mid_price <= 0:
            raise ValueError("invalid calibration initial price")
        _identity({"exchange": self.exchange, "symbol": self.symbol})
        _positive_int(self.sample_interval_us, "sample_interval_us", 86_400_000_000)
        for name in ("train_start_us", "train_end_us", "fit_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
                raise ValueError(f"{name} must be a nonnegative signed-64-bit integer")
        if not self.train_start_us < self.train_end_us:
            raise ValueError("invalid calibration training timestamp bounds")
        for name in ("train_rows", "train_valid_rows", "train_joint_rows"):
            _positive_int(getattr(self, name), name, 5_000_000)
        if not len(self.pool) <= self.train_joint_rows <= self.train_valid_rows <= self.train_rows:
            raise ValueError("inconsistent calibration training row counts")
        for digest in (*self.train_source_sha256, self.train_frame_sha256):
            if not isinstance(digest, str) or len(digest) != 64 or any(
                    char not in "0123456789abcdef" for char in digest):
                raise ValueError("invalid calibration provenance hash")
        if self.limitations != LIMITATIONS:
            raise ValueError("calibration scope limitations must be retained")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def model_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()


def fit_calibration(
    frame: pd.DataFrame, metadata: Mapping | None = None, *, max_pool_rows: int = 4096,
    seed: int = 0,
) -> EmpiricalCalibrationModel:
    """Freeze train-only empirical marginals and a bounded joint bootstrap pool.

    At least 32 complete return/observable rows are required. Sampling parameters
    and diagnostics should be declared before examining holdout performance.
    The full train frame hash and source hashes bind the exact fit input.
    """
    valid = _validate_frame(frame)
    meta = _metadata(frame, metadata)
    interval = _interval(frame, meta)
    _positive_int(max_pool_rows, "max_pool_rows", 100_000)
    if max_pool_rows < 32:
        raise ValueError("max_pool_rows must be at least 32")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative signed-64-bit integer")
    values = frame.loc[valid, list(FEATURES)].to_numpy(dtype=float)
    joint = values[np.isfinite(values).all(axis=1)]
    if len(joint) < 32:
        raise ValueError("calibration requires at least 32 complete joint observations")
    if len(joint) > max_pool_rows:
        indices = np.sort(np.random.default_rng(seed).choice(len(joint), max_pool_rows, replace=False))
        pool = joint[indices]
    else:
        pool = joint
    probabilities = np.linspace(0, 1, 101)
    distributions = [column[np.isfinite(column)] for column in values.T]
    hashes = _source_hashes(meta)
    return EmpiricalCalibrationModel(
        schema_version=1, model_type="empirical_joint_iid_observables",
        features=FEATURES, exchange=_identity(meta)[0], symbol=_identity(meta)[1],
        sample_interval_us=interval, train_start_us=int(frame.timestamp_us.iloc[0]),
        train_end_us=int(frame.timestamp_us.iloc[-1]), train_rows=len(frame),
        train_valid_rows=int(valid.sum()), train_joint_rows=len(joint),
        train_source_sha256=hashes, train_frame_sha256=_frame_hash(frame),
        initial_mid_price=float(frame.loc[valid, "mid_price"].iloc[-1]),
        probabilities=tuple(float(value) for value in probabilities),
        quantiles=tuple(tuple(float(value) for value in np.quantile(column, probabilities))
                        for column in distributions),
        means=tuple(float(column.mean()) for column in distributions),
        standard_deviations=tuple(float(column.std()) for column in distributions),
        pool=tuple(tuple(float(value) for value in row) for row in pool), fit_seed=seed,
    )


def evaluate_calibration(
    model: EmpiricalCalibrationModel, frame: pd.DataFrame, metadata: Mapping | None = None,
) -> dict:
    """Score a strictly later holdout without refitting or changing the model.

    Normalized quantile distance uses train IQR/std, with a small relative-scale
    floor for constant marginals. Coverage measures the frozen train 95% range;
    discreteness may make nominal coverage exceed 95%. No independence-based
    p-values are reported for autocorrelated high-frequency observations.
    """
    valid = _validate_frame(frame)
    meta = _metadata(frame, metadata)
    if int(frame.timestamp_us.iloc[0]) <= model.train_end_us:
        raise ValueError("holdout must be strictly later than every training observation")
    if _interval(frame, meta) != model.sample_interval_us:
        raise ValueError("train and holdout sampling intervals differ")
    if _identity(meta) != (model.exchange, model.symbol):
        raise ValueError("train and holdout exchange/symbol differ")
    if not valid.any():
        raise ValueError("holdout has no valid observations")
    result = {}
    statuses = []
    for index, feature in enumerate(FEATURES):
        observed = frame.loc[valid, feature].to_numpy(dtype=float)
        observed = observed[np.isfinite(observed)]
        if len(observed) < 32:
            raise ValueError(f"holdout {feature} requires at least 32 finite observations")
        predicted = np.asarray(model.quantiles[index])
        scale = max(predicted[75] - predicted[25], model.standard_deviations[index],
                    abs(model.means[index]) * .01, 1e-12)
        holdout_quantiles = np.quantile(observed, model.probabilities)
        distance = float(np.mean(np.abs(holdout_quantiles - predicted)) / scale)
        lower, upper = np.interp([.025, .975], model.probabilities, predicted)
        coverage = float(np.mean((observed >= lower) & (observed <= upper)))
        status = "FAIL" if distance > .5 or coverage < .75 else (
            "WARNING" if distance > .25 or coverage < .90 else "PASS")
        statuses.append(status)
        result[feature] = {
            "status": status, "holdout_count": len(observed), "train_mean": model.means[index],
            "holdout_mean": float(observed.mean()), "train_std": model.standard_deviations[index],
            "holdout_std": float(observed.std()), "normalization_scale": float(scale),
            "normalized_quantile_distance": distance, "train_95pct_range": [float(lower), float(upper)],
            "holdout_coverage_of_train_95pct_range": coverage,
            "holdout_quantiles": [float(value) for value in holdout_quantiles],
        }
    valid_fraction = float(valid.mean())
    if valid_fraction < .90:
        statuses.append("FAIL")
    elif valid_fraction < .99:
        statuses.append("WARNING")
    return {
        "schema_version": 1, "status": "FAIL" if "FAIL" in statuses else (
            "WARNING" if "WARNING" in statuses else "PASS"),
        "model_sha256": model.model_sha256, "fit_on_holdout": False,
        "holdout_kind": "synthetic_model_sample" if meta.get("synthetic") else "observed_feature_table",
        "holdout_source_sha256": list(_source_hashes(meta)), "holdout_frame_sha256": _frame_hash(frame),
        "train_start_us": model.train_start_us, "train_end_us": model.train_end_us,
        "holdout_start_us": int(frame.timestamp_us.iloc[0]),
        "holdout_end_us": int(frame.timestamp_us.iloc[-1]), "holdout_samples": len(frame),
        "valid_sample_fraction": valid_fraction, "metrics": result,
        "thresholds": {"normalized_distance_warning_above": .25, "normalized_distance_fail_above": .5,
                       "coverage_warning_below": .90, "coverage_fail_below": .75,
                       "valid_fraction_warning_below": .99, "valid_fraction_fail_below": .90},
        "units": dict(UNITS), "limitations": list(model.limitations),
    }


def generate_features(model: EmpiricalCalibrationModel, n_samples: int, *, seed: int = 0) -> pd.DataFrame:
    """Generate bounded IID joint observations and compound their sampled returns.

    Output is hypothetical and must never be fed to an execution/fill backtest.
    A fitted empirical row cannot generate a price outside floating-point range;
    pathological accumulated paths raise instead of silently clipping prices.
    """
    _positive_int(n_samples, "n_samples", 1_000_000)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative signed-64-bit integer")
    if model.train_end_us + n_samples * model.sample_interval_us > 2**63 - 1:
        raise ValueError("generated timestamps exceed signed-64-bit range")
    rng = np.random.default_rng(seed)
    values = np.asarray(model.pool)[rng.integers(0, len(model.pool), size=n_samples)]
    frame = pd.DataFrame(values, columns=FEATURES)
    prices = math.log(model.initial_mid_price) + np.cumsum(frame.log_return.to_numpy())
    if (prices > 700).any() or (prices < -700).any():
        raise ValueError("generated price path exceeds supported numeric range")
    frame.insert(0, "timestamp_us", model.train_end_us +
                 np.arange(1, n_samples + 1, dtype=np.int64) * model.sample_interval_us)
    frame["mid_price"] = np.exp(prices)
    frame["valid"] = True
    frame["synthetic"] = True
    # The table is self-contained; the first simulated return depends on the
    # model's initial price outside this table, so mask it at this boundary.
    frame.loc[0, "log_return"] = math.nan
    frame.attrs["model_sha256"] = model.model_sha256
    frame.attrs["limitations"] = list(model.limitations)
    frame.attrs["calibration_metadata"] = {
        "exchange": model.exchange, "symbol": model.symbol,
        "sample_interval_us": model.sample_interval_us, "synthetic": True,
    }
    return frame


def save_calibration(model: EmpiricalCalibrationModel, path: str | Path) -> Path:
    """Write once; existing model artifacts are never silently overwritten."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"model": model.to_dict(), "model_sha256": model.model_sha256}
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(envelope, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return destination


def load_calibration(path: str | Path) -> EmpiricalCalibrationModel:
    source = Path(path)
    if source.stat().st_size > 64 * 1024**2:
        raise ValueError("calibration artifact exceeds 64 MiB")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate calibration key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
        if set(value) != {"model", "model_sha256"}:
            raise ValueError("invalid calibration envelope")
        data = dict(value["model"])
        for name in ("features", "train_source_sha256", "probabilities", "means", "standard_deviations",
                     "limitations"):
            data[name] = tuple(data[name])
        for name in ("quantiles", "pool"):
            data[name] = tuple(tuple(row) for row in data[name])
        model = EmpiricalCalibrationModel(**data)
        if model.model_sha256 != value["model_sha256"]:
            raise ValueError("calibration digest mismatch")
        return model
    except (TypeError, KeyError, json.JSONDecodeError, OverflowError) as exc:
        raise ValueError(f"invalid calibration artifact: {exc}") from exc
