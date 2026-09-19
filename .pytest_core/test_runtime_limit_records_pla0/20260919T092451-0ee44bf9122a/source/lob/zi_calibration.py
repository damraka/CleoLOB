"""Simulation-based calibration of the *execution* simulator to observable L2.

Published top-five snapshots are sampled on a causal capture-time grid. Amounts
retain native units until one train-only median side-depth is mapped to 1,000
synthetic lots. This is a shape comparison, not a contract/currency conversion.
Snapshot OFI is a best-quote change proxy, not observed order-level event flow.
Visible-depth book walks are instantaneous impact proxies, not causal impact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .engine import ExchangeSimulator, SimConfig


DEPTH_LOTS = 1000.0
QUANTILES = (.1, .5, .9)
IMPACT_FRACTIONS = (.1, .5, 1.0)
FAMILY_FEATURES = {
    "spread": ("spread_bps",),
    "depth": ("bid_depth", "ask_depth"),
    "imbalance": ("imbalance",),
    "ofi": ("abs_snapshot_ofi",),
    "impact": ("impact_0.1_bps", "impact_0.5_bps", "impact_1.0_bps"),
    "volatility": ("return_bps",),
}
# Fixed diagnostic tolerances: they are not p-values or a profitability test.
DEFAULT_GATE = {
    "maximum_family_error": .60,
    "minimum_valid_fraction": .95,
    "minimum_target_samples": 1000,
    "quantiles": list(QUANTILES),
    "spread_impact_scale_floor_bps": .05,
    "ofi_scale_floor": .05,
    "volatility_scale_floor_bps": .05,
}
LIMITATIONS = [
    "L2 does not identify FIFO queues, hidden liquidity, or individual arrivals/cancellations.",
    "Amounts map to synthetic lots using each instrument's training-only median top-five side depth.",
    "The mapping does not convert inverse contracts to shares or linear monetary P&L.",
    "Snapshot OFI is a one-second best-quote proxy, not event-by-event exchange OFI.",
    "Book-walk impact excludes replenishment and is not an estimated causal market-impact function.",
    "One representative tick geometry may fail for instruments with different relative tick sizes.",
    "Calibration PASS is an observable-distribution diagnostic, not counterfactual fill validation.",
]


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _measure(prices: np.ndarray, amounts: np.ndarray, timestamps: np.ndarray,
             valid: np.ndarray) -> pd.DataFrame:
    """Arrays have columns bid[0:5], ask[0:5], descending/ascending respectively."""
    bp, ap = prices[:, :5], prices[:, 5:]
    bq, aq = amounts[:, :5], amounts[:, 5:]
    mid = (bp[:, 0] + ap[:, 0]) / 2
    bd, ad = bq.sum(axis=1), aq.sum(axis=1)
    frame = pd.DataFrame({"timestamp_us": timestamps, "valid": valid, "mid_price": mid,
                          "spread_bps": (ap[:, 0] - bp[:, 0]) / mid * 10000,
                          "bid_depth_native": bd, "ask_depth_native": ad,
                          "imbalance": (bd - ad) / (bd + ad)})
    for fraction in IMPACT_FRACTIONS:
        costs = []
        for ladder, volumes, total, side in ((bp, bq, bd, -1), (ap, aq, ad, 1)):
            wanted = total * fraction
            before = np.cumsum(volumes, axis=1) - volumes
            takes = np.minimum(volumes, np.maximum(0, wanted[:, None] - before))
            vwap = np.sum(takes * ladder, axis=1) / wanted
            costs.append(side * (vwap - mid) / mid * 10000)
        frame[f"impact_{fraction:.1f}_bps"] = (costs[0] + costs[1]) / 2
    ofi = np.full(len(frame), np.nan)
    returns = np.full(len(frame), np.nan)
    if len(frame) > 1:
        change = ((bp[1:, 0] >= bp[:-1, 0]) * bq[1:, 0]
                  - (bp[1:, 0] <= bp[:-1, 0]) * bq[:-1, 0]
                  - (ap[1:, 0] <= ap[:-1, 0]) * aq[1:, 0]
                  + (ap[1:, 0] >= ap[:-1, 0]) * aq[:-1, 0])
        denominator = (bq[1:, 0] + bq[:-1, 0] + aq[1:, 0] + aq[:-1, 0]) / 2
        consecutive = valid[1:] & valid[:-1] & (np.diff(timestamps) == 1_000_000)
        ofi[1:] = np.where(consecutive, change / denominator, np.nan)
        returns[1:] = np.where(consecutive, np.log(mid[1:] / mid[:-1]) * 10000, np.nan)
    frame["snapshot_ofi"] = ofi
    frame["abs_snapshot_ofi"] = np.abs(ofi)
    frame["return_bps"] = returns
    for column in frame.columns.difference(["timestamp_us", "valid"]):
        frame.loc[~valid, column] = np.nan
    return frame


def extract_top5(path: str | Path, *, max_staleness_seconds: float = 5.0,
                 max_rows: int = 20_000_000) -> tuple[pd.DataFrame, dict]:
    """Read a complete published top-five file, using the last row at/before each second.

    All chunks are consumed (including gzip CRC validation), even after the last
    useful grid point. No state is carried across files or invalid observations.
    """
    path = Path(path)
    if not math.isfinite(max_staleness_seconds) or max_staleness_seconds <= 0:
        raise ValueError("max_staleness_seconds must be positive")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("max_rows must be positive")
    before_hash = _sha(path)
    price_cols = [f"{side}s[{level}].price" for side in ("bid", "ask") for level in range(5)]
    amount_cols = [f"{side}s[{level}].amount" for side in ("bid", "ask") for level in range(5)]
    columns = ["exchange", "symbol", "timestamp", "local_timestamp", *price_cols, *amount_cols]
    sampled, sample_grids = [], []
    previous, next_grid, last_timestamp, identity, first_timestamp = None, None, None, None, None
    total_rows = 0
    for chunk in pd.read_csv(path, usecols=columns, chunksize=100_000):
        total_rows += len(chunk)
        if total_rows > max_rows:
            raise ValueError("snapshot row resource limit exceeded")
        if chunk.empty:
            continue
        identities = chunk[["exchange", "symbol"]].drop_duplicates()
        if len(identities) != 1:
            raise ValueError("top-five input must contain exactly one exchange/instrument")
        current_identity = tuple(identities.iloc[0])
        if identity is not None and current_identity != identity:
            raise ValueError("top-five input instrument changes")
        identity = current_identity
        # Enforce integral capture timestamps before converting to signed int64.
        raw_times = chunk.local_timestamp.to_numpy()
        if not np.all(np.isfinite(raw_times)) or np.any(raw_times != np.floor(raw_times)):
            raise ValueError("capture timestamps must be finite integers")
        times = raw_times.astype(np.int64)
        if np.any(times < 0) or np.any(np.diff(times) < 0) or (
                last_timestamp is not None and times[0] < last_timestamp):
            raise ValueError("capture timestamps must be nondecreasing")
        if first_timestamp is None:
            first_timestamp = int(times[0])
        last_timestamp = int(times[-1])
        if previous is not None:
            chunk = pd.concat([previous, chunk], ignore_index=True)
            times = chunk.local_timestamp.to_numpy(dtype=np.int64)
        if next_grid is None:
            next_grid = ((int(times[0]) + 999_999) // 1_000_000) * 1_000_000
        # A same-timestamp group can cross chunk boundaries. Delay its grid until
        # the next timestamp (or EOF) proves that the whole group was consumed.
        grids = np.arange(next_grid, int(times[-1]), 1_000_000, dtype=np.int64)
        if len(grids):
            indices = np.searchsorted(times, grids, side="right") - 1
            sampled.append(chunk.iloc[indices].copy())
            sample_grids.append(grids)
            next_grid = int(grids[-1]) + 1_000_000
        previous = chunk.iloc[[-1]].copy()
    if previous is not None and next_grid == last_timestamp:
        sampled.append(previous.copy())
        sample_grids.append(np.asarray([next_grid], dtype=np.int64))
    if not sampled or identity is None:
        raise ValueError("no complete one-second samples")
    selected = pd.concat(sampled, ignore_index=True)
    grids = np.concatenate(sample_grids)
    prices = selected[price_cols].to_numpy(dtype=float)
    amounts = selected[amount_cols].to_numpy(dtype=float)
    valid = (np.isfinite(prices).all(axis=1) & np.isfinite(amounts).all(axis=1)
             & (prices > 0).all(axis=1) & (amounts > 0).all(axis=1)
             & (np.diff(prices[:, :5], axis=1) < 0).all(axis=1)
             & (np.diff(prices[:, 5:], axis=1) > 0).all(axis=1)
             & (prices[:, 0] < prices[:, 5])
             & (grids - selected.local_timestamp.to_numpy() <= max_staleness_seconds * 1_000_000))
    with np.errstate(divide="ignore", invalid="ignore"):
        frame = _measure(prices, amounts, grids, valid)
    good_prices = prices[valid]
    if not len(good_prices):
        raise ValueError("no valid top-five snapshots")
    gaps = np.concatenate([-np.diff(good_prices[:, :5], axis=1).ravel(),
                           np.diff(good_prices[:, 5:], axis=1).ravel()])
    native_tick = float(np.quantile(gaps[gaps > 0], .05))
    after_hash = _sha(path)
    if before_hash != after_hash:
        raise ValueError("source changed while extracting observations")
    frame["instrument"] = f"{identity[0]}:{identity[1]}"
    metadata = {"path": str(path.resolve()), "sha256": after_hash, "source_complete": True,
                "exchange": str(identity[0]), "symbol": str(identity[1]),
                "instrument": f"{identity[0]}:{identity[1]}", "rows": total_rows,
                "samples": len(frame), "valid_samples": int(valid.sum()),
                "first_capture_us": first_timestamp, "last_capture_us": last_timestamp,
                "sample_interval_us": 1_000_000, "inferred_native_tick": native_tick,
                "inferred_tick_method": "5th percentile of sampled positive adjacent price gaps"}
    return frame, metadata


def _training_scales(frames: list[pd.DataFrame], metadata: list[dict]) -> dict:
    combined = pd.concat(frames, ignore_index=True)
    scales = {}
    for instrument, rows in combined[combined.valid].groupby("instrument", sort=True):
        depth = float(np.median((rows.bid_depth_native + rows.ask_depth_native) / 2))
        mid = float(rows.mid_price.median())
        tick = float(np.median([m["inferred_native_tick"] for m in metadata
                                if m["instrument"] == instrument]))
        scales[instrument] = {"native_depth_per_1000_lots": depth, "native_mid_price": mid,
                              "native_tick": tick, "tick_bps": tick / mid * 10000}
    return scales


def _normalize(frame: pd.DataFrame, scales: dict) -> pd.DataFrame:
    result = frame.copy()
    missing = sorted(set(result.instrument) - set(scales))
    if missing:
        raise ValueError(f"no training-only quantity scale for {missing}")
    factors = result.instrument.map({key: value["native_depth_per_1000_lots"]
                                    for key, value in scales.items()}).to_numpy()
    result["bid_depth"] = result.bid_depth_native / factors
    result["ask_depth"] = result.ask_depth_native / factors
    return result


def summarize(frame: pd.DataFrame) -> dict:
    result = {"samples": len(frame), "valid_samples": int(frame.valid.sum()),
              "valid_fraction": float(frame.valid.mean()), "features": {}}
    for family, names in FAMILY_FEATURES.items():
        for name in names:
            values = frame.loc[frame.valid, name].dropna().to_numpy()
            if not len(values):
                raise ValueError(f"no valid measurements for {name}")
            result["features"][name] = {"quantiles": np.quantile(values, QUANTILES).tolist(),
                                        "mean": float(values.mean()),
                                        "std": float(values.std()),
                                        "nonzero_fraction": float(np.mean(values != 0))}
    return result


def compare_summaries(target: dict, simulated: dict, gate: dict | None = None) -> dict:
    """Predefined dimensionless moment errors; every family must pass separately."""
    gate = dict(DEFAULT_GATE if gate is None else gate)
    families = {}
    for family, names in FAMILY_FEATURES.items():
        errors = []
        for name in names:
            obs, sim = target["features"][name], simulated["features"][name]
            observed_q, simulated_q = np.asarray(obs["quantiles"]), np.asarray(sim["quantiles"])
            if family == "volatility":
                scale = max(obs["std"], gate["volatility_scale_floor_bps"])
                error = max(abs(obs["std"] - sim["std"]) / scale,
                            abs(obs["nonzero_fraction"] - sim["nonzero_fraction"]))
            else:
                scale = (1.0 if family in {"depth", "imbalance"} else
                         max(observed_q[-1], gate["ofi_scale_floor"]) if family == "ofi" else
                         max(observed_q[1], gate["spread_impact_scale_floor_bps"]))
                error = float(np.mean(np.abs(observed_q - simulated_q)) / scale)
            errors.append(float(error))
        families[family] = {"error": float(max(errors)),
                            "passed": bool(max(errors) <= gate["maximum_family_error"])}
    valid = (target["valid_fraction"] >= gate["minimum_valid_fraction"]
             and simulated["valid_fraction"] >= gate["minimum_valid_fraction"])
    sufficient = target["samples"] >= gate["minimum_target_samples"]
    passed = valid and sufficient and all(item["passed"] for item in families.values())
    return {"status": "PASS" if passed else "FAIL", "families": families,
            "valid_fraction_passed": valid, "sample_count_passed": sufficient,
            "objective": float(np.mean([value["error"] for value in families.values()])
                               + 10 * (1 - simulated["valid_fraction"])), "gate": gate}


def simulate_features(config: SimConfig, *, seeds: Iterable[int] = (41001, 41002),
                      seconds: int = 120, warmup_seconds: float = 30) -> pd.DataFrame:
    if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds < 2:
        raise ValueError("seconds must be an integer >= 2")
    if not math.isfinite(warmup_seconds) or warmup_seconds < 0:
        raise ValueError("warmup_seconds must be finite and nonnegative")
    frames = []
    seed_values = tuple(seeds)
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("simulation seeds must be distinct and nonempty")
    for seed in seed_values:
        sim = ExchangeSimulator(replace(config, seed=seed, record_events=False, check_invariants=False))
        sim.step(warmup_seconds)
        prices, quantities, valid = [], [], []
        for _ in range(seconds):
            sim.step(1.0)
            bids, asks = sim.book.depth(5)
            ok = len(bids) == len(asks) == 5
            row = bids + [(np.nan, np.nan)] * (5 - len(bids))
            row += asks + [(np.nan, np.nan)] * (5 - len(asks))
            prices.append([p * config.tick_size for p, _q in row])
            quantities.append([q for _p, q in row])
            valid.append(ok and bids[0][0] < asks[0][0])
        with np.errstate(divide="ignore", invalid="ignore"):
            frame = _measure(np.asarray(prices), np.asarray(quantities),
                             np.arange(seconds, dtype=np.int64) * 1_000_000, np.asarray(valid))
        frame["bid_depth"] = frame.bid_depth_native / DEPTH_LOTS
        frame["ask_depth"] = frame.ask_depth_native / DEPTH_LOTS
        frame["seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def candidate_configs(scales: dict, count: int = 24, search_seed: int = 40100) -> list[SimConfig]:
    """A bounded, seeded design fixed before observing simulated fit scores."""
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 256:
        raise ValueError("candidate count must be in [1, 256]")
    relative_tick = float(np.median([entry["tick_bps"] for entry in scales.values()]))
    mid_ticks = max(21, round(10000 / relative_tick))
    base = SimConfig(initial_mid_ticks=mid_ticks, tick_size=100 / mid_ticks,
                     target_level_vol=200, limit_rate=4, market_rate=2,
                     cancel_rate=.1, limit_qty_mean=35, market_qty_mean=40,
                     resilience=.15, record_events=False, max_events=2_000_000)
    rng = np.random.default_rng(search_seed)
    def log_draw(lower: float, upper: float) -> float:
        return float(np.exp(rng.uniform(np.log(lower), np.log(upper))))
    candidates = [base]
    for _ in range(count - 1):
        candidates.append(replace(base, target_level_vol=int(rng.integers(60, 251)),
                                  limit_rate=log_draw(.5, 12), market_rate=log_draw(.2, 8),
                                  cancel_rate=log_draw(.03, .5), offset_p=float(rng.uniform(.15, .8)),
                                  limit_qty_mean=log_draw(10, 120), market_qty_mean=log_draw(10, 120),
                                  resilience=log_draw(.03, 1.0)))
    return candidates


def calibration_protocol(*, candidate_count: int = 24, seconds: int = 120,
                         warmup_seconds: float = 30, seeds: Iterable[int] = (41001, 41002),
                         search_seed: int = 40100) -> dict:
    return {"schema_version": 1, "method": "simulation-based minimum moment distance",
            "candidate_count": candidate_count, "simulation_seconds_per_seed": seconds,
            "warmup_seconds": warmup_seconds, "simulation_seeds": list(seeds),
            "search_seed": search_seed, "gate": dict(DEFAULT_GATE),
            "gate_registered_before_fit": True, "target_depth_lots": DEPTH_LOTS,
            "features": {key: list(value) for key, value in FAMILY_FEATURES.items()},
            "candidate_bounds": {"target_level_vol": [60, 250], "limit_rate": [.5, 12],
                                 "market_rate": [.2, 8], "cancel_rate": [.03, .5],
                                 "offset_p": [.15, .8], "limit_qty_mean": [10, 120],
                                 "market_qty_mean": [10, 120], "resilience": [.03, 1.0]},
            "limitations": LIMITATIONS}


def fit_model(paths: Iterable[str | Path], *, candidate_count: int = 24,
              simulation_seconds: int = 120, warmup_seconds: float = 30,
              seeds: Iterable[int] = (41001, 41002), search_seed: int = 40100,
              out: str | Path | None = None) -> dict:
    paths = tuple(Path(path) for path in paths)
    if not paths or len(set(path.resolve() for path in paths)) != len(paths):
        raise ValueError("distinct training paths are required")
    protocol = calibration_protocol(candidate_count=candidate_count, seconds=simulation_seconds,
                                    warmup_seconds=warmup_seconds, seeds=seeds, search_seed=search_seed)
    output = Path(out) if out is not None else None
    if output is not None:
        if output.exists() and any(output.iterdir()):
            raise ValueError("fit output must be a new or empty directory")
        _write(output / "protocol.json", protocol)
    extracted = [extract_top5(path) for path in paths]
    frames, metadata = [item[0] for item in extracted], [item[1] for item in extracted]
    if len({item["sha256"] for item in metadata}) != len(metadata):
        raise ValueError("duplicate training content")
    scales = _training_scales(frames, metadata)
    target_frames = [_normalize(frame, scales) for frame in frames]
    target = summarize(pd.concat(target_frames, ignore_index=True))
    candidates = candidate_configs(scales, candidate_count, search_seed)
    if output is not None:
        _write(output / "search_design.json", {"candidates": [asdict(cfg) for cfg in candidates],
                                               "training_sources": metadata, "scales": scales})
    records, best = [], None
    for index, config in enumerate(candidates):
        try:
            simulation = summarize(simulate_features(config, seeds=protocol["simulation_seeds"],
                                   seconds=simulation_seconds, warmup_seconds=warmup_seconds))
            comparison = compare_summaries(target, simulation, protocol["gate"])
            record = {"candidate": index, "config": asdict(config), "comparison": comparison,
                      "simulated_summary": simulation, "error": None}
            if best is None or comparison["objective"] < best["comparison"]["objective"]:
                best = record
        except (ValueError, RuntimeError, AssertionError) as exc:
            record = {"candidate": index, "config": asdict(config), "comparison": None,
                      "simulated_summary": None, "error": f"{type(exc).__name__}: {exc}"}
        records.append(record)
        if output is not None:
            with (output / "candidate_results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    if best is None:
        raise ValueError("all simulator calibration candidates failed")
    model = {"schema_version": 1, "protocol": protocol, "protocol_sha256": _digest(protocol),
             "training_sources": metadata, "scales": scales, "training_summary": target,
             "simulator_config": best["config"], "selected_candidate": best["candidate"],
             "training_comparison": best["comparison"], "simulated_summary": best["simulated_summary"],
             "training_instrument_comparisons": {
                 instrument: compare_summaries(summarize(pd.concat(
                     [frame.loc[frame.instrument == instrument] for frame in target_frames], ignore_index=True)),
                     best["simulated_summary"], protocol["gate"]) for instrument in scales},
             "limitations": LIMITATIONS}
    model["model_sha256"] = _digest(model)
    if output is not None:
        _write(output / "model.json", model)
    return model


def load_model(path: str | Path) -> dict:
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = model.pop("model_sha256")
    actual = _digest(model)
    model["model_sha256"] = expected
    if actual != expected or model["protocol_sha256"] != _digest(model["protocol"]):
        raise ValueError("calibration model digest mismatch")
    SimConfig(**model["simulator_config"])
    return model


def evaluate_model(model: dict, paths: Iterable[str | Path], *,
                   seeds: Iterable[int] = (42001, 42002),
                   simulation_seconds: int | None = None) -> dict:
    paths, seeds = tuple(paths), tuple(seeds)
    if not paths or len(set(Path(path).resolve() for path in paths)) != len(paths):
        raise ValueError("distinct evaluation paths are required")
    if set(seeds) & set(model["protocol"]["simulation_seeds"]):
        raise ValueError("evaluation simulator seeds must be independent of fitting")
    before_model = _digest(model)
    training_hashes = {source["sha256"] for source in model["training_sources"]}
    # Check content and chronology before extracting any target features.
    for path in paths:
        if _sha(Path(path)) in training_hashes:
            raise ValueError("evaluation reuses training content")
    extracted = [extract_top5(path) for path in paths]
    if len({item[1]["sha256"] for item in extracted}) != len(extracted):
        raise ValueError("duplicate evaluation content")
    for _frame, meta in extracted:
        prior = [source for source in model["training_sources"] if source["instrument"] == meta["instrument"]]
        if not prior or meta["first_capture_us"] <= max(source["last_capture_us"] for source in prior):
            raise ValueError("evaluation instrument must have strictly earlier training data")
    frames = [_normalize(frame, model["scales"]) for frame, _meta in extracted]
    simulation = summarize(simulate_features(SimConfig(**model["simulator_config"]), seeds=seeds,
                           seconds=simulation_seconds or model["protocol"]["simulation_seconds_per_seed"],
                           warmup_seconds=model["protocol"]["warmup_seconds"]))
    target = summarize(pd.concat(frames, ignore_index=True))
    instruments = {}
    for frame, meta in zip(frames, [item[1] for item in extracted]):
        instruments[f"{meta['instrument']}@{meta['first_capture_us']}"] = {
            "source": meta, "target_summary": summarize(frame),
            "comparison": compare_summaries(summarize(frame), simulation, model["protocol"]["gate"])}
    combined = compare_summaries(target, simulation, model["protocol"]["gate"])
    if _digest(model) != before_model:
        raise RuntimeError("evaluation modified frozen model")
    return {"schema_version": 1, "model_sha256": model["model_sha256"],
            "simulation_seeds": list(seeds), "model_unchanged": True,
            "status": "PASS" if combined["status"] == "PASS" and all(
                item["comparison"]["status"] == "PASS" for item in instruments.values()) else "FAIL",
            "comparison": combined, "instruments": instruments,
            "target_summary": target, "simulated_summary": simulation, "limitations": LIMITATIONS}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--train", nargs="+", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--candidates", type=int, default=24)
    fit.add_argument("--seconds", type=int, default=120)
    fit.add_argument("--warmup", type=float, default=30)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--data", nargs="+", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--seeds", nargs="+", type=int, default=[42001, 42002])
    args = parser.parse_args(argv)
    if args.command == "fit":
        result = fit_model(args.train, out=args.out, candidate_count=args.candidates,
                           simulation_seconds=args.seconds, warmup_seconds=args.warmup)
        print(json.dumps({"model": str(Path(args.out) / "model.json"),
                          "training_status": result["training_comparison"]["status"],
                          "candidate": result["selected_candidate"]}))
    else:
        output = Path(args.out)
        if output.exists():
            raise ValueError("evaluation output already exists")
        result = evaluate_model(load_model(args.model), args.data, seeds=args.seeds)
        _write(output, result)
        print(json.dumps({"evaluation": str(output), "status": result["status"]}))


if __name__ == "__main__":
    main()
