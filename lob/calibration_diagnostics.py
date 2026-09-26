"""Descriptive, observable-level failure attribution without fitting a holdout.

Block-bootstrap intervals describe mean drift under local stationarity, not an
identified cause or independent-event inference. Original model gates are never
replaced by these diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import _frame_hash
from .generalization import OBSERVABLES, observe


UNSUPPORTED = {
    "arrival_intensity": "Sampled aggregate depth does not identify individual arrivals.",
    "cancellation_intensity": "Net level changes do not identify cancellations.",
    "trade_intensity": "This feature schema has no trade-event stream.",
    "book_update_intensity": "Fixed-grid samples are not source event counts.",
    "order_size_distribution": "Visible level quantities are not individual order sizes.",
    "top_level_depth": "The canonical calibration table retains top-five sums only.",
    "queue_position": "Aggregate L2 has no source-native order identities or FIFO state.",
    "causal_price_impact": "Observed co-movement does not identify counterfactual impact.",
}


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3 or left[finite].std() == 0 or right[finite].std() == 0:
        return None
    return float(np.corrcoef(left[finite], right[finite])[0, 1])


def _block_means(values: np.ndarray, block_rows: int) -> np.ndarray:
    # Nonoverlapping chronological blocks preserve gaps and invalid observations.
    # Empty blocks remain missing; no concatenation across missing source rows.
    result = []
    for start in range(0, len(values), block_rows):
        block = values[start:start + block_rows]
        if len(block) != block_rows or np.isfinite(block).sum() < .8 * block_rows:
            continue
        result.append(float(np.nanmean(block)))
    return np.asarray(result)


def mean_drift_interval(reference: np.ndarray, target: np.ndarray, *, block_rows: int,
                        repetitions: int, seed: int, family_size: int = 8,
                        alpha: float = .05) -> dict:
    """Independent chronological block resampling, Bonferroni across observables.

    The interval concerns the difference in means of eligible complete blocks.
    At least eight blocks per period are required. Blocks are not asserted to be
    independent; remaining long memory/nonstationarity limits coverage.
    """
    if isinstance(block_rows, bool) or not isinstance(block_rows, int) or block_rows < 2:
        raise ValueError("block_rows must be an integer >= 2")
    if (isinstance(repetitions, bool) or not isinstance(repetitions, int)
            or not 99 <= repetitions <= 10000):
        raise ValueError("repetitions must be an integer in [99, 10000]")
    if not 0 < alpha < 1 or family_size < 1:
        raise ValueError("invalid interval confidence/family size")
    left, right = _block_means(reference, block_rows), _block_means(target, block_rows)
    result = {"reference_blocks": len(left), "target_blocks": len(right),
              "block_rows": block_rows, "bootstrap_repetitions": repetitions,
              "family_size": family_size, "family_confidence": 1 - alpha,
              "correction": "Bonferroni", "seed": seed}
    if min(len(left), len(right)) < 8:
        return {**result, "status": "INCONCLUSIVE", "reason": "fewer than eight eligible blocks",
                "estimate": None, "interval": None}
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for i in range(repetitions):
        estimates[i] = (rng.choice(right, len(right), replace=True).mean()
                        - rng.choice(left, len(left), replace=True).mean())
    tail = alpha / (2 * family_size)
    return {**result, "status": "ESTIMATED", "estimate": float(right.mean() - left.mean()),
            "interval": np.quantile(estimates, [tail, 1 - tail]).tolist()}


def _dynamics(frame: pd.DataFrame, values: pd.DataFrame, interval_us: int,
              spread_threshold: float) -> dict:
    adjacent = np.diff(frame.timestamp_us.to_numpy()) == interval_us
    acf = {}
    for name in OBSERVABLES:
        column = values[name].to_numpy()
        left, right = column[:-1].copy(), column[1:].copy()
        left[~adjacent] = np.nan
        acf[name] = _correlation(left, right)
    spread = values.spread_bps.to_numpy()
    eligible = adjacent & np.isfinite(spread[:-1]) & np.isfinite(spread[1:])
    states = (spread > spread_threshold).astype(int)
    counts = np.zeros((2, 2), dtype=int)
    for i in np.flatnonzero(eligible):
        counts[states[i], states[i + 1]] += 1
    probabilities = [[float(v / row.sum()) for v in row] if row.sum() else [None, None]
                     for row in counts]
    dependence = {}
    names = list(OBSERVABLES)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            dependence[f"{name}:{other}"] = _correlation(
                values[name].to_numpy(), values[other].to_numpy())
    return {"lag1_autocorrelation": acf, "pairwise_correlation": dependence,
            "spread_state_threshold_from_training": spread_threshold,
            "spread_transition_counts": counts.tolist(),
            "spread_transition_probabilities": probabilities}


def diagnose_period(reference: pd.DataFrame, target: pd.DataFrame, *, interval_us: int,
                    original_status: str, generated: pd.DataFrame | None = None,
                    block_rows: int = 60, repetitions: int = 999, seed: int = 94101) -> dict:
    """Compare a target with training; diagnose associations, never change status.

    ``generated`` contains observable columns, e.g. ``generalization.generate_observables``.
    It is used for marginal model discrepancies only. Temporal dependence is
    estimated on the timestamped real/reference periods without joining seeds.
    """
    from .capabilities import Capability, L2_CONTRACT
    L2_CONTRACT.require(Capability.AGGREGATE_DEPTH)
    if original_status not in {"PASS", "WARNING", "FAIL", "FAILED", "NOT_ESTABLISHED"}:
        raise ValueError("original_status must be a recognized fixed assessment")
    training, actual = observe(reference, interval_us), observe(target, interval_us)
    probabilities = np.array([.01, .05, .25, .5, .75, .95, .99])
    metrics = {}
    for index, name in enumerate(OBSERVABLES):
        reference_values, target_values = training[name].to_numpy(), actual[name].to_numpy()
        left, right = reference_values[np.isfinite(reference_values)], target_values[np.isfinite(target_values)]
        if min(len(left), len(right)) < 32:
            metrics[name] = {"status": "INCONCLUSIVE", "reference_n": len(left), "target_n": len(right),
                             "attribution": "insufficient_data"}
            continue
        scale = float(max(np.quantile(left, .75) - np.quantile(left, .25), left.std(),
                          abs(left.mean()) * .01, 1e-12))
        reference_q, target_q = np.quantile(left, probabilities), np.quantile(right, probabilities)
        quantile_distance = float(np.mean(np.abs(np.quantile(left, np.linspace(.01, .99, 51))
                                                  - np.quantile(right, np.linspace(.01, .99, 51)))) / scale)
        lower, upper = np.quantile(left, [.025, .975])
        coverage = float(np.mean((right >= lower) & (right <= upper)))
        drift_ci = mean_drift_interval(reference_values, target_values, block_rows=block_rows,
                                       repetitions=repetitions, seed=seed + index,
                                       family_size=len(OBSERVABLES))
        model_distance = None
        if generated is not None:
            values = generated[name].dropna().to_numpy()
            if len(values) >= 32:
                model_distance = float(np.mean(np.abs(np.quantile(values, np.linspace(.01, .99, 51))
                                           - np.quantile(right, np.linspace(.01, .99, 51)))) / scale)
        attribution = ("distribution_shift_association" if quantile_distance > .5 else
                       "model_misspecification_or_sampling_error" if model_distance is not None
                       and model_distance > .5 else "no_large_marginal_discrepancy")
        metrics[name] = {"status": "DESCRIPTIVE", "reference_n": len(left), "target_n": len(right),
                         "train_scale": scale, "normalized_quantile_distance": quantile_distance,
                         "central_training_95pct_coverage": coverage,
                         "standardized_mean_drift": float((right.mean() - left.mean()) / scale),
                         "reference_variance": float(left.var()), "target_variance": float(right.var()),
                         "variance_ratio": float(right.var() / left.var()) if left.var() > 0 else None,
                         "quantile_probabilities": probabilities.tolist(),
                         "quantile_errors_train_scaled": ((target_q - reference_q) / scale).tolist(),
                         "outside_training_98pct_fraction": float(np.mean(
                             (right < reference_q[0]) | (right > reference_q[-1]))),
                         "mean_drift_uncertainty": drift_ci,
                         "model_target_normalized_distance": model_distance, "attribution": attribution}
    threshold = float(training.spread_bps.median())
    reference_dynamics = _dynamics(reference, training, interval_us, threshold)
    target_dynamics = _dynamics(target, actual, interval_us, threshold)
    changes = {}
    for statistic in ("lag1_autocorrelation", "pairwise_correlation"):
        changes[statistic] = {name: (target_dynamics[statistic][name] - value
                                    if value is not None and target_dynamics[statistic][name] is not None
                                    else None) for name, value in reference_dynamics[statistic].items()}
    return {"schema_version": 1, "original_status": original_status,
            "overall_status": original_status, "status_changed": False,
            "reference_frame_sha256": _frame_hash(reference), "target_frame_sha256": _frame_hash(target),
            "reference_rows": len(reference), "target_rows": len(target),
            "reference_valid_fraction": float(reference.valid.mean()),
            "target_valid_fraction": float(target.valid.mean()), "observables": metrics,
            "reference_dynamics": reference_dynamics,
            "target_dynamics": target_dynamics, "dynamics_change": changes,
            "unsupported_observability": UNSUPPORTED,
            "causal_attribution": "NOT_ESTABLISHED",
            "parameter_instability": "NOT_ESTABLISHED: no target refits permitted",
            "limitations": ["Associations cannot distinguish causal regime shift from model misspecification.",
                            "Block intervals require approximate stationarity and weak dependence between blocks.",
                            "Bootstrap tail precision is limited by the registered repetition budget.",
                            "No event independence, equivalence, profitability or historical fill claim."]}


def report_markdown(result: dict) -> str:
    lines = ["# Calibration failure diagnostics", "",
             f"Original assessment: **{result['original_status']}** (unchanged).", "",
             f"Reference: {result['reference_rows']} rows; target: {result['target_rows']} rows.", "",
             "| Observable | Distance | Mean drift / train scale | Attribution |",
             "|---|---:|---:|---|"]
    for name, value in result["observables"].items():
        if value["status"] == "INCONCLUSIVE":
            lines.append(f"| {name} | unavailable | unavailable | insufficient data |")
        else:
            lines.append(f"| {name} | {value['normalized_quantile_distance']:.4f} | "
                         f"{value['standardized_mean_drift']:.4f} | {value['attribution']} |")
    lines += ["", "Causal attribution and parameter instability remain NOT_ESTABLISHED.", "",
              "Mean-drift intervals in the JSON use chronological blocks with Bonferroni correction",
              "across eight observables. They are descriptive and assume approximately stationary blocks.", "",
              "Unsupported: " + ", ".join(UNSUPPORTED) + ".", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--original-status", required=True)
    parser.add_argument("--interval-us", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    from .evidence import finalize_evidence, register_evidence
    from .experiments.registry import sha256_file
    output = Path(args.out)
    inputs = [{"identity": role, "role": role, "adapter_version": "canonical-features-v1",
               "kind": "retained_evidence", "sha256": sha256_file(Path(path))}
              for role, path in (("reference", args.reference), ("target", args.target))]
    register_evidence(output, kind="calibration-diagnostics", inputs=inputs,
                      config={"original_status": args.original_status, "interval_us": args.interval_us,
                              "block_rows": 60, "repetitions": 999},
                      seeds={"bootstrap": list(range(94101, 94109))},
                      hypotheses=["Attribute observable discrepancies without changing the original assessment."])
    result = diagnose_period(pd.read_csv(args.reference), pd.read_csv(args.target),
                             interval_us=args.interval_us, original_status=args.original_status)
    for path, record in zip((args.reference, args.target), inputs):
        if sha256_file(Path(path)) != record["sha256"]:
            raise ValueError("diagnostic source changed after registration")
    (output / "diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(report_markdown(result), encoding="utf-8")
    finalize_evidence(output, status="FAILED" if result["overall_status"] in {"FAIL", "FAILED"}
                      else "INCONCLUSIVE", metrics=result)
    print(json.dumps({"out": str(output), "status": result["overall_status"]}))


if __name__ == "__main__":
    main()
