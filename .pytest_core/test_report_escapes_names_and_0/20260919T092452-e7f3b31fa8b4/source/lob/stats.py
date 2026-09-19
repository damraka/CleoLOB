"""Statistics for comparing execution agents.

All comparisons are *paired by seed*: agents share exogenous random streams, while
their orders can change the realized book. Seeds are the independent sampling unit.
Bootstrap intervals and an exact sign test keep this dependency-free (numpy/pandas only).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def bootstrap_ci(x: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise ValueError("bootstrap samples must be a finite one-dimensional sequence")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if arr.size == 0:
        return (math.nan, math.nan)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    # Bound temporary allocation instead of n_boot × episode_count in memory.
    means = np.empty(n_boot)
    batch = max(1, min(n_boot, 1_000_000 // arr.size))
    for start in range(0, n_boot, batch):
        stop = min(start + batch, n_boot)
        idx = rng.integers(0, arr.size, size=(stop - start, arr.size))
        means[start:stop] = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test p-value (ties excluded)."""
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (wins, losses)):
        raise ValueError("wins and losses must be nonnegative integers")
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def _validate_episodes(df: pd.DataFrame, metric: str) -> None:
    required = {"scenario", "agent", "seed", "label", metric}
    if not required <= set(df):
        raise ValueError(f"missing episode columns: {sorted(required - set(df))}")
    if df[list(required)].isna().any().any():
        raise ValueError("episode identifiers and outcomes must not be missing")
    if df.duplicated(["scenario", "agent", "seed"]).any():
        raise ValueError("duplicate scenario/agent/seed episodes are not independent samples")
    if not np.isfinite(df[metric].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite economic outcomes cannot enter statistical tests")


def adjust_pvalues(values: Sequence[float], method: str = "holm") -> np.ndarray:
    """Bonferroni/Holm family-wise correction or Benjamini-Hochberg FDR.

    The family must include all planned comparisons; callers must not prefilter
    p-values by significance. Returned values preserve input order.
    """
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be finite and in [0, 1]")
    if method not in {"bonferroni", "holm", "fdr_bh"}:
        raise ValueError("unknown multiple-testing correction")
    n = len(p)
    if not n:
        return p.copy()
    if method == "bonferroni":
        return np.minimum(1.0, p * n)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    if method == "holm":
        adjusted = np.maximum.accumulate(ranked * np.arange(n, 0, -1))
    else:
        adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    result = np.empty(n)
    result[order] = np.minimum(1.0, adjusted)
    return result


def summarize(df: pd.DataFrame, metric: str = "effective_bps", *, n_boot: int = 2000,
              alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Per (scenario, agent) descriptive statistics with a bootstrap CI of the mean."""
    _validate_episodes(df, metric)
    rows: List[Dict[str, object]] = []
    for (scenario, agent), g in df.groupby(["scenario", "agent"], sort=False, observed=True):
        x = g[metric].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(x, n_boot=n_boot, alpha=alpha, seed=seed)
        rows.append({
            "scenario": scenario, "agent": agent, "label": g["label"].iloc[0], "n": int(x.size),
            "mean": x.mean(), "ci_lo": lo, "ci_hi": hi, "median": float(np.median(x)),
            "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "standard_error": float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else None,
            "p5": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)),
            "worst": float(x.max()),
            "best": float(x.min()),
            "fill_frac": float(g["fill_frac"].mean()),
            "filled_only_mean": float(g["shortfall_bps"].mean()),
            "children": float(g["children"].mean()),
        })
    return pd.DataFrame(rows)


def paired_vs_reference(df: pd.DataFrame, reference: str = "ac",
                        metric: str = "effective_bps", *, correction: str = "holm",
                        n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Paired-by-seed comparison of every agent against ``reference``.

    delta = agent − reference (negative = agent cheaper). Reports mean delta with a
    bootstrap CI, win rate (share of seeds where the agent was strictly cheaper) and an
    exact sign-test p-value.
    """
    _validate_episodes(df, metric)
    rows: List[Dict[str, object]] = []
    for scenario, g in df.groupby("scenario", sort=False, observed=True):
        ref = g[g["agent"] == reference].set_index("seed")[metric]
        if ref.empty:
            raise ValueError(f"missing reference agent in scenario {scenario}")
        for agent, ga in g.groupby("agent", sort=False, observed=True):
            if agent == reference:
                continue
            a = ga.set_index("seed")[metric]
            if set(ref.index) != set(a.index):
                raise ValueError(f"incomplete paired seeds for {scenario}/{agent}; retain failures and withhold inference")
            common = ref.index.sort_values()
            delta = (a.loc[common] - ref.loc[common]).to_numpy(dtype=float)
            wins = int((delta < 0).sum())
            losses = int((delta > 0).sum())
            lo, hi = bootstrap_ci(delta, n_boot=n_boot, alpha=alpha, seed=seed)
            rows.append({
                "scenario": scenario, "agent": agent, "label": ga["label"].iloc[0],
                "reference": reference, "n": int(delta.size),
                "delta_mean": float(delta.mean()), "delta_ci_lo": lo, "delta_ci_hi": hi,
                "delta_median": float(np.median(delta)),
                "win_rate": wins / delta.size, "wins": wins, "losses": losses,
                "p_sign": sign_test_p(wins, losses),
                "ties": int(delta.size) - wins - losses,
            })
    columns = ["scenario", "agent", "label", "reference", "n", "delta_mean", "delta_ci_lo",
               "delta_ci_hi", "delta_median", "win_rate", "wins", "losses", "p_sign", "ties"]
    result = pd.DataFrame(rows, columns=columns)
    result["p_adjusted"] = adjust_pvalues(result["p_sign"], correction)
    result["correction"] = correction
    result["family_size"] = len(result)
    result["significant_5pct"] = result["p_adjusted"] < 0.05
    result["significant_at_alpha"] = result["p_adjusted"] < alpha
    return result
