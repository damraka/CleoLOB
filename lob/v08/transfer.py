"""Ranking transfer and simulator decision utility (M32).

Compares a *source* ranking of policies (e.g. synthetic predictions) with a
*target* (e.g. bounded historical outcomes, another period, instrument or
venue):

* Kendall tau-b and Spearman correlation of mean costs;
* pairwise direction accuracy on the pairs the target can actually decide
  (``determinate`` pairs: bounded historical intervals that exclude zero), with
  an exact binomial test against chance (0.5);
* top-k retention (the source's top k among the target's top k);
* regret transfer: target regret of the source-chosen policy;
* rank uncertainty (bootstrap over target units);
* dominance stability across several target cells (fraction of pairs whose
  direction is the same in every cell where it is determinate).

A simulator can be generatively poor yet decision-useful; this module measures
the decision side directly and never infers it from realism scores.
"""
from __future__ import annotations

import math

import numpy as np

from .statistics import kendall_tau, rank_uncertainty, spearman


def binomial_greater(k: int, n: int, p: float = 0.5) -> float:
    """One-sided exact P(X >= k) for X ~ Bin(n, p)."""
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def compare(source: dict[str, float], target: dict[str, float], *, determinate: dict[str, str] | None = None,
            k: int = 3, target_units: np.ndarray | None = None, names: list[str] | None = None,
            seed: int = 0) -> dict:
    """``determinate``: pair key 'A|B' -> 'A_LOWER' / 'B_LOWER' / 'INCONCLUSIVE' for the target."""
    names = names or sorted(set(source) & set(target))
    s = np.asarray([source[n] for n in names], float)
    t = np.asarray([target[n] for n in names], float)
    out = {"policies": names, "kendall_tau": kendall_tau(s, t), "spearman": spearman(s, t)}
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = f"{a}|{b}"
            verdict = (determinate or {}).get(key)
            if verdict is None:
                verdict = "A_LOWER" if t[i] < t[names.index(b)] else "B_LOWER" if t[i] > t[names.index(b)] else \
                    "INCONCLUSIVE"
                inferred = True
            else:
                inferred = False
            if verdict == "INCONCLUSIVE":
                continue
            predicted = "A_LOWER" if source[a] < source[b] else "B_LOWER"
            pairs.append({"pair": key, "target": verdict, "predicted": predicted, "agree": predicted == verdict,
                          "target_from_point_estimates": inferred})
    agree = sum(p["agree"] for p in pairs)
    out["pairwise"] = {"determinate_pairs": len(pairs), "agree": agree,
                       "accuracy": agree / len(pairs) if pairs else None,
                       "p_value_vs_chance": binomial_greater(agree, len(pairs)) if pairs else None,
                       "pairs": pairs}
    top_s = set(sorted(names, key=lambda n: source[n])[:k])
    top_t = set(sorted(names, key=lambda n: target[n])[:k])
    out["top_k"] = {"k": k, "retained": len(top_s & top_t), "retention": len(top_s & top_t) / k}
    chosen = min(names, key=lambda n: source[n])
    out["regret_transfer"] = {"source_choice": chosen, "target_best": min(names, key=lambda n: target[n]),
                              "target_regret": float(target[chosen] - min(target.values()))}
    if target_units is not None:
        out["rank_uncertainty"] = rank_uncertainty(np.asarray(target_units, float), names, n_boot=500, seed=seed)
    return out


def dominance_stability(cells: dict[str, dict[str, str]]) -> dict:
    """``cells``: cell -> {pair: verdict}. A pair is stable if every determinate verdict agrees."""
    pairs = sorted({p for c in cells.values() for p in c})
    stable, unstable, never = [], [], []
    for p in pairs:
        verdicts = {c[p] for c in cells.values() if c.get(p) in ("A_LOWER", "B_LOWER")}
        if not verdicts:
            never.append(p)
        elif len(verdicts) == 1:
            stable.append(p)
        else:
            unstable.append(p)
    decided = len(stable) + len(unstable)
    return {"stable": stable, "unstable": unstable, "never_determinate": never,
            "stability": len(stable) / decided if decided else None, "cells": len(cells)}


def matrix(cells: dict[str, dict[str, float]], *, names: list[str]) -> dict:
    """Kendall tau between every pair of cells (transfer heatmap)."""
    keys = sorted(cells)
    m = np.ones((len(keys), len(keys)))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i < j:
                m[i, j] = m[j, i] = kendall_tau([cells[a][n] for n in names], [cells[b][n] for n in names])
    return {"cells": keys, "kendall_tau": m.tolist()}
