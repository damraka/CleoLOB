"""Policy selection, mixture of experts, abstention and safe fallback (M27).

A *case* is one decision situation: a context (regime label, regime
probabilities, OOD score, model-uncertainty score) and, for evaluation, the
true mean cost of every candidate policy in that situation (from simulation
with known truth). Selectors are trained on training cases only:

* ``global`` — the single policy with the lowest mean training cost;
* ``regime`` — the best training policy within the case's regime label
  (falls back to ``global`` for regimes with fewer than ``min_cases``);
* ``moe`` — mixture of experts: minimizes expected cost under the case's regime
  probabilities, using per-regime training means;
* ``confident`` — the regime choice only if its bootstrap probability of being
  best is at least ``confidence``, else the registered safe baseline.

``Abstention`` wraps any selector: if the OOD score or the uncertainty score
exceeds its registered threshold, the selector abstains and the registered safe
baseline is used (``fallback_used``). ``evaluate`` measures incorrect decisions
(regret above ``margin``), regret, abstention rate and fallback rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SELECTORS = ("global", "regime", "moe", "confident")


@dataclass
class Case:
    case_id: str
    regime: int
    regime_probs: list[float] | None = None
    ood_score: float = 0.0
    uncertainty: float = 0.0
    costs: dict[str, float] = field(default_factory=dict)       # true (or training) mean cost per policy
    cost_samples: dict[str, list[float]] = field(default_factory=dict)


class Selector:
    def __init__(self, kind: str, policies: list[str], *, safe_baseline: str, min_cases: int = 3,
                 confidence: float = 0.6, seed: int = 0) -> None:
        if kind not in SELECTORS:
            raise ValueError(f"unknown selector {kind!r}")
        if safe_baseline not in policies:
            raise ValueError("the safe baseline must be a registered candidate")
        self.kind, self.policies, self.safe = kind, list(policies), safe_baseline
        self.min_cases, self.confidence, self.seed = min_cases, confidence, seed

    def fit(self, cases: list[Case]) -> Selector:
        p = self.policies
        all_costs = np.asarray([[c.costs[x] for x in p] for c in cases], float)
        self.global_choice = p[int(np.argmin(all_costs.mean(0)))]
        self.regime_means: dict[int, np.ndarray] = {}
        self.regime_pbest: dict[int, np.ndarray] = {}
        rng = np.random.default_rng(self.seed)
        for r in sorted({c.regime for c in cases}):
            rows = all_costs[[c.regime == r for c in cases]]
            if len(rows) < self.min_cases:
                continue
            self.regime_means[r] = rows.mean(0)
            wins = np.zeros(len(p))
            for _ in range(400):
                wins[np.argmin(rows[rng.integers(0, len(rows), len(rows))].mean(0))] += 1
            self.regime_pbest[r] = wins / 400
        return self

    def choose(self, case: Case) -> tuple[str, dict]:
        p = self.policies
        if self.kind == "global":
            return self.global_choice, {}
        if self.kind == "moe" and case.regime_probs is not None and self.regime_means:
            labels = sorted(self.regime_means)
            probs = np.asarray([case.regime_probs[r] if r < len(case.regime_probs) else 0.0 for r in labels])
            if probs.sum() > 0:
                expected = (probs / probs.sum()) @ np.asarray([self.regime_means[r] for r in labels])
                return p[int(np.argmin(expected))], {"moe_expected": expected.tolist()}
        if case.regime not in self.regime_means:
            return self.global_choice, {"regime_fallback": "global (regime unseen in training)"}
        i = int(np.argmin(self.regime_means[case.regime]))
        if self.kind == "confident" and self.regime_pbest[case.regime][i] < self.confidence:
            return self.safe, {"low_confidence": float(self.regime_pbest[case.regime][i])}
        return p[i], {}


@dataclass(frozen=True)
class Abstention:
    ood_threshold: float
    uncertainty_threshold: float

    def abstain(self, case: Case) -> bool:
        return case.ood_score > self.ood_threshold or case.uncertainty > self.uncertainty_threshold


def evaluate(selector: Selector, cases: list[Case], *, margin: float, abstention: Abstention | None = None) -> dict:
    rows = []
    for c in cases:
        choice, info = selector.choose(c)
        abstained = bool(abstention and abstention.abstain(c))
        if abstained:
            choice = selector.safe
        best = min(c.costs.values())
        regret = c.costs[choice] - best
        rows.append({"case": c.case_id, "choice": choice, "regret": regret, "incorrect": regret > margin,
                     "abstained": abstained, "fallback_used": abstained or "low_confidence" in info,
                     "ood_score": c.ood_score, "uncertainty": c.uncertainty, "regime": c.regime})
    regrets = np.asarray([r["regret"] for r in rows], float)
    return {"selector": selector.kind, "n": len(rows), "incorrect_rate": float(np.mean([r["incorrect"] for r in rows])),
            "mean_regret": float(regrets.mean()), "p95_regret": float(np.quantile(regrets, 0.95)),
            "abstention_rate": float(np.mean([r["abstained"] for r in rows])),
            "fallback_rate": float(np.mean([r["fallback_used"] for r in rows])), "margin": margin, "rows": rows}
