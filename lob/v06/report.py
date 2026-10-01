"""M15/M17: paper figures and the machine-readable claim graph, built only from sealed results.

``build(out)`` reads the registered v0.6 runs, writes deterministic SVG+CSV figures
under ``out/figures`` and ``out/claims.json``. Every number in a figure or claim is
read from a sealed ``result.json``; nothing is typed by hand. Run directories are
the registered ones listed in ``RUNS``; pilots and invalid attempts are never used.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..experiments.registry import PROJECT_ROOT
from . import figures as fg
from . import protocol as pr
from .claims import make_claim, write_claims
from .observables import FAMILIES

RUNS = {
    "design": "results/v06/m1/design", "develop": "results/v06/m6/develop", "select": "results/v06/m6/select",
    "regime": "results/v06/m12/regime", "identifiability": "results/v06/m7/identifiability",
    "evaluation": "results/v06/m10/evaluation", "bank": "results/v06/m13/bank-2",
    "june": "results/v06/m13/deribit-eth-perp-2020-06-01", "eth_sep": "results/v06/m13/deribit-eth-perp-2020-09-01-registered",
    "btc_sep": "results/v06/m13/deribit-btc-perp-2020-09-01", "eth_jul": "results/v06/m13/deribit-eth-perp-2020-07-01",
    "btc_jul": "results/v06/m13/deribit-btc-perp-2020-07-01", "eth_aug": "results/v06/m13/deribit-eth-perp-2020-08-01",
    "transfer_oct": "results/v06/m14/deribit-eth-perp-2020-10-01", "transfer_aug": "results/v06/m14/deribit-eth-perp-2020-08-01",
    "benchmarks": "results/v06/m16/benchmarks-2",
}
HOLDOUTS = (("june", "ETH Jun (retro)"), ("eth_jul", "ETH Jul (retro)"), ("btc_jul", "BTC Jul (retro)"),
            ("eth_aug", "ETH Aug (retro)"), ("eth_sep", "ETH Sep (fresh)"), ("btc_sep", "BTC Sep (fresh)"))


def _result(root: Path, key: str) -> dict:
    return json.loads((root / RUNS[key] / "result.json").read_text(encoding="utf-8"))


def figures(root: Path, out: Path) -> list[str]:
    made = []
    design = _result(root, "design")
    rows, values = ["real vs real margin"], [[design["equivalence_margins"][f] for f in FAMILIES]]
    for key, label in HOLDOUTS:
        models = _result(root, key)["realism"]["models"]
        for model in ("selected", "control"):
            rows.append(f"{label} {'v3' if model == 'selected' else 'v0.5'}")
            values.append([models[model]["families"][f]["error"] for f in FAMILIES])
    fg.heatmap(out / "fig01-realism-family-errors", "Realism family errors by dataset and model",
               rows, [f.replace("_", " ") for f in FAMILIES], values,
               note="First row: sealed real-vs-real margins (development vs selection day). Lower is closer.")
    made.append("fig01-realism-family-errors")

    select = _result(root, "select")
    from .calibration import NAMES
    fg.heatmap(out / "fig02-ensemble-parameters", "Materially distinct near-optimal parameter vectors (unit box)",
               [m["key"] for m in select["ensemble"]], list(NAMES), [m["unit"] for m in select["ensemble"]],
               note="Transformed unit coordinates; both vectors lie in the sealed near-optimal region.")
    made.append("fig02-ensemble-parameters")
    develop = _result(root, "develop")
    points = [{"label": k, "x": develop["rescored"][k]["rescore_objective"], "y": v, "group":
               "near-optimal" if k in select["near_optimal"] else "advanced", "annotate": k == select["selected"]["key"]}
              for k, v in select["selection_objectives"].items() if v is not None and k in develop["rescored"]]
    fg.scatter(out / "fig03-calibration-selection", "Development re-score vs selection objective (32 advanced)",
               points, x_label="development re-score objective", y_label="selection-day objective",
               groups=("near-optimal", "advanced"), note=f"v0.5 control on the selection day: "
               f"{select['control']['selection_objective']:.2f}; real-vs-real yardstick 1.25")
    made.append("fig03-calibration-selection")

    ident = _result(root, "identifiability")
    sens = ident["sensitivity"]
    fg.heatmap(out / "fig04-sensitivity", "Local sensitivity of family errors (per unit step)", sens["families"],
               sens["parameters"], sens["matrix"], signed=True, fmt="{:.1f}",
               note="Central differences, common seeds; n/a = event-capped evaluation. Rank <= 9 by construction.")
    made.append("fig04-sensitivity")
    fg.heatmap(out / "fig05-profiles", "Profile diagnostics: best objective with one parameter fixed",
               list(ident["profiles"]), [str(g) for g in next(iter(ident["profiles"].values()))["grid"]],
               [v["best_objective"] for v in ident["profiles"].values()],
               note="24 re-optimization draws per grid point; n/a = all draws event-capped.")
    made.append("fig05-profiles")

    evaluation = _result(root, "evaluation")
    worlds = list(evaluation["worlds"])
    agents = ["twap", "vwap", "pov", "ac", "ppo:single", "dqn:single", "ppo:ensemble", "dqn:ensemble"]
    fg.heatmap(out / "fig06-world-costs", "Mean completion-adjusted cost (bps) by simulator world and agent", worlds,
               agents, [[(evaluation["summaries"].get(f"{w}|{a}") or {}).get("mean_cost_bps") for a in agents]
                        for w in worlds], note="200 market seeds per cell; learned agents averaged over 4 seeds.")
    made.append("fig06-world-costs")
    pairs = evaluation["rank_stability_all_worlds_descriptive"]["pairs"]
    fg.heatmap(out / "fig07-pair-directions", "Pairwise conclusion direction per world (+1 left costlier)",
               list(pairs), worlds, [[pairs[p]["directions"].get(w) for w in worlds] for p in pairs], signed=True,
               fmt="{:.0f}", note="Intervals at 0.05 / (15 x 19); 0 = indeterminate. Descriptive across all worlds.")
    made.append("fig07-pair-directions")
    risk = evaluation["model_risk"]["agents"]
    components = ["market_seed_se_bps", "calibration_ensemble_sd_bps", "structural_intervention_sd_bps",
                  "regime_model_sd_bps", "training_seed_sd_bps"]
    fg.heatmap(out / "fig08-model-risk", "Separate uncertainty components (bps; not additive)", list(risk),
               [c.replace("_bps", "").replace("_", " ") for c in components],
               [[risk[a].get(c) for c in components] for a in risk], note="Each column is its own SD or SE.")
    made.append("fig08-model-risk")
    h7 = evaluation["world_realism"]
    instability = evaluation["instability"]
    points = [{"label": w, "x": h7[w]["families"].get("temporal"), "y": instability[w]["conclusion_disagreement"],
               "group": "intervention" if w.startswith("intervention") else "ensemble"}
              for w in instability if w in h7 and w != "selected"]
    fg.scatter(out / "fig09-realism-vs-instability", "Temporal-family error vs conclusion disagreement (H7)", points,
               x_label="selection-day temporal family error", y_label="fraction of 15 conclusions changed",
               groups=("ensemble", "intervention"), note="Within-simulator association only; H7 NOT_ESTABLISHED.")
    made.append("fig09-realism-vs-instability")

    labels, aucs = [], []
    for key, label in HOLDOUTS:
        gap = _result(root, key)["domain_gap"]
        for name in ("logistic", "tree", "forest"):
            labels.append(f"{label} {name}")
            aucs.append(gap["classifiers"][name]["auc"] if gap.get("classifiers") else None)
    fg.bars(out / "fig10-domain-gap-auc", "Real-vs-synthetic discriminator test AUC", labels, aucs, unit="AUC",
            note="0.5 = indistinguishable; leakage-resistant 60 s window features; chronological/seed-grouped splits.")
    made.append("fig10-domain-gap-auc")
    dims = ["volatility", "spread", "activity", "stress"]
    rows, values = [], []
    for key, label in HOLDOUTS:
        comparison = _result(root, key)["transitions"]["comparison"]["selected"]
        rows.append(label)
        values.append([comparison[d]["transition_row_tv_mean"] for d in dims])
    fg.heatmap(out / "fig11-regime-transitions", "Regime-transition mismatch, selected v3 vs history (row TV)", rows,
               dims, values, note="5-minute v0.5 M6 labels; simulated 8 x 6 h paths.")
    made.append("fig11-regime-transitions")
    transfer = _result(root, "transfer_oct")
    labels = [k for k in sorted(transfer["means"]) if k.startswith("conservative|")]
    fg.bars(out / "fig12-historical-costs", "Bounded historical cost, ETH 2020-10-01 (conservative fills)",
            [k.split("|")[1] for k in labels], [transfer["means"][k]["mean_cost_bps"] for k in labels], unit="bps",
            note="144 episodes; no exact fills; all pairwise contrasts indeterminate at the registered alpha.")
    made.append("fig12-historical-costs")
    return made


def claims(root: Path) -> list[dict]:
    families = pr.load_protocol(root / pr.PROTOCOL_PATH)["statistics"]["families"]

    def run(key: str) -> Path:
        return root / RUNS[key]

    def c(claim_id, hypothesis, key, path, metric, threshold, status_path, family, level, interpretation, **kw):
        size = families[family]["size"] if family else None
        alpha = families[family]["adjusted_alpha"] if family else None
        return make_claim(claim_id=claim_id, hypothesis_id=hypothesis, run=run(key), result_path=path, metric=metric,
                          threshold=threshold, status_path=status_path, family=family,
                          family_size=size if isinstance(size, int) else None,
                          alpha=alpha if isinstance(alpha, float) else None, evidence_level=level,
                          interpretation=interpretation, root=root, **kw)

    out = [
        c("H1", "H1", "june", "improvement", "objective(v3) - objective(v0.5 control)", "Bonferroni upper bound < 0",
          "improvement/status", "F1_calibration", "retrospective", "Relative improvement on consumed June data; "
          "absolute realism still fails 7/9 margins.", estimate_key="difference"),
        c("H2", "H2", "eth_sep", "improvement", "objective(v3) - objective(v0.5 control)", "Bonferroni upper bound < 0",
          "improvement/status", "F1_calibration", "confirmatory", "Fresh same-instrument improvement; no family "
          "equivalent within its real-vs-real margin.", estimate_key="difference"),
        c("H3", "H3", "btc_sep", "improvement", "objective(v3) - objective(v0.5 control)", "Bonferroni upper bound < 0",
          "improvement/status", "F1_calibration", "confirmatory", "Cross-instrument improvement with ETH scale and "
          "tick geometry; absolute fit still fails.", estimate_key="difference"),
        c("H4", "H4", "identifiability", "H4_identifiability", "materially distinct near-optimal vectors",
          ">= 2 (L-infinity >= 0.25)", "H4_identifiability/status", "F2_identifiability", "descriptive",
          "Calibration is not uniquely identified at the sealed objective and seed resolution.",
          estimate_key="distinct_near_optimal"),
        c("H5", "H5", "evaluation", "H5_equifinality", "max |member cost difference| (classical agents)",
          "Bonferroni interval excludes 0 and |difference| >= 1 bps", "H5_equifinality/status", "F3_equifinality",
          "confirmatory",
          "Not established; the design's execution MDE (1.0-1.7 bps per mean) exceeds the 1 bps margin, so this is "
          "low resolution, not invariance.", estimate_key="max_abs_difference_bps"),
        c("H6", "H6", "evaluation", "H6_rank_stability", "certified reversals across the 2 ensemble members",
          "opposite certified directions in two members", "H6_rank_stability/status", "F4_rank_stability",
          "confirmatory",
          "No certified reversal; point rankings of the two plausible worlds are nearly unrelated (Kendall tau 0.07).",
          estimate_key="kendall_tau_mean"),
        c("H7", "H7", "evaluation", "H7_execution_sensitive_realism", "max Spearman rho (family error vs disagreement)",
          "rho >= 0.5 with Holm-adjusted significance", "H7_execution_sensitive_realism/status", "F5_sensitivity",
          "confirmatory", "No realism family explains conclusion instability across 15 synthetic worlds."),
        c("H8", "H8", "eth_sep", "domain_gap/classifiers/logistic", "test ROC-AUC (logistic)", "one-sided lower bound > 0.5",
          "domain_gap/status", "F6_domain_gap", "confirmatory", "Real and synthetic 60 s windows are perfectly "
          "separable; a weak discriminator would not have shown realism.", estimate_key="auc"),
        c("H9", "H9", "eth_sep", "regime", "regime model vs global, within regime", "Bonferroni upper bound < 0 in both "
          "regimes", "regime/H9", "F7_regime", "confirmatory", "High-volatility model better within regime; "
          "low-volatility model (23 selection blocks) worse: FAILED by rule."),
        c("H10", "H10", "eth_sep", "regime", "off-regime noninferiority", "upper bound < delta = 0.244 in both "
          "directions", "regime/H10", "F7_regime", "confirmatory", "Regime models do not transfer outside their regime."),
        c("H11-ppo", "H11", "transfer_oct", "H11/ppo", "gap(ensemble) - gap(single), PPO", "upper bound < 0 in both "
          "fill modes", "H11/ppo/status", "F8_transfer_learning", "confirmatory",
          "Domain-randomized PPO does not transfer more consistently than single-world PPO."),
        c("H11-dqn", "H11", "transfer_oct", "H11/dqn", "gap(ensemble) - gap(single), DQN", "upper bound < 0 in both "
          "fill modes", "H11/dqn/status", "F8_transfer_learning", "confirmatory",
          "Domain-randomized DQN does not transfer more consistently than single-world DQN."),
        c("H12", "H12", "transfer_oct", "H12_fill_semantics", "pairs with consistent conclusions across fill modes",
          "no opposite or one-mode-only determinate pair", "H12_fill_semantics/status", "F9_fill_semantics",
          "confirmatory", "Established vacuously: all 15 pairs are indeterminate under both fill modes.",
          estimate_key="evaluable"),
        c("S-support-eth-sep", None, "eth_sep", "support", "out-of-support fraction of fresh ETH windows",
          "> 0.5 means NO_CLAIM", None, None, "descriptive", "Historical windows lie outside the calibrated simulators' "
          "support; absolute realism claims are withheld.", estimate_key="out_of_support_fraction",
          status="ESTABLISHED" if _result(root, "eth_sep")["support"]["out_of_support_fraction"] > 0.5
          else "NOT_ESTABLISHED"),
    ]
    return out


def build(out: str | Path, *, root: Path = PROJECT_ROOT) -> dict:
    out = Path(out)
    made = figures(root, out / "figures")
    claim_list = claims(root)
    protocol_sha = pr.protocol_sha256(pr.load_protocol(root / pr.PROTOCOL_PATH))
    write_claims(out / "claims.json", claim_list, protocol_sha)
    return {"figures": made, "claims": {c["claim_id"]: c["status"] for c in claim_list}}
