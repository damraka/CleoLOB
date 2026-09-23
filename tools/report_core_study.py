"""Render sealed outcomes as standalone Plotly HTML; never rerun policies."""
from __future__ import annotations
import argparse
from collections import Counter
from html import escape
import json
import math
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lob.core_study import verify_study
from lob.experiments.registry import PROJECT_ROOT, sha256_file


def _interval_text(interval: dict | None) -> str:
    if not interval:
        return "unavailable"
    return (f"{interval['mean_delta_bps']:+.3f} bps "
            f"({100 * interval['confidence']:g}% CI "
            f"[{interval['ci_low_bps']:+.3f}, {interval['ci_high_bps']:+.3f}])")


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _number(value: float | None, digits: int = 3) -> str:
    return "unavailable" if value is None else f"{value:.{digits}f}"


def _extras(study: Path, out: Path, result: dict, plan: dict) -> None:
    context = {}
    for name, digest in plan.get("evidence", {}).items():
        if Path(name).name not in {"PARALLEL_REPLICATION.json", "export-provenance.json"}:
            continue
        path = (PROJECT_ROOT / name).resolve()
        if not path.is_relative_to(PROJECT_ROOT) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Registered report context changed or missing: {name}")
        context[name] = {"sha256": digest, "document": json.loads(path.read_text(encoding="utf-8"))}
    rows = [json.loads(line) for line in (study / "final_episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    selectors = [("Risk-neutral AC", "ac", 0.0), ("Fixed risk-sensitive AC", "ac_risk", 0.0)]
    selectors += [(f"PPO / {penalty:g} bps", "ppo", penalty) for penalty in plan["penalties_bps"]]
    groups = []
    for label, agent, penalty in selectors:
        selected = [row for row in rows if row["agent"] == agent and row["penalty_bps"] == penalty]
        if not selected:
            continue
        priced = [row for row in selected if row["status"] != "INVALID"
                  and row.get("net_effective_bps") is not None]
        components = {}
        for title, key in (("Actual fill shortfall", "gross_cost"), ("Actual fees", "total_fees"),
                           ("Hypothetical residual shortfall", "hypothetical_liquidation_cost"),
                           ("Hypothetical residual fees", "hypothetical_liquidation_fees")):
            components[title] = _mean([row[key] / (row["target_quantity"] * row["arrival"]) * 1e4
                                      for row in priced if row.get(key) is not None])
        groups.append({"label": label, "episodes": len(selected), "priced_episodes": len(priced),
                       "statuses": dict(Counter(row["status"] for row in selected)),
                       "mean_net_cost_bps": _mean([row["net_effective_bps"] for row in priced]),
                       "mean_fill_fraction": _mean([row.get("fill_frac") for row in selected]),
                       "mean_completion_penalty_bps": _mean([row.get("completion_penalty_bps") for row in selected]),
                       "cost_components_bps": components})
    models = result["training_summary"]["models"]
    rewards = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
                            subplot_titles=["Signed training reward components", "Completion penalty share of absolute reward components"])
    labels = [f"{model['penalty_bps']:g} bps / {model['training_seed']}" for model in models]
    terms = sorted({term for model in models for term in model["reward_summary"]["mean_reward_terms_bps"]})
    for term in terms:
        rewards.add_trace(go.Bar(name=term.replace("_", " "), x=labels,
                                 y=[model["reward_summary"]["mean_reward_terms_bps"].get(term, 0) for model in models]),
                          row=1, col=1)
    rewards.add_trace(go.Bar(name="Completion share", x=labels, showlegend=False,
                             y=[100 * model["reward_summary"]["completion_penalty_absolute_share"] for model in models],
                             marker_color="#8c65a8"), row=2, col=1)
    rewards.update_yaxes(title_text="Mean contribution (bps)", row=1, col=1)
    rewards.update_yaxes(title_text="Absolute share (%)", range=[0, 100], row=2, col=1)
    rewards.update_xaxes(tickangle=-55)
    rewards.update_layout(template="plotly_white", height=960, barmode="relative",
                          title="Training objective composition per optimizer seed",
                          legend={"orientation": "h", "y": 1.07}, margin={"b": 160})
    rewards.write_html(out / "reward-decomposition.html", include_plotlyjs=True, auto_open=False)
    execution = make_subplots(rows=2, cols=1, vertical_spacing=0.2,
                              subplot_titles=["Final net cost components (priced episodes)", "Actual fill and economic outcome validity"])
    labels = [group["label"] for group in groups]
    for component in groups[0]["cost_components_bps"]:
        execution.add_trace(go.Bar(name=component, x=labels,
                                   y=[group["cost_components_bps"][component] for group in groups]), row=1, col=1)
    execution.add_trace(go.Scatter(name="Actual fill (%)", x=labels, mode="markers+lines",
                                   y=[100 * group["mean_fill_fraction"] if group["mean_fill_fraction"] is not None else None
                                      for group in groups]), row=2, col=1)
    execution.add_trace(go.Scatter(name="Priced episodes (%)", x=labels, mode="markers+lines",
                                   y=[100 * group["priced_episodes"] / group["episodes"] for group in groups]), row=2, col=1)
    execution.update_yaxes(title_text="Mean cost contribution (bps)", row=1, col=1)
    execution.update_yaxes(title_text="Percent", range=[0, 105], row=2, col=1)
    execution.update_layout(template="plotly_white", height=840, barmode="relative",
                            title={"text": "Execution economics: actual fills and hypothetical residual liquidation",
                                   "y": .98, "yanchor": "top"},
                            legend={"orientation": "h", "y": 1.05, "yanchor": "bottom"},
                            margin={"t": 150, "b": 80})
    execution.write_html(out / "execution-decomposition.html", include_plotlyjs=True, auto_open=False)
    report = {"result": result, "economic_groups": groups, "registered_context": context,
              "training_reward_components": [{"penalty_bps": model["penalty_bps"], "training_seed": model["training_seed"],
                                                **model["reward_summary"]} for model in models],
              "aggregation": "Balanced final crossed design; policy costs use priced episodes. Training reward means are per optimizer seed."}
    (out / "report-data.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    primary = result["primary"]
    lines = ["# Fixed-budget PPO versus simulator-fitted AC", "",
             f"Primary, 0 bps completion penalty: **{_interval_text(primary.get('economic_interval'))}**.",
             "Positive differences mean PPO costs more. Economic cost includes actual fills, fees and",
             "hypothetical visible-book residual liquidation; it excludes the completion penalty.", "",
             f"Inference status: `{primary['inference_status']}`. These are conditional synthetic results;",
             "the separately frozen real-data calibration failed its external fidelity gates.", "",
             f"The study fitted **{len(models)} PPO models**: {len(plan['training_seeds'])} optimizer seeds × "
             f"{len(plan['penalties_bps'])} penalties, {plan['timesteps_per_model']:,} steps each. "
             "This fixed budget is not convergence evidence.", "",
             f"Final evaluation retained **{result['actual_final_episodes']:,}/{result['planned_final_episodes']:,} "
             f"planned episodes**, including **{result['invalid_final_episodes']} INVALID**, across "
             f"**{result['power']['final_market_count']} common market seeds**.", "",
             "## Penalty ablation", "",
             "Intervals independently resample optimizer and market seeds. The simultaneous family contains",
             "four penalty arms and the prespecified risk-sensitive AC comparison.", "",
             "| Training penalty | PPO minus risk-neutral AC | Complete markets | INVALID PPO episodes | Mean actual fill |",
             "|---|---|---:|---:|---:|"]
    for arm in result["penalty_ablations"]:
        fill = arm.get("mean_fill_fraction")
        lines.append(f"| {arm['penalty_bps']:g} bps | {_interval_text(arm.get('family_interval'))} | "
                     f"{arm['complete_markets']}/{arm['planned_markets']} | {arm['invalid_episodes']} | "
                     f"{_number(100 * fill if fill is not None else None, 2)}% |")
    secondary = result["ac_risk_sensitivity"]
    power = result["power"]
    lines += ["", "Risk-neutral AC equals TWAP analytically. The secondary comparator fixes κT=1 using",
              "simulator-fitted impact and volatility; it is not selected from final results.", "",
              f"Primary PPO minus fixed risk-sensitive AC: **{_interval_text(secondary.get('family_interval'))}**",
              f"(`{secondary['status']}`).", "", "## Power and execution", "",
              f"Diagnostic power analysis locked {power['final_market_count']} markets for a "
              f"{plan['mde_bps']:g} bps effect and {100 * plan['target_power']:g}% target power.",
              f"Approximate achieved power: {_number(power.get('approximate_achievable_power'))}; "
              f"resource/training-variance cap active: {power['capped']}.",
              "Five optimizer seeds impose a variance floor that extra market episodes cannot remove.", "",
              "| Policy | Priced / total episodes | Mean net cost (bps) | Actual fill | Completion penalty (bps) |",
              "|---|---:|---:|---:|---:|"]
    for group in groups:
        fill = group["mean_fill_fraction"]
        lines.append(f"| {group['label']} | {group['priced_episodes']}/{group['episodes']} | "
                     f"{_number(group['mean_net_cost_bps'])} | "
                     f"{_number(100 * fill if fill is not None else None, 2)}% | "
                     f"{_number(group['mean_completion_penalty_bps'])} |")
    lines += ["", "Priceable residuals are hypothetical valuations, not actual filled quantities. Cost means",
              "use priced episodes only; missing outcomes and their denominators remain visible.", "",
              "The 100/500 bps residual-price sensitivities in `report-data.json` are assumptions, not",
              "observed liquidation prices or guaranteed worst cases. If every outcome is valid, they",
              "reproduce the raw comparisons. No policy or test episode was rerun for this report.", ""]
    lines += ["## Training reward decomposition", "",
              "Each table entry averages the five optimizer-level summaries equally. The share uses",
              "absolute reward components before averaging, so offsetting gains/losses do not hide",
              "the completion penalty. These are training returns, not final economic costs.", "",
              "| Training penalty | Mean return (bps) | Mean penalty contribution (bps) | Mean absolute penalty share |",
              "|---|---:|---:|---:|"]
    for penalty in plan["penalties_bps"]:
        summaries = [model["reward_summary"] for model in models if model["penalty_bps"] == penalty]
        mean_return = _mean([summary.get("mean_return_bps") for summary in summaries])
        penalty_term = _mean([summary["mean_reward_terms_bps"].get("completion_penalty", 0) for summary in summaries])
        share = _mean([100 * summary["completion_penalty_absolute_share"] for summary in summaries])
        lines.append(f"| {penalty:g} bps | {_number(mean_return)} | {_number(penalty_term)} | {_number(share, 2)}% |")
    lines += ["", "## Replication scope", "",
              "Intervals use this run's five optimizer seeds and common market paths. Repeating a fixed",
              "design on the same seeds is a computational replication, not a fresh independent holdout.", ""]
    if any(Path(name).name == "PARALLEL_REPLICATION.json" for name in context):
        lines += ["This retained Windows run repeats the design already executed in Linux CI runs",
                  "35649272649 and 35649544066. Those prior computations are not pooled as extra seeds.",
                  "The original local serial attempt was preserved after one completed model and an",
                  "interrupted second fit. A new source snapshot and registration introduced isolated",
                  "process scheduling without changing the 20-model design, seeds, budget or settings.",
                  "The registered computational amendment and prior-run export provenance are included",
                  "with verified hashes in `report-data.json`.", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    pages = [("Comparison intervals", "penalty-comparison.html", 450),
             ("Training curves", "training-curves.html", 760),
             ("Training reward decomposition", "reward-decomposition.html", 980),
             ("Final execution decomposition", "execution-decomposition.html", 860)]
    links = "".join(f'<li><a href="{page}">{escape(title)}</a></li>' for title, page, _ in pages)
    frames = "".join(f'<section><h2>{escape(title)}</h2><iframe title="{escape(title)}" '
                     f'style="height:{height}px" src="{page}"></iframe></section>'
                     for title, page, height in pages)
    summary = escape(_interval_text(primary.get("economic_interval")))
    fill = primary.get("mean_fill_fraction")
    completion = (f"The primary policy actually filled {100 * fill:.2f}% of the parent order on average; "
                  f"the remaining {100 * (1 - fill):.2f}% is hypothetically valued at visible-book liquidation cost."
                  if fill is not None else "Actual fill fraction is unavailable; consult the outcome table.")
    html = ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>CleoLOB sealed execution study</title><style>"
            "body{font:16px/1.6 system-ui,sans-serif;color:#182d40;max-width:1400px;margin:32px auto;padding:0 24px}"
            "h1,h2{line-height:1.2}iframe{width:100%;border:1px solid #dce3e8;border-radius:8px}"
            "a{color:#286a94}.finding{font-size:22px;font-weight:600}section{margin-top:48px}</style>"
            f"<h1>PPO versus simulator-fitted AC</h1><p class=\"finding\">Primary: {summary}</p>"
            f"<p>{completion} Completion penalties are excluded from this economic endpoint.</p>"
            "<p>Negative differences favor PPO. This is a fixed-budget synthetic result. Real-data calibration "
            "failed its external fidelity gates; these results do not establish historical execution performance.</p>"
            "<p>The retained local computation repeats a fixed design. Earlier CI computations are not extra "
            "independent seeds or a fresh holdout. See the summary for execution chronology.</p>"
            f"<ul>{links}<li><a href=\"summary.md\">Study summary and limitations</a></li>"
            "<li><a href=\"report-data.json\">Derived report data</a></li></ul>"
            f"{frames}</html>")
    (out / "index.html").write_text(html, encoding="utf-8")


def render(study: Path, out: Path) -> None:
    study, out = study.resolve(), out.resolve()
    if out == study or out.is_relative_to(study):
        raise ValueError("Reports must be outside the sealed study directory")
    verified = verify_study(study)
    if not verified["valid"]:
        raise ValueError(f"Study verification failed: {verified}")
    result = json.loads((study / "result.json").read_text(encoding="utf-8"))
    plan = json.loads((study / "preregistration.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    arms = [arm for arm in result["penalty_ablations"] if arm.get("family_interval")]
    comparisons = [(f"PPO / {arm['penalty_bps']:g} bps vs AC", arm["family_interval"]) for arm in arms]
    secondary = result["ac_risk_sensitivity"].get("family_interval")
    if secondary:
        comparisons.append(("PPO / 0 bps vs risk-sensitive AC", secondary))
    figure = go.Figure()
    for index, (label, interval) in enumerate(comparisons):
        mean, low, high = [interval[key] for key in ("mean_delta_bps", "ci_low_bps", "ci_high_bps")]
        figure.add_trace(go.Scatter(x=[low, high], y=[index, index], mode="lines",
                                   line={"color": "#286a94", "width": 4}, showlegend=False,
                                   hovertemplate=f"CI [{low:+.3f}, {high:+.3f}]<extra></extra>"))
        figure.add_trace(go.Scatter(x=[mean], y=[index], mode="markers", showlegend=False,
                                   marker={"color": "#122d40", "size": 11},
                                   hovertemplate=f"Mean {mean:+.3f} bps<extra></extra>"))
    figure.add_vline(x=0, line_dash="dash", line_color="#9b4a39")
    confidence = 100 * comparisons[0][1]["confidence"] if comparisons else 0
    figure.update_layout(template="plotly_white", height=430,
                         title=f"Synthetic execution: {confidence:g}% crossed bootstrap intervals",
                         xaxis_title="PPO minus comparator net cost (bps); negative favors PPO",
                         yaxis={"tickvals": list(range(len(comparisons))),
                                "ticktext": [label for label, _ in comparisons]},
                         annotations=[{"text": "Conditional on the simulator. Consult the real-data fidelity gates.",
                                       "xref": "paper", "yref": "paper", "x": 0, "y": -0.28,
                                       "showarrow": False}], margin={"b": 105, "l": 245})
    figure.write_html(out / "penalty-comparison.html", include_plotlyjs=True, auto_open=False)
    curves = make_subplots(rows=2, cols=2, shared_xaxes=True,
                           subplot_titles=[f"Completion penalty {p:g} bps" for p in plan["penalties_bps"]])
    colors = ["#286a94", "#ca7444", "#47946b", "#8c65a8", "#655747"]
    for index, penalty in enumerate(plan["penalties_bps"]):
        for seed_index, seed in enumerate(plan["training_seeds"]):
            trace = json.loads((study / "models" / f"penalty-{penalty:g}-seed-{seed}.training.json")
                               .read_text(encoding="utf-8"))
            episodes = trace["episodes"]
            rewards = [episode["reward"] for episode in episodes]
            smooth = [float(np.mean(rewards[max(0, i - 19):i + 1])) for i in range(len(rewards))]
            curves.add_trace(go.Scatter(x=[e["step"] for e in episodes], y=smooth,
                                       name=str(seed), legendgroup=str(seed), showlegend=index == 0,
                                       mode="lines", line={"color": colors[seed_index], "width": 1.5}),
                             row=index // 2 + 1, col=index % 2 + 1)
    curves.update_xaxes(title_text="Training steps")
    curves.update_yaxes(title_text="20-episode mean return (bps)")
    curves.update_layout(template="plotly_white", height=740,
                         title="Training returns include completion penalties; curves do not prove convergence")
    curves.write_html(out / "training-curves.html", include_plotlyjs=True, auto_open=False)
    _extras(study, out, result, plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.study, args.out)
