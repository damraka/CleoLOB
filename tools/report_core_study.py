"""Render sealed outcomes as standalone Plotly HTML; never rerun policies."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lob.core_study import verify_study


def render(study: Path, out: Path) -> None:
    verified = verify_study(study)
    if not verified["valid"]:
        raise ValueError(f"Study verification failed: {verified}")
    result = json.loads((study / "result.json").read_text(encoding="utf-8"))
    plan = json.loads((study / "preregistration.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    arms = [arm for arm in result["penalty_ablations"] if arm.get("family_interval")]
    figure = go.Figure()
    for index, arm in enumerate(arms):
        interval = arm["family_interval"]
        mean, low, high = [interval[key] for key in ("mean_delta_bps", "ci_low_bps", "ci_high_bps")]
        figure.add_trace(go.Scatter(x=[low, high], y=[index, index], mode="lines",
                                   line={"color": "#286a94", "width": 4}, showlegend=False,
                                   hovertemplate=f"CI [{low:+.3f}, {high:+.3f}]<extra></extra>"))
        figure.add_trace(go.Scatter(x=[mean], y=[index], mode="markers", showlegend=False,
                                   marker={"color": "#122d40", "size": 11},
                                   hovertemplate=f"Mean {mean:+.3f} bps<extra></extra>"))
    figure.add_vline(x=0, line_dash="dash", line_color="#9b4a39")
    confidence = 100 * arms[0]["family_interval"]["confidence"] if arms else 0
    figure.update_layout(template="plotly_white", height=430,
                         title=f"Synthetic execution: {confidence:g}% crossed bootstrap intervals",
                         xaxis_title="PPO minus risk-neutral AC net cost (bps); negative favors PPO",
                         yaxis={"title": "Training completion penalty", "tickvals": list(range(len(arms))),
                                "ticktext": [f"{arm['penalty_bps']:g} bps" for arm in arms]},
                         annotations=[{"text": "Conditional on the simulator. Consult the real-data fidelity gates.",
                                       "xref": "paper", "yref": "paper", "x": 0, "y": -0.28,
                                       "showarrow": False}], margin={"b": 105})
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.study, args.out)
