"""Large-scale, reproducible evaluation of execution agents.

Every agent runs on every (scenario, seed) pair, so comparisons are paired by market.
Writes ``episodes.csv`` (raw), ``summary.csv``, ``paired.csv``, ``report.md``,
``report.html`` (distribution plots) and ``meta.json`` (reproducibility record).

Examples
--------
    python evaluate.py                                   # 100 seeds × all scenarios × all agents
    python evaluate.py --episodes 500 --scenarios calm,thin --agents ac,twap,pov,ppo
    python evaluate.py --episodes 20 --workers 1         # quick smoke run
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from lob.runner import AGENTS, load_policy_model, run_episode
from lob.scenarios import DESCRIPTIONS, SCENARIOS, scenario_params
from lob.stats import paired_vs_reference, summarize
from lob.experiments.registry import source_manifest

_MODEL: Optional[Any] = None          # per-process PPO model (loaded once per worker)


def _init_worker(model_path: str) -> None:
    global _MODEL
    _MODEL, _ = load_policy_model(model_path)


def _run_task(scenario: str, seed: int, agents: List[str], overrides: Dict[str, Any]
              ) -> List[Dict[str, Any]]:
    p = scenario_params(scenario, seed, **overrides, strict_model=True)
    rows = []
    for agent in agents:
        try:
            row = run_episode(agent, p, model=_MODEL if agent == "ppo" else None)
        except Exception as exc:
            row = {"agent": agent, "label": agent, "seed": seed, "status": "FAILED",
                   "error": f"{type(exc).__name__}: {exc}"}
        row["scenario"] = scenario
        rows.append(row)
    return rows


# ------------------------------------------------------------------ reporting
def _fmt(x: float, d: int = 2) -> str:
    return f"{x:+.{d}f}" if isinstance(x, float) else str(x)


def write_markdown(summary: pd.DataFrame, paired: pd.DataFrame, meta: Dict[str, Any],
                   out: Path) -> None:
    lines = ["# CLEO evaluation report", "",
             f"*{meta['timestamp']} · {meta['episodes']} seeds/scenario · "
             f"{len(meta['scenarios'])} scenarios · {len(meta['agents'])} agents · "
             f"model: `{meta['model_label']}`*", "",
             "**Effective implementation shortfall** vs arrival mid, in bps (positive = cost): "
             "unfilled inventory at the horizon is priced at the walk-the-book liquidation VWAP, "
             "so no agent can look cheap by not finishing. `filled-only` is the lenient view. "
             "CIs are 95% percentile bootstrap. Paired tables compare each agent with "
             f"`{meta['reference']}` on identical seeds; win rate = share of seeds where "
             "the agent was strictly cheaper; raw p from an exact two-sided sign test, "
             "with Holm correction across the complete scenario/candidate family. "
             "CIs are marginal rather than simultaneous. This synthetic study does not establish alpha.", "",
             f"**Run status: {meta.get('status', 'WARNING')}**. "
             "If any planned outcome is failed or unpriced, all comparative inference is withheld. "
             "Use `cleo evaluate --config ...` for the immutable registry workflow.", ""]
    for scenario in meta["scenarios"]:
        lines += [f"## {scenario} — {DESCRIPTIONS.get(scenario, '')}", "",
                  "| agent | n | mean bps | 95% CI | median | std | p5 | p95 | worst | fill | filled-only | children |",
                  "|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        s = summary[summary["scenario"] == scenario].sort_values("mean")
        for r in s.itertuples():
            lines.append(f"| {r.label} | {r.n} | {_fmt(r.mean)} | [{_fmt(r.ci_lo)}, {_fmt(r.ci_hi)}] "
                         f"| {_fmt(r.median)} | {r.std:.2f} | {_fmt(r.p5)} | {_fmt(r.p95)} "
                         f"| {_fmt(r.worst)} | {r.fill_frac:.0%} | {_fmt(r.filled_only_mean)} "
                         f"| {r.children:.0f} |")
        pr = paired[paired["scenario"] == scenario]
        if not pr.empty:
            lines += ["", f"Paired vs `{meta['reference']}` (Δ = agent − reference, bps):", "",
                      "| agent | Δ mean | 95% CI | Δ median | win rate | raw p | Holm p | sig. |",
                      "|---|---:|:---:|---:|---:|---:|---:|:---:|"]
            for r in pr.sort_values("delta_mean").itertuples():
                lines.append(f"| {r.label} | {_fmt(r.delta_mean)} | [{_fmt(r.delta_ci_lo)}, "
                             f"{_fmt(r.delta_ci_hi)}] | {_fmt(r.delta_median)} | {r.win_rate:.0%} "
                             f"| {r.p_sign:.3f} | {r.p_adjusted:.3f} | {'✓' if r.significant_5pct else '–'} |")
        lines.append("")
    lines += ["## Reproducibility", "", "```json", json.dumps(meta, indent=1), "```", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


def write_html(episodes: pd.DataFrame, paired: pd.DataFrame, meta: Dict[str, Any],
               out: Path) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover
        return
    parts = [f"<h1>CLEO evaluation — {meta['timestamp']}</h1>",
             f"<p>{meta['episodes']} seeds per scenario · model: {meta['model_label']}</p>"]
    for scenario in meta["scenarios"]:
        e = episodes[episodes["scenario"] == scenario]
        fig = go.Figure()
        for agent, g in e.groupby("agent", sort=False):
            fig.add_trace(go.Box(y=g["effective_bps"], name=g["label"].iloc[0],
                                 boxmean="sd", boxpoints="outliers"))
        fig.update_layout(title=f"{scenario}: effective implementation shortfall (bps)",
                          yaxis_title="bps (positive = cost)", height=420)
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        pr = paired[paired["scenario"] == scenario]
        if not pr.empty:
            fig2 = go.Figure(go.Bar(
                x=pr["label"], y=pr["delta_mean"],
                error_y=dict(type="data", symmetric=False,
                             array=pr["delta_ci_hi"] - pr["delta_mean"],
                             arrayminus=pr["delta_mean"] - pr["delta_ci_lo"])))
            fig2.update_layout(title=f"{scenario}: Δ vs {meta['reference']} (bps, 95% CI; "
                                     "negative = cheaper)", height=360)
            parts.append(fig2.to_html(full_html=False, include_plotlyjs=False))
    out.write_text("<html><head><meta charset='utf-8'><title>CLEO evaluation</title></head>"
                   "<body style='font-family:sans-serif;max-width:1100px;margin:auto'>"
                   + "".join(parts) + "</body></html>", encoding="utf-8")


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


# ------------------------------------------------------------------ main
def main(argv: Optional[List[str]] = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=100, help="seeds per scenario")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS), help="comma-separated")
    ap.add_argument("--agents", default=",".join(a for a in AGENTS if a != "ppo"), help="comma-separated; PPO requires a checkpoint")
    ap.add_argument("--reference", default="ac", help="agent used for paired comparisons")
    ap.add_argument("--seed0", type=int, default=1000, help="first seed")
    ap.add_argument("--model-path", default="models/ppo_lob")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    if args.episodes <= 0 or args.seed0 < 0 or args.workers <= 0:
        ap.error("episodes/workers must be positive and seed0 nonnegative")

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    if not scenarios or not agents or len(set(agents)) != len(agents) or len(set(scenarios)) != len(scenarios):
        ap.error("agents and scenarios must be nonempty lists without duplicates")
    for a in agents:
        if a not in AGENTS:
            ap.error(f"unknown agent '{a}' (choose from {', '.join(AGENTS)})")
    for s in scenarios:
        if s not in SCENARIOS:
            ap.error(f"unknown scenario '{s}' (choose from {', '.join(SCENARIOS)})")
    if args.reference not in agents:
        ap.error("--reference must be one of --agents")

    _, model_label = load_policy_model(args.model_path)
    if "ppo" in agents and "PPO" not in model_label:
        print(f"! no trained PPO at {args.model_path}; PPO episodes will be marked FAILED, never substituted")

    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    tasks = [(s, seed) for s in scenarios for seed in seeds]
    overrides = {"model_path": args.model_path}
    t_start = time.time()
    rows: List[Dict[str, Any]] = []
    print(f"evaluating {len(agents)} agents × {len(scenarios)} scenarios × {len(seeds)} seeds "
          f"= {len(tasks) * len(agents)} episodes on {args.workers} worker(s)")

    if args.workers <= 1:
        _init_worker(args.model_path)
        for i, (s, seed) in enumerate(tasks, 1):
            rows += _run_task(s, seed, agents, overrides)
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} markets · {time.time() - t_start:.0f}s")
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=(args.model_path,)) as ex:
            futs = [ex.submit(_run_task, s, seed, agents, overrides) for s, seed in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                rows += f.result()
                if i % 20 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} markets · {time.time() - t_start:.0f}s")

    episodes = pd.DataFrame(rows)
    episodes["scenario"] = pd.Categorical(episodes["scenario"], categories=scenarios, ordered=True)
    episodes = episodes.sort_values(["scenario", "seed", "agent"]).reset_index(drop=True)
    complete = episodes["status"].isin(["VALID", "WARNING"]).all()
    if complete:
        summary = summarize(episodes)
        paired = paired_vs_reference(episodes, reference=args.reference)
    else:
        summary = pd.DataFrame(columns=["scenario", "label", "n", "mean", "ci_lo", "ci_hi", "median", "worst", "fill_frac"])
        paired = pd.DataFrame(columns=["scenario"])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    out = Path(args.out) / stamp
    out.mkdir(parents=True, exist_ok=False)
    meta = {
        "timestamp": stamp, "episodes": args.episodes, "seed0": args.seed0,
        "scenarios": scenarios, "agents": agents, "reference": args.reference,
        "model_path": args.model_path, "model_label": model_label,
        "git_commit": _git_commit(), "python": platform.python_version(),
        "elapsed_s": round(time.time() - t_start, 1),
        "packages": {},
        "status": "WARNING" if complete else "INVALID",
        "resolved_scenarios": {s: scenario_params(s, args.seed0, strict_model=True, **overrides) for s in scenarios},
        "seeds": seeds, "source_manifest": source_manifest(),
        "inference": "complete paired family, Holm-adjusted" if complete else "WITHHELD — failed or unpriced planned outcomes",
    }
    for pkg in ("numpy", "pandas", "gymnasium", "stable_baselines3"):
        try:
            meta["packages"][pkg] = __import__(pkg).__version__
        except Exception:
            meta["packages"][pkg] = None

    episodes.to_csv(out / "episodes.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    paired.to_csv(out / "paired.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    write_markdown(summary, paired, meta, out / "report.md")
    if complete:
        write_html(episodes, paired, meta, out / "report.html")
    else:
        (out / "report.html").write_text("<h1>INVALID / INCOMPLETE comparison</h1><p>Statistical inference withheld. "
                                         "See episodes.csv for every failure and report.md for details.</p>", encoding="utf-8")

    print(f"\ndone in {meta['elapsed_s']}s → {out}\n")
    cols = ["scenario", "label", "n", "mean", "ci_lo", "ci_hi", "median", "worst", "fill_frac"]
    with pd.option_context("display.width", 140, "display.float_format", "{:+.2f}".format):
        print(summary[cols].to_string(index=False))
    if not paired.empty:
        print(f"\npaired vs {args.reference}:")
        pcols = ["scenario", "label", "delta_mean", "delta_ci_lo", "delta_ci_hi", "win_rate", "p_sign"]
        with pd.option_context("display.width", 140, "display.float_format", "{:+.3f}".format):
            print(paired[pcols].to_string(index=False))
    return out


if __name__ == "__main__":
    result_path = main()
    result_meta = json.loads((result_path / "meta.json").read_text(encoding="utf-8"))
    sys.exit(0 if result_meta["status"] in {"VALID", "WARNING"} else 1)
