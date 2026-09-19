"""Offline HTML evidence report, built only from recorded episode outcomes."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_report(run: Path, metadata: dict[str, Any], result: dict[str, Any],
                 episodes: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    def table(data: pd.DataFrame, columns: list[str] | None = None) -> str:
        if data.empty:
            return "<p>NOT RUN — no complete eligible comparison.</p>"
        if columns:
            data = data[[key for key in columns if key in data]]
        return data.to_html(index=False, escape=True, float_format=lambda x: f"{x:.5g}", na_rep="UNPRICED")

    body = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>CLEO research report</title>",
        "<style>body{font:16px system-ui;margin:40px auto;max-width:1200px;padding:0 24px;color:#182534;background:#f7f9fc}"
        "h1,h2{color:#183c59}table{border-collapse:collapse;background:white;width:100%;font-size:13px}"
        "th,td{padding:9px;border:1px solid #d8e2e9;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "section{overflow:auto;margin:24px 0}.badge{font-weight:bold;color:#8a4600}</style></head><body>",
        f"<h1>{html.escape(result['name'])}</h1><p>{html.escape(metadata['experiment_id'])}</p>",
        f"<p class='badge'>Status: {html.escape(result['status'])} · INSUFFICIENT EVIDENCE FOR ALPHA</p>",
        "<h2>Objective and methodology</h2><p>Compare execution costs for a frozen set of strategies on paired synthetic seeds. "
        "Lower economic implementation shortfall is cheaper. Fees apply to actual fills. Unfilled quantity is valued "
        "hypothetically only when visible depth is sufficient; it is never recorded as a real execution. The noncompletion "
        "reward penalty is reported separately from economic cost.</p>",
        "<p>The market is an uncalibrated Poisson order-flow model with FIFO, message latency and endogenous book feedback. "
        "Shared exogenous seeds do not imply identical realized books. Observations are current exchange state; market-data "
        "dissemination latency is not modeled. No training, independent historical validation or live-trading test occurs here.</p>",
        "<h2>Design coverage</h2><pre>" + html.escape(json.dumps(result["coverage"], indent=2)) + "</pre>",
        "<h2>Research guardrails</h2><pre>" + html.escape(json.dumps(result["guardrails"], indent=2)) + "</pre>",
        "<section><h2>Economic results</h2>" + table(summary) + "</section>",
        "<p>Means and paired differences use percentile bootstrap intervals over independent seeds. "
        "Intervals are marginal, not simultaneous, and a singleton interval is not evidence of precision. "
        "Sign tests exclude exact ties and test direction rather than the mean effect. "
        "Adjusted p-values cover all planned candidate-versus-reference tests in this run. "
        "No alpha verdict follows from statistical significance.</p>",
        "<section><h2>Paired comparisons</h2>" + table(paired) + "</section>",
        "<section><h2>Every planned episode, including failures</h2>" + table(episodes, [
            "agent", "seed", "status", "effective_bps", "gross_effective_bps", "fee_bps", "fill_frac",
            "unpriced_leftover_qty", "outstanding_qty", "invalid_reasons", "error"]) + "</section>",
        "<h2>Stress, OOD, ablations and sensitivity</h2><p>NOT RUN. Changing the frozen market configuration "
        "creates a separate experiment; this report does not assert robustness across regimes.</p>",
        "<h2>Conclusion and limitations</h2><p>" + html.escape(result["conclusion"]) + "</p><ul>",
        *["<li>" + html.escape(warning) + "</li>" for warning in result["warnings"]],
        "<li>Historical queue counterfactuals, impact calibration, feed latency, exchange auctions, leverage, "
        "and portfolio risk are not established by this study.</li></ul>",
        "<h2>Provenance</h2><pre>" + html.escape(json.dumps(metadata, indent=2)) + "</pre>",
        "<p>Resolved settings, source snapshot, per-episode order/fill/risk logs, and SHA-256 manifest accompany this report. "
        "Checksums detect accidental change; they are not cryptographic signatures.</p></body></html>",
    ]
    with (run / "report.html").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(body))
