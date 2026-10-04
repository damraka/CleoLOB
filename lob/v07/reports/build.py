"""Report generator v2 and read-only research explorer (workstreams 69, 88, 90).

``build(out)`` regenerates every report artifact from sealed runs and the ledger (reproduction tier 2):
result registry, hypothesis table, negative results, claim graph, figures (with CSV sources and manifest),
and a static, read-only HTML explorer. ``render_markdown`` returns the hypothesis, attempt, consumed-holdout
and negative-result tables as Markdown for the documentation; nothing is filtered by outcome.
"""
from __future__ import annotations

from html import escape
import json
from pathlib import Path

from ...experiments.registry import PROJECT_ROOT
from . import figures, registry


def _fmt(v, digits: int = 4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x, digits) for x in v) + "]"
    return str(v)


def render_markdown(table: list[dict], reg: dict) -> dict[str, str]:
    lines = ["| Hypothesis | Dataset | Model | Estimate | Interval | Margin | Alpha (adjusted) | Family | MDE | Status |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        dataset = ", ".join(r["dataset"]) if isinstance(r["dataset"], list) else r["dataset"]
        lines.append(f"| {r['hypothesis']} | {dataset} | {r['model']} | {_fmt(r['estimate'])} | {_fmt(r['interval'])} | "
                     f"{_fmt(r['margin'])} | {_fmt(r['alpha'])} | {r['family']} | {_fmt(r['power_or_mde'])} | "
                     f"**{r['status']}** |")
    members = ["| Hypothesis | Member | Estimate | Interval | Status |", "|---|---|---|---|---|"]
    for r in table:
        for m in r["members"]:
            label = m.get("member") or m.get("dataset") or ""
            est = m.get("estimate", m.get("mean", m.get("ratio", m.get("difference"))))
            ci = m.get("interval") or m.get("ci")
            members.append(f"| {r['hypothesis']} | {label} | {_fmt(est)} | {_fmt(ci)} | "
                           f"{m.get('status') or m.get('edge') or '—'} |")
    attempts = ["| Ledger index | Design | Outcome | Directory | Note |", "|---|---|---|---|---|"]
    for a in reg["attempts"]:
        attempts.append(f"| {a['ledger_index']} | {a['design']} | {a['outcome']} | {a.get('directory') or '—'} | "
                        f"{(a.get('note') or '').replace('|', '/')} |")
    consumed = "\n".join(f"- `{d}`" for d in reg["consumed"])
    fresh = "\n".join(f"- `{d}`" for d in reg["fresh_remaining"]) or "- none"
    return {"hypotheses": "\n".join(lines), "members": "\n".join(members), "attempts": "\n".join(attempts),
            "consumed": consumed, "fresh": fresh}


def explorer_html(reg: dict, table: list[dict]) -> str:
    rows = "".join(f"<tr><td>{escape(r['hypothesis'])}</td><td>{escape(r['status'])}</td>"
                   f"<td>{escape(_fmt(r['estimate']))}</td><td>{escape(_fmt(r['interval']))}</td>"
                   f"<td>{escape(str(r['run']))}</td></tr>" for r in table)
    runs = "".join(f"<tr><td>{escape(x['run'])}</td><td>{escape(x['analysis'])}</td><td>{'valid' if x['valid'] else 'INVALID'}"
                   f"</td><td>{escape(str(x['git_commit'])[:10])}</td></tr>" for x in reg["runs"])
    attempts = "".join(f"<tr><td>{a['ledger_index']}</td><td>{escape(a['design'])}</td><td>{escape(a['outcome'])}</td></tr>"
                       for a in reg["attempts"])
    return ("<!doctype html><html><head><meta charset='utf-8'><title>CleoLOB v0.7 explorer</title><style>"
            "body{font-family:system-ui,sans-serif;margin:24px;background:#fcfcfb;color:#0b0b0b}"
            "table{border-collapse:collapse;margin-bottom:24px}td,th{border:1px solid #e4e3df;padding:4px 8px;"
            "font-size:13px}</style></head><body><h1>CleoLOB v0.7 research explorer (read-only)</h1>"
            f"<p>Protocol {escape(str(reg['protocol_sha256']))[:16]}…, ledger head {escape(reg['ledger_head'][:16])}…, "
            f"{reg['ledger_entries']} entries. Static page generated from sealed files; it cannot modify evidence.</p>"
            "<h2>Hypotheses</h2><table><tr><th>ID</th><th>Status</th><th>Estimate</th><th>Interval</th><th>Run</th></tr>"
            f"{rows}</table><h2>Runs</h2><table><tr><th>Run</th><th>Analysis</th><th>Verification</th><th>Commit</th></tr>"
            f"{runs}</table><h2>Attempts (failed, aborted and others)</h2><table><tr><th>Index</th><th>Design</th>"
            f"<th>Outcome</th></tr>{attempts}</table></body></html>\n")


def build(out: str | Path, *, root: Path = PROJECT_ROOT, docs: list[Path] | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    reg = registry.build_registry(root)
    table = registry.hypothesis_table(root)
    docs = docs or sorted((root / "docs").glob("v07-*.md"))
    graph = registry.claim_graph(table, docs, root)
    negative = registry.negative_results(table, reg)
    for name, value in (("registry.json", reg), ("hypotheses.json", table), ("claim-graph.json", graph),
                        ("negative-results.json", negative)):
        (out / name).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    figure_dir = out / "figures"
    entries = []
    for row in table:
        items = []
        for m in row["members"]:
            est = m.get("estimate", m.get("mean", m.get("difference")))
            ci = m.get("interval") or m.get("ci")
            if isinstance(ci, list) and len(ci) == 2 and None not in ci and est is not None:
                items.append({"label": m.get("member") or m.get("dataset") or "", "estimate": est, "low": ci[0],
                              "high": ci[1], "status": m.get("status") or m.get("edge")})
        if row["interval"] and None not in row["interval"] and row["estimate"] is not None:
            items.append({"label": "summary", "estimate": row["estimate"], "low": row["interval"][0],
                          "high": row["interval"][1], "status": row["status"]})
        if items:
            runs = row["run"] if isinstance(row["run"], list) else [row["run"]]
            entries.append(figures.interval_plot(figure_dir / row["hypothesis"].lower(), f"{row['hypothesis']}: "
                                                 f"estimates with intervals ({row['status']})", items, unit="estimate",
                                                 margin=row["margin"] if isinstance(row["margin"], (int, float)) else None,
                                                 runs=runs))
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures.write_manifest(figure_dir, entries)
    (out / "explorer.html").write_text(explorer_html(reg, table), encoding="utf-8", newline="\n")
    markdown = render_markdown(table, reg)
    (out / "tables.md").write_text("\n\n".join(f"## {k}\n\n{v}" for k, v in markdown.items()) + "\n",
                                   encoding="utf-8", newline="\n")
    return {"registry": reg, "table": table, "graph": graph, "negative": negative, "figures": entries,
            "markdown": markdown}
