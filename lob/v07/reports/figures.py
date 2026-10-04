"""Uncertainty-aware figures with reproducible sources (workstreams 65, 89).

Every figure is an SVG plus a CSV of exactly the plotted numbers and a manifest entry with the run ids, the
producing function and the SHA-256 of the CSV, so ``check_manifest`` confirms that stored inputs reproduce
each figure. Interval plots always draw the interval; a point is never shown alone when an interval exists.
"""
from __future__ import annotations

import csv
import hashlib
from html import escape
import io
import json
from pathlib import Path

from ...v06.figures import FONT, GRID, INK, MUTED, SURFACE

BLUE = "#2a78d6"


def _csv(rows: list[list], header: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def interval_plot(path: Path, title: str, items: list[dict], *, unit: str, zero: float = 0.0,
                  margin: float | None = None, runs: list[str] | None = None) -> dict:
    """items: {label, estimate, low, high, status}. Dot plus interval per row, dashed reference at ``zero``."""
    width, row_h, left = 760, 26, 300
    height = 70 + row_h * max(1, len(items))
    values = [v for i in items for v in (i.get("low"), i.get("high"), i.get("estimate")) if v is not None] + [zero]
    if margin is not None:
        values += [-margin, margin]
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.1 or 1.0
    lo, hi = lo - pad, hi + pad

    def x(v: float) -> float:
        return left + (v - lo) / (hi - lo) * (width - left - 30)
    svg = [f"<rect width='{width}' height='{height}' fill='{SURFACE}'/>",
           f"<text x='16' y='26' {FONT} font-size='15' fill='{INK}'>{escape(title)}</text>",
           f"<line x1='{x(zero):.1f}' y1='40' x2='{x(zero):.1f}' y2='{height - 20}' stroke='{MUTED}' "
           f"stroke-dasharray='3 3'/>"]
    if margin is not None:
        for m in (-margin, margin):
            svg.append(f"<line x1='{x(m):.1f}' y1='40' x2='{x(m):.1f}' y2='{height - 20}' stroke='{GRID}'/>")
    rows = []
    for k, item in enumerate(items):
        y = 55 + k * row_h
        label = f"{item['label']} ({item.get('status', '')})"
        svg.append(f"<text x='16' y='{y + 4}' {FONT} font-size='11' fill='{INK}'>{escape(label)}</text>")
        if item.get("low") is not None and item.get("high") is not None:
            svg.append(f"<line x1='{x(item['low']):.1f}' y1='{y}' x2='{x(item['high']):.1f}' y2='{y}' "
                       f"stroke='{BLUE}' stroke-width='2'><title>{item['low']:.4g} to {item['high']:.4g} "
                       f"{escape(unit)}</title></line>")
        if item.get("estimate") is not None:
            svg.append(f"<circle cx='{x(item['estimate']):.1f}' cy='{y}' r='4' fill='{BLUE}'><title>"
                       f"{item['estimate']:.4g} {escape(unit)}</title></circle>")
        rows.append([item["label"], item.get("estimate"), item.get("low"), item.get("high"), item.get("status")])
    text = (f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
            f"viewBox='0 0 {width} {height}'>" + "".join(svg) + "</svg>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".svg").write_text(text, encoding="utf-8", newline="\n")
    table = _csv(rows, ["label", "estimate", "low", "high", "status"])
    path.with_suffix(".csv").write_text(table, encoding="utf-8", newline="\n")
    return {"figure": path.with_suffix(".svg").name, "source": path.with_suffix(".csv").name,
            "source_sha256": hashlib.sha256(table.encode()).hexdigest(), "rows": len(rows), "unit": unit,
            "runs": runs or [], "script": "lob.v07.reports.figures.interval_plot"}


def write_manifest(directory: Path, entries: list[dict]) -> Path:
    path = directory / "figures-manifest.json"
    path.write_text(json.dumps({"figures": entries}, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def check_manifest(directory: Path) -> list[str]:
    issues = []
    manifest = json.loads((directory / "figures-manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["figures"]:
        text = (directory / entry["source"]).read_text(encoding="utf-8").replace("\r\n", "\n")
        if hashlib.sha256(text.encode()).hexdigest() != entry["source_sha256"]:
            issues.append(f"{entry['figure']}: source table changed")
        if len(list(csv.reader(io.StringIO(text)))) - 1 != entry["rows"]:
            issues.append(f"{entry['figure']}: row count differs")
    return issues
