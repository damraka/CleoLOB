"""Dependency-free SVG figures with their data beside them (M35).

Every figure is generated from stored evidence only: ``<name>.svg`` plus ``<name>.json`` holding the plotted data,
the source run and the JSON pointer the data came from (figure provenance). Colours come from a fixed palette;
text uses the system sans-serif stack. No plotting library is required.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

PALETTE = ("#2f6db5", "#d1782c", "#3f9a5b", "#b8434c", "#7a5ab5", "#8c6d46", "#c25aa0", "#6b7280")
W, H, PAD = 720, 380, 56


def _esc(s) -> str:
    return html.escape(str(s))


def _scale(values, lo=None, hi=None):
    v = [x for x in values if x is not None and math.isfinite(x)]
    lo = min(v) if lo is None and v else (lo if lo is not None else 0.0)
    hi = max(v) if hi is None and v else (hi if hi is not None else 1.0)
    if hi == lo:
        hi, lo = hi + 1.0, lo - 1.0
    return lo, hi


def _frame(title: str, xlabel: str, ylabel: str, body: str, ylo: float, yhi: float) -> str:
    ticks = "".join(
        f'<line x1="{PAD}" x2="{W - 20}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        f'<text x="{PAD - 6}" y="{y + 4:.1f}" font-size="11" text-anchor="end" fill="#374151">{v:.3g}</text>'
        for v, y in ((ylo + (yhi - ylo) * k / 4, H - PAD - (H - 2 * PAD) * k / 4) for k in range(5)))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'font-family="system-ui, -apple-system, Segoe UI, sans-serif" role="img" aria-label="{_esc(title)}">'
            f'<rect width="{W}" height="{H}" fill="#ffffff"/>'
            f'<text x="{W / 2}" y="22" text-anchor="middle" font-size="15" fill="#111827">{_esc(title)}</text>{ticks}'
            f'<line x1="{PAD}" y1="{H - PAD}" x2="{W - 20}" y2="{H - PAD}" stroke="#6b7280"/>'
            f'<line x1="{PAD}" y1="30" x2="{PAD}" y2="{H - PAD}" stroke="#6b7280"/>'
            f'<text x="{W / 2}" y="{H - 10}" text-anchor="middle" font-size="12" fill="#374151">{_esc(xlabel)}</text>'
            f'<text x="16" y="{H / 2}" text-anchor="middle" font-size="12" fill="#374151" '
            f'transform="rotate(-90 16 {H / 2})">{_esc(ylabel)}</text>{body}</svg>\n')


def _y(v, lo, hi):
    return H - PAD - (H - 2 * PAD) * (v - lo) / (hi - lo)


def _write(out: Path, name: str, svg: str, data: dict, provenance: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.svg").write_text(svg, encoding="utf-8", newline="\n")
    (out / f"{name}.json").write_text(json.dumps({"data": data, "provenance": provenance}, indent=1, default=str) + "\n",
                                      encoding="utf-8", newline="\n")
    return {"figure": f"{name}.svg", "data": f"{name}.json", **provenance}


def bar(out: Path, name: str, labels: list, values: list, *, title: str, ylabel: str, errors: list | None = None,
        provenance: dict, xlabel: str = "") -> dict:
    lo, hi = _scale([*(v for v in values if v is not None)] + [0.0] +
                    [v + e[1] for v, e in zip(values, errors or []) if v is not None and e] +
                    [v - e[0] for v, e in zip(values, errors or []) if v is not None and e])
    n = max(len(labels), 1)
    width = (W - PAD - 30) / n
    body = ""
    for i, (lab, v) in enumerate(zip(labels, values)):
        x = PAD + 5 + i * width
        if v is None or not math.isfinite(v):
            body += f'<text x="{x + width / 2:.1f}" y="{H - PAD - 6}" font-size="10" text-anchor="middle">n/a</text>'
        else:
            y0, y1 = _y(0.0, lo, hi), _y(v, lo, hi)
            body += (f'<rect x="{x:.1f}" y="{min(y0, y1):.1f}" width="{width * 0.7:.1f}" height="{abs(y1 - y0):.1f}" '
                     f'fill="{PALETTE[i % len(PALETTE)]}"><title>{_esc(lab)}: {v:.4g}</title></rect>')
            if errors and errors[i]:
                e0, e1 = errors[i]
                cx = x + width * 0.35
                body += (f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{_y(v - e0, lo, hi):.1f}" y2="{_y(v + e1, lo, hi):.1f}" '
                         f'stroke="#111827"/>')
        body += (f'<text x="{x + width * 0.35:.1f}" y="{H - PAD + 14}" font-size="10" text-anchor="end" '
                 f'transform="rotate(-35 {x + width * 0.35:.1f} {H - PAD + 14})">{_esc(lab)}</text>')
    data = {"labels": labels, "values": values, "errors": errors}
    return _write(out, name, _frame(title, xlabel, ylabel, body, lo, hi), data, provenance)


def lines(out: Path, name: str, series: dict, *, title: str, xlabel: str, ylabel: str, provenance: dict) -> dict:
    xs = [x for s in series.values() for x in s[0]]
    ys = [y for s in series.values() for y in s[1] if y is not None and math.isfinite(y)]
    xlo, xhi = _scale(xs)
    lo, hi = _scale(ys)
    body = ""
    for i, (label, (x, y)) in enumerate(series.items()):
        pts = " ".join(f"{PAD + (W - PAD - 20) * (a - xlo) / (xhi - xlo):.1f},{_y(b, lo, hi):.1f}"
                       for a, b in zip(x, y) if b is not None and math.isfinite(b))
        color = PALETTE[i % len(PALETTE)]
        body += f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>'
        body += f'<text x="{W - 24}" y="{40 + 14 * i}" font-size="11" text-anchor="end" fill="{color}">{_esc(label)}</text>'
    return _write(out, name, _frame(title, xlabel, ylabel, body, lo, hi),
                  {k: {"x": list(v[0]), "y": list(v[1])} for k, v in series.items()}, provenance)


def scatter(out: Path, name: str, x: list, y: list, *, title: str, xlabel: str, ylabel: str, provenance: dict,
            labels: list | None = None) -> dict:
    xlo, xhi = _scale(x)
    lo, hi = _scale(y)
    body = ""
    for i, (a, b) in enumerate(zip(x, y)):
        if a is None or b is None:
            continue
        cx, cy = PAD + (W - PAD - 20) * (a - xlo) / (xhi - xlo), _y(b, lo, hi)
        tip = f"{labels[i]}: " if labels else ""
        body += (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{PALETTE[0]}" fill-opacity="0.7">'
                 f'<title>{_esc(tip)}({a:.3g}, {b:.3g})</title></circle>')
    for k in range(5):
        v = xlo + (xhi - xlo) * k / 4
        body += (f'<text x="{PAD + (W - PAD - 20) * k / 4:.1f}" y="{H - PAD + 14}" font-size="11" text-anchor="middle">'
                 f'{v:.3g}</text>')
    return _write(out, name, _frame(title, xlabel, ylabel, body, lo, hi), {"x": x, "y": y, "labels": labels}, provenance)


def heatmap(out: Path, name: str, matrix: list, rows: list, cols: list, *, title: str, provenance: dict) -> dict:
    vals = [v for r in matrix for v in r if v is not None]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    n, m = max(len(rows), 1), max(len(cols), 1)
    cw, ch = (W - 2 * PAD - 60) / m, (H - 2 * PAD) / n
    body = ""
    for i, r in enumerate(matrix):
        for j, v in enumerate(r):
            t = 0.5 if v is None or hi == lo else (v - lo) / (hi - lo)
            color = f"rgb({int(240 - 190 * t)},{int(244 - 130 * t)},{int(250 - 60 * t)})"
            x, y = PAD + 60 + j * cw, PAD + i * ch
            body += (f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw:.1f}" height="{ch:.1f}" fill="{color}" stroke="#ffffff"/>'
                     f'<text x="{x + cw / 2:.1f}" y="{y + ch / 2 + 4:.1f}" font-size="11" text-anchor="middle">'
                     f'{"n/a" if v is None else f"{v:.2f}"}</text>')
        body += f'<text x="{PAD + 56}" y="{PAD + i * ch + ch / 2 + 4:.1f}" font-size="11" text-anchor="end">{_esc(rows[i])}</text>'
    for j, c in enumerate(cols):
        body += f'<text x="{PAD + 60 + j * cw + cw / 2:.1f}" y="{PAD - 6}" font-size="11" text-anchor="middle">{_esc(c)}</text>'
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="system-ui, -apple-system, Segoe UI, sans-serif" role="img" aria-label="{_esc(title)}">'
           f'<rect width="{W}" height="{H}" fill="#ffffff"/><text x="{W / 2}" y="22" text-anchor="middle" font-size="15">'
           f'{_esc(title)}</text>{body}</svg>\n')
    return _write(out, name, svg, {"matrix": matrix, "rows": rows, "cols": cols}, provenance)
