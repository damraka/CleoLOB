"""M17: paper figures rendered deterministically from sealed result files.

Each figure is written as an SVG (with a per-mark ``<title>`` tooltip) and a CSV
table view of exactly the plotted numbers, so every figure is reproducible from
stored data and never hand-edited. Colours: sequential single-hue blue for
magnitudes, blue/red with a neutral grey midpoint for signed quantities, at most
three categorical slots for point identities (validated reference palette).
"""
from __future__ import annotations

import csv
from html import escape
import math
from pathlib import Path

SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS = "#2a78d6", "#f0efec", "#e34948"
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FONT = "font-family='system-ui, sans-serif'"


def _mix(a: str, b: str, t: float) -> str:
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


def sequential(value: float | None, low: float, high: float) -> str:
    if value is None or not math.isfinite(value):
        return "#ffffff"
    t = 0.0 if high <= low else min(1.0, max(0.0, (value - low) / (high - low)))
    position = t * (len(SEQUENTIAL) - 1)
    i = min(int(position), len(SEQUENTIAL) - 2)
    return _mix(SEQUENTIAL[i], SEQUENTIAL[i + 1], position - i)


def diverging(value: float | None, limit: float) -> str:
    if value is None or not math.isfinite(value):
        return "#ffffff"
    t = 0.0 if limit <= 0 else max(-1.0, min(1.0, value / limit))
    return _mix(DIVERGING_MID, DIVERGING_POS if t > 0 else DIVERGING_NEG, abs(t))


def _text(x: float, y: float, s: str, *, size: int = 11, anchor: str = "start", color: str = INK,
          rotate: float | None = None) -> str:
    transform = f" transform='rotate({rotate} {x} {y})'" if rotate is not None else ""
    return (f"<text x='{x:.1f}' y='{y:.1f}' {FONT} font-size='{size}' fill='{color}' "
            f"text-anchor='{anchor}'{transform}>{escape(s)}</text>")


def _write(path: Path, svg: list[str], width: float, height: float, rows: list[list], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(svg)
    path.with_suffix(".svg").write_text(
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width:.0f}' height='{height:.0f}' "
        f"viewBox='0 0 {width:.0f} {height:.0f}'><rect width='100%' height='100%' fill='{SURFACE}'/>\n{body}\n</svg>\n",
        encoding="utf-8", newline="\n")
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def heatmap(path: Path, title: str, rows: list[str], columns: list[str], values: list[list[float | None]], *,
            signed: bool = False, fmt: str = "{:.2f}", note: str = "") -> None:
    cell_w, cell_h, left, top = 64, 24, 200, 70 + 90
    width = left + cell_w * len(columns) + 20
    height = top + cell_h * len(rows) + 50
    finite = [v for row in values for v in row if v is not None and math.isfinite(v)]
    low, high = (min(finite), max(finite)) if finite else (0.0, 1.0)
    limit = max(abs(low), abs(high)) if finite else 1.0
    svg = [_text(16, 24, title, size=15), _text(16, 44, note, size=11, color=MUTED)]
    for j, column in enumerate(columns):
        svg.append(_text(left + cell_w * j + cell_w / 2, top - 8, column, size=10, color=MUTED, rotate=-45))
    table = []
    for i, row in enumerate(rows):
        y = top + cell_h * i
        svg.append(_text(left - 8, y + cell_h / 2 + 4, row, size=11, anchor="end", color=MUTED))
        for j, column in enumerate(columns):
            value = values[i][j]
            fill = diverging(value, limit) if signed else sequential(value, low, high)
            label = "n/a" if value is None or not math.isfinite(value) else fmt.format(value)
            x = left + cell_w * j
            svg.append(f"<rect x='{x + 1}' y='{y + 1}' width='{cell_w - 2}' height='{cell_h - 2}' rx='3' "
                       f"fill='{fill}' stroke='{GRID}'><title>{escape(row)} / {escape(column)}: {label}</title></rect>")
            dark = (not signed and value is not None and math.isfinite(value) and high > low
                    and (value - low) / (high - low) > 0.55)
            svg.append(_text(x + cell_w / 2, y + cell_h / 2 + 4, label, size=10, anchor="middle",
                             color="#ffffff" if dark else INK))
            table.append([row, column, None if label == "n/a" else value])
    _write(path, svg, width, height, table, ["row", "column", "value"])


def scatter(path: Path, title: str, points: list[dict], *, x_label: str, y_label: str, note: str = "",
            groups: tuple[str, ...] = ()) -> None:
    width, height, left, right, top, bottom = 640, 440, 70, 30, 70, 60
    xs = [p["x"] for p in points if p["x"] is not None]
    ys = [p["y"] for p in points if p["y"] is not None]
    if not xs or not ys:
        xs, ys = [0.0, 1.0], [0.0, 1.0]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad_x, pad_y = (x1 - x0) * 0.08 or 1.0, (y1 - y0) * 0.08 or 1.0
    x0, x1, y0, y1 = x0 - pad_x, x1 + pad_x, y0 - pad_y, y1 + pad_y

    def px(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * (width - left - right)

    def py(y: float) -> float:
        return height - bottom - (y - y0) / (y1 - y0) * (height - top - bottom)

    svg = [_text(16, 24, title, size=15), _text(16, 44, note, size=11, color=MUTED)]
    for k in range(5):
        gy = y0 + (y1 - y0) * k / 4
        svg.append(f"<line x1='{left}' x2='{width - right}' y1='{py(gy):.1f}' y2='{py(gy):.1f}' stroke='{GRID}'/>")
        svg.append(_text(left - 6, py(gy) + 4, f"{gy:.2f}", size=10, anchor="end", color=MUTED))
        gx = x0 + (x1 - x0) * k / 4
        svg.append(_text(px(gx), height - bottom + 16, f"{gx:.2f}", size=10, anchor="middle", color=MUTED))
    svg.append(_text((left + width - right) / 2, height - 16, x_label, anchor="middle", color=MUTED))
    svg.append(_text(18, (top + height - bottom) / 2, y_label, anchor="middle", color=MUTED, rotate=-90))
    table = []
    for p in points:
        if p["x"] is None or p["y"] is None:
            continue
        color = CATEGORICAL[groups.index(p.get("group"))] if p.get("group") in groups else CATEGORICAL[0]
        svg.append(f"<circle cx='{px(p['x']):.1f}' cy='{py(p['y']):.1f}' r='5' fill='{color}' stroke='{SURFACE}' "
                   f"stroke-width='2'><title>{escape(p['label'])}: ({p['x']:.3f}, {p['y']:.3f})</title></circle>")
        if p.get("annotate"):
            svg.append(_text(px(p["x"]) + 8, py(p["y"]) - 6, p["label"], size=10, color=MUTED))
        table.append([p["label"], p.get("group", ""), p["x"], p["y"]])
    for k, group in enumerate(groups):
        svg.append(f"<circle cx='{width - 160}' cy='{24 + 16 * k}' r='5' fill='{CATEGORICAL[k]}'/>")
        svg.append(_text(width - 150, 28 + 16 * k, group, size=11, color=MUTED))
    _write(path, svg, width, height, table, ["label", "group", "x", "y"])


def bars(path: Path, title: str, labels: list[str], values: list[float | None], *, unit: str, note: str = "") -> None:
    """Horizontal bars of one measure (single series: no legend)."""
    bar_h, left, top = 20, 260, 70
    width, height = 720, top + (bar_h + 6) * len(labels) + 40
    finite = [v for v in values if v is not None and math.isfinite(v)]
    high = max(finite) if finite else 1.0
    scale = (width - left - 80) / high if high > 0 else 1.0
    svg = [_text(16, 24, title, size=15), _text(16, 44, note, size=11, color=MUTED)]
    table = []
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + i * (bar_h + 6)
        svg.append(_text(left - 8, y + bar_h / 2 + 4, label, size=11, anchor="end", color=MUTED))
        if value is None or not math.isfinite(value):
            svg.append(_text(left + 4, y + bar_h / 2 + 4, "n/a", size=10, color=MUTED))
            table.append([label, None])
            continue
        w = max(1.0, value * scale)
        svg.append(f"<rect x='{left}' y='{y}' width='{w:.1f}' height='{bar_h}' rx='4' fill='{SEQUENTIAL[4]}'>"
                   f"<title>{escape(label)}: {value:.3f} {escape(unit)}</title></rect>")
        svg.append(_text(left + w + 6, y + bar_h / 2 + 4, f"{value:.2f}", size=10, color=INK))
        table.append([label, value])
    _write(path, svg, width, height, table, ["label", f"value ({unit})"])
