"""v0.7 study templates, leaderboard and reproducible uncertainty-aware figures."""
from __future__ import annotations

from pathlib import Path

from lob.v07.reports import figures, leaderboard, templates

ROOT = Path(__file__).resolve().parents[1]


def test_template_validates_and_catches_problems() -> None:
    t = templates.load_template(ROOT)
    assert templates.validate(t) == []
    bad = {**t, "statistical_family": {"kind": "confirmatory", "correction": "none"},
           "compute_budget": {"early_stopping": "when significant"}, "datasets": [{"role": "whatever"}]}
    assert len(templates.validate(bad)) == 3


def test_leaderboard_has_no_composite_score() -> None:
    board = leaderboard.leaderboard({"b": {"selection_objective": 2.0}, "a": {"selection_objective": 3.0}})
    assert [r["model"] for r in board["rows"]] == ["a", "b"]
    assert not any("score" in c or "rank" in c for c in board["columns"])


def test_figures_reproduce_from_sources(tmp_path) -> None:
    entry = figures.interval_plot(tmp_path / "h1", "H1", [{"label": "day1", "estimate": -0.2, "low": -0.4,
                                                           "high": 0.1, "status": "NOT_ESTABLISHED"}], unit="objective")
    figures.write_manifest(tmp_path, [entry])
    assert figures.check_manifest(tmp_path) == []
    assert (tmp_path / "h1.svg").read_text(encoding="utf-8").startswith("<svg")
    (tmp_path / "h1.csv").write_text("label,estimate\nx,1\n", encoding="utf-8")
    assert figures.check_manifest(tmp_path)
