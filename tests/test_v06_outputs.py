"""v0.6 figures (deterministic SVG/CSV), public export guard and local benchmarks."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from lob.v06 import export as ex
from lob.v06 import figures as fg
from lob.artifacts import verify_artifacts


def test_colour_scales() -> None:
    assert fg.sequential(0.0, 0.0, 1.0) == fg.SEQUENTIAL[0] and fg.sequential(1.0, 0.0, 1.0) == fg.SEQUENTIAL[-1]
    assert fg.sequential(None, 0, 1) == "#ffffff"
    assert fg.diverging(0.0, 1.0) == fg.DIVERGING_MID
    assert fg.diverging(1.0, 1.0) == fg.DIVERGING_POS and fg.diverging(-5.0, 1.0) == fg.DIVERGING_NEG


def test_figures_are_deterministic_and_carry_tables(tmp_path: Path) -> None:
    for name in ("a", "b"):
        fg.heatmap(tmp_path / name / "heat", "Family errors", ["selected", "control"], ["spread", "depth"],
                   [[0.5, None], [1.5, 2.0]], note="test")
        fg.scatter(tmp_path / name / "scatter", "Scatter", [{"label": "w1", "x": 0.1, "y": 0.2, "group": "member"},
                                                              {"label": "w2", "x": None, "y": 0.3}],
                   x_label="x", y_label="y", groups=("member",))
        fg.bars(tmp_path / name / "bars", "Bars", ["a", "b"], [1.0, None], unit="bps")
    for stem in ("heat", "scatter", "bars"):
        a = (tmp_path / "a" / f"{stem}.svg").read_bytes()
        assert a == (tmp_path / "b" / f"{stem}.svg").read_bytes()
        assert a.startswith(b"<svg") and b"<title>" in a
    rows = list(csv.reader((tmp_path / "a" / "heat.csv").open(encoding="utf-8")))
    assert rows[0] == ["row", "column", "value"] and ["selected", "depth", ""] in rows


def test_export_guard_and_bundle(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ex.check_text("x", "path C:\\Users\\someone\\data")
    with pytest.raises(ValueError):
        ex.check_text("x", "api_key=abc")
    ex.check_text("x", '{"alpha": 0.05, "status": "NOT_ESTABLISHED"}')
    run = tmp_path / "results/v06/m1/design"
    run.mkdir(parents=True)
    for name, text in (("result.json", '{"a": 1}'), ("binding.json", "{}"), ("config.json", "{}"),
                       ("provenance.json", "{}"), ("design.json", '{"restricted": [1, 2]}')):
        (run / name).write_text(text, encoding="utf-8")
    pilot = tmp_path / "results/v06/pilot/x"
    pilot.mkdir(parents=True)
    (pilot / "binding.json").write_text("{}", encoding="utf-8")
    (tmp_path / "configs/v06").mkdir(parents=True)
    (tmp_path / "configs/v06/protocol.json").write_text("{}", encoding="utf-8")
    manifest = ex.export(tmp_path / "bundle", root=tmp_path)
    assert "runs/m1/design/result.json" in manifest["included"]
    assert "registration/protocol.json" in manifest["included"]
    assert [e["path"] for e in manifest["excluded"]] == ["results/v06/m1/design/design.json"]
    assert not any("pilot" in p for p in manifest["included"])
    assert verify_artifacts(tmp_path / "bundle")["valid"]
    with pytest.raises(FileExistsError):
        ex.export(tmp_path / "bundle", root=tmp_path)
    (run / "result.json").write_text('{"p": "/home/user/x"}', encoding="utf-8")
    with pytest.raises(ValueError, match="refusing"):
        ex.export(tmp_path / "bundle2", root=tmp_path)
    assert json.loads((tmp_path / "bundle" / "export.json").read_text(encoding="utf-8"))["schema"]
