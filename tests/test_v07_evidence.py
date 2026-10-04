"""v0.7 run evidence: normalized source hashing, write-once runs, binding and tamper detection."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from lob.v07.evidence import runs
from lob.v07.protocol import core as pr
from tests.test_v07_protocol import COPIED, ROOT


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    for name in COPIED + tuple(f"configs/v07/{p.name}" for p in (ROOT / "configs/v07").glob("*.json")):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, tmp_path / name)
    pr.initialize_ledger(tmp_path)
    return tmp_path


def test_normalized_hash_ignores_line_endings(tmp_path) -> None:
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_bytes(b"x = 1\ny = 2\n")
    b.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert runs.normalized_sha256(a) == runs.normalized_sha256(b)
    b.write_bytes(b"x = 1\r\ny = 3\r\n")
    assert runs.normalized_sha256(a) != runs.normalized_sha256(b)


def test_run_roundtrip_and_tamper_detection(root) -> None:
    out = runs.new_run(root / "results/v07/t")
    runs.write_json(out, "detail.json", {"k": [1, 2]})
    runs.finalize(out, analysis="t", dataset_ids=[], config={"a": 1}, result={"r": 1}, root=root)
    assert runs.verify_run(out, root=root)["valid"]
    with pytest.raises(FileExistsError):
        runs.new_run(out)
    (out / "result.json").write_text(json.dumps({"r": 2}) + "\n", encoding="utf-8")
    assert not runs.verify_run(out, root=root)["valid"]
    assert runs.verify_tree(root / "results/v07", root=root)["invalid"] == ["t"]


def test_finalize_refuses_source_change_during_run(root, monkeypatch) -> None:
    out = runs.new_run(root / "results/v07/s")
    original = runs.source_manifest

    def changed(r=ROOT):
        files = dict(original())
        files["lob/engine.py"] = "0" * 64
        return files
    monkeypatch.setattr(runs, "source_manifest", changed)
    with pytest.raises(ValueError, match="implementation changed"):
        runs.finalize(out, analysis="s", dataset_ids=[], config={}, result={}, root=root)


def test_provenance_records_normalized_hashing() -> None:
    prov = runs.provenance(ROOT)
    assert "CRLF" in prov["source_hashing"] and "lob/v07/evidence/runs.py" in prov["source_files"]
    assert not any(":" in k or k.startswith("/") for k in prov["source_files"])
