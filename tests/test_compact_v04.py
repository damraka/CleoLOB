import json
from types import SimpleNamespace

import pytest

from tools import compact_v04 as compact


@pytest.mark.parametrize("family", ["policy", "multiperiod", "mbo", "scaling", "calls", "smoke"])
def test_compact_export_rejects_historical_sources_before_copying(tmp_path, monkeypatch, family):
    # Byte integrity alone never grants redistribution permission.
    for name in ("verify_policy", "verify_study", "verify_evidence", "verify_artifacts"):
        monkeypatch.setattr(compact, name, lambda _: {"valid": True})
    documents = {
        "policy": ("preregistration.json", {"dataset": {"kind": "synthetic"}}),
        "multiperiod": ("plan.json", {"config": {"kind": "synthetic"}}),
        "mbo": ("result.json", {"evidence_kind": "synthetic"}),
        "scaling": ("datasets.json", [{"kind": "synthetic"}]),
        "calls": ("result.json", {"dataset": {"kind": "synthetic"}}),
        "smoke": ("result.json", {"dataset": "generated synthetic fixtures; no historical date"}),
    }
    paths = {}
    for name, (filename, content) in documents.items():
        path = tmp_path / name
        path.mkdir()
        text = json.dumps(content)
        if name == family:
            text = text.replace("synthetic", "historical")
        (path / filename).write_text(text)
        paths[name] = path
    output = tmp_path / "public-export"
    with pytest.raises(ValueError):
        compact.export(SimpleNamespace(**paths, out=output))
    assert not output.exists()
