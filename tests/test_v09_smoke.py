"""v0.9 smoke and repository-level verification commands (generated data only)."""
from __future__ import annotations

from lob.v09.smoke_steps import run_smoke


def test_smoke_is_protocol_bound_and_verifies(tmp_path) -> None:
    out = run_smoke(tmp_path / "smoke", workers=2)
    assert out["valid"] and out["mode"] == "protocol_bound"
    assert all(out["verification"].values())
    assert set(out["steps"]) >= {"runner", "execution", "promotion", "decisions", "sbc", "substitution"}
