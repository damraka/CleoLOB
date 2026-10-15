"""v0.9 study aggregation helpers (pure functions; no data access)."""
from __future__ import annotations

import json

from lob.config import canonical_json
from lob.v09.studies import continual, fills

from .v09_fixtures import v09_root, v09_template  # noqa: F401


def test_continual_units_accept_json_string_day_keys():
    row = {"days": ["d0", "d1", "d2"],
           "methods": {m: {"forgetting": {0: 0.1 * i, 1: 0.2 * i}} for i, m in enumerate(("naive", "replay", "ewc"))}}
    # task results pass through canonical JSON in the runner: integer keys become strings
    roundtrip = json.loads(canonical_json(row))
    units = continual.units_of([roundtrip])
    assert list(units) == ["d0", "d1"]
    assert units["d1"] == {"naive": 0.0, "replay": 0.2, "ewc": 0.4}


def test_continual_units_order_is_chronological_beyond_ten_days():
    days = [f"d{k:02d}" for k in range(12)]
    row = {"days": days, "methods": {m: {"forgetting": {str(k): float(k) for k in range(11)}}
                                     for m in ("naive", "replay", "ewc")}}
    assert list(continual.units_of([row])) == days[:11]


def test_fill_scores_are_json_safe_when_a_tercile_is_not_estimable():
    import json
    import numpy as np
    from lob.config import canonical_json
    tercile = np.array([0, 0, 1, 1, 2, 2])
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    window = np.array([0, 0, 0, 0, 0, 0])                 # every probe in an even window: odd half is empty
    out = fills.score([0.5, np.nan, 0.4], [0.6, 0.5, np.nan], tercile, y, window)
    assert out["probes"] == 2 and out["odd"] is None and out["even"] == out["effect"]
    none = fills.score([np.nan] * 3, [0.5] * 3, tercile, y, window)
    assert none["effect"] is None and none["probes"] == 0
    assert fills._probs([0.2, np.nan, 0.3]) == [0.2, None, 0.3]
    json.loads(canonical_json({"a": out, "b": none}))       # strict JSON: no NaN anywhere


def test_bounds_confirm_branch_follows_the_design_not_the_namespace(v09_root):
    from types import SimpleNamespace
    import pytest
    from lob.v09.studies import bounds
    base = dict(root=str(v09_root), phase="confirm", namespace="explore", out="results/v09/dryrun/b", workers=1,
                seed=0, dataset=None, input=[])
    with pytest.raises(ValueError, match="requires its sealed design"):
        bounds.run(SimpleNamespace(**base, design=None))
    # an explore-namespace dry run with a design passes the branch check (and then needs that design to be sealed)
    with pytest.raises(Exception) as err:
        bounds.run(SimpleNamespace(**base, design="dryrun-bounds-missing"))
    assert "requires" not in str(err.value)


def test_transfer_cli_takes_no_phase_as_the_pipeline_invokes_it():
    import argparse
    from lob.v09 import cli
    p = argparse.ArgumentParser()
    cli.add_commands(p.add_subparsers(dest="command"))
    a = p.parse_args(["transfer-study-v09", "--namespace", "confirm", "--design", "confirm-bounds", "--out", "x",
                      "--input", "bounds=r", "--input", "late=s"])
    assert a.input == [("bounds", "r"), ("late", "s")] and not hasattr(a, "phase")
