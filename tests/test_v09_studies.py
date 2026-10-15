"""v0.9 study aggregation helpers (pure functions; no data access)."""
from __future__ import annotations

import json

from lob.config import canonical_json
from lob.v09.studies import continual, fills


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
