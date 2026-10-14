"""v0.9 study aggregation helpers (pure functions; no data access)."""
from __future__ import annotations

import json

from lob.config import canonical_json
from lob.v09.studies import continual


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
