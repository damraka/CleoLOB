"""Fresh-data eligibility and the historical consumption firewall (M19).

Rules (registered in ``configs/v1/protocol.json`` and re-stated here as code):

1. **Dataset consumption.** Every dataset accessed by v0.1-v0.9, as recorded in the v0.9 ledger (accesses plus its
   ``import_consumed`` entries for v0.5-v0.8), is consumed forever.
2. **Period consumption.** v0.9 saw market conditions across all its pools, which run to 2025-08-28 with archive
   probes to 2025-08-31. Every calendar date on or before ``CONSUMED_THROUGH`` is period-consumed **for every venue
   and instrument**. Same-date streams are dependent (audit Issue B), so a date seen through one stream is not fresh
   through another.
3. **Excluded dates.** Later dates on which earlier versions read any data: 2025-09-20 (v0.9 spot format day) and
   2026-07-01, 2026-08-01 and 2026-09-01 (v0.8 Deribit first-of-month samples).
4. **v1 single use.** Within v1, a dataset or date assigned to one role is never reassigned to another; a guarded
   (confirm, replicate, vault) date is used by exactly one pool, across all streams.

Consumed and period-consumed data may serve ``development`` and ``explore`` only, and are labelled as such.
"""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path

CONSUMED_THROUGH = "2025-08-31"
EXCLUDED_DATES = {
    "2025-09-20": "v0.9 Bybit spot BTCUSDT format day (development access)",
    "2026-07-01": "v0.8 Deribit BTC/ETH-PERPETUAL first-of-month sample",
    "2026-08-01": "v0.8 Deribit BTC/ETH-PERPETUAL first-of-month sample",
    "2026-09-01": "v0.8 Deribit BTC/ETH-PERPETUAL first-of-month sample",
}
GUARDED_ROLES = frozenset({"confirm", "replicate", "vault"})


def _date_of(dataset: str) -> str | None:
    tail = dataset[-10:]
    try:
        date.fromisoformat(tail)
        return tail
    except ValueError:
        return None


def inventory(root: Path) -> dict:
    """Every dataset and calendar date consumed before v1, from the committed v0.9 ledger (read-only)."""
    ledger = root / "configs/v09/consumption-ledger.jsonl"
    datasets: dict[str, dict] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        ds = e.get("dataset")
        if not ds:
            continue
        if e["event"] == "import_consumed":
            datasets.setdefault(ds, {"version": "v0.5-v0.8", "evidence": e["payload"].get("evidence")})
        elif e["event"] == "access":
            rec = datasets.setdefault(ds, {"version": "v0.9", "roles": set()})
            if isinstance(rec.get("roles"), set):
                rec["roles"].add(e.get("role"))
    for rec in datasets.values():
        if isinstance(rec.get("roles"), set):
            rec["roles"] = sorted(r for r in rec["roles"] if r)
    dates = sorted({d for d in (_date_of(k) for k in datasets) if d})
    return {"schema": "cleolob-v1-consumed-inventory-1", "source": "configs/v09/consumption-ledger.jsonl",
            "consumed_through": CONSUMED_THROUGH, "excluded_dates": EXCLUDED_DATES,
            "datasets": dict(sorted(datasets.items())), "dataset_count": len(datasets), "dates_with_reads": dates,
            "rule": "dates <= consumed_through are period-consumed for every venue; excluded dates likewise"}


def date_fresh(day: str) -> tuple[bool, str]:
    if day <= CONSUMED_THROUGH:
        return False, f"period-consumed (on or before {CONSUMED_THROUGH})"
    if day in EXCLUDED_DATES:
        return False, f"excluded date: {EXCLUDED_DATES[day]}"
    return True, "fresh"


def eligible(dataset: str, day: str, role: str, *, consumed: set[str], v1_dates: dict[str, str]) -> tuple[bool, str]:
    """May ``dataset`` (on ``day``) take ``role`` in v1? ``v1_dates`` maps guarded dates to the pool role using them."""
    if role in ("development", "explore", "consumed"):
        return True, "non-evidence use"
    if dataset in consumed:
        return False, "dataset consumed by an earlier version"
    ok, why = date_fresh(day)
    if not ok:
        return False, why
    if role in GUARDED_ROLES:
        prior = v1_dates.get(day)
        if prior is not None and prior != role:
            return False, f"date already assigned to v1 role {prior}"
    return True, "fresh"


def import_into_ledger(ws, inv: dict) -> int:
    """Record every pre-v1 consumed dataset and the period/date exclusions as ``import_consumed`` (idempotent)."""
    state = ws.state()
    n = 0
    for ds, rec in inv["datasets"].items():
        if ds in state.consumed:
            continue
        ws.append("import_consumed", dataset=ds, reason=f"consumed by {rec['version']}",
                  payload={"version": rec["version"], "evidence": "configs/v09/consumption-ledger.jsonl"})
        n += 1
    period = f"period:all-venues:<={inv['consumed_through']}"
    if period not in state.consumed:
        ws.append("import_consumed", dataset=period, reason="v0.9 saw market conditions through its pools",
                  payload={"version": "v0.9", "through": inv["consumed_through"]})
        n += 1
    for day, why in inv["excluded_dates"].items():
        key = f"date:all-venues:{day}"
        if key not in state.consumed:
            ws.append("import_consumed", dataset=key, reason=why, payload={"version": "v0.8/v0.9", "date": day})
            n += 1
    return n


__all__ = ["CONSUMED_THROUGH", "EXCLUDED_DATES", "GUARDED_ROLES", "inventory", "date_fresh", "eligible",
           "import_into_ledger"]
