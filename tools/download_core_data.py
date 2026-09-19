"""Fetch only the six predeclared public inputs; never inspect their observations."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from datetime import datetime, timezone

from lob.data_download import download_tardis_sample


def fetch(item):
    role, date, symbol = item
    try:
        path = download_tardis_sample("deribit", symbol, date, "book_snapshot_5",
                                      "data/public", max_bytes=512 * 1024**2)
        provenance = json.loads(path.with_name(path.name + ".provenance.json").read_text())
        return dict(role=role, date=date, symbol=symbol, status="DOWNLOADED",
                    path=str(path), provenance=provenance)
    except Exception as exc:
        return dict(role=role, date=date, symbol=symbol, status="UNAVAILABLE",
                    error=f"{type(exc).__name__}: {exc}")


def main():
    plan = json.loads(Path("examples/studies/core/DATA_PLAN.json").read_text())
    tasks = [(role, date, symbol) for role in ("train", "validation", "holdout")
             for date in plan[role + "_dates"] for symbol in plan["symbols"]]
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(fetch, item) for item in tasks]):
            result = future.result()
            results.append(result)
            print(json.dumps({key: value for key, value in result.items() if key != "provenance"}), flush=True)
    out = Path("examples/studies/core/data_downloads.json")
    out.write_text(json.dumps(dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
                                  files=results), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
