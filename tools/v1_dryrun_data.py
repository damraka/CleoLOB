"""Make consumed v0.9 explore days available to v1 engineering dry runs (never evidence).

The raw files v0.9 already downloaded (``data/v09/raw/<venue-segment>/<SYMBOL>/<day>/``) are hashed into the v1
content-addressed store under their original provider URLs and recorded as ``acquire`` entries of *consumed*
datasets. The firewall makes them readable only in the ``dev`` and ``explore`` namespaces.

    python tools/v1_dryrun_data.py --stream bybit-linear:BTCUSDT --stream bybit-linear:ETHUSDT --days 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lob.v1.data import store as S  # noqa: E402
from lob.v1.data.connectors import base as CB  # noqa: E402
from lob.v1.protocol.document import Workspace  # noqa: E402

V09_EXPLORE = ("2023-08-19", "2023-08-21", "2023-08-27", "2023-08-29", "2023-09-02", "2023-09-12", "2023-10-08",
               "2023-10-17")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--stream", action="append", required=True)
    p.add_argument("--days", type=int, default=8)
    a = p.parse_args(argv)
    ws = Workspace(a.root)
    raw = S.RawStore(a.root / "data/v1/raw")
    state = ws.state()
    done = []
    for stream in a.stream:
        venue_seg, sym = stream.split(":")
        conn_name, seg = venue_seg.split("-")
        conn = CB.get(conn_name)
        for day in V09_EXPLORE[: a.days]:
            ds = f"{conn_name}-{seg}-{sym.lower()}-{day}"
            if ds not in state.consumed:
                raise SystemExit(f"{ds} is not a consumed dataset")
            src_dir = a.root / "data/v09/raw" / venue_seg / sym / day
            files = {"l2": src_dir / "l2.zip", "trades": src_dir / "trades.csv.gz"}
            digests = {}
            for s in conn.sources(seg, sym, day):
                got = raw.put_file(files[s.name], s.url)
                digests[s.name] = got.sha256
            if ds not in state.acquired:
                ws.append("acquire", dataset=ds, reason="consumed v0.9 raw files for v1 engineering dry runs",
                          payload={"sha256": digests, "role": "consumed", "pool": "dryrun"})
            done.append(ds)
    print(json.dumps({"datasets": done}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
