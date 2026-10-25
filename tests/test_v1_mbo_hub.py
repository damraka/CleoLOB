"""v1 MBO hub (H15): exact queue position vs L2 bounds on a constructed event stream."""
import numpy as np
import pytest

from lob.v1.data.schema import EVENT_DTYPE, EventKind
from lob.v1.research.studies import mbo_hub as H

S = 10 ** 9


def stream():
    rows = []

    def put(t, kind, side=0, price=0, size=0.0, oid=-1):
        rows.append((t, t, -1, kind, side, price, size, oid, -1))
    put(0, EventKind.SNAPSHOT_BEGIN)
    put(0, EventKind.ORDER_ADD, 1, 100, 1.0, 1)
    put(0, EventKind.ORDER_ADD, 1, 100, 2.0, 2)
    put(0, EventKind.ORDER_ADD, -1, 101, 5.0, 3)
    put(0, EventKind.SNAPSHOT_END)
    put(1 * S, EventKind.ORDER_ADD, -1, 104, 1.0, 8)           # first event starts the sampling clock
    put(60 * S, EventKind.ORDER_ADD, -1, 102, 1.0, 9)          # triggers the sample at t = 60 s (queue ahead 3)
    put(61 * S, EventKind.ORDER_DELETE, 1, 100, 0.0, 2)        # cancel of 2 lots ahead
    put(62 * S, EventKind.TRADE, -1, 100, 0.5, 1)
    put(62 * S, EventKind.ORDER_MODIFY, 1, 100, 0.5, 1)        # 0.5 executed from order 1
    put(95 * S, EventKind.ORDER_ADD, -1, 103, 1.0, 10)         # after the 30 s horizon: evaluation
    return [np.array(rows, dtype=EVENT_DTYPE)]


def test_mbo_hub_l2_bounds_contain_exact_position():
    r = H.h15(stream(), every_s=60.0)
    assert r["samples"] == 2 and r["containment_rate"] == 1.0
    # bid: Q0 = 3, decreases 2.5 -> optimistic 0.5; prints 0.5 -> pessimistic min(level 0.5, 2.5) = 0.5; exact 0.5
    # ask: no change -> width 0; so H15 = mean(0, 0) = 0 here, exact = optimistic = pessimistic
    assert r["H15"] == pytest.approx(0.0)
    assert r["mean_exact_fraction"] == pytest.approx((0.5 / 3 + 1.0) / 2)
