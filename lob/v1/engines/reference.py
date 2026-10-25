"""Slow, obviously-correct reference matching engine for differential testing (M38, M40).

A flat list of resting orders; every command re-sorts the opposite side by (price priority, entry sequence) and
walks it. Same price-time semantics as ``MatchingEngine`` (price-time allocation only), written independently
without levels, indexes or incremental best tracking, so agreement between the two is evidence about both.
"""
from __future__ import annotations

from .matching import BUY, SELL, Order, Report
from .rules import Rules


class ReferenceEngine:
    def __init__(self, rules: Rules | None = None) -> None:
        self.rules = rules or Rules()
        if self.rules.allocation != "price_time":
            raise ValueError("the reference engine implements price-time allocation only")
        self.resting: list[dict] = []
        self.status: dict[int, list] = {}
        self.seq = 0

    def _opp_sorted(self, side: int) -> list[dict]:
        opp = [r for r in self.resting if r["side"] == -side]
        return sorted(opp, key=lambda r: (r["price"] if side == BUY else -r["price"], r["seq"]))

    def best(self, side: int):
        ps = [r["price"] for r in self.resting if r["side"] == side]
        if not ps:
            return None
        return max(ps) if side == BUY else min(ps)

    def _cross(self, side: int, price) -> bool:
        b = self.best(-side)
        return b is not None and (price is None or (price >= b if side == BUY else price <= b))

    def submit(self, o: Order, ts: int = 0) -> list[Report]:
        if o.oid in self.status:
            self.status[o.oid] = ["rejected", 0]
            return [Report("reject", o.oid, o.owner, ts, o.price, o.qty, reason="duplicate_order_id")]
        bad = (o.side not in (BUY, SELL) or not self.rules.valid_qty(o.qty) or (o.kind == "limit" and o.price is None)
               or (o.kind == "market" and (o.price is not None or o.post_only)) or o.tif not in ("GTC", "IOC", "FOK"))
        if bad:
            self.status[o.oid] = ["rejected", 0]
            return [Report("reject", o.oid, o.owner, ts, o.price, o.qty, reason="invalid")]
        price, rem = o.price, o.qty
        reps = [Report("ack", o.oid, o.owner, ts, price, o.qty, remaining=o.qty)]
        if o.post_only and self._cross(o.side, price):
            if self.rules.post_only == "reject":
                self.status[o.oid] = ["rejected", 0]
                return reps + [Report("reject", o.oid, o.owner, ts, price, o.qty, reason="post_only_would_cross")]
            b = self.best(-o.side)
            price = b - 1 if o.side == BUY else b + 1
            reps.append(Report("replace", o.oid, o.owner, ts, price, rem, reason="post_only_slide"))
        book = self._opp_sorted(o.side)
        ok = [r for r in book if price is None or (r["price"] <= price if o.side == BUY else r["price"] >= price)]
        if o.tif == "FOK":
            avail = 0
            for r in ok:
                if r["owner"] == o.owner and self.rules.self_trade == "cancel_taker":
                    break
                if r["owner"] == o.owner and self.rules.self_trade == "cancel_maker":
                    continue
                avail += r["rem"]
            if avail < rem:
                self.status[o.oid] = ["expired", 0]
                return reps + [Report("expire", o.oid, o.owner, ts, price, rem, reason="fok_not_fillable")]
        levels_seen = []
        for r in ok:
            if rem <= 0:
                break
            if r["price"] not in levels_seen:
                levels_seen.append(r["price"])
                if len(levels_seen) > self.rules.max_walk_levels:
                    break
            if r["owner"] == o.owner and self.rules.self_trade != "allow":
                if self.rules.self_trade == "cancel_taker":
                    reps.append(Report("cancel", o.oid, o.owner, ts, price, rem, reason="self_trade"))
                    self.status[o.oid] = ["cancelled", 0]
                    return reps
                reps.append(Report("cancel", r["oid"], r["owner"], ts, r["price"], r["rem"], reason="self_trade"))
                self.status[r["oid"]] = ["cancelled", 0]
                self.resting.remove(r)
                continue
            q = min(rem, r["rem"])
            r["rem"] -= q
            rem -= q
            reps.append(Report("fill", r["oid"], r["owner"], ts, r["price"], q, "maker", o.oid, r["rem"]))
            reps.append(Report("fill", o.oid, o.owner, ts, r["price"], q, "taker", r["oid"], rem))
            if r["rem"] == 0:
                self.status[r["oid"]] = ["filled", 0]
                self.resting.remove(r)
            else:
                self.status[r["oid"]] = ["live", r["rem"]]
        if rem > 0:
            if o.kind == "market" or o.tif in ("IOC", "FOK"):
                reps.append(Report("expire", o.oid, o.owner, ts, price, rem,
                                   reason="market_unfilled" if o.kind == "market" else "ioc_remainder"))
                self.status[o.oid] = ["expired", 0]
            else:
                self.seq += 1
                self.resting.append({"oid": o.oid, "owner": o.owner, "side": o.side, "price": price, "rem": rem,
                                     "seq": self.seq})
                self.status[o.oid] = ["live", rem]
        else:
            self.status[o.oid] = ["filled", 0]
        return reps

    def cancel(self, oid: int, ts: int = 0, owner: str | None = None) -> list[Report]:
        r = next((x for x in self.resting if x["oid"] == oid), None)
        if r is None:
            return [Report("reject", oid, owner or "?", ts, reason="too_late" if oid in self.status else "unknown")]
        self.resting.remove(r)
        self.status[oid] = ["cancelled", 0]
        return [Report("cancel", oid, r["owner"], ts, r["price"], r["rem"])]

    def depth(self, side: int, n: int = 10) -> list[tuple[int, int]]:
        levels: dict[int, int] = {}
        for r in self.resting:
            if r["side"] == side:
                levels[r["price"]] = levels.get(r["price"], 0) + r["rem"]
        return sorted(levels.items(), key=lambda kv: -kv[0] if side == BUY else kv[0])[:n]


__all__ = ["ReferenceEngine"]
