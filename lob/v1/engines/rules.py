"""Configurable matching-rule adapters (M31).

``Rules`` collects every venue-dependent choice the matching engine must make, so a simulation states its
mechanics explicitly instead of inheriting them silently:

* ``allocation`` - ``price_time`` (FIFO within a level) or ``pro_rata`` (size-proportional with FIFO for the
  integer remainder; ``pro_rata_min`` lots per order before the remainder pass);
* ``self_trade`` - ``cancel_taker`` (the incoming order's remaining quantity is cancelled when it would trade with
  its owner's resting order), ``cancel_maker`` (the resting order is cancelled and matching continues) or ``allow``;
* ``post_only`` - ``reject`` a post-only order that would cross, or ``slide`` it to one tick behind the opposite
  touch;
* ``lot`` / ``min_qty`` - quantities are integer lots and must be multiples of ``lot`` (in lots, usually 1);
* ``max_walk_levels`` - market-order price protection (levels a market order may sweep);
* ``modify_priority`` - when a replace keeps priority: ``decrease_only`` (size decrease at the same price) or
  ``never``.

``VENUES`` registers presets used in studies; each preset is a documented assumption, not a venue certification.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

ALLOCATIONS = ("price_time", "pro_rata")
SELF_TRADE = ("cancel_taker", "cancel_maker", "allow")
POST_ONLY = ("reject", "slide")
MODIFY_PRIORITY = ("decrease_only", "never")


@dataclass(frozen=True)
class Rules:
    allocation: str = "price_time"
    self_trade: str = "cancel_taker"
    post_only: str = "reject"
    lot: int = 1
    min_qty: int = 1
    max_walk_levels: int = 10_000
    modify_priority: str = "decrease_only"
    pro_rata_min: int = 0

    def __post_init__(self) -> None:
        for name, allowed in (("allocation", ALLOCATIONS), ("self_trade", SELF_TRADE), ("post_only", POST_ONLY),
                              ("modify_priority", MODIFY_PRIORITY)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        if self.lot < 1 or self.min_qty < 1 or self.max_walk_levels < 1 or self.pro_rata_min < 0:
            raise ValueError("lot, min_qty and max_walk_levels must be positive")

    def valid_qty(self, qty: int) -> bool:
        return qty >= self.min_qty and qty % self.lot == 0

    def to_dict(self) -> dict:
        return asdict(self)


VENUES = {
    "fifo_default": Rules(),
    "crypto_perp": Rules(allocation="price_time", self_trade="cancel_taker", post_only="reject"),
    "pro_rata_futures": Rules(allocation="pro_rata", self_trade="cancel_maker", pro_rata_min=1),
}

__all__ = ["Rules", "VENUES", "ALLOCATIONS", "SELF_TRADE", "POST_ONLY", "MODIFY_PRIORITY"]
