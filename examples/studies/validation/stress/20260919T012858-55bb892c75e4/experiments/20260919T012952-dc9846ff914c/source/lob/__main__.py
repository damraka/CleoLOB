"""Smoke test: 10 simulated seconds of background flow, then a liquidity shock."""
from __future__ import annotations

from .engine import ExchangeSimulator, Order, OrderType, Side, SimConfig

if __name__ == "__main__":
    sim = ExchangeSimulator(SimConfig())
    for _ in range(300):
        sim.step(0.1)
    ts = sim.cfg.tick_size
    bids, asks = sim.book.depth(5)
    print(f"[t={sim.t:.0f}s] trades={len(sim.book.trades)}  mid={sim.book.mid() * ts:.2f}  "
          f"spread={sim.book.spread()} tick(s)")
    print("  bids:", [(round(p * ts, 2), v) for p, v in bids])
    print("  asks:", [(round(p * ts, 2), v) for p, v in asks])

    # Liquidity shock: 3,000-share market sell, then watch resilience refill the bid side.
    mid_before = sim.book.mid()
    trades = sim.book.process(Order(10**9, Side.SELL, 3000, OrderType.MARKET, "DEMO", sim.t),
                              sim.t)
    vwap = sum(t.price * t.qty for t in trades) / sum(t.qty for t in trades)
    print(f"\n[shock] sold {sum(t.qty for t in trades):,} @ vwap {vwap * ts:.4f} "
          f"(mid was {mid_before * ts:.4f} -> slippage "
          f"{(mid_before - vwap) / mid_before * 1e4:.1f} bps)")
    for k in range(1, 4):
        sim.step(2.0)
        bids, _ = sim.book.depth(3)
        print(f"[t=+{2 * k}s] best bids after refill:",
              [(round(p * ts, 2), v) for p, v in bids])
