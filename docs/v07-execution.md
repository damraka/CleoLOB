# v0.7 exchange, replay, queues and execution

This document covers the simulator-side mechanics that execution results depend on.
Results appear in `docs/v07-paper.md`.

## Venue specification layer (workstream 3)

`lob.v07.exchange.venue.ExchangeRules` declares, for each venue:
- tick and lot grid, minimum size
- order types and time-in-force
- post-only behaviour (reject or slide)
- self-trade prevention
- fees

**Validation.** `validate()` refuses off-grid prices, sizes below the minimum or off the lot
grid, unsupported time-in-force, and crossing post-only orders.

**Declared, not verified.**
- Rules for Deribit and BitMEX come from public contract specifications. They are
  **declared and unverified** (`verified=False`). Only the simulator venue is exact.
- Fees default to the v0.6 mandate convention (maker 0 bps, taker 1 bps), not to any venue's
  fee schedule. No fee schedule is claimed.

## Deterministic replay, book hashes, checkpoints (workstream 4)

`lob.v07.replay.deterministic.BookReplay` applies canonical L2 records. After each
local-timestamp group it extends a SHA-256 chain over the top 10 levels.

**Checkpoints.**
- `checkpoint()` serializes the book, the record index and the chain head, with an
  integrity hash.
- `resume()` continues from a checkpoint and reproduces the uninterrupted chain exactly
  (tested).
- A tampered checkpoint is refused.

**Correctness.** The final book equals the reference `lob.replay.l2.L2Replay`
reconstruction. Crossed books are counted, never repaired.

## Latency and asynchronous actions (workstreams 51, 52)

The frozen v0.5 engine delays strategy actions by `base + Exp(jitter)`.
`lob.v07.exchange.latency.attach(sim, model)` swaps in another model per simulator instance.
Draws still come from the simulator's own latency stream, so runs remain seed-deterministic.

| Model | Definition |
|---|---|
| `engine_default` | 5 ms + Exp(5 ms) (unchanged v0.5 default) |
| `zero` | 0 |
| `slow_fixed` | 50 ms |
| `heavy_tail` | 2 ms + LogNormal(log 10 ms, 1.0) |
| `spiky` | 5 ms + Exp(5 ms), plus 500 ms with probability 2% |

**Delayed feed.** `DelayedFeed` gives an agent the book as of `now − delay`.

**Race accounting.** `race_report` classifies asynchronous outcomes from each order's status
history:
- filled
- cancelled
- cancels that lost the race to a fill
- still in flight
- resting

**Scope.** These are simulator semantics. Nothing here measures or claims real venue
latency, colocation or HFT behaviour.
