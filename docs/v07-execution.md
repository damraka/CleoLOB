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

## Queue-position uncertainty (workstream 5)

`lob.v07.queue.models` covers hypothetical passive orders on aggregate L2. The order joins
the back of its level, and the level's later history is a sequence of three event types:
- prints at the price
- prints through the price
- displayed-size observations

A size decrease not explained by prints is a cancellation of unknown position. The five
models differ only in where those cancellations sit and what fills the order:

| Model | Cancellations | Fills from |
|---|---|---|
| `conservative` | irrelevant | prints strictly through the price only |
| `fifo_lower` | behind the order | prints at/through, after the queue ahead |
| `probabilistic` | share `F(a)` ahead (`a` = fraction of the level ahead) | prints at/through, after the queue ahead |
| `fifo_upper` | ahead of the order | prints at/through, after the queue ahead |
| `optimistic` | — (front of queue) | every print at or through the price |

**Ordering.** Filled quantity is ordered `conservative ≤ fifo_lower ≤ probabilistic ≤
fifo_upper ≤ optimistic` for any monotone `F`. A property test checks this on 200 random
level histories with random Beta-shaped `F`. Seeded draws (`fill_distribution`) give a fill
distribution inside the FIFO bounds.

**Assumptions.** Every model assumes no impact of the hypothetical order, no hidden liquidity
and price-time priority. Aggregate L2 establishes none of these, so the results are
`ASSUMPTION_DEPENDENT` bounds, never exact fills.

**Learned cancellation positions (genuine order-level data only).**
- `lob.v07.queue.learned` measures, on the Bitstamp captures (consumed in v0.5), where each
  cancelled order sat in its level: `u` = volume ahead / level volume.
- This requires tracked price-time priority: additions join the back, and size increases or
  reprices re-queue. The feed does not establish this, so the result is
  **ASSUMPTION_DEPENDENT**.
- Levels holding a single order are excluded, because their `u` is always 0.
- Run `results/v07/m3/queue`, retrospective and descriptive:

| Capture | Cancellations | Mean `u` [95% CI] | In front quintile | In back quintile | KS from uniform |
|---|---|---|---|---|---|
| development (900 s) | 13,039 | 0.073 [0.070, 0.077] | 87.3% | 2.1% | 0.87 |
| validation (1,800 s) | 16,284 | 0.081 [0.078, 0.084] | 85.5% | 2.1% | 0.85 |

The two captures agree closely (two-sample KS 0.021). Under the tracking assumption,
cancellations come overwhelmingly from the front of the queue, far from the pro-rata
(uniform) rule.

**Interpretation.** Front cancellations reduce the queue ahead of a newly joined order. The
learned `F` therefore moves `probabilistic` toward `fifo_upper`. This is one venue (spot
Bitstamp), two short consumed captures and one priority assumption. It is not evidence about
Deribit or BitMEX queues.

**Queue-model sensitivity (development day).**
- Setup: 400 hypothetical 14-lot passive buys at the best bid, 60 s lifetime, seed 70601.
- Mean fill fraction:

| Model | Mean fill fraction |
|---|---|
| conservative | 0.134 |
| fifo_lower | 0.152 |
| probabilistic, pro-rata | 0.173 |
| probabilistic, learned | 0.184 |
| fifo_upper | 0.184 |
| optimistic | 0.206 |

- The conservative-to-optimistic width is 0.072, about half the conservative value. Queue
  assumptions alone move passive fills by this much on this data. Historical transfer (M16)
  therefore reports both fill bounds and the queue sensitivity.
