# v0.5 M2 — bounded historical counterfactual execution

A hypothetical passive order in historical data never gets a binary fill label by default.
[`lob/fill_bounds.py`](../lob/fill_bounds.py) produces nested bounds under explicit assumption
sets:

```text
conservative_lower  <=  fifo_lower  <=  fifo_upper  <=  optimistic_upper
```

| Bound | Assumes | Uses |
|---|---|---|
| Conservative | price priority only | prints strictly *through* the order's price |
| Observable FIFO (L2) | price-time priority and no hidden liquidity | displayed queue ahead; non-trade decreases attributed behind (lower) or ahead (upper); net level changes can mask simultaneous additions and cancellations |
| Identity FIFO (order-level) | price-time priority | exact source identities ahead, with their executions and cancellations (a point value) |
| Optimistic | none about queue position | front of the level: every print at or through the price |

All sets assume the hypothetical order does not change anyone else's behaviour (no market
impact). The class vocabulary is frozen:

| Class | Meaning |
|---|---|
| `OBSERVED_FILL` | real source orders only |
| `GUARANTEED_FILL` | conservative lower bound = size |
| `POSSIBLE_FILL` | anything between the two guaranteed classes |
| `GUARANTEED_NON_FILL` | optimistic upper bound = 0 |
| `INDETERMINATE` | an invalid or stale book at submission, a capture gap, or right-censoring |
| `UNSUPPORTED` | a question the capability cannot express |

Aggregate L2 trackers refuse identity, FIFO and hidden-size capabilities. Their queue
evidence reports exact queue position as `UNSUPPORTED`.

## Frozen design

Sealed in ledger entry 20 (`configs/v05/m1-m2-design.json`) before any external-holdout
access:
- orders every 60 s, from 60 s after the first capture until 120 s before its end;
- both sides, joining the best price observed strictly before submission;
- sizes of 1,000 and 10,000 USD contracts (Deribit);
- lifetimes of 10 and 60 s;
- a capture gap over 10 s or a book stale over 5 s → INDETERMINATE.

That gives 11,496 orders per day.

## Aggregate-L2 results (registered runs, `results/v05/m2/*`)

| Period | Role | Guaranteed fill | Possible | Guaranteed non-fill | Indeterminate | Determinate (conservative) | Width, conservative→optimistic, given possible | FIFO width, given possible |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ETH 2020-04-01 | development (consumed) | 633 | 3,721 | 7,142 | 0 | 67.6% | 0.189 | 0.028 |
| ETH 2020-05-01 | selection (consumed) | 1,275 | 4,259 | 5,962 | 0 | 63.0% | 0.170 | 0.013 |
| ETH 2020-06-01 | internal (retrospective) | 1,205 | 3,903 | 6,380 | 8 | 66.0% | 0.170 | 0.020 |
| **ETH 2020-07-01** | **external holdout** | 750 | 3,492 | 7,254 | 0 | **69.6%** | 0.214 | 0.031 |
| **BTC 2020-07-01** | **cross-instrument holdout** | 1,125 | 4,725 | 5,646 | 0 | **58.9%** | 0.525 | 0.225 |

Widths are fractions of order size.

**Reading the results:**
- **Determinate outcomes are mostly non-fills.** Roughly a third to two fifths of
  hypothetical join-the-best orders remain genuinely undetermined. Most determinate outcomes
  are guaranteed non-fills, because no opposite print reached the price during the lifetime.
- **ETH uncertainty.** Among possible fills, the honest conservative-to-optimistic interval
  spans about 17–21% of order size on ETH. The observable-FIFO interval is only 1–3% wide,
  but that narrowing rests entirely on price-time priority and no-hidden-liquidity
  assumptions that aggregate L2 cannot establish.
- **BTC uncertainty.** On BTC-PERPETUAL the bounds are much wider (52% and 23%): deeper
  displayed queues relative to print sizes leave more room between the assumption sets.
- **Mean fill fractions.** The mean conservative fill fraction is 8–16% and the mean
  optimistic one 14–32%. These are ranges, never point estimates.

## Order-level evidence (Bitstamp)

With genuine identities, the identity-FIFO value is a point value (zero width). The
conservative and optimistic bounds do not assume FIFO, so they stay as wide as on L2:
identity alone does not narrow them. The registered validation outcome is **INVALID** (see
[v05-mbo.md](v05-mbo.md)); the numbers below are exploratory or development evidence.

| Capture | Hypothetical orders | Determinate (conservative) | Identity-FIFO classes (fill / possible / non-fill) |
|---|---:|---:|---|
| Development, 900 s | 104 | 60.6% | 14 / 12 / 78 |
| Validation, post-hoc exploratory, 1,800 s | 224 | 55.4% | 44 / 44 / 136 |

**Observed-order coverage test.** Genuine source orders that joined the back of the best
price were re-evaluated from their queue position at entry and the public flow at their
price.

| Result | Development | Post-hoc validation |
|---|---:|---:|
| Actual fills inside [conservative, optimistic] | 466 / 466 | 2,702 / 2,702 |
| Fills equal to the identity-FIFO value | all | all |
| Later identity executed while an earlier one rested | 0 | 0 |

On these captures, therefore:
- the bound assumptions were never falsified;
- price-time priority was never contradicted.

This is consistent with FIFO, not proof that the venue guarantees it.

## Limits

- **No market impact.** Real orders change others' behaviour.
- **Unobservable queue details.** Hidden liquidity and exact queue position remain
  unobservable in L2, and the bounds span them.
- **Timing.** Prints and book updates are ordered by provider capture time. Same-timestamp
  ordering is declared (prints first), not observed.
- **Scope.** One venue per capability level, with short windows. Consumed periods are
  labeled as such.
