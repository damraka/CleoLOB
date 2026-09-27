# v0.5 M1 — genuine order-level (MBO) validation

## Status

| Evidence | Status |
|---|---|
| **Frozen M1 validation gate** on the registered Bitstamp validation capture | **INVALID**: the frozen adapter could not process the capture |
| Genuine *historical vendor* MBO archive (e.g. LOBSTER, Databento) | **NOT_AVAILABLE**: access requires terms acceptance or credentials |
| Development capture (900 s) under the final development adapter | development evidence only; gate would be **FAILED** (E1 = 0.887) |
| Post-hoc exploratory analysis of the validation capture | **EXPLORATORY_POST_HOC**; would be **FAILED** under the frozen gate (E1 = 0.867) |
| Synthetic MBO fixtures (v0.4 and v0.5 tests) | synthetic mechanics only; never historical evidence |

No gate outcome was upgraded, and no capture was relabeled.

## What was done

1. **Freeze.** The M0 protocol (`9de49d3`) registered two future Bitstamp btcusd
   captures as fresh:
   - a 900-second development capture for adapter work;
   - a 1,800-second validation capture, to be recorded only after the adapter was frozen,
     with one retry allowed.
2. **Development.** The adapter rules were derived from the development capture and a
   20-second tool smoke capture (see [v05-data.md](v05-data.md)).
3. **Seal.** The design `configs/v05/m1-m2-design.json`, adapter version
   `bitstamp-capture-4`, the lifecycle semantics and the implementation hashes were sealed
   in ledger entry 20 before the validation capture started.
4. **Validation attempt 1.** It ended with a **GAP** after 1,148 s (`ConnectionClosedError`)
   and is retained unevaluated (`data/v05/bitstamp/validation-1800s`, SHA-256 `7eb524b1…`).
5. **Retry.** The single permitted retry completed all 1,800 s (SHA-256 `e9e39a62…`). It
   was run exactly once under the unchanged frozen implementation, whose hashes were
   re-verified against the seal.
6. **Frozen outcome: INVALID.** The capture contains venue message types never seen during
   development:
   - `order_subtype` 1 or 2 creations at a sentinel price of 999,999,999.00, some with zero amount;
   - an `order_subtype` 7 creation with zero price and amount.

   The frozen adapter correctly refused to guess, so the replay could not proceed. Outcome
   counts beyond the failure are unknown. The sealed record is in
   `results/v05/m1/validation-retry1`.

The main negative finding: adapter rules derived from a 15-minute development window
did not cover the venue's message vocabulary seen over the following 30 minutes.

## Post-hoc exploratory analysis (not the registered gate)

One adapter rule was added *after* the frozen outcome, as `bitstamp-capture-5-posthoc`.
Creations with zero amount, zero price or a sentinel price are treated as non-resting
instant orders. The ledger forbids amendments after access, so these results are only
exploratory (`results/v05/m1/validation-retry1-posthoc`):

| Quantity | Development (900 s) | Validation, post-hoc (1,800 s) |
|---|---:|---:|
| Order-channel messages | 93,514 | 271,290 |
| `event_id` chain links / breaks | 93,513 / 0 | 271,289 / 0 |
| Normalized events | 93,157 | 271,607 |
| Unexplained lifecycle anomalies | 0 | 0 |
| Deterministic replay | yes | yes |
| Periodic REST census checks: exact order matches | 8,103/8,103; 8,102/8,102; 8,083/8,083 (one extra order each) | 6 checks, each within one order |
| Census adjacent priority pairs concordant / discordant | 5,780 / 0 | 11,533 / 1 |
| Trade prints linked to a maker execution at the same key | 186 / 191 | 539 / 583 |
| **E1: exact top-10 agreement with `order_book` references** | **0.887** (5,554 / 6,262) | **0.867** (13,512 / 15,585) |
| Secondary: agreement with any state within the prior 100 ms | 0.950 | 0.958 |
| Frozen gate result (for comparison only) | FAILED | FAILED |

**Reading the evidence:**
- **Replayed order state.** The replay agrees with every complete REST census to within one
  order, and census listing order agrees with the tracked price-time priority in all but one
  of 17,314 adjacent pairs.
- **E1 below 0.90.** The published `order_book` channel is a lagged snapshot. Mismatched
  references typically match a replay state 20–40 ms earlier, and order events carry
  millisecond timestamps.
- **What E1 measures.** E1 therefore tests alignment with the reference channel more than
  the correctness of the reconstructed state. It stays the frozen endpoint and it is not
  met. The secondary diagnostic cannot upgrade the result.

## Semantics validated or refuted

- **Identities:** source IDs are used unchanged; duplicate, unknown and after-completion
  events are counted.
- **Priority:** evidence is consistent with price-time priority. In the M2 observed-order
  test on the post-hoc validation stream, no later identity at a price executed while an
  earlier identity still rested (2,702 evaluable orders). The feed itself still does not
  *establish* FIFO, so `fifo_established` stays false and FIFO positions are refused.
- **Executions:**
  - `order_changed` always reported an execution;
  - `amount_traded` was inconsistent (sometimes cumulative, sometimes per event) and serves
    only as a cross-check;
  - deletions with zero remaining are executions;
  - aggressive creations are transient taker orders and never rest while crossing.
- **Censoring:**
  - the initial REST census left-censors every resting order;
  - boundary events in the census millisecond are resolved idempotently;
  - the file end right-censors live orders;
  - a chain break or disconnect is a GAP.

## Reproduce

```sh
python tools/capture_bitstamp.py --seconds 1800 --snapshot-every 300 --out data/v05/bitstamp/<new-dir>
python -m lob.mbo_study data/v05/bitstamp/<dir> --out results/v05/m1/<new-run> --dataset-id <registered id>
python -m lob.mbo_study data/v05/bitstamp/<dir> --out results/v05/m1/<new-run> --dataset-id <id> --posthoc
```

A new capture is a new dataset. It must be declared fresh in a protocol amendment before
it is recorded, and it can never replace the INVALID registered outcome above.
