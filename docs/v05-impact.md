# v0.5 M4 — market impact and resilience validation

**Question.** Do simulator responses to aggressive trades agree with history in direction,
magnitude and persistence at preregistered horizons?

**Answer.** Under the frozen gate (H4), agreement is **NOT_ESTABLISHED** at every horizon, on
every dataset, for both the M3-selected simulator and the v0.4 control.

## Frozen design

Defined in `configs/v05/m4-design.json` and sealed in ledger entry 37 before any holdout
access.

**Events and responses.**
- Events are aggressive trades on the shared 100 ms tape.
- Signed mid-price responses are measured at 0.1, 1, 10 and 60 s. Gate horizons are 1, 10
  and 60 s.
- Also recorded: spread, imbalance and opposite-depth responses, a volatility response, and
  recovery of opposite best-level depth to 90% (censored at 60 s).
- Conditioning variables are measured at the last sample at or before each event, so they
  never use post-event information.

**Conditioning.**
- Size and participation terciles, from development data only.
- Median splits for spread, trailing volatility, depth and activity.
- Imbalance alignment and direction.

**Simulators.**
- The M3 selected (combined) model and the v0.4 control, exactly as sealed.
- 10 seeds × 1,800 simulated seconds each.

**Statistics.**
- Historical means: 600 s block bootstrap. Simulated means: seed bootstrap.
- Bonferroni over 3 size buckets × 3 horizons × 4 datasets (α = 0.05/36).
- H4 per horizon requires, among evaluable size buckets:
  - at least two thirds agree in sign with a |log magnitude ratio| ≤ log 2;
  - at least half have Bonferroni-nonzero historical means.
- Buckets with fewer than 100 events are NOT_AVAILABLE.

## Results

In every table, "k/n" means k agreeing out of n evaluable size buckets.

| Dataset | Events | Selected (1 s / 10 s / 60 s) | Control (1 s / 10 s / 60 s) | H4 |
|---|---:|---|---|---|
| ETH Apr 2020 (development) | 8,483 | 1/2, 1/2, 1/2 | 0/2, 0/2, 0/2 | NOT_ESTABLISHED |
| ETH Jun 2020 (retrospective) | 18,002 | 1/2, 1/2, 1/2 | 0/2, 0/2, 0/2 | NOT_ESTABLISHED |
| **ETH Jul 2020 (external)** | 11,231 | 1/2, 1/2, 1/2 | 0/2, 0/2, 0/2 | **NOT_ESTABLISHED** |
| BTC Jul 2020 (cross-instrument) | 20,217 | 1/2, 0/2, 1/2 | 1/2, 0/2, 0/2 | NOT_ESTABLISHED |

**Findings by size bucket:**
- **Smallest tercile.** Historical events below 0.25 lots are NOT_AVAILABLE for comparison:
  neither simulator ever generates aggressive events that small.
- **Largest tercile.** The selected model matches history in sign and magnitude at 1 s and
  60 s on ETH. Examples: July, 1 s: 2.92 bps historical vs 3.50 bps simulated; 60 s: 4.38 vs
  4.58.
- **Middle tercile.** The selected model's immediate response is far too small (0.16 bps vs
  1.28–3.67 bps historical), and its 60 s response reverses sign.
- **Control.** It underestimates magnitudes almost everywhere, e.g. 0.75 vs 2.92 bps in the
  large bucket at 1 s in July.

**Persistence (mean 60 s response / mean 1 s response):**

| | ETH Apr | ETH Jun | ETH Jul | BTC Jul |
|---|---:|---:|---:|---:|
| Historical | 1.10 | 1.48 | 1.41 | 2.72 |
| Selected simulator | 0.85 | 0.85 | 0.85 | 0.85 |
| Control | 1.08 | 1.08 | 1.08 | 1.08 |

In history, impact grows or persists after the trade. In the simulators it partially reverts.

**Resilience.** Historical opposite best-level depth recovers to 90% within 60 s after
98.8–99.7% of depleting events, with a median of 65–91 ms. Both simulators recover in
99.6–100% of cases. Recovery share therefore does not discriminate; the timing differences
are in the sealed results.

**Almgren–Chriss proxies** (empirical proxies, not structural identification):

| | σ (bps/√s) | Temporary proxy (bps per unit participation) | Persistent proxy (bps per lot) | Liquidity scale (lots) |
|---|---:|---:|---:|---:|
| ETH Apr | 1.16 | 6.66 | 0.012 | 1,604 |
| ETH Jul | 0.60 | 4.81 | 0.028 | 1,097 |
| BTC Jul | 0.32 | 0.55 | 0.004 | 2,728 |
| Selected simulator | 2.45 | 6.57 | 0.085 | — |

The selected simulator is roughly twice as volatile as history per √s, and its persistent
impact proxy is 3–20 times larger.

## Interpretation and limits

- The combined simulator reproduces the large-trade response better than the v0.4 class.
- It misses the size gradient, sub-lot flow and the persistence of historical impact.
- Responses are associations with signed aggressive flow; they mix information, concurrent
  flow and mechanical impact. No causal impact model is identified.
- Net depth changes cannot separate replenishment from cancellations.
- BTC is evaluated with the ETH development scale, so it is a transfer test.
