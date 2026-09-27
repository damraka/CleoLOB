# v0.5 M5 — strict execution mandates and completion

v0.4 counted a parent order as complete once settlement had removed its residual:
- 1,061 of 1,080 episodes completed inside the decision horizon;
- 19 completed only during post-horizon settlement.

v0.5 makes the distinction a formal primary endpoint (E5). The rules are frozen in the
[protocol](v05-research-protocol.md) and implemented once in
[`lob/mandate.py`](../lob/mandate.py).

## Rules (identical for every policy and control)

- Orders may be submitted only while elapsed time is below the horizon, and never at or
  after it. The v0.4 urgency rule is unchanged and shared.
- **Within-horizon completion (E5):** actual fills with fill time ≤ horizon end reach the
  target. The boundary is inclusive.
- **Settlement:** after the horizon, only cancellations and exchange events are processed.
  Fills in this window are *post-horizon fills*.
- **Final settlement completion:** reported separately. `settlement_only_completion` marks
  parents completed only after the horizon, which the mandate never counts.
- **Unresolved settlement:** withholds the final outcome (`None`), but not the within-horizon
  outcome.
- **Hypothetical residual valuation:** the terminal book walk. It never creates a fill and
  never changes completion.

Every execution row (`run_episode`, the RL environment and all classical controls)
carries a `mandate_*` block. Classical controls and learned policies use the same function:

| Field | Meaning |
|---|---|
| `mandate_within_horizon_completion` | E5 |
| `mandate_within_horizon_filled_qty`, `mandate_fill_fraction_at_horizon` | actual fills by horizon end |
| `mandate_residual_inventory_at_horizon` | target minus within-horizon fills |
| `mandate_post_horizon_filled_qty` | settlement fills |
| `mandate_final_settlement_completion`, `mandate_settlement_only_completion` | final outcome, reported separately |
| `mandate_residual_inventory_after_settlement`, `mandate_final_fill_fraction` | after settlement |
| `mandate_time_to_completion`, `mandate_lateness_seconds` | completion time; lateness is 0 within horizon and positive if completed during settlement |
| `mandate_fees_within_horizon`, `mandate_fees_post_horizon`, `mandate_fees_total` | fees split by window |
| `mandate_realized_fill_cost_bps` | actual fills only |
| `mandate_hypothetical_residual_valuation_bps` | hypothetical; never a fill |
| `mandate_completion_adjusted_cost_bps` (E6) | realized plus hypothetical residual; the components are always also reported |
| `mandate_participation` | own fills / (own fills + other trades) |

## Tests

[`tests/test_v05_mandate.py`](../tests/test_v05_mandate.py) covers:
- a fill exactly at the horizon;
- fills one event before and one event after the horizon;
- latency pushing fills past the horizon;
- delayed cancellation during settlement;
- missing opposite liquidity;
- kill switch and order-size limits;
- unresolved settlement;
- valuation without fills;
- overfill rejection;
- identical mandate blocks for TWAP, VWAP, POV, AC and the policy runner.

The v0.4 completion and policy-study tests still pass unchanged. Existing v0.4 row keys
and values are unaltered; the `mandate_*` keys are additions.

## Empirical use

The registered M7 study ([v05-rl.md](v05-rl.md)) uses E5 as its completion endpoint and E6
as its cost endpoint. It reports settlement-only completions, residuals and lateness
separately for every regime, arm, training seed and control.
