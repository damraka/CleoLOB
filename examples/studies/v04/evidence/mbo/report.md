# MBO validation

Status: **ESTABLISHED**; source kind: **synthetic**.
Real historical MBO validation: **NOT_AVAILABLE**.
Source aggregate agreement: **ESTABLISHED**.

declared-source reconstruction consistency, not exchange truth or counterfactual fills

| Diagnostic | Count |
| --- | ---: |
| events | 12 |
| orders_observed | 5 |
| snapshot_order_records | 2 |
| adds | 3 |
| modifications | 1 |
| cancels | 1 |
| deletes | 1 |
| executions | 3 |
| partial_executions | 1 |
| terminal_executions | 2 |
| executed_quantity | 12 |
| trade_prints | 1 |
| priority_resets | 1 |
| left_censored_order_records | 2 |
| census_censored_trajectories | 1 |
| known_terminal_lifetimes | 2 |
| terminal_lifetime_sum_ns | 120 |
| internal_aggregate_mismatches | 0 |
| queue_checks | 26 |
| queue_inconsistencies | 0 |
| reference_comparisons | 5 |
| aggregate_book_mismatches | 0 |
| price_level_mismatches | 0 |
| volume_mismatches | 0 |
| absolute_volume_difference_lots | 0 |
| reference_timestamp_mismatches | 0 |
| sequence_gaps | 0 |
| duplicate_sequences | 0 |
| duplicate_order_ids | 0 |
| timestamp_reversals | 0 |
| invalid_transitions | 0 |
| unmatched_references | 0 |
| right_censored_at_eof | 1 |

validated prefix and first rejected event; no skipped or repaired suffix


Snapshot FIFO and venue amendment rules are declared source obligations.
Independent oracle checks implementation consistency, not exchange correctness.
Recorded executions behind visible orders are retained; they are not reassigned to FIFO head.
Sequence continuity of normalized indices cannot establish original feed completeness.
No hidden liquidity, passive counterfactual fills or profitability established.
