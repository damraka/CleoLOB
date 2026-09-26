# Calibration failure diagnostics

Original assessment: **FAIL** (unchanged).

Reference: 1024 rows; target: 512 rows.

| Observable | Distance | Mean drift / train scale | Attribution |
|---|---:|---:|---|
| spread_bps | 0.4381 | -0.4091 | no_large_marginal_discrepancy |
| bid_depth5 | 1.2065 | 1.2143 | distribution_shift_association |
| ask_depth5 | 1.1845 | 1.1881 | distribution_shift_association |
| imbalance5 | 0.0450 | 0.0445 | no_large_marginal_discrepancy |
| log_return | 0.2189 | 0.0470 | no_large_marginal_discrepancy |
| rms_return_10 | 0.4033 | -0.4019 | no_large_marginal_discrepancy |
| spread_change_bps | 0.1805 | 0.0050 | no_large_marginal_discrepancy |
| relative_depth_change | 0.0362 | 0.0036 | no_large_marginal_discrepancy |

Causal attribution and parameter instability remain NOT_ESTABLISHED.

Mean-drift intervals in the JSON use chronological blocks with Bonferroni correction
across eight observables. They are descriptive and assume approximately stationary blocks.

Unsupported: arrival_intensity, cancellation_intensity, trade_intensity, book_update_intensity, order_size_distribution, top_level_depth, queue_position, causal_price_impact.
