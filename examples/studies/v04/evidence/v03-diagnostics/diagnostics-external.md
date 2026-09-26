# Calibration failure diagnostics

Original assessment: **FAIL** (unchanged).

Reference: 838 rows; target: 360 rows.

| Observable | Distance | Mean drift / train scale | Attribution |
|---|---:|---:|---|
| spread_bps | 1.0691 | 1.0663 | distribution_shift_association |
| bid_depth5 | 0.9905 | -0.9910 | distribution_shift_association |
| ask_depth5 | 0.8794 | -0.8815 | distribution_shift_association |
| imbalance5 | 0.0267 | -0.0263 | no_large_marginal_discrepancy |
| log_return | 0.7014 | -0.1077 | distribution_shift_association |
| rms_return_10 | 1.0626 | 1.0617 | distribution_shift_association |
| spread_change_bps | 0.3139 | 0.0249 | no_large_marginal_discrepancy |
| relative_depth_change | 0.0455 | -0.0127 | no_large_marginal_discrepancy |

Causal attribution and parameter instability remain NOT_ESTABLISHED.

Mean-drift intervals in the JSON use chronological blocks with Bonferroni correction
across eight observables. They are descriptive and assume approximately stationary blocks.

Unsupported: arrival_intensity, cancellation_intensity, trade_intensity, book_update_intensity, order_size_distribution, top_level_depth, queue_position, causal_price_impact.
