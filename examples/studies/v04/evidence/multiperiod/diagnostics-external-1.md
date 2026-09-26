# Calibration failure diagnostics

Original assessment: **FAIL** (unchanged).

Reference: 1024 rows; target: 512 rows.

| Observable | Distance | Mean drift / train scale | Attribution |
|---|---:|---:|---|
| spread_bps | 1.0370 | 1.0404 | distribution_shift_association |
| bid_depth5 | 0.9265 | -0.9263 | distribution_shift_association |
| ask_depth5 | 0.9394 | -0.9416 | distribution_shift_association |
| imbalance5 | 0.0393 | -0.0366 | no_large_marginal_discrepancy |
| log_return | 0.7668 | -0.0476 | distribution_shift_association |
| rms_return_10 | 1.3312 | 1.3300 | distribution_shift_association |
| spread_change_bps | 0.3387 | 0.0150 | no_large_marginal_discrepancy |
| relative_depth_change | 0.0756 | -0.0038 | no_large_marginal_discrepancy |

Causal attribution and parameter instability remain NOT_ESTABLISHED.

Mean-drift intervals in the JSON use chronological blocks with Bonferroni correction
across eight observables. They are descriptive and assume approximately stationary blocks.

Unsupported: arrival_intensity, cancellation_intensity, trade_intensity, book_update_intensity, order_size_distribution, top_level_depth, queue_position, causal_price_impact.
