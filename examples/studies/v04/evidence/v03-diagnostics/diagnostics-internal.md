# Calibration failure diagnostics

Original assessment: **WARNING** (unchanged).

Reference: 838 rows; target: 240 rows.

| Observable | Distance | Mean drift / train scale | Attribution |
|---|---:|---:|---|
| spread_bps | 0.3296 | 0.3211 | no_large_marginal_discrepancy |
| bid_depth5 | 0.3224 | -0.3182 | no_large_marginal_discrepancy |
| ask_depth5 | 0.3009 | -0.3005 | no_large_marginal_discrepancy |
| imbalance5 | 0.0433 | 0.0102 | no_large_marginal_discrepancy |
| log_return | 0.3011 | 0.0585 | no_large_marginal_discrepancy |
| rms_return_10 | 0.3914 | 0.3908 | no_large_marginal_discrepancy |
| spread_change_bps | 0.0269 | 0.0006 | no_large_marginal_discrepancy |
| relative_depth_change | 0.0383 | -0.0014 | no_large_marginal_discrepancy |

Causal attribution and parameter instability remain NOT_ESTABLISHED.

Mean-drift intervals in the JSON use chronological blocks with Bonferroni correction
across eight observables. They are descriptive and assume approximately stationary blocks.

Unsupported: arrival_intensity, cancellation_intensity, trade_intensity, book_update_intensity, order_size_distribution, top_level_depth, queue_position, causal_price_impact.
