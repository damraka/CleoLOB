# Calibration failure diagnostics

Original assessment: **PASS** (unchanged).

Reference: 1024 rows; target: 512 rows.

| Observable | Distance | Mean drift / train scale | Attribution |
|---|---:|---:|---|
| spread_bps | 0.0210 | 0.0166 | no_large_marginal_discrepancy |
| bid_depth5 | 0.0336 | -0.0223 | no_large_marginal_discrepancy |
| ask_depth5 | 0.0439 | -0.0255 | no_large_marginal_discrepancy |
| imbalance5 | 0.0273 | 0.0119 | no_large_marginal_discrepancy |
| log_return | 0.0476 | -0.0313 | no_large_marginal_discrepancy |
| rms_return_10 | 0.0439 | -0.0287 | no_large_marginal_discrepancy |
| spread_change_bps | 0.0152 | 0.0084 | no_large_marginal_discrepancy |
| relative_depth_change | 0.0714 | 0.0140 | no_large_marginal_discrepancy |

Causal attribution and parameter instability remain NOT_ESTABLISHED.

Mean-drift intervals in the JSON use chronological blocks with Bonferroni correction
across eight observables. They are descriptive and assume approximately stationary blocks.

Unsupported: arrival_intensity, cancellation_intensity, trade_intensity, book_update_intensity, order_size_distribution, top_level_depth, queue_position, causal_price_impact.
