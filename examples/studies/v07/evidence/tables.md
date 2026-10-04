## hypotheses

| Hypothesis | Dataset | Model | Estimate | Interval | Margin | Alpha (adjusted) | Family | MDE | Status |
|---|---|---|---|---|---|---|---|---|---|
| H1 | deribit-eth-perp-2020-11-01, deribit-eth-perp-2020-12-01 | G0_post vs G0_point | — | — | — | 0.025 | F1_posterior | 0.2158 | **FAILED** |
| H2 | deribit-eth-perp-2020-11-01, deribit-eth-perp-2020-12-01 | G1-G4 vs G0_point | — | — | — | 0.00625 | F2_distinguishability | 0.07152 | **NOT_ESTABLISHED** |
| H3 | deribit-eth-perp-2020-11-01, deribit-eth-perp-2020-12-01 | G1-G4 vs G0_point | — | — | 0.05 | 0.00625 | F3_support | 0.07152 | **NOT_ESTABLISHED** |
| H4 | deribit-btc-perp-2020-11-01 | G3_conditional_ar | 0.7172 | [0.6432, 0.7676] | — | 0.025 | F4_cross_market | 0.2158 | **FAILED** |
| H5 | bitmex-xbtusd-2020-11-01 | G3_conditional_ar | 0.6162 | [0.5275, 0.6629] | — | 0.025 | F4_cross_market | 0.2158 | **FAILED** |
| H6 | development | G0_post | — | — | — | — | D1_posterior_shape | — | **ESTABLISHED** |
| H7 | simulation (development-calibrated worlds) | plausible world set | — | — | 0.5 | 0.00625 | F7_model_uncertainty | — | **ESTABLISHED** |
| H8 | simulation (development-calibrated worlds) | plausible world set | — | — | 1 | 0.001786 | F8_robust_pairs | 0.9242 | **NOT_ESTABLISHED** |
| H9 | simulation (development-calibrated worlds) | plausible world set | — | — | — | 0.05 | D2_ranking_topology | — | **INCONCLUSIVE** |
| H10 | deribit-eth-perp-2020-11-01, deribit-eth-perp-2020-12-01 | EA vs G0_point | — | — | — | 0.025 | F10_execution_realism | 0.4625 | **NOT_ESTABLISHED** |
| H11 | deribit-eth-perp-2020-11-01, deribit-eth-perp-2020-12-01 | EA vs G0_point | — | — | delta = 0.10 x G0 selection-day objective (computed and sealed before holdout access) | 0.025 | F11_noninferiority | 0.1961 | **FAILED** |
| H12 | deribit-eth-perp-2021-01-01 | learned/classical policies | — | — | — | 0.0125 | F12_transfer | 2.585 | **NOT_ESTABLISHED** |
| H13 | deribit-eth-perp-2021-01-01 | learned/classical policies | — | — | — | 0.05 | D3_historical_survival | — | **INCONCLUSIVE** |

## members

| Hypothesis | Member | Estimate | Interval | Status |
|---|---|---|---|---|
| H1 | deribit-eth-perp-2020-11-01 | 0.3868 | [0.211, 0.6269] | FAILED |
| H1 | deribit-eth-perp-2020-12-01 | 0.5678 | [0.3026, 0.9321] | FAILED |
| H2 | G1_state_hawkes@deribit-eth-perp-2020-11-01 | -2.543e-05 | [-9.366e-05, 0] | NOT_ESTABLISHED |
| H2 | G2_regime_switching@deribit-eth-perp-2020-11-01 | 0 | [0, 0] | NOT_ESTABLISHED |
| H2 | G3_conditional_ar@deribit-eth-perp-2020-11-01 | 0 | [0, 0] | NOT_ESTABLISHED |
| H2 | G4_neural_temporal@deribit-eth-perp-2020-11-01 | 0 | [0, 0] | NOT_ESTABLISHED |
| H2 | G1_state_hawkes@deribit-eth-perp-2020-12-01 | 0.00213 | [0, 0.006671] | NOT_ESTABLISHED |
| H2 | G2_regime_switching@deribit-eth-perp-2020-12-01 | 0.0021 | [-2.347e-05, 0.00658] | NOT_ESTABLISHED |
| H2 | G3_conditional_ar@deribit-eth-perp-2020-12-01 | 0.0005194 | [0, 0.001997] | NOT_ESTABLISHED |
| H2 | G4_neural_temporal@deribit-eth-perp-2020-12-01 | 0.002048 | [0, 0.006381] | NOT_ESTABLISHED |
| H3 | G1_state_hawkes@deribit-eth-perp-2020-11-01 | 0.00278 | [0, 0.04672] | NOT_ESTABLISHED |
| H3 | G2_regime_switching@deribit-eth-perp-2020-11-01 | 0 | [0, 0] | NOT_ESTABLISHED |
| H3 | G3_conditional_ar@deribit-eth-perp-2020-11-01 | 0 | [0, 0.002778] | NOT_ESTABLISHED |
| H3 | G4_neural_temporal@deribit-eth-perp-2020-11-01 | 0.01529 | [0.002778, 0.04778] | NOT_ESTABLISHED |
| H3 | G1_state_hawkes@deribit-eth-perp-2020-12-01 | 0 | [0, 0] | NOT_ESTABLISHED |
| H3 | G2_regime_switching@deribit-eth-perp-2020-12-01 | 0 | [0, 0] | NOT_ESTABLISHED |
| H3 | G3_conditional_ar@deribit-eth-perp-2020-12-01 | 0 | [0, 0] | NOT_ESTABLISHED |
| H3 | G4_neural_temporal@deribit-eth-perp-2020-12-01 | 0 | [0, 0] | NOT_ESTABLISHED |
| H6 |  | — | — | — |
| H7 | ac | 1.496 | — | NOT_ESTABLISHED |
| H7 | imbalance_aware | 1.426 | — | NOT_ESTABLISHED |
| H7 | liquidity_sensitive | 1.435 | — | NOT_ESTABLISHED |
| H7 | pov | 2.845 | — | ESTABLISHED |
| H7 | spread_aware | 1.189 | — | NOT_ESTABLISHED |
| H7 | twap | 1.368 | — | NOT_ESTABLISHED |
| H7 | urgency | 1.216 | — | NOT_ESTABLISHED |
| H7 | vwap | 1.173 | — | NOT_ESTABLISHED |
| H8 | ac|imbalance_aware | 0.1134 | [-0.1486, 0.4239] | EQUIVALENT_WITHIN_MARGIN |
| H8 | ac|liquidity_sensitive | 0.05033 | [-0.4196, 0.4198] | EQUIVALENT_WITHIN_MARGIN |
| H8 | ac|spread_aware | 0.06188 | [-0.2686, 0.3307] | EQUIVALENT_WITHIN_MARGIN |
| H8 | ac|urgency | 0.0163 | [-0.387, 0.332] | EQUIVALENT_WITHIN_MARGIN |
| H8 | imbalance_aware|spread_aware | -0.05147 | [-0.4978, 0.2007] | EQUIVALENT_WITHIN_MARGIN |
| H8 | liquidity_sensitive|imbalance_aware | 0.06302 | [-0.3159, 0.6078] | EQUIVALENT_WITHIN_MARGIN |
| H8 | liquidity_sensitive|spread_aware | 0.01155 | [-0.3355, 0.4774] | EQUIVALENT_WITHIN_MARGIN |
| H8 | liquidity_sensitive|urgency | -0.03403 | [-0.1936, 0.1489] | EQUIVALENT_WITHIN_MARGIN |
| H8 | pov|ac | 0.2663 | [-0.1643, 0.8901] | EQUIVALENT_WITHIN_MARGIN |
| H8 | pov|imbalance_aware | 0.3797 | [-0.04421, 0.9795] | EQUIVALENT_WITHIN_MARGIN |
| H8 | pov|liquidity_sensitive | 0.3167 | [-0.0887, 0.9334] | EQUIVALENT_WITHIN_MARGIN |
| H8 | pov|spread_aware | 0.3282 | [-0.125, 0.8847] | EQUIVALENT_WITHIN_MARGIN |
| H8 | pov|urgency | 0.2826 | [-0.1765, 0.9031] | EQUIVALENT_WITHIN_MARGIN |
| H8 | twap|ac | -0.05682 | [-0.3929, 0.2162] | EQUIVALENT_WITHIN_MARGIN |
| H8 | twap|imbalance_aware | 0.05653 | [-0.2989, 0.3937] | EQUIVALENT_WITHIN_MARGIN |
| H8 | twap|liquidity_sensitive | -0.006488 | [-0.6904, 0.3748] | EQUIVALENT_WITHIN_MARGIN |
| H8 | twap|pov | -0.3232 | [-1.097, 0.1596] | INDETERMINATE |
| H8 | twap|spread_aware | 0.00506 | [-0.4716, 0.3272] | EQUIVALENT_WITHIN_MARGIN |
| H8 | twap|urgency | -0.04052 | [-0.5293, 0.2914] | EQUIVALENT_WITHIN_MARGIN |
| H8 | twap|vwap | -0.09318 | [-0.4168, 0.1422] | EQUIVALENT_WITHIN_MARGIN |
| H8 | urgency|imbalance_aware | 0.09705 | [-0.1982, 0.6163] | EQUIVALENT_WITHIN_MARGIN |
| H8 | urgency|spread_aware | 0.04558 | [-0.2954, 0.4323] | EQUIVALENT_WITHIN_MARGIN |
| H8 | vwap|ac | 0.03635 | [-0.2597, 0.3308] | EQUIVALENT_WITHIN_MARGIN |
| H8 | vwap|imbalance_aware | 0.1497 | [-0.1146, 0.5372] | EQUIVALENT_WITHIN_MARGIN |
| H8 | vwap|liquidity_sensitive | 0.08669 | [-0.388, 0.494] | EQUIVALENT_WITHIN_MARGIN |
| H8 | vwap|pov | -0.23 | [-0.9168, 0.2878] | EQUIVALENT_WITHIN_MARGIN |
| H8 | vwap|spread_aware | 0.09824 | [-0.2327, 0.3965] | EQUIVALENT_WITHIN_MARGIN |
| H8 | vwap|urgency | 0.05266 | [-0.2992, 0.3841] | EQUIVALENT_WITHIN_MARGIN |
| H10 | deribit-eth-perp-2020-11-01 | 0.1891 | [-0.001502, 0.3112] | NOT_ESTABLISHED |
| H10 | deribit-eth-perp-2020-12-01 | 0.08816 | [-0.06755, 0.2363] | NOT_ESTABLISHED |
| H11 | deribit-eth-perp-2020-11-01 | 0.4362 | — | FAILED |
| H11 | deribit-eth-perp-2020-12-01 | 0.3952 | — | FAILED |
| H12 | dqn@conservative | -0.006434 | [-2.511, 1.467] | NOT_ESTABLISHED |
| H12 | dqn@optimistic | 0.03046 | [-2.672, 1.54] | NOT_ESTABLISHED |
| H12 | ppo@conservative | 0.0271 | [-2.489, 1.602] | NOT_ESTABLISHED |
| H12 | ppo@optimistic | -0.09658 | [-2.756, 1.584] | NOT_ESTABLISHED |

## attempts

| Ledger index | Design | Outcome | Directory | Note |
|---|---|---|---|---|
| 90 | m6-recovery-registered | FAILED | results/v07/m6/recovery | third truth (seeded prior draw) violated the 500 events/s guard while simulating its synthetic target; the run aborted unsealed after 2 of 3 truths. Fix: implausible truths are rejected and redrawn (seeded, recorded); rerun in results/v07/m6/recovery-2. |
| 92 | m6-posterior-registered | ABORTED | results/v07/m6/posterior | process stopped by the host (system low on memory) during an SMC-ABC run; no result sealed; selection and execution-aware steps never started. Recovery-2 completed and is sealed (ASSUMPTION_DEPENDENT). |
| 100 | m8-identifiability-v2 | FAILED | results/v07/m8/identifiability | components unevaluable in single-seed noise runs produced NaN noise scales; eigendecomposition failed. Fixed (drop rows without a finite noise estimate) with a regression test; rerun in results/v07/m8/identifiability-2. |
| 110 | m8-identifiability-v2 | INVALID | results/v07/m8/identifiability-2 | sealed run retained; a zero-seed-noise observable (exceed_spread_p99) hit the 1e-6 noise floor and dominated the standardized Jacobian (row norm 2.6e7 vs median 149), making rank and sloppiness meaningless. Fixed: noise floored at 10% of the median noise (regression test); rerun in identifiability-3. Descriptive, development-only analysis; no holdout involved. |
| 113 | m17-holdout-realism | ABORTED | results/v07/m17/deribit-eth-perp-2020-12-01 | stopped by the host for low system memory after the downloaded and opened stages; no tape built, no statistic computed, no outcome inspected. Rerun under the same sealed design in results/v07/m17/deribit-eth-perp-2020-12-01-2 with one job at a time. |
| 114 | m15-execution-model-risk-registered | ABORTED | results/v07/m15/execution | stopped by the host for low system memory during episode simulation; nothing sealed. Rerun with identical seeds and settings, 4 workers, in results/v07/m15/execution-2; policy training and predictions had not started. |
| 118 | m17-holdout-realism | ABORTED | results/v07/m17/deribit-eth-perp-2020-12-01 | correction to entry 113: the aborted run had also completed the parsed stage (entry 112, tape built) before the host stopped it; no statistic was computed, no evaluated stage was recorded and no outcome was inspected. The rerun uses the same sealed design. |
| 186 | m16-transfer | ABORTED | results/v07/m16/transfer | stopped for low system memory after the downloaded, opened and parsed stages (144 episodes), while replay workers held pickled episode slices (about 3 GB each); no replay row written, no statistic computed, no outcome inspected. Entry 185 recorded git_dirty true: the only uncommitted change in the main checkout was a README draft (no code). Rerun under the same sealed design with a lower-memory episode representation, in results/v07/m16/transfer-2. |
| 189 | m16-transfer | ABORTED | results/v07/m16/transfer-2 | rerun with CLEOLOB_WORKERS=4 launched at commit 65b6058; stopped by the operator after the downloaded and opened stages (entries 187-188) when the instruction to use one worker arrived; no episodes parsed, no replay row, no statistic, no outcome inspected. The empty write-once directory is retained. Rerun under the same sealed design with CLEOLOB_WORKERS=1 in results/v07/m16/transfer-3. |
| 195 | m22-final-analysis | INVALID | results/v07/m22/final | the M3 queue sensitivity is a conservative-to-optimistic width of the mean passive fill fraction, but the decision benchmark stored it as queue_sensitivity_bps. Values are unchanged; the field is renamed queue_sensitivity_fill_fraction with a regression test. The sealed run is retained; rerun in results/v07/m22/final-2. Synthesis of sealed runs only; no data access. |

## consumed

- `bitmex-xbtusd-2020-11-01`
- `bitstamp-btcusd-mbo-dev`
- `bitstamp-btcusd-mbo-validation`
- `deribit-btc-perp-2020-07-01`
- `deribit-btc-perp-2020-09-01`
- `deribit-btc-perp-2020-11-01`
- `deribit-eth-perp-2020-04-01`
- `deribit-eth-perp-2020-05-01`
- `deribit-eth-perp-2020-06-01`
- `deribit-eth-perp-2020-07-01`
- `deribit-eth-perp-2020-08-01`
- `deribit-eth-perp-2020-09-01`
- `deribit-eth-perp-2020-10-01`
- `deribit-eth-perp-2020-11-01`
- `deribit-eth-perp-2020-12-01`
- `deribit-eth-perp-2021-01-01`
- `deribit:BTC-PERPETUAL:2026-07-01`
- `deribit:BTC-PERPETUAL:2026-08-01`
- `deribit:BTC-PERPETUAL:2026-09-01`
- `deribit:ETH-PERPETUAL:2020-04-01`
- `deribit:ETH-PERPETUAL:2020-05-01`
- `deribit:ETH-PERPETUAL:2020-06-01`
- `deribit:ETH-PERPETUAL:2026-07-01`
- `deribit:ETH-PERPETUAL:2026-08-01`
- `deribit:ETH-PERPETUAL:2026-09-01`

## fresh

- none
