# v0.4 completion-constrained policy protocol

The v0.4 experiment is a new **bounded synthetic study**. It preserves the v0.3
registration, checkpoints, failures and null findings. The v0.3 environment remains
the default; `protocol_version: v04` explicitly opts into this experiment. No
historical passive-fill counterfactual or profitability claim follows from it.

## Execution objective and completion

The same completion rule applies to PPO, DQN, TWAP, VWAP, POV, Almgren–Chriss,
heuristic and random policies. During the final 20% of the decision horizon it
replaces each normal decision with an aggressive schedule: cancel existing orders,
reserve them until cancellation acknowledgment, and submit available residual
divided by remaining decisions, rounded up to a lot. The last decision requests
all available residual. No new order is submitted after the horizon.

These are real simulator orders, subject to the same liquidity, parent inventory,
order size, position, notional, loss, kill-switch and latency limits as other orders.
They can partially fill, fail, or remain unresolved. Completion is **not guaranteed**.
Ordinary settlement reconciles late fills after the decision horizon and is
reported separately. Terminal book-walk and midpoint valuations never create fills.

Episode rows include actual fills, residual inventory, settled actual completion,
time of the last actual fill when complete, realized fee-inclusive fill cost,
realized gross slippage, actual fees, participation and hypothetical residual
midpoint cost. Participation is own fills divided by own fills plus other trades
between arrival and final settlement. It is an observed simulator ratio, not a
hard participation cap. Gross effective shortfall remains a noncausal impact proxy
that also mixes spread, timing and exogenous market movement. Unresolved settlement
withholds final completion; absence of executable residual depth withholds the
economic endpoint but does not erase an observed failure to complete. A residual
midpoint valuation is unavailable without a true two-sided terminal quote; the
simulator's last-midpoint fallback is not presented as an observed quote.

## State and normalization audit

The 32-feature v0.4 state contains current five-level bid/ask relative prices in
ticks and quantities; microprice distance; imbalance; remaining inventory and
time fractions; own reserved, available, resting and cancellation-pending quantity
fractions; own quantity-weighted limit-price distance from current midpoint;
oldest live child age relative to horizon; risk-halt state; and opposite-liquidity
availability. Empty levels have zero volume and explicitly artificial price
markers. The feature names are persisted in registration and normalization files.

This is **simulator-internal observation**, with instantaneous aggregate depth
and own-order state. No queue quantity, FIFO position, hidden quantity, future
event, future price, market seed or terminal value is supplied to the policy.
The state does not establish historical observability or exchange feed realism;
it is not claimed to be a sufficient Markov state. Missing feed data is outside
this synthetic experiment; empty books and no-event periods are tested explicitly.

Normalization fits mean and population standard deviation using eight registered
original-regime random exploration episodes from the training seed domain only.
Scales have a floor of one raw feature unit. Standardized values are clipped at
±10. Means, scales, feature schema, exploration seeds, sample count and observation
hash are recorded and sealed **before policy fitting**. Checkpoint metadata binds
the normalization hash; evaluation cannot update it. Test seeds are never used to
fit a scaler. Clipping may hide extreme magnitude and is an explicit limitation.

The `no_book` ablation masks the first 22 market features and the own-price-distance
and opposite-liquidity fields, retaining own-order quantities/age, inventory/time
and halt state. `no_terminal` removes only the extra completion reward penalty.
It retains the identical completion rule and hypothetical economic accounting.
Thus that ablation addresses a reward term, not the completion mechanism itself.
These finite ablations are not a complete search over state or reward designs.

## Frozen design and inference

`configs/v04-policy-study.json` specifies four independent training seeds, two
algorithms, three arms, 2,048 training steps per model, twelve common unseen
evaluation market seeds, and original/shifted/stress regimes. That is 24 models,
49,152 model training steps and **1,080 evaluation episodes**: three regimes ×
twelve markets × (six controls + 24 learned instances). The eight normalization
exploration episodes are additional and separately recorded. The v0.3 design had
three training seeds, 1,024 steps per model and eight evaluation markets.

The final fixed-budget checkpoint is the only evaluated checkpoint. Training
episode returns, reward components, update metrics and seed-wise outcome means
expose learning curves and instability without claiming convergence. Failed
normalization/training attempts, failed control identification and INVALID
evaluation rows are retained and cannot be overwritten. AC identification uses
separate seeds per known regime; failed fits use registered defaults and retain
FAIL. VWAP uses separate synthetic auxiliary paths. These controls know each
regime generator, as in v0.3; evaluation outcomes never select their parameters.

Registration freezes configuration, source, primary endpoints, hypotheses,
comparison family, correction, completion margin and resource budget before any
model fitting or final evaluation. The 48 registered pair definitions each have
two primary endpoints: economic cost and actual completion, making **96 planned
hypotheses**. Each receives a crossed training-seed/market-seed bootstrap interval
at `1 - 0.05/96`, with 20,000 registered resamples.
These are two-sided central intervals with each tail `0.05/(2*96)`; using their
bounds for one-sided cost/noninferiority gates is conservative relative to a
one-sided interval at `0.05/96`. Comparison grids remain paired;
missing/invalid costs withhold cost comparisons, while missing/unsettled actual
completion withholds completion comparisons. Neither shrinks the planned family.

A per-contrast joint gate passes only if the adjusted cost interval's upper bound
is below zero **and** the adjusted completion-difference interval's lower bound is
at least −0.05. This is a preregistered five-percentage-point completion
noninferiority margin, not equality. Cost excludes the extra completion penalty;
actual completion is never inferred from cost or hypothetical liquidation.
Completion summaries include settled outcomes even when economics are INVALID.
The raw valid-cost-only v0.3 summary remains labeled separately.

Constant paired completion differences produce degenerate bootstrap bounds. Such
comparisons are **INCONCLUSIVE**, even if every observed episode completed, and
cannot pass the joint success gate: unobserved future failures are not ruled out
by a `[0, 0]` difference interval. The observed completion rate remains visible.

**Power is NOT_ESTABLISHED.** The fixed CPU budget is not based on prospective
variance-based power calibration. Four training seeds and twelve evaluation seeds
still provide coarse estimates; approximately extreme Bonferroni quantiles with
20,000 resamples are noisy. Larger counts than v0.3 do not establish convergence,
equivalence, reliable tail behavior, general learned superiority or historical
external validity. There is no global “best policy” label from sample means.

## Reproduction

Use a fresh directory and the frozen source revision with the RL extra installed:

```sh
python -m lob.policy_study register --config configs/v04-policy-study.json --out results/v04/policy
python -m lob.policy_study train --out results/v04/policy
python -m lob.policy_study evaluate --out results/v04/policy
python -m lob.policy_study verify --out results/v04/policy
python -m pytest tests/test_completion_v04.py tests/test_policy_study.py tests/test_settlement.py
```

Source drift after registration is refused. A local SHA-256 seal detects mutation;
it does not supply an external timestamp, signature or independent replication.
The v0.4 verifier also recomputes registered paired summaries from the episode
journal and checks evaluation-lock, training and control-result links; a rehashed
but numerically altered result cannot pass this semantic check.
Results are reported only after an actual frozen run, never inferred from tests.

## Registered run obtained on 2026-09-26

The study at `results/v04/policy-completion-20260926` ran from clean source commit
`9072bb80c4a823d940d2b752e34f08881a4841de`. All 24 models completed 49,152 training
steps and all 1,080 evaluation episodes were retained. There were zero INVALID or
WARNING rows. Artifact and semantic verification returned `valid: true`.

Every episode completed by the end of settlement, with zero final residual. This
held across every regime, arm, training seed and control. **19 episodes completed
after the 2-second decision horizon**, during settlement; 1,061 completed within
the horizon. The latest completion was 2.056912619 seconds. This bounded synthetic
result does not guarantee completion at other quantities, liquidity or latencies,
and cannot isolate the constraint's causal effect from the changed state, budget
and seeds relative to v0.3.

Of the 48 economic contrasts, 47 multiplicity-adjusted intervals included zero.
The exception was main DQN minus POV in stress: mean −2.133149109 bps, adjusted
interval [−4.549744191, −0.075293572] bps. All 48 completion contrasts were
**INCONCLUSIVE** because observed paired completion differences were constant.
Consequently **zero joint success gates passed**. Learned-policy superiority
remains **NOT_ESTABLISHED**; the single economic contrast must not be promoted to
a general winning-policy claim. All three separate AC identification fits passed.

Main-arm mean economic cost, in bps, was:

| Regime | TWAP | POV | PPO | DQN |
| --- | ---: | ---: | ---: | ---: |
| Original | 1.7963 | 1.8388 | 1.8280 | 1.7700 |
| Shifted | 2.0782 | 2.0568 | 2.1159 | 1.9036 |
| Stress | 2.5834 | 3.5446 | 2.1587 | 1.4114 |

These sample means are descriptive. All main PPO/DQN training-seed completion
rates were 100%, but stress PPO training-seed cost means ranged from 1.1736 to
3.1013 bps, illustrating remaining seed sensitivity. Full seed-level outcomes and
learning traces remain in the sealed run. Power and convergence are unestablished.

The plan SHA-256 is
`c037f9f87070adeac98078bcb1886bcfcf980b542ec2b4cdf6236528a2be0b56`;
the canonical source-manifest digest is
`2c58cf1fe9fa0fdaa006548addc2411309bd2fe66f275442206c8a7655e5ccff`.
