# v0.3 execution-policy protocol and environment audit

**IMPLEMENTED / TESTED:** `lob.policy_study` adds Stable Baselines3 PPO and DQN,
three independent training seeds, fixed paired market seeds, classical controls,
two bounded ablations, regime shifts, training traces, checkpoint provenance and
crossed-seed uncertainty. DQN fits the existing five discrete actions. The
simulator, economic accounting, settlement, classical agents and bootstrap
implementation are reused; `lob.core_study` and its historical results remain intact.

**LIMITATION:** the checked-in design is a CPU smoke study, with 1,024 steps per
model. It cannot establish convergence, reliable tail risk, algorithm superiority,
state-of-the-art execution, historical counterfactual fills or live alpha. Three
training seeds and eight market seeds provide coarse uncertainty estimates.
Selecting `evidence_level=research` changes the label, not the strength of evidence.

## Reproduce

From a clean checkout with the RL extra installed:

```sh
python -m pip install -e '.[rl,dev]'
python -m lob.policy_study register --config configs/v03-policy-study.json --out results/v03/policy-smoke
python -m lob.policy_study train --out results/v03/policy-smoke
python -m lob.policy_study evaluate --out results/v03/policy-smoke
python -m lob.policy_study verify --out results/v03/policy-smoke
python -m pytest tests/test_policy_study.py
```

Choose a fresh output directory for a new run. Registration writes the entire
resolved design, thresholds, model hyperparameters, source snapshot and hashes,
git commit and dirty-tree indicator, runtime/package versions and metric definitions
**before fitting**.
Training and evaluation refuse a changed source snapshot. Use the recorded commit
to reproduce an old run; verification can inspect sealed artifacts with later code.
The seal detects accidental changes; it is not an externally timestamped registry
or a cryptographic signature establishing preregistration priority.

The supplied study trains 18 models (two algorithms × three arms × three training
seeds) and evaluates 576 episodes (three regimes × eight market seeds × 24 policy
instances). Every final fixed-budget model is retained. Evaluation is deterministic.
There is no selection among seeds, checkpoints, arms, market seeds or regimes.
Failed/interrupted attempts cannot be overwritten or silently retried. Training
aborts on an INVALID economic terminal state. Evaluation exceptions and INVALID
episodes are retained as rows; affected statistical comparisons are withheld.

The three regimes are original, shifted and stress. The market overlays in the
registered JSON define them exactly. Policies train only on the original regime.
**LIMITATION:** this protocol does not claim any historically calibrated regime is
valid. It explicitly reports calibrated-regime evaluation as unavailable. Applying
a failed calibration and calling it validated would not strengthen the study.

## Environment audit

| Component | Definition and audit finding |
| --- | --- |
| Observation | Float32 vector of length 24: five bids and asks, each with midpoint-relative price divided by 10 ticks and volume divided by 500; microprice minus midpoint in ticks; top-five imbalance; remaining parent fraction; remaining horizon fraction. |
| Normalization | Fixed constants only. No fitted scaler, future statistics, train/test pooled normalization or running evaluation updates. Values need not be bounded by one. Missing depth uses zero quantity and an artificial relative price marker, not a claimed actual quote. |
| Available information | Current simulated book and own inventory/time. No market seed, future arrivals, future prices, episode return or terminal mark is given to the policy. The simulator exposes instantaneous book observations; feed delay is not modeled. Parent side is fixed by configuration and not an observation feature. |
| Action | Discrete(5): wait/cancel, join own-side touch, improve one tick when feasible, marketable limit at opposite touch, market order at 1.5× nominal child size. Nominal child size depends on parent quantity and decision count. |
| Cancellation and latency | Each decision requests cancellation of old children. Outstanding reservations remain until acknowledgment. Message latency is configured base plus exponentially distributed jitter; cancellation can race fills. This is a stylized transport model, not measured exchange latency. |
| Reward | Negative side-adjusted, parent-notional-weighted execution shortfall in bps; actual maker/taker/per-share fees; actual late fills/fees; terminal residual walk-the-book cost/fees; an extra completion objective penalty proportional to residual quantity. Training records these components separately. |
| Terminal inventory | Cancellation/settlement follows the final decision or completion. Actual late fills count. Unfilled parent inventory is hypothetically valued against terminal executable depth; valuation is not an actual fill. Insufficient depth or incomplete settlement makes the economic outcome INVALID and withholds the endpoint. |
| Constraints | Shared execution-risk layer enforces parent quantity and reservations, configured order/position/notional/loss bounds, lot alignment and kill switch. The parent cannot overfill. |
| Horizon | `ceil(horizon/decision_dt)` decisions; the final step is shortened to end at the horizon. Settlement can extend elapsed exchange time, which is separately reported. |
| Fees | Maker and taker basis points and optional currency-per-share fees use the common ledger. Smoke uses a 1 bp taker fee and zero maker fee. |
| Impact | Orders consume or add simulated FIFO liquidity and change the endogenous book. Background Poisson flow and replenishment provide stylized feedback. There is no independently validated permanent-impact model. Gross effective shortfall is an impact-related proxy that also includes spread, timing and exogenous price changes. |
| Randomization | Stable Baselines3 seeds initialize policy/optimizer and Gym episode RNG. Training markets are drawn from `[1000000,2000000)`. Evaluation and AC-identification seed lists are disjoint from that range and each other; synthetic VWAP auxiliary seeds are checked for overlap. NumPy/PyTorch CPU operations use one thread and deterministic algorithms. |
| Pairing | Same market seeds share independent exogenous simulator random streams. Actions alter realized books, so pairing never means identical endogenous states or identical passive fill opportunities. |

Tests check shape/action contracts, inventory reservations, fee/reward accounting,
terminal reward ablation, current-state observation masking, deterministic seed
randomization, disjoint seed domains, real PPO/DQN fitting, DQN reproducibility and
save/load behavior. The existing execution, settlement and risk tests remain the
primary simulator integrity suite.

## Controls and finite ablations

TWAP, synthetic-volume-profile VWAP, POV, Almgren–Chriss, the existing
passive-when-on-schedule heuristic, and random actions run on every regime/market
seed. They use the same economic report, costs, market mechanics and parent order.
The synthetic VWAP volume curve is estimated on five separate paths at
`market_seed + 1000 + day`, not on the evaluated path. It uses the known generator
of each regime; this is favorable information for the control, disclosed here.

Before opening evaluated paths, AC's depth slope and arithmetic volatility are
estimated on the separate identification seeds for each known regime using
`controls.estimate_ac_parameters`. The existing PASS gate requires positive slope,
nonnegative spread intercept, R² ≥ 0.2 and probe coverage ≥ 0.8. Passing fits use
`kappa × horizon = 1` when volatility is positive; zero volatility has its usual
risk-neutral degeneracy. Failed fits remain in `evaluation_lock.json`, with their
FAIL status, and the prespecified ResearchConfig AC parameters are used. Neither
fallback nor risk choice is optimized against evaluation. This local executable
book approximation is not evidence that the simulator satisfies the AC process.

The finite arms are `main`, `no_book` (mask the first 22 features, retain inventory
and time), and `no_terminal` (remove only the extra completion penalty, retaining
terminal economic liquidation cost/fees). Both algorithms retrain under each arm
with the same independent training seeds. Training and evaluation use the same
mask. Other reward components, risk coefficients and decision frequency can be
configured in a separately registered study but are not claimed as ablations in
this design.

## Endpoints and inference

The economic endpoint is fee-inclusive `net_effective_bps`, including hypothetical
residual liquidation but excluding the additional completion training penalty.
Lower is cheaper. Tables report mean, median, sample standard deviation, empirical
p95, worst cost, mean beyond p95, completion rate, fill fraction, terminal residual
quantity and gross effective shortfall as an explicitly noncausal impact proxy.
Conditional descriptive results exclude invalid economics and label that exclusion;
they never turn a missing terminal valuation into zero. Summary confidence
intervals are withheld when the planned grid is incomplete.
Economically defined WARNING outcomes, such as risk-rejected children, remain in
cost estimates and receive separate warning counts; adverse outcomes are not dropped.
Unregistered seeds, regimes, agents or arms are rejected before summarization;
missing classical market cells withhold their intervals just as missing learned
training/market cells do.

Learned-policy intervals resample training seeds and market seeds independently,
retaining both variance axes. Classical-control intervals resample markets.
Training-seed mean costs and their sample standard deviation disclose seed
sensitivity; episode returns and optimizer metrics provide training curves without
claiming convergence.

The planned family contains 48 comparisons: main PPO/DQN versus six controls in
three regimes (36), plus each of two ablations versus its algorithm's main arm in
three regimes (12). Paired cost differences use the existing two-way crossed
bootstrap with Bonferroni intervals at `alpha = 0.05/48`, including all comparisons
regardless of direction. Ablation contrasts pair both axes. Any missing or INVALID
cell withholds the affected comparison rather than choosing complete cases after
seeing outcomes. The fixed 20,000 resamples provide only approximate percentile
intervals; multiple-comparison correction does not cure small seed counts or an
undertrained policy. No significance or superiority label is emitted automatically.

## Artifacts and limitations

`preregistration.json`, `evaluation_lock.json`, `result.json` and the final manifest
are compact evidence. Model ZIPs, per-model training traces, episode journals and
source snapshots remain under ignored `results/v03/` by default. The manifest
records their hashes for separate archival. Portable metadata omits the local
Python executable path. Never commit restricted exchange data or treat synthetic
trajectories as historical data.

**LIMITATION:** this is a young research implementation with limited independent
review. No historical L2 counterfactual fills, hidden-liquidity reconstruction,
production latency or live alpha are established. Losses to any classical control,
failed AC fits, incomplete execution, unstable training seeds and stress failures
are legitimate results and must remain visible.
