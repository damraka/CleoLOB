# Research methodology and limitations

The platform should make a strategy falsifiable. A positive synthetic result is
not evidence of live profitability. A failed execution strategy, an unpriced
terminal position or an invalid dataset is a result to preserve.

Claims distinguish **IMPLEMENTED** functionality, **TESTED** behavior,
**EMPIRICALLY OBSERVED** study outcomes, **FAILED** gates or experiments,
**LIMITATION** of inference and **PLANNED** work. Tests and hash verification do
not establish market realism. This young project has limited independent external
validation; it has no demonstrated live alpha or production execution capability.

## Mechanics and units

Synthetic prices are positive integer ticks. A tick has configured currency
value; quantities are integer shares in multiples of a lot. FIFO is determined
by exchange arrival and queue sequence. The event clock uses float seconds;
canonical historical events use exact integer nanoseconds. Direct synthetic
book calls and order-state transitions reject backward timestamps.

Limit and market orders, GTC/IOC/FOK/GTD, post-only, partial fills, cancellations,
modifications and conditional cancel/replace have explicit lifecycle behavior.
Market GTC requests normalize to IOC. FOK checks executable depth before any
fill. Post-only rejects crossing arrivals. GTD expiry removes resting orders
before other matching at its deadline. Size reductions keep FIFO; increases and
price changes lose priority. Synthetic MODIFY quantity is **total lifetime
quantity including existing fills**; canonical historical MODIFY replaces
**remaining** quantity. Adapters must respect this distinction.

Exchange events tie-break by enqueue sequence, with GTD expiry effective before
matching at the same timestamp. A cancel reaching the exchange before its
original order is not accepted and cannot magically cancel a future arrival.
An already filled original causes conditional replacement rejection. Old
orders remain reserved until their terminal state, not when a cancel is sent.

Self-trades are not prevented at the book layer; aggregate background owner ZI
can appear on both sides. The strategy accounting ledger rejects same-owner
dual-side fills. The current execution agents trade one side per parent. A
multi-sided market-making agent requires explicit self-trade prevention first.

## Market model and randomness

The default market is a zero-intelligence Poisson model. Simulation-based fitting
is implemented separately in `zi_calibration.py`, but **FAILED** external gates
leave its real-market fidelity unvalidated; fitting is not validation.
Limit/market arrivals have persistent exponential clocks; background resting
orders have individual exponential cancellation lifetimes. State-dependent
resilience accepts proposals based on missing volume in price slots near the
opposite touch. Consuming a whole price level leaves a measurable deficit.
This is an assumption about replenishment, not evidence of real market impact.

NumPy SeedSequence derives five streams: regular order marks (size/offset),
arrival times, latency, cancellation and resilience. Explicit stream seed
overrides are available on `SimConfig`; registered research derives them from
each frozen episode seed and records their mapping in audit logs. Random-agent
actions use the separate legacy `episode_seed + 7` RNG. VWAP's synthetic volume
forecast uses five shifted seeds; it is not a fitted historical volume curve.

With no intervening actions, changing `step(dt)` partitions preserves the event
path. Agents share exogenous streams in paired comparisons, but their actions
alter prices, depth, cancellation outcomes and refill behavior. It is incorrect
to call their realized books identical. Strategy IDs are separate from negative
background IDs. Changing the engine's sampling algorithm changes the historical
seed-to-path mapping, so old benchmark numbers are not current validation.

## Accounting and terminal valuation

An average-cost ledger tracks cash, signed inventory, realized/unrealized PnL
and fees. Initial sell inventory is endowed at arrival price, not credited as
trading profit. The identity `equity - initial_equity = realized + unrealized -
fees` is checked. Market orders consume actual depth; recorded fills never
exceed available liquidity or parent reservations.

Filled shortfall is signed execution cost versus arrival midpoint, plus actual
fees. Economic effective shortfall adds hypothetical terminal liquidation cost
and fees to the unfilled portion. These estimates are kept separate from cash
and actual fills. If visible depth cannot price all leftover shares, effective
economic metrics are **null** and the episode is **INVALID**. Partial-depth
VWAP is never extrapolated to unavailable shares.

The RL objective adds an explicit 25 bps default noncompletion penalty weighted
by leftover fraction. `optimization_cost_bps` includes this penalty;
`effective_bps` does not. Reward terms expose filled shortfall, actual fees,
terminal shortfall, hypothetical terminal fees and completion penalty. When
terminal depth is unknown the finite reward remains a diagnostic, not a valid
economic result. Decision logs capture the observation **before** the action,
then record the next observation separately.

Outstanding orders at the horizon enter bounded post-decision settlement. The
exchange processes delayed messages, cancellation retries and late fills while
the ledger continues reconciling cash, inventory and fees. Reports separate
decision and settlement durations; unresolved leaves make economic results
INVALID/null and RL episodes truncate. Hypothetical leftover valuation occurs
after successful settlement, and remains an estimate rather than an actual
liquidation fill. No new strategy children are sent during settlement.

## Statistics and evidence

Seeds, not fills, are the independent sampling unit. Summaries include mean,
median, standard deviation, standard error, quantiles and worst/best outcome.
Percentile bootstrap confidence intervals describe the mean; paired intervals
describe candidate-minus-reference differences. Negative differences favor the
candidate. One seed produces a point interval, which is not evidence of precision.

Exact two-sided sign tests exclude ties and address the direction of paired
outcomes, not a hypothesis about the mean difference. Reports show raw and
adjusted p-values. Holm is the default family-wise correction; Bonferroni and
Benjamini-Hochberg FDR are available. BH requires its usual independence or
positive-dependence assumptions. Ordinary single-run bootstrap intervals remain
marginal, not simultaneous adjusted confidence intervals. Registered core/v0.3
policy studies separately report their declared family-level Bonferroni intervals.
Methods are consistent with the
[statsmodels correction definitions](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)
and the [SciPy exact binomial-test definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

The registered comparison requires every planned agent/seed outcome. Duplicate
episodes, missing pairs, NaNs and infinities fail validation. If any planned
episode is failed, unpriced or not run, comparative inference is withheld for
the whole run; it is never rescued by dropping unfavorable seeds. Failure rows
remain in the artifact. Corrections apply within the declared run; selecting
the best of many separate reports requires a larger declared hypothesis family.

`design` computes the finite Cartesian size first and either enumerates a small
space or uniformly samples unique indices without allocating the entire space.
Its coverage is a fraction of specified discrete combinations, not assurance
of coverage of the continuous market parameter space. It does not launch a
sweep, perform optimization, or claim sensitivity estimates.

## Generalization and remaining limits

PPO training and validation sample disjoint explicit seed domains. A validation
set used during training is not a final test set. Repeated synthetic seeds
and the same simulator family still share modeling assumptions. Existing
checkpoints trained under previous mechanics have an environment shift; their
shape compatibility is not evidence of comparability.

**EMPIRICALLY OBSERVED:** The preserved September 22 core study completed 20 PPO
fits and 704 final episodes. Its primary economic comparison and actual fill
fraction are recorded in
[`ppo-final-20260922/result.json`](../examples/studies/core/ppo-final-20260922/result.json).
Its synthetic result does not overcome the simulator's failed historical
calibration gates. It is incorrect to describe PPO as an untested interface or
to promote that bounded study into general algorithm superiority.

**IMPLEMENTED / TESTED:** The separate v0.3 policy protocol trains PPO and DQN for
the existing five-action environment, using three independent training seeds and
fixed disjoint evaluation markets. It retains six controls, original/shifted/stress
regimes and two retrained ablations: masking book observations and removing only
the additional terminal completion penalty. Net economic cost always retains
actual fees and hypothetical residual valuation. Crossed resampling preserves
both training-seed and market-seed variability; all 48 planned contrasts belong
to one Bonferroni family. Missing/INVALID cells withhold affected inference.
See [the full environment audit](rl-v03.md) and
[measured v0.3 outcomes](v03-final-report.md). SAC and continuous-action policies
remain unimplemented. Short smoke training does not establish convergence.

**EMPIRICALLY OBSERVED / FAILED:** The v0.3 run completed all 18 fits (18,432
steps) and 576 evaluations, with zero INVALID/WARNING economic states. Every
one of 48 corrected cost intervals includes zero. Main PPO completed every
evaluated parent; main DQN completion was 0%, 4.17% and 4.17% across original,
shifted and stress regimes. Mean DQN residuals were 272.125, 205.375 and 108.0833
of 300 units. Economic validity means the residual can be valued; it does not
turn hypothetical liquidation into fills or certify execution completion.
PPO's stress mean cost (3.7378 bps) exceeded TWAP (2.9858) and AC (3.4108);
corrected uncertainty does not establish a difference. These observed failures
remain in the result rather than prompting evaluation-seed selection or retraining.

**IMPLEMENTED / TESTED:** `generalization.py` evaluates IID joint resampling and
a train-fitted two-state spread Markov model over both expanding and rolling
chronological folds. Every observable has a declared definition, units, finite
history requirement, missing-data behavior and loss weight. The protocol fixes
diagnostic gates before evaluation; validation selects family/window, then a
seal freezes that selection before internal and external scoring. Those later
scores cannot feed back into selection. Failure attribution is a heuristic,
not causal identification. See [calibration details](calibration-v03.md).

**EMPIRICALLY OBSERVED / FAILED:** Validation selected `spread_markov/expanding`
at mean loss 0.163811 versus 0.487916 for `iid_joint/expanding`. The selected
model then returned internal WARNING (0.290986) and external FAIL (0.937720).
Its sealed parameters and thresholds remained unchanged. Lower validation loss
does not override a failed external score or establish a realistic simulator.

**FAILED / LIMITATION:** April/May 2020 observable and July–September 2026 simulator
studies retain their WARNING/FAIL outcomes. Consumed holdouts cannot become fresh
holdouts for later changes. The new two-state fixture exercises chronological
software behavior; repeated synthetic fixture partitions are not independent,
previously uninspected real-market evidence. A failed simulator calibration is
not used to label an RL evaluation regime historically validated.

Market-data dissemination delay is absent; agents observe current exchange state.
Message latency models only fixed plus exponential delay. No opening/closing
auction, halt/price-limit model in the synthetic exchange, queue-jump rule, hidden
liquidity, venue fee schedules, corporate actions, operational
outage model, Hawkes process, calibrated volatility or calibrated impact is
implemented in the FIFO exchange. A separate funded linear portfolio module now
models multi-currency exposure, reservations, margin capacity, leverage, loss
limits and scenario/empirical tail risk. Automatic flattening, borrow and
nonlinear derivative settlement remain outside its scope.

Historical replay reconstructs a supplied feed, not the counterfactual future
after an agent's order changes it. Real data can have survivorship and selection
bias, omissions, venue-specific ordering and licensing restrictions. The included
synthetic feeds verify canonical mechanics only. A separate Tardis aggregate
L2 adapter has now matched two full days of real published snapshots; see
[the assessment](../examples/studies/historical/README.md). This adds ingestion
evidence without counterfactual fills. The subsequent frozen empirical observable
calibration, purged walk-forward diagnostics and registered execution stresses are
documented in [validation and risk](validation-and-risk.md). Later-date observable
diagnostics failed; all stress-family inference was withheld where terminal depth
was unavailable. Instrument adapters, latent flow/impact calibration and broader
independent datasets remain necessary before any robust execution claim. Existing
capacity and bounded ablation studies answer their declared questions only.

**IMPLEMENTED / TESTED:** MBO source identities follow a separate reconstruction
path. Exact same-price queue measurements require source-complete identities
and snapshot priority. Recorded maker executions establish observed fill times;
unlinked trade prints do not establish an order fill. Watch cohorts preserve
submission origins, cancellation and censoring. Their complete-case frequencies
can suffer selection/censoring bias and are not counterfactual fill probabilities.
No actual historical MBO dataset has been validated. Aggregate L2 refuses these
identity-dependent metrics; hidden liquidity is unavailable in both paths.

**IMPLEMENTED / TESTED:** The performance suite distinguishes research workload
throughput, full synthetic-episode throughput and unmeasured production latency.
It records warmup/repeats, book depth, p50/p95/p99, OS/Python/CPU/package versions
and source identity. Python allocations are measured separately from timed calls
and are not process RSS. Local timings include scheduler/timer overhead and have
no worst-case or colocation guarantee. Profiling alone is not an achieved speedup.

The [reproduction guide](reproduction.md) specifies installation, principal study
commands and artifact verification. Retained compact results identify their
dataset, period, symbol, configuration, seeds and runtime. Checkpoints, episode
journals, raw market data and large generated reports stay outside Git.

No broker or live-trading connection is included or authorized.
