# Research methodology and limitations

The platform should make a strategy falsifiable. A positive synthetic result is
not evidence of live profitability. A failed execution strategy, an unpriced
terminal position or an invalid dataset is a result to preserve.

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

The market is a zero-intelligence Poisson model, **not empirically calibrated**.
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
positive-dependence assumptions. Bootstrap intervals remain marginal, not
simultaneous adjusted confidence intervals. Methods are consistent with the
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

PPO training and validation now sample disjoint explicit seed domains. Validation
is used during training, so it is not a final test set. Repeated synthetic seeds
and the same simulator family still share modeling assumptions. Existing
checkpoints trained under previous mechanics have an environment shift; their
shape compatibility is not evidence of comparability. No new PPO/SAC research
training or held-out strategy study is claimed in this batch.

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
fabricated feed verifies canonical mechanics only. A separate Tardis aggregate
L2 adapter has now matched two full days of real published snapshots; see
[the assessment](../examples/studies/historical/README.md). This adds ingestion
evidence without counterfactual fills. The subsequent frozen empirical observable
calibration, purged walk-forward diagnostics and registered execution stresses are
documented in [validation and risk](validation-and-risk.md). Later-date observable
diagnostics failed; all stress-family inference was withheld where terminal depth
was unavailable. Instrument adapters, latent flow/impact calibration, broader
independent datasets, capacity tests and ablations remain necessary before any
robust execution claim.

No broker or live-trading connection is included or authorized.
