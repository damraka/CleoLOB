# v0.8 counterfactual historical execution bounds

Aggregate L2 cannot reveal where a hypothetical order would have queued. `lob/v08/counterfactual.py` therefore
replays an unchanged policy against recorded frames under explicit assumptions. Its outputs are labelled as
follows:
- every replay number is HYPOTHETICAL;
- every bound is BOUNDED and ASSUMPTION_DEPENDENT.

None of these numbers is an actual historical fill.

## Assumption grid (72 registered combinations)

**Queue rule.** How cancellations at the order's price are treated relative to `ahead`, the volume queued in
front of the order:
- optimistic: all of them are credited to `ahead`;
- neutral: a proportional share is credited;
- conservative: none is credited.

**Latency.** 0 or 0.1 s, applied to both observation and order arrival.

**Hidden fraction.** 0 or 0.5 of the visible size, assumed to sit ahead of the order. This applies under the
conservative rule.

**Impact.** What happens to the liquidity the strategy takes:
- none;
- transient, with a 10 s half-life;
- persistent.

**Permanent shift.** 0 or 0.05 ticks per lot executed.

## Synthetic validation (section 38)

Each simulated world is run twice with the same seed:
1. **With the strategy.** This is the ground truth, including the strategy's impact and feedback.
2. **Without the strategy.** This is the "historical" tape that the bounds see.

Two coverages are reported:
- **Per-path coverage:** the fraction of true outcomes that fall inside the bounds.
- **Expected-value coverage:** bootstrap intervals of mean(truth − lower) and mean(upper − truth).

## Pair bounds

For a pair of policies, assumptions are shared between the two, since both face the same market. The pair
bound is the range, over assumptions, of the mean paired difference. Each end of that range is widened by
bootstrap sampling uncertainty.

A pair is determinate only if the whole interval excludes zero. Otherwise it is INCONCLUSIVE.

Results are in [v08-final-report.md](v08-final-report.md).
