# v0.6 calibration v3, identifiability, ensemble and regime conditioning (M6–M8, M12)

## Model family

The model is `lob.sim_v2.ExtendedSimulator`, unchanged from v0.5, with every v0.5 flow
extension available. The empirical event and addition size tables are fixed from development
data (the v0.5 rule).

**Protocol amendment 1.** With those tables active, `limit_qty_mean` and `market_qty_mean`
have no effect on the simulator. v0.5's search varied them anyway. Searching them would
manufacture non-identifiability, so the search runs in a **14-dimensional** transformed unit
box:
- `target_level_vol`, `limit_rate`, `market_rate`, `cancel_rate`, `offset_p`, `resilience`;
- `inside_spread_prob`, `cancel_depth_exponent`, `imbalance_beta`;
- `regime_multiplier`, `regime_switch_rate`, `regime_high_share`;
- Hawkes branching ratio, `hawkes_decay`.

Bounds are in `lob.v06.calibration.PARAMETERS`.

**Protocol amendment 2.** A path ends deterministically after 500 engine events per simulated
second, which is 5.6× the v0.5 selected model. The candidate becomes a retained failure with
objective +∞. Development timing had shown random candidates running at about 100× the
historical activity.

**Objective.** The objective is the mean over the nine families of family error ÷ development
scale ([v06-realism.md](v06-realism.md)). It is a labelled optimization target, not a
validity score.

## Registered search and selection

| Step | Data | Result |
|---|---|---|
| Search: 8 seeded starts × (96 global + 3 × 64 refinements) = 2,304 candidates, 3 seeds × 1,800 s | development (ETH 2020-04-01) | 468 retained event-cap failures; best objective 2.20 |
| Re-score the 64 best on 5 fresh seeds × 3,600 s; 32 advance | development | best re-score 2.17 |
| Score the 32 candidates and the v0.5 control on 5 selection seeds × 3,600 s | selection (ETH 2020-05-01) | selected `c07-round3-009`: **2.44**; v0.5 control **3.82** |
| Near-optimal region: ≤ min + max(0.10·min, 2·SE), with SE = 0.030 | selection | threshold 2.69; **4** candidates |
| Materially distinct set: greedy, L∞ ≥ 0.25 in the unit box | selection | **2** vectors (L∞ = 0.66). These form the ensemble. |

The selection was sealed in the ledger (`m6-calibration-selection`) before any holdout access.

**Real-versus-real yardstick.** The development and selection days score **1.25** against
each other on the same objective. On the selection day, the selected v3 model is about twice
as far from history as the next real month is.

**Bound contacts.** These are reported, not hidden:
- `offset_p` sits at its lower bound (0.05) in both ensemble members.
- `resilience` is near its lower bound in the selected model (0.0010).
- `regime_multiplier` reaches its upper bound in the near-optimal region.

## Identifiability (M7; H4 **ESTABLISHED**)

The two materially distinct vectors score 2.44 and 2.59 on the selection day. Both fall within
the preregistered near-optimal tolerance (threshold 2.69 = best + 10%). They are not
statistically equal fits: with common selection seeds, the second is worse by 0.15 (paired SD
0.075 over 5 seeds, about 4.6 SE). The 10% tolerance, not seed noise (2 SE = 0.06), sets the
region.

The count of two is bounded by the frozen pipeline:
- 2,304 search candidates;
- the top 64 re-scored;
- the top 32 scored on the selection day;
- 4 near-optimal;
- 2 materially distinct (L∞ ≥ 0.25).

Three of the four near-optimal candidates come from the same search start and lie within
L∞ 0.12–0.19 of each other. The ensemble limit of 8 was not binding. The next candidate
(2.701) misses the threshold by 0.013. Two members is therefore not evidence that only two
plausible configurations exist.

The two vectors encode different mechanisms:

| Parameter | Selected | Member 2 |
|---|---|---|
| Hawkes branching ratio | 0.04 | 0.66 |
| Hawkes decay (1/s) | 1.95 | 0.26 |
| inside-spread placement probability | 0.008 | 0.48 |
| high-activity regime share | 0.12 | 0.41 |
| regime switch rate (1/s) | 0.038 | 0.20 |
| target level volume (lots) | 26 | 90 |
| market rate (1/s per side) | 0.142 | 0.057 |

Diagnostics run on development data with the fit seeds (`results/v06/m7/identifiability`):

- **Profiles.** Each parameter is fixed on a 5-point grid and the others are re-optimized
  with 24 draws.
  - Strongly constrained: `cancel_rate` (range 2.70), `market_rate` (2.48), `offset_p`
    (1.44) and `limit_rate` (1.31).
  - Weakly constrained: `resilience` (0.22), `target_level_vol` (0.25), `hawkes_decay` (0.27)
    and `regime_multiplier` (0.29). Their profile ranges are only 3–4 × the evaluation-noise
    SE (0.065).
  - No profile is flat *within* noise.
  - 593 of the 1,680 profile draws hit the event cap. A missing grid value means every draw
    at that point did.
- **Local sensitivity** (±0.1 unit steps, common seeds).
  - `offset_p` dominates book state, returns and resilience.
  - `cancel_rate` dominates the tail and temporal families.
  - `limit_rate` dominates event activity and sizes.
  - `market_rate` dominates the dependence and event-process families.

  Caveat: the family-level matrix is 9 × 14 and has rank at most 9. Without the missing
  `cancel_depth_exponent` column (its downward step hit the event cap), it is 9 × 13. Its
  singular values are 13.1, 4.8, 3.1, 1.8, 1.4, 0.86, 0.63, 0.08 and 0.05, so the effective
  rank is about 5–7 at 10%, 5% and 1% of the largest value. The near-zero directions are
  partly structural and are not reported as market findings. They mix:
  - regime switching, regime multiplier and depth target;
  - imbalance response, Hawkes branching and regime share;
  - Hawkes decay and regime parameters.

  Several selected coordinates sit at or near bounds, so those steps are one-sided:
  - `offset_p` 0.00, `resilience` 0.006 and `inside_spread_prob` 0.009;
  - `imbalance_beta` 0.97, `regime_multiplier` 0.97 and Hawkes branching 0.04.

  `offset_p`'s large influence partly reflects its boundary step.
- **Morris elementary effects** (12 trajectories) are reported as `EXPLORATORY`.

**Interpretation.** At the resolution of the sealed objective and seeds, the observable
statistics do not pin down the excitation and regime mechanisms. Calibration is not uniquely
identified. No claim of full parameter identification is made. This is a statement about the
calibration problem, not about real market mechanisms.

## Ensemble (M8)

The ensemble consists of the two materially distinct members. The selected model comes
first. For each member, `results/v06/m6/select` keeps:
- the configuration hash;
- the parameter vector;
- the development re-score and selection objectives;
- the family errors;
- the seeds.

An ensemble of two gives only coarse coverage of calibration uncertainty. Structural
interventions (M9) add synthetic worlds for sensitivity analyses.

## Regime-conditioned calibration (M12)

**Setup.**
- Labels: the v0.5 M6 development thresholds, unchanged, on 5-minute blocks, volatility
  dimension.
- Search: 4 starts and 1,152 candidates per regime, with the same re-score and selection
  pipeline restricted to each regime's blocks.

**Registered models** (sealed as `m12-regime-calibration`):
- High-volatility model: selection objective 2.37, chosen on 264 selection blocks.
- Low-volatility model: selection objective 2.45, chosen on **only 23** selection blocks,
  because May 2020 was mostly high-volatility by April's thresholds.

**Fresh ETH 2020-09-01 (H9, H10):**

| Regime blocks | Regime model − global (H9) | Other regime's model − global (H10) |
|---|---|---|
| high volatility (240) | **−0.32** [−0.44, −0.18], better | low model: **+1.51** [+1.32, +1.71], worse than δ = 0.244 |
| low volatility (47) | **+0.96** [+0.78, +1.14], worse | high model: +0.23 [+0.14, +0.34], not noninferior |

H9 is **FAILED** by the preregistered rule: one regime's lower bound exceeds 0. H10 is
**FAILED**. Regime conditioning helped only where enough selection data existed, and neither
regime model transfers outside its regime.
