# CleoLOB v0.8 research report: decision uncertainty

Branch `research/v0.8-decision-uncertainty`, version `0.8.0.dev0`. This report is filled from registered results
only. The full hypothesis table, run pointers and engineering record are in
[v08-final-report.md](v08-final-report.md).

## 1. Question

> Can execution decisions remain scientifically reliable when market models are imperfect, regimes drift through
> time, model/parameter/queue/latency uncertainty is explicit, execution completion is mandatory or constrained, and
> conclusions must transfer across time, instruments, venues and plausible market worlds?

## 2. Preregistration

- **Freeze.** The protocol `configs/v08/protocol.json` was frozen in an append-only ledger that pins the v0.7
  protocol and ledger head.
- **Registered before any confirmatory seed was drawn:** hypotheses H1–H21, statistical families, the generator
  gate, abstention and completion rules, the mandate (buy 40 lots over 120 s with a hard completion constraint), and
  two fresh real holdouts.
- **Designs.** Each confirmatory study ran from a sealed design that opened the `confirm` seed namespace. The
  synthetic multi-axis holdout (`holdout` namespace) could be opened exactly once, and only if a conclusion was
  carried into it.
- **Amendments.** Amendment 1 added a secondary 400-lot mandate, reported but never decisive, before any
  confirmatory run. Amendment 2 re-pointed non-hypothesis claim-registry entries after all confirmatory runs had
  sealed; it changed no rule.

## 3. What the data could and could not say

**Real data.** The fresh real data never arrived. The registered acquisition rule probes a consumed control file
first. It returned HTTP 403 for the 2021-02-01 transfer days and for the 2021-03-01 final holdout, so neither
holdout was read. Five hypotheses that depended on fresh data are NOT_AVAILABLE:
- [C08-H13] ranking transfer, NOT_AVAILABLE;
- [C08-H15] temporal transfer, NOT_AVAILABLE;
- [C08-H16] instrument transfer, NOT_AVAILABLE;
- [C08-H17] venue transfer, NOT_AVAILABLE;
- [C08-H20] final real holdout, NOT_AVAILABLE.

Retrospective analyses on consumed days are EXPLORATORY and do not stand in for them ([C08-DATA-BLOCKED]).

**Simulated worlds.** The synthetic worlds are not acceptable models of the market. Every generator family failed
the registered gate (discriminator AUC 1.0, support coverage 0.0), and the gate result is NO_ACCEPTABLE_MODEL
([C08-GATE]). The confirmatory studies still ran, as registered, on plausible worlds drawn from the posterior.
Everything below is therefore a statement about those SYNTHETIC worlds and the decision machinery. Nothing below is
a statement about real markets.

## 4. Results

### 4.1 Calibration under drift (M6, M7, M14)

- Adaptive calibration beat frozen calibration on drifting worlds ([C08-H1] ESTABLISHED; 0.061 [0.055, 0.069]).
- Regime-conditioned calibration beat global calibration ([C08-H2] ESTABLISHED; 0.018 [0.007, 0.030]).
- Dynamic model averaging beat the best single generator under model switching ([C08-H3] ESTABLISHED; 0.017
  [0.013, 0.021]).
- The post-drift gain from updating is clear, but a finite half-life was estimable in only one of the two gradual
  scenarios, so the joint hypothesis is NOT_ESTABLISHED ([C08-H18] NOT_ESTABLISHED).

### 4.2 Inference and uncertainty (M12–M18)

- Synthetic parameter recovery came close but did not meet the bar ([C08-H4] NOT_ESTABLISHED). Every parameter had
  coverage ≥ 0.83, but the Hawkes decay's Wilson lower bound (0.55) is below the registered 0.60.
- Model and parameter uncertainty is material for some execution comparisons ([C08-H5] ESTABLISHED). It explains
  55% of the TWAP vs MPC cost variance and 62% of TWAP vs POV.
- Identifiability is weak locally ([C08-ID-V3]), and posterior predictive checks fail at several horizons
  ([C08-PPC]).

### 4.3 Does better calibration mean better decisions? (M21)

No association was found ([C08-H6] NOT_ESTABLISHED): ρ −0.02 [−0.27, 0.23]. Under structural misspecification,
the predicted policy ranking agreed only weakly with the true one: Kendall τ 0.14 ([C08-MISSPEC]).

### 4.4 Execution under constraints and uncertainty (M22–M27)

- **Completion constraints.** They raised completion strongly for POV (+0.72) within the cost margin, but only by
  +0.11 for the heuristic, whose lower bound misses the registered 0.05 ([C08-H7] NOT_ESTABLISHED).
- **CVaR vs nominal selection.** The two criteria picked different schedules, with no determinate tail-regret
  difference ([C08-H8] NOT_ESTABLISHED).
- **MPC and robust MPC.** Both lowered simulated CVaR cost vs TWAP by about 1.5–1.7 bps ([C08-H9] ESTABLISHED,
  Holm over 3). The development-tuned adaptive variant did not.
- **Regime-aware selection.** It did not beat a single global policy ([C08-H10] NOT_ESTABLISHED).
- **OOD-aware abstention.** It made selection worse: more incorrect choices, and mean regret rose from 211 to 1,499
  bps ([C08-H11] FAILED).
- **Safe fallback.** It increased the rate of large degradations under OOD shifts instead of preventing them
  ([C08-H12] FAILED).

### 4.5 Bounds, conformal coverage and stress (M20, M28–M30)

- Bounded historical replays on synthetic tapes separated one registered pair, TWAP vs heuristic ([C08-H14]
  ESTABLISHED). The other four pairs stayed indeterminate, and none was determinate on the 400-lot mandate.
  Per-path coverage of the bounds varies by policy ([C08-CF-COVERAGE]).
- Split-conformal cost intervals reached 0.90 coverage on exchangeable data ([C08-H19] ESTABLISHED). They fell to
  0.51–0.59 under every registered shift.
- Stress scenarios reorder simulated policy rankings, up to τ ≈ 0 for resilience collapse ([C08-STRESS],
  [C08-STRESS-SEARCH]).
- Latency profiles move simulated mean costs by at most 0.25 bps ([C08-LATENCY]).

### 4.6 The sealed synthetic holdout (H21)

The only conclusions carried forward were H9's: MPC and robust MPC vs TWAP. Both survived the sealed synthetic
holdout of unseen ecology/endogenous worlds, regimes and seeds ([C08-H21] ESTABLISHED). The CVaR improvements were
+9.4 and +8.7 bps, with lower bounds above 4 bps. This is the strongest result of v0.8, and it is SYNTHETIC: the
holdout family is another simulator, not a market.

### 4.7 Infrastructure

Exploratory and infrastructure results:
- cross-venue routing and basket execution ([C08-PORTFOLIO], [C08-PORTFOLIO-REAL], [C08-BASKET]);
- a real-chronology half-life that is not identifiable ([C08-HALF-LIFE-REAL]);
- local research-workload timings ([C08-BENCH]).

None of them supports a market claim.

## 5. What this means

Within simulation, v0.8 supports three narrow statements:
1. Calibration that adapts to drift, regime or model class predicts better.
2. Model uncertainty is large enough to matter for some execution comparisons.
3. Model-predictive execution reduced tail cost relative to TWAP across the registered synthetic world design and
   a sealed synthetic holdout.

It also reports two clear negative results: OOD-aware abstention and safe fallback, as registered, made decisions
worse.

What v0.8 cannot say matters as much. No generator passed the gate. No fresh real data could be acquired. No
result here is evidence about real-market execution quality, and none of it is a trading or production claim.

## 6. Limits stated in advance

- **SYNTHETIC evidence only.** Every confirmatory status is SYNTHETIC or SYNTHETIC_HOLDOUT, and the generator gate
  rejected every family.
- **Aggregate L2 data.** Queue-position and hidden-liquidity effects are modelled assumptions, never observed.
- **Counterfactual replays are bounds.** They are not historical fills.
- **Retrospective analyses are EXPLORATORY.** Analyses on consumed days are capped at that level.
- **Conformal validity assumes exchangeability.** The shifted-coverage results show what happens without it.
