# v0.8 calibration: adaptive, regime-conditioned, decision-focused

## Adaptive and regime-conditioned calibration (M14; H1, H2, H18)

`cleo adaptive-calibration-v08` runs four benchmarks.

**`drift` (H1, H18).** Four registered scenarios:
- market/add step ×2;
- Hawkes branching 0.3→0.7;
- gradual rates ×(1 + 0.08k);
- gradual calm/active mixture.

Each scenario uses 12 windows of 600 bins with a `state_dependent` template. Endpoints:
- H1: relative loss reduction, rolling (3 windows) vs frozen (first 2 windows).
- H18(b): the same comparison, after the change only.
- H18(a): a half-life fit to the mean degradation curve of a frozen window-1 fit.

**`regime` (H2).** Four persistent regime-switching truths. Regime-conditioned Poisson fits (threshold labeler
fitted on the initial windows) are compared with one global fit.

**`switch` (H3).** The generating family changes at windows 4 and 8. Components are fitted on windows 0–3, and
the best single family is chosen on windows 4–5. Dynamic model averaging (forgetting 0.99) is then compared with
that family on windows 6–11.

**`real`.** The same schemes run on the consumed ETH chronology. The evidence level is RETROSPECTIVE, capped at
EXPLORATORY.

## Calibration half-life (M7)

`cleo change-point-study-v08` fits `D(h) = D_inf (1 − exp(−h/τ))` to degradation curves of three kinds:
- observable fit;
- execution-sensitive fit (TWAP cost error against bounded historical replays);
- policy-ranking utility (Kendall tau).

A half-life is reported only when the fit is informative. Otherwise the status is NOT_ESTABLISHED and no
half-life is claimed.

## Decision-focused calibration (M21; H6) and misspecification (section 34)

`cleo calibration-utility-v08` builds (truth, candidate) pairs:
- **Truths:** the six development fits.
- **Candidates:** the six families fitted on one 10-minute frame of each truth, plus posterior-draw variants.

For every pair it measures:
- calibration losses (generic, and over execution-sensitive summaries only), scaled by the truth's own
  summary noise with a floor;
- predictive fit;
- cost and completion errors;
- Kendall tau, pairwise accuracy and top-policy retention;
- regret.

H6 is the Spearman correlation between execution-aware loss and regret.

The misspecification summary compares well-specified fits with misspecified ones on:
- regret and ranking error;
- fit degradation;
- market-rate bias;
- completion error;
- OOD response.

Results are reported in [v08-final-report.md](v08-final-report.md).
