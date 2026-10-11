# v0.8 uncertainty: decomposition, model averaging, OOD and conformal

## Uncertainty decomposition (M18; H5)

`cleo uncertainty-study-v08` varies one world at a time and holds seeds in common across all worlds.

**Base worlds:**
- the registered selection posterior draws (model family × parameter sample);
- Poisson refits on four disjoint blocks of development windows (calibration-window source).

**Factors.** Each base world gets three variants, with a balanced assignment of:
- queue rule: uniform, front, back;
- latency: 0, 50 or 250 ms;
- hidden-liquidity knob: 0 or 0.5.

**Variance split.** For each policy and each registered pair, the variance is split into:
- aleatoric (seed) and epistemic parts;
- first-order shares of the epistemic part for model family, parameter sample, calibration window, queue,
  latency and hidden liquidity.

**Limits of the design.**
- The shares describe this registered design only.
- Regime is not varied in this design, so its share is NOT_EVALUABLE here.

## Model averaging (M17; H3)

`cleo model-averaging-v08` reports:
- posterior model probabilities (BIC), Akaike weights and stacking weights;
- regime-conditioned stacking weights;
- dynamic weight paths;
- mixture posterior predictive sampling.

Model uncertainty counts as material when no family holds at least 0.9 of the weight. When it is material,
no single generator is selected.

## OOD detection (M19)

`cleo ood-study-v08` evaluates six detectors:
- support;
- Mahalanobis (shrinkage);
- kNN;
- conformal p-value;
- ensemble disagreement of the development fits;
- posterior predictive surprise.

**Procedure.**
- Thresholds are set at a 5% false-positive rate on a separate in-distribution calibration set.
- Detection rates are reported for all 13 registered stress scenarios.
- Decision degradation is the Spearman correlation between the OOD score and selector regret.

## Abstention and fallback (M27; H11, H12)

`cleo policy-abstention-v08` trains a regime selector on development cases. It abstains when either score
exceeds its threshold:
- OOD (Mahalanobis);
- model disagreement.

On abstention it falls back to the registered safe baseline (TWAP).

**Measured on in-distribution and registered OOD cases:**
- incorrect decisions (regret > 0.5 bps);
- large degradation (regret > 2 bps);
- abstention and fallback rates.

## Conformal coverage (M20; H19)

`cleo conformal-study-v08` uses split-conformal 90% intervals for episode economic cost.

**Data.**
- Predictor: the plausible-world mean cost per policy.
- Calibration: held-out worlds, development seeds.
- Evaluation: the same worlds with confirm seeds (exchangeable), and registered stress worlds (shifted).

**Secondary variants.** Mondrian intervals (grouped by family) and rolling intervals.

**Validity.** Coverage is distribution-free only under exchangeability. Shifted worlds violate exchangeability
by design.
