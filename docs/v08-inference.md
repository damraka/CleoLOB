# v0.8 inference, identifiability and recovery

## Likelihood-free inference (`lob/v08/inference/abc.py`)

**Priors.** Priors are boxes over flattened generator parameters, uniform or log-uniform. Proposals move in a
logit-transformed space, so kernels never leave the support.

**Methods.**
- ABC rejection.
- Adaptive ABC-SMC: the tolerance is a quantile of the previous distances, the kernel uses twice the weighted
  covariance, and importance weights are prior / mixture kernel.
- Gaussian synthetic (surrogate) likelihood, with a seeded random-walk Metropolis sampler.
- Posterior importance reweighting with a forgetting exponent.
- Local-linear regression adjustment.

**Distance scaling.** Summaries are standardized by **replicate noise at a reference parameter**, not by
prior-predictive spread. A development pilot showed that prior-predictive scaling down-weights the
informative summaries.

**Determinism.** Simulations are seed-addressable. Candidate lists are generated before evaluation, so a
process pool gives identical results.

## Hierarchical calibration (`hierarchical.py`)

- DerSimonian–Laird random-effects pooling, with shrinkage of group estimates.
- Recursive pooling across the levels global, instrument, venue, regime, day.
- A node whose source posterior did not contract is flagged NOT_IDENTIFIED.

## Adaptive calibration (`adaptive.py`)

Schemes: frozen, rolling, expanding, forgetting (parameter smoothing), regime-conditioned and
change-triggered (CUSUM on the predictive loss).

Windows are scored one step ahead. A window boundary never carries a predictive term.

## Identifiability v3 (`identifiability.py`)

**Local analysis.**
- Finite-difference Jacobians with common random numbers.
- A Gaussian synthetic-likelihood Fisher approximation, from which: effective rank, participation ratio,
  sloppiness and condition number.
- These results are local and approximate, and are labelled as such.

**Global and posterior analysis.**
- Binning first-order indices over the prior.
- A profile-distance proxy from ABC populations.
- Posterior contraction, correlation and mode counts.
- Structural flags per family.

**Verdicts per parameter:**
- PRACTICALLY_IDENTIFIED
- WEAKLY_IDENTIFIED
- NOT_IDENTIFIED

## Synthetic recovery (section 14) and H4

`cleo inference-v08 recovery`:
- Registered truths come from the recovery priors for Poisson, Hawkes and queue-reactive parameter subsets.
- ABC-SMC is the primary method. ABC rejection and the analytic Poisson MLE are secondary.
- Outputs: 90% credible-interval coverage with Wilson intervals, median relative error, SBC rank histograms
  and contraction.
- Model-family recovery is reported as a confusion matrix (held-out log score).
- The misspecification benchmark (truth family differs from the fitted family) is part of
  `calibration-utility-v08`; see [v08-calibration.md](v08-calibration.md).

Results and statuses are reported in [v08-final-report.md](v08-final-report.md).
