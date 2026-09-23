# MBO post-fix validation evidence

This directory retains the original v0.3 evidence and adds a separate
post-fix validation run for the corrected MBO research semantics.

## Code under validation

- Git commit: `817961d7743176476e579010eb18e7402f00235e`
- `lob/mbo.py` SHA-256:
  `da7cddd13ef0518998e5efd2dd306584ddafd0ada9e847ec55f58fea37713bea`

The correction separates:

- cumulative execution reaching the originally submitted quantity;
- observed terminal full fill of the resting order identity;
- explicit cancellation before terminal execution;
- queue movement before versus after a priority reset caused by repricing
  or same-price size increase.

## Evidence

- `benchmark-mbo-fix/`
  - regenerated local Python research benchmark;
  - includes the MBO workload;
  - generated from a clean working tree at the commit above.

- `smoke-mbo-fix/`
  - regenerated end-to-end synthetic smoke evidence;
  - generated from a clean working tree at the commit above;
  - the calibration smoke result remains `FAIL`;
  - no failed scientific result was rewritten into a pass.

Both artifact directories pass the repository artifact verifier.

These runs are reproducibility and regression evidence. They are not
independent scientific validation, real historical MBO validation,
production-HFT latency evidence, or evidence of profitability.

The original v0.3 evidence directories are intentionally retained unchanged.
