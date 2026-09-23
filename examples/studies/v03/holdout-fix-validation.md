# Calibration holdout-isolation post-fix evidence

This evidence was regenerated after the v0.3 holdout-isolation hardening.

## Code under validation

- Git commit: `fec1525aebe9a6a3bd7715f672ad1ae3b47bd7a3`
- `lob/generalization.py` SHA-256:
  `2c93569b9d792999a541288fb5605a9d5f3a767098b021b78a5b9c06e5fadc46`

## What changed

The hardening ensures that:

- candidate selection uses only train/validation observations;
- the final fitted model uses only the registered selection period;
- the internal holdout is materialized only after the selection seal exists;
- `selection.json` hashes the selection period rather than the complete
  development dataset;
- changing internal-holdout values cannot change validation, model,
  selection, or selection-seal artifacts;
- changing external-holdout values cannot change the selected model;
- the first return of a feature-CSV partition cannot carry information from
  outside the registered partition.

## Result

The regenerated synthetic calibration preserves the previous scientific
conclusion:

- selected candidate: `spread_markov/expanding`
- internal status: `WARNING`
- external status: `FAIL`
- overall status: `FAIL`
- external used for selection: `false`
- real-market generalization: `NOT_ESTABLISHED`

The accompanying smoke study also remains `COMPLETE` while retaining the
calibration `FAIL`.

These artifacts are reproducibility and regression evidence. They are not
independent scientific validation and do not establish real-market
generalization, historical fill validity, profitability, or production-HFT
performance.

Earlier v0.3 evidence is intentionally retained unchanged.
