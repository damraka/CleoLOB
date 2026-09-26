# v0.4 chronology and calibration diagnostics

The preserved v0.3 external calibration failures remain failures. v0.4 adds
separate-file holdout access, multiple registered periods, a consumed-source
registry, and observable-level diagnostics. It does not calibrate historical
counterfactual fills or establish broad real-market generalization.

## Registered multi-period workflow

```sh
python -m lob.multiperiod --config configs/v04-multiperiod-smoke.json --out results/v04/multiperiod
python -m lob.multiperiod --out results/v04/multiperiod --verify
```

The redistributable smoke design has one development period, two selection
periods, one internal test and two external periods. All are **synthetic**.
IID-joint and two-state spread-Markov candidates fit only development data;
selection minimizes mean loss across the chronological selection periods. No
refit follows selection. All gates retain the existing observable-study values.

Each historical period must be a separate canonical feature CSV. Registration
validates descriptors without opening, hashing or statting holdout files.
`selection-seal.json` is saved before the first internal/external read. Corrupt
or missing holdouts therefore leave the same model and selection artifacts and
produce `INVALID` evaluations. Model errors, failed gates and attempted accesses
remain in the new, write-once output directory.

The full source manifest and portable runtime/git provenance are recorded and
checked during execution. Verification checks byte integrity, required files,
the selection seal, model digest, plan/result binding and every period result.
A hash seal is an accidental-change detector; it is not an independent timestamp
authority or proof of researcher non-inspection.

For historical use set `kind` to `feature_csv`. Each descriptor contains:

| Field | Meaning |
|---|---|
| `id`, `role` | Unique safe identifier; development, selection, internal or external |
| `path`, `sha256` | Portable path relative to `--data-root`; canonical CSV byte hash |
| `source_sha256` | Raw source byte hash before normalization |
| `independent_period_id` | Stable venue/instrument/date identity, shared by alternate encodings |
| `start_us`, `rows` | Exact regular-grid UTC capture-clock interval |
| `inspection` | `consumed` or `fresh`; fresh is reserved for final external evaluation |
| `provenance` | Source, permission, adapter version and transformation declaration |

Canonical columns remain `timestamp_us`, `valid`, `mid_price`, `spread_bps`,
`bid_depth5`, `ask_depth5`, `imbalance5`, and `log_return`. First returns must be
NaN; invalid rows remain present and temporal measurements never bridge gaps.
Every period starts after the preceding period plus the preregistered embargo.
The feature schema observes top-five depth only; top-level depth, trade/update
intensity and order-event intensities require additional source fields.

Include known inspection history in `consumed_periods`, beginning with
[`v04-consumed-data.json`](../configs/v04-consumed-data.json), and carry forward
each completed or failed run's access records and `consumption-registry.json`.
Raw-source and stable-period aliases cannot turn consumed data into fresh data.
Duplicating a fresh source under another file or period name is rejected.
Independent-period counts exclude repeated source identities. User declarations
remain necessary for inspection outside this repository.

## Failure attribution

`lob.calibration_diagnostics.diagnose_period` compares a target with training
without fitting parameters or changing the original assessment. It reports:

- Quantile-grid distribution distance divided by training IQR/std scale.
- Coverage of the training central 95% range; 1/5/25/50/75/95/99% quantile errors.
- Standardized mean drift, variances and the fraction outside training's 98% range.
- Lag-one autocorrelation and pairwise dependence changes.
- Spread-state transition counts/probabilities, with the state threshold fixed
  to the training spread median and no transitions across missing observations.
- Chronological block-bootstrap uncertainty for mean drift. The interval uses
  independent resampling of complete nonoverlapping block means, at least eight
  eligible blocks per period and Bonferroni correction across eight observables
  within each target diagnosis. These exploratory intervals are not a simultaneous
  guarantee across all targets or studies.

Partial blocks and blocks with less than 80% finite values are excluded from the
interval estimator; their observations remain in marginal metrics and validity
denominators. The interval estimates eligible-block mean differences. Reported
block counts disclose its effective sample size. Approximate stationarity and
weak dependence between blocks are assumptions, not established facts; long
memory and the finite resampling budget limit tail coverage. No p-values or
equivalence conclusions are inferred.

Large train-relative distance is labeled `distribution_shift_association`.
A large model-target discrepancy with smaller train-relative drift is labeled
`model_misspecification_or_sampling_error`. These are diagnostic associations;
causal attribution remains `NOT_ESTABLISHED`. Parameter instability is also
unidentified because this workflow does not refit on targets. Unsupported
arrival/cancellation/trade intensity, order sizes, queues and causal impact are
listed explicitly rather than inferred from net aggregate depth changes.

Standalone, sealed diagnostics from existing feature CSVs:

```sh
python -m lob.calibration_diagnostics --reference TRAIN.csv --target TARGET.csv --original-status FAIL --out results/v04/diagnosis
```

This command registers source hashes and settings before loading observations,
then creates a verifiable research bundle. It does not make either input fresh.

## Preserved and new empirical work

```sh
python -m lob.historical_generalization --v03-study examples/studies/v03/calibration-holdout-fix --out results/v04/v03-failure-diagnostics
python -m lob.historical_generalization --config configs/v04-calibration-diagnostics.json --out results/v04/historical-diagnostics
```

The first command reconstructs the already inspected v0.3 synthetic fixture and
checks its reference/target hashes against the unchanged original artifacts.
The second requires the retained local April/May model and raw data; their exact
hashes and original split bounds are checked. Neither command trains a model.
April validation, April internal and May external statuses must reproduce their
preserved assessments before fresh evaluation can proceed.

The historical protocol predeclares June 1, 2020 Deribit ETH-PERPETUAL incremental
L2, the next first-of-month period after consumed May. The source was absent from
the inspected repository inventory and references before registration. Download
integrity and an intake hash are persisted before feature extraction, as is a
consumption marker beside the raw file. Existing/consumed June data are rejected
as fresh. A failed download is retained without substituting another date.
One fresh day is not multiple independent regimes, venues or instruments.

The provider documents [first-of-month public access](https://docs.tardis.dev/historical-data-details/deribit).
Its [terms, checked September 26, 2026](https://docs.tardis.dev/legal/terms-of-service)
restrict internal quantitative-model use and distribution of fitted models and
derived data. v0.4 keeps new raw data, sampled rows, fitted historical parameters
and detailed empirical outputs under ignored local directories; public export
requires the appropriate permission. Existing v0.3 evidence is preserved.
The checked-in protocol contains the exact source, access date, checksums and
reproduction prerequisites. It does not silently recreate a missing original fit.

## Executed synthetic checks

The registered source commit was `9072bb80c4a823d940d2b752e34f08881a4841de`.
The multi-period run selected `spread_markov` on development/selection data only.
Its selection loss was 0.073398; final results were:

| Registered synthetic period | Gate | Loss |
|---|---|---:|
| Internal | PASS | 0.108912 |
| External scale 1.8 | FAIL | 0.976637 |
| External scale 0.6 | FAIL | 0.552638 |

The original v0.3 fixture independently reproduced its internal WARNING and
external FAIL, with original frame hashes and model bytes unchanged. Its external
discrepancies were associated with spread, side-depth and return/volatility
distribution shifts. This is consumed synthetic evidence, not real-market proof.
All multi-period and v0.3 diagnostic bundle hashes verified.

The local historical diagnostic run reproduced April validation WARNING,
April internal FAIL and May external FAIL using the unchanged original model
and exact preserved frame hashes. Its first June acquisition attempt was denied
by the sandbox network; that sealed `NOT_AVAILABLE` attempt was retained. An
unsandboxed acquisition retry used a new output directory and the same frozen
scientific configuration. The genuine, preregistered June 1 external period also
**FAILED**: spread and both side-depth gates failed; imbalance and return gates
warned. The old model and all scientific source hashes remained unchanged.
Distribution shifts were associated with spread/depth discrepancies; causal
attribution remains unidentified. The new failure strengthens the negative
external evidence without establishing generalization across regimes.

The original attempt's 13 files and the retry's 16 files passed exact-file/hash
verification. Detailed results remain in `results/v04/historical-diagnostics`
and `results/v04/historical-diagnostics-network-retry`. June is now recorded as
consumed in the current registry. Further fresh evidence and redistribution
permission remain separate requirements.
