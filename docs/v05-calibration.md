# v0.5 M3 — calibration v2 and model-class comparison

v0.4 showed that the calibrated simulator failed fresh external calibration. v0.5 asked
whether a richer, justified simulator family improves *out-of-sample* fit over the v0.4
model class, and whether it passes the frozen gate. The answer is split:

- **Relative improvement:** yes, on the same instrument. It is statistically established on
  the fresh July 2020 ETH external holdout.
- **Absolute generalization:** no. Every holdout FAILED the per-family gate, for both the
  selected model and the control.
- **Cross-instrument improvement:** not established on BTC-PERPETUAL.

## Frozen design

Frozen in `configs/v05/m3-design.json`; the procedure and implementation hashes were sealed
in ledger entry 29.

**Measurement.**
- A single operator (`lob.observables_v2`) is applied identically to historical and simulated
  100 ms top-5 tapes.
- It measures 12 observable families: spread, depth, imbalance, volatility, trade intensity,
  update intensity, cancellation intensity (net proxy), sizes, interarrival, resilience,
  autocorrelation and dependence.
- Family error follows the v0.4 `zi_calibration` convention, capped at 10. E3 is the mean
  family error; the gate is every family ≤ 0.60.

**Scaling.** Development median top-5 side depth = 1,000 lots. This is applied unchanged
everywhere, including BTC.

**Model families.**
- The control is the v0.4 ZI simulator class.
- There are six extensions in `lob.sim_v2`, all opt-in and exact under thinning:
  - empirical size tables;
  - spread-conditioned arrivals;
  - depth-conditioned cancellation;
  - imbalance-conditioned market flow;
  - a Markov activity regime;
  - self-exciting market orders. Their admission rule is a development dispersion index
    above 1.5; the observed value was 3.30, so they were admitted.
- With no extension active, the extended simulator is bit-identical to the v0.4 engine.

**Budget.** Identical for every family: 48 candidates in two batches, 3 seeds × 900
simulated seconds.

**Chronology.**
1. Fit on April (development).
2. Select on May.
3. Combine the extensions that beat the control on May.
4. Seal the selection in the ledger.
5. Evaluate unchanged on June (retrospective internal), July ETH (fresh external) and July BTC
   (cross-instrument).

**Inference.**
- H3a: paired 600 s block bootstrap of the selected-minus-control E3 (2,000 resamples,
  α = 0.05/3).
- H3b: the per-family gate.

## Results

**Training fit (development E3; never evidence of generalization).**

| Family | Training E3 |
|---|---:|
| v0.4 ZI control | 1.996 |
| depth-conditioned cancellation | 1.129 |
| spread-conditioned arrivals | 1.327 |
| Markov activity regime | 1.367 |
| empirical sizes | 1.372 |
| self-exciting market orders | 1.443 |
| imbalance-conditioned market flow | 1.476 |

**Selection (May).**
- Every extension beat the control: 1.90–2.35 against 2.946.
- The combined model reached 0.938 and was selected. Its training E3 was 0.558.
- 7 of its 48 candidates hit the 5-million-event cap and were retained as failures.

**Holdouts.**

| Holdout | Selected E3 | Control E3 | H3a (selected − control), α = 0.05/3 | H3b gate, selected / control |
|---|---:|---:|---|---|
| June 2020 ETH (retrospective) | 1.594 | 2.355 | **ESTABLISHED**, [−1.32, −0.39] | FAILED / FAILED |
| **July 2020 ETH (fresh external)** | **1.390** | **2.567** | **ESTABLISHED**, [−1.35, −0.70] | FAILED / FAILED |
| July 2020 BTC (cross-instrument) | 2.688 | 2.763 | NOT_ESTABLISHED, [−0.26, 0.04] | FAILED / FAILED |

**Failed families on July ETH.**
- Selected model: 10 of 12. Only `dependence` and `imbalance` pass. The worst errors are
  sizes 3.39, volatility 3.44, interarrival 2.63 and update intensity 2.21.
- Control: 8 of 12. Interarrival and sizes are at the cap of 10.

**Failed families on BTC.** Both models fail spread (5.96) and volatility and interarrival
(capped at 10). The ETH tick geometry and scale do not transfer, which is exactly what this
layer tests.

**Regime-conditioned results (M6).**
- The selected model beats the control in every regime label on June and July ETH, but not on
  BTC.
- It passes the gate in no regime.
- Every June and July block falls in the development "low depth" regime, a distribution shift
  the development data never covered.

## Interpretation

The extensions address real misspecification of the v0.4 class: sizes, interarrival
clustering, depth-conditioned cancellation and activity regimes. The improvement survives a
fresh period on the same instrument.

The simulator still does not reproduce real observables within the preregistered tolerance.
Calibrated-simulator conclusions (M7) therefore remain synthetic evidence. Diagnoses are
associations; no causal attribution of calibration failure is claimed.

## Deviations (disclosed)

1. **Selection seal after July access.**
   - The M3 design stated that the selection would be sealed before any holdout access.
   - The selection process started at 11:08 (entry 29 precedes it) and reads only April and
     May. It ran until 13:33 because some combined candidates were very slow.
   - July was first accessed for the frozen M2 study at 12:58 (entries 40–43). The
     `m3-selection` seal (entry 44) therefore *follows* that first access.
   - The selection could not read July. The stated ordering was nevertheless not met.
2. **Performance revision.**
   - After selection and before holdout evaluation, `lob/sim_v2.py` was changed so the regime
     lookup precomputes its switch array (ledger entry 45).
   - Outputs on the selected model are bit-identical; it runs 33–40× faster.
3. **Budget cap.** Seven combined candidates hit the 5-million-event cap. As frozen, they were
   retained as failed candidates.

## Reproduce

See [v05-reproduction.md](v05-reproduction.md). Fitted historical parameters and detailed
historical summaries stay local under the provider terms. The public bundle carries losses,
gates, intervals and hashes.
