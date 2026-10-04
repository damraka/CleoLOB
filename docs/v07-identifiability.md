# v0.7 identifiability v2 and observable design (M8)

Code: `lob/v07/identifiability/` (`analysis.py`, `study.py`).
Sealed run: `results/v07/m8/identifiability-3`. Development data only, through the frozen v0.6
development target.

## Attempts (retained)

| Run | Outcome |
|---|---|
| `results/v07/m8/identifiability` | **FAILED** (engineering bug): components that cannot be evaluated in single-seed noise runs produced NaN noise scales, and the eigendecomposition failed. Fixed with a regression test. |
| `results/v07/m8/identifiability-2` | sealed but **INVALID** (analysis artifact): one observable with zero seed noise (`exceed_spread_p99`) hit the 1e-6 noise floor. Its standardized row (norm 2.6e7, median 149) dominated every eigen-analysis. Fixed: noise is floored at 10% of the median noise, with a regression test. |
| `results/v07/m8/identifiability-3` | sealed; reported below |

## Method (workstreams 12, 13)

**Jacobian.** Central differences (step 0.05 in the unit box, shrunk at the boundary) of all 85
v0.6 component errors with respect to the 14 parameters. The center is the best particle of the
pooled G0 posterior, and every evaluation uses common random numbers (3 seeds × 1,800 s). This
removes the v0.6 cap, where nine family errors limited the rank to 9.

**Standardization.** Each row is divided by its measured seed noise, floored as above.

**Observables used.** 72 of 85. The 13 dropped components could not be evaluated in at least
two single-seed runs:
- 60 s returns and volatility
- several tail quantiles and Hill indices
- resilience durations

**Analyses.**
- sloppiness: eigenvalues of J'J, effective rank, participation ratio
- posterior widths, correlations and profiles (`results/v07/m6/posterior-2`)
- identification classes
- the observable × parameter influence map

## Results

**Sloppiness.**
- Effective rank 8 (eigenvalues above 1e-3 of the largest).
- Participation ratio 1.37: one combination of parameters dominates the observable response.
- The eigenvalues span 17.8 decades. The smallest eigenvalue is 0, from `inside_spread_prob`.

**Parameters.** All 14 are not identified at this resolution:

| Parameter | Relative influence | Posterior 90% width | Status |
|---|---|---|---|
| inside_spread_prob | 0.00 | 0.91 | STRUCTURALLY_NOT_IDENTIFIED (no observable responds at the center) |
| offset_p | 1.00 | 0.51 | PRACTICALLY_NOT_IDENTIFIED |
| regime_switch_rate | 1.00 | 0.91 | PRACTICALLY_NOT_IDENTIFIED |
| hawkes_branching | 0.67 | 0.87 | PRACTICALLY_NOT_IDENTIFIED |
| imbalance_beta | 0.67 | 0.85 | PRACTICALLY_NOT_IDENTIFIED |
| market_rate | 0.57 | 0.71 | PRACTICALLY_NOT_IDENTIFIED |
| other 8 parameters | 0.02–0.20 | 0.61–0.93 | PRACTICALLY_NOT_IDENTIFIED |

The observables do respond to most parameters locally, but the ABC posterior leaves each
parameter spread over at least half its range. Practical non-identification therefore comes
from the posterior width, which is itself a consequence of the high final ABC tolerance. These
statements are limited to this simulator family, these observables and the 100 ms resolution.

**Compensation.** No posterior parameter pair reaches |ρ| ≥ 0.7. The pooled particles are too
diffuse for strong pairwise compensation.

**Observable design.**
- **Weak observables:** 59 of 72 have row norms below 5% of the maximum. The largest rows
  belong to:
  - spread tail exceedance (`exceed_spread_p99`)
  - 10 s signed-volume autocorrelation
  - return autocorrelation at lag 2
  - imbalance tail exceedance
  - trade Fano factor
- **Uninformative family:** returns. No component of the returns family moves more than 5%
  of the strongest observable.
- **Redundant pairs:** two
  - trade inter-arrival vs. 10 s trade Fano factor (cosine −0.97)
  - lag-5 return ACF vs. lag-5 absolute-return ACF (−0.998)
- **Complementary observables:**
  - lag-5 return ACFs are dominated by `cancel_rate`
  - trade Fano factor, replenishment probability and inter-arrival are dominated by
    `market_rate`

**Equifinality.** Together with H6 (multimodal posterior, ESTABLISHED, descriptive) and the
diffuse posterior, the development data at this tolerance do not single out one parameter
vector. This extends the v0.6 equifinality finding; it is not a statement about real market
mechanisms.

**Synthetic validation is kept separate.** Known identifiable and non-identifiable linear
models, recovered exactly, are in `tests/test_v07_identifiability.py`.
