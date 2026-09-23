# Frozen simulator external evaluation

All three July-only calibrated simulator models failed both August validation
and the previously unopened September holdout. The external assessment is
complete; it does not establish real-market validity for the execution study.

The evaluation consumed four distinct BTC/ETH files: **7,461,728 published
top-five snapshots**, yielding **345,595 causal one-second samples**, of which
**324,867** passed the unchanged observation-validity rules. Pooled rows reuse
these observations and must not be counted as additional independent data.

| Stage | Cohort | Valid observations | Failed moment families | Outcome |
|---|---|---:|---|---|
| August validation | ETH | 99.970% | depth | FAIL |
| August validation | BTC | 99.638% | OFI, impact, volatility | FAIL |
| August validation | pooled | 99.804% | depth, OFI, impact | FAIL |
| September holdout | ETH | 88.221% | spread, impact | FAIL |
| September holdout | BTC | 88.180% | depth, impact, volatility | FAIL |
| September holdout | pooled | 88.201% | impact, volatility | FAIL |

Every September cohort also failed the preregistered 95% valid-observation
requirement. Invalid observations were retained in those denominators. The
moment-family error threshold remained 0.60. For the primary ETH instrument,
September spread error was 1.0534 and impact error was 1.3693; the favorable
depth, OFI, imbalance and volatility diagnostics do not reverse the overall
failure.

The frozen amended models were evaluated first on all August cohorts using
simulation seeds 42001 and 42002, then on all September cohorts using seeds
43001 and 43002, as specified in `SEARCH_AMENDMENT.json`. Each simulation seed
used 1,800 measured seconds after a 30-second warmup. No model was retuned and
no gate changed after either external phase. Training-only quantity scales
were reused without refitting. Download sidecar SHA-256 hashes and byte sizes
matched all four raw files; extraction also consumed each gzip stream through
EOF and checked source hashes before and after extraction.

The first ETH validation result was successfully saved before a reporting-only
dictionary-key error stopped orchestration. That result was reused without
re-evaluating it. The reporting repair and original/result hashes are retained
in `frozen-external-evaluation/orchestration-resume.json`; scientific simulator
and calibration source hashes remained unchanged.

Machine-readable evidence is in
[`frozen-external-evaluation/summary.json`](frozen-external-evaluation/summary.json),
with six full cohort results, a plan containing model/source/download hashes,
and nine verified JSON hashes in `checksums.json`. Execution findings from this
simulator must consequently be described as conditional synthetic findings.
Top-five snapshot book walks measure visible instantaneous costs; they do not
identify historical counterfactual fills or causal market impact.
