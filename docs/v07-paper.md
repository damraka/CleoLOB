# CleoLOB v0.7 research report

Branch `research/v0.7-generative-market-validation`, version `0.7.0.dev0`. This report is
filled from registered results only. Sections without results say so.

## 1. Question

> When richer generative market models, posterior calibration, parameter uncertainty,
> model-class uncertainty, regime drift, queue uncertainty, structural misspecification and
> historical counterfactual uncertainty are all accounted for, which execution conclusions
> remain scientifically defensible across time, instruments, venues and market conditions?

## 2. Preregistration (M0)

The protocol is in `configs/v07/protocol.json`. It references seven further documents:
- dataset registry
- hypotheses H1–H13
- statistical families
- compute budget
- result taxonomy
- claim registry
- scenario taxonomy

The protocol is frozen in the append-only ledger `configs/v07/consumption-ledger.jsonl`.
That ledger pins the v0.5 and v0.6 protocols and ledger heads, and imports every dataset
and calendar period they consumed. Run `cleo protocol-v07 verify` to check the current state.

**Freeze record.**
- Protocol `63fb6e56…`, frozen at ledger entry 40 on commit `0ef1939`.
- Entries 1–40 record `git_dirty: true`. The only change was the new, still-untracked
  ledger file. Later entries exclude the ledger file from the dirty check.

**Fresh holdouts at the freeze.** None had been accessed by any CleoLOB version.

| Role | Dataset |
|---|---|
| fresh temporal | Deribit ETH-PERPETUAL 2020-11-01, 2020-12-01 |
| fresh cross-instrument | Deribit BTC-PERPETUAL 2020-11-01 |
| fresh cross-venue | BitMEX XBTUSD 2020-11-01 |
| final transfer | Deribit ETH-PERPETUAL 2021-01-01 (after the policy evaluation lock) |

**Hypotheses.** Each hypothesis has its estimator, sample unit, uncertainty method,
multiplicity family, threshold, margin and failure semantics fixed in
`configs/v07/hypotheses.json`.

| ID | Question (short) | Family |
|---|---|---|
| H1 | Posterior calibration beats the v0.6 point baseline (fresh ETH) | F1, Bonferroni 2 |
| H2 | A richer generator reduces classifier distinguishability | F2, Bonferroni 8 |
| H3 | A richer generator increases support coverage | F3, Bonferroni 8 |
| H4 | The selected generator transfers cross-instrument | F4, Bonferroni 2 |
| H5 | The selected generator transfers cross-venue | F4, Bonferroni 2 |
| H6 | The posterior is multimodal | descriptive |
| H7 | Model uncertainty is material to execution cost | F7, Holm 8 |
| H8 | At least one policy comparison is robust across model uncertainty | F8, Bonferroni 28 |
| H9 | A single-world ranking becomes model-dependent or inconclusive | descriptive |
| H10 | Execution-aware calibration improves execution-sensitive realism | F10, Bonferroni 2 |
| H11 | H10 is not metric overfitting (noninferiority, δ = 0.10 × baseline) | F11, Bonferroni 2 |
| H12 | Posterior training improves historical transfer | F12, Bonferroni 4 |
| H13 | Robust conclusions survive both historical fill bounds | descriptive, conditional |

## 3. Results

Every status below is copied from a sealed run. `docs/v07-final-report.md` has the full claim
table, the attempt history and the run hashes. Effects come first, then status. Lower objective
values are better; a positive difference "model minus G0 point" means the model is worse.

### 3.1 Generators and selection (M4–M6)

Five families were fitted on the development day and scored on development and selection data
(`docs/v07-generators.md`). Diffusion generators are NOT_AVAILABLE (no GPU). The sealed
selection rule picked **G3** (conditional AR resampling; selection objective 2.005, vs. 2.461
for the v0.6 point model G0).

### 3.2 Posterior calibration (M6)

- **Recovery is ASSUMPTION_DEPENDENT.** Marginal 90% coverage was 0.714, 0.714 and 1.0 for the
  three synthetic truths, against the registered 80%. Every conclusion that uses the posterior
  inherits this limitation.
- **H6 is ESTABLISHED (descriptive).** [C-H6] All three SMC-ABC runs are multimodal, but each major
  component holds only 5–13% of mass and the posterior is diffuse at the reached tolerance.
- **H1 is FAILED.** [C-H1] Posterior-predictive realism was *worse* than the point model on both fresh
  ETH days: +0.387 [0.211, 0.627] in November and +0.568 [0.303, 0.932] in December.
- The first posterior attempt was ABORTED by the host for low memory and is kept in the ledger.

### 3.3 Identifiability (M8)

None of the 14 parameters is identified at this resolution (effective rank 8 of 14; the
sensitivity eigenvalues span 17.8 decades). `inside_spread_prob` is structurally not identified; the
other 13 are practically not identified (`docs/v07-identifiability.md`). Two earlier attempts
(FAILED, INVALID) are kept.

### 3.4 Realism, domain gap and transfer of realism (M10, M17)

- **H2 is NOT_ESTABLISHED and VACUOUS.** [C-H2] Every family, including G0, is separated from
  history with AUC ≈ 1 at 10 s, 60 s and 300 s, so there is no room for a reduction.
- **H3 is NOT_ESTABLISHED.** [C-H3] Support coverage of fresh windows is 0–1.5% for every family.
- **H4 is FAILED.** [C-H4] G3 is worse than G0 cross-instrument (BTC): +0.717 [0.643, 0.768].
- **H5 is FAILED.** [C-H5] G3 is worse than G0 cross-venue (BitMEX XBTUSD): +0.616 [0.527, 0.663].
- The added capacity of G3 and G4 is not justified on untouched data. Their development and
  selection advantage disappears on fresh same-instrument days and reverses out of domain.

### 3.5 Execution-aware calibration (M7, M17)

- **H10 is NOT_ESTABLISHED.** [C-H10] The execution-sensitive objective of the execution-aware model was
  not better than G0's on either fresh day: +0.189 [−0.002, 0.311] and +0.088 [−0.068, 0.236].
- **H11 is FAILED.** [C-H11] Its generic realism loss exceeded the registered noninferiority margin
  δ = 0.246 on both days (one-sided upper bounds 0.486 and 0.453).

### 3.6 Model risk across 21 plausible worlds (M15)

- **H7 is ESTABLISHED for POV only.** [C-H7] Its between-world SD of mean cost is 0.96 bps, 2.85 times
  the within-world seed SE. For the other 7 policies it is NOT_ESTABLISHED.
- **H8 is NOT_ESTABLISHED.** [C-H8] No policy pair is robustly ordered. 27 of 28 pairs have pooled
  intervals inside ±1 bps (descriptive equivalence within the margin, averaged over worlds),
  and 1 is INDETERMINATE. The most adverse plausible world moves each pair by 0.75–8.5 bps, so
  the average does not describe every world.
- **H9 is INCONCLUSIVE.** [C-H9] No pair is determinate even in the single G0 world.
- 21 worlds give 21 distinct policy rankings. Market randomness dominates the variance of
  single episodes; among model components, the simulator family contributes more than the
  posterior spread.

### 3.7 Historical bounded transfer, ETH 2021-01-01 (M14, M16)

- **H12 is NOT_ESTABLISHED.** [C-H12] On the 144 replay episodes, the gap between simulated and bounded
  historical cost of posterior-trained policies did not differ from that of single-world
  policies for any member. Differences range from −0.10 to +0.03 bps, with intervals of about
  [−2.7, 1.6] bps (alpha 0.0125 each). This is not equivalence.
- **H13 is INCONCLUSIVE.** [C-H13] No H8-robust pair exists, which is the registered NOT_EVALUABLE case.
- Descriptively, no classical pair is determinate in history under either fill bound. Learned
  policies cost more than every classical policy in point estimate.
- Two earlier attempts of this evaluation were ABORTED for memory and are kept in the ledger.

### 3.8 Decision benchmark and synthesis (M22) — `results/v07/m22/final-2`

- **Certification.** All 28 policy pairs end NOT_ESTABLISHED: the framework abstains on every
  ranking. Robust-selection criteria over the 21 worlds pick different policies:
  - expectation: imbalance-aware
  - worst case: urgency
  - CVaR over worlds: liquidity-sensitive
  - distributionally robust: urgency
- **Realism to decision.** World realism did not predict the decision deviation (median Spearman
  −0.35 over 21 worlds; UNRESOLVED).
- **Ablations.** None of the six complexity components helped in point estimate: posterior
  calibration, neural generator, continuous conditioning, discrete regimes, execution-aware
  calibration and posterior-world policy training. The complexity penalty finds no family whose
  fresh realism beats every simpler family.
- **Overfitting.** The more flexible generators deteriorate most from development to fresh
  data: +2.36 for G4 and +2.29 for G3, against +1.64 for G0.
- The first M22 run (`m22/final`) is retained as INVALID: it labelled a fill-fraction queue
  width as bps.

### 3.9 What this means

Under the registered tests, no v0.7 complexity (posterior calibration, richer generators,
execution-aware calibration) improved fresh-data realism over the v0.6 point model, and no
execution conclusion is robust across the plausible worlds. The positive results are
descriptive (H6) or about uncertainty itself (H7 for POV). No difference in historical
transfer from posterior-world policy training was established (H12), and no robust conclusion
existed to carry to history (H13). These are negative and null results
for these designs, data and budgets; they are not evidence that such methods cannot work.

## 4. Limits stated in advance

- Aggregate L2 cannot establish exact FIFO, order identity, queue position, hidden
  liquidity or exact passive fills. Those capabilities are `NOT_AVAILABLE`.
- Event resolution is 100 ms.
- There is no GPU. A diffusion generator is `NOT_AVAILABLE`.
- No result supports claims about live profitability, alpha, production trading, colocated
  latency or causal real-market mechanisms.
