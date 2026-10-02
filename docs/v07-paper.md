# CleoLOB v0.7 research report (in progress)

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

Pending. No v0.7 study has run.

## 4. Limits stated in advance

- Aggregate L2 cannot establish exact FIFO, order identity, queue position, hidden
  liquidity or exact passive fills. Those capabilities are `NOT_AVAILABLE`.
- Event resolution is 100 ms.
- There is no GPU. A diffusion generator is `NOT_AVAILABLE`.
- No result supports claims about live profitability, alpha, production trading, colocated
  latency or causal real-market mechanisms.
