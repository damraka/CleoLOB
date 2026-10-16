# CleoLOB v0.9 — Reality-Calibrated Market Twins, Causal Execution Science & Safe Autonomous Adaptation (release notes; causal claims limited to registered estimands)

CleoLOB v0.9.0 asks which execution conclusions remain scientifically defensible once all of the following are
considered together:
- market reconstruction uncertainty;
- latent liquidity and self-impact;
- strategic response and causal ambiguity (what observational data cannot identify);
- multi-fidelity simulation;
- distribution shift and tail events;
- model uncertainty and online adaptation;
- real historical validation.

**Preregistration.** Every candidate hypothesis (H1–H36), execution conclusion (C1–C6), statistical family,
threshold, data pool, backup order, seed namespace and holdout role was registered in `configs/v09/` and frozen in
an append-only, hash-chained ledger before any exploratory access.

**Sequence.**
1. A registered gate promoted hypotheses once, from exploratory pilots.
2. The confirmatory registry was committed before any confirmatory day was read.
3. Confirmatory designs were sealed after dry runs on explore days.
4. The untouched 2025 vault was opened exactly once, after the confirmatory phase closed.

**What the release changes.** The studies ran on the research branch at development version `0.9.0.dev0`, and the
release changes only version metadata and release wording.

## Scientific limitations

Read these before any result below.

- **Bounded, hypothetical replay.**
  - The execution conclusions are tested in bounded historical counterfactual replay, on the conservative
    (worst-assumption) bound of a registered grid of queue, latency, hidden-liquidity and impact assumptions.
  - The costs are HYPOTHETICAL. Public L2 never shows queue positions, so no exact fill is claimed.
  - Nothing here supports live trading, profit, HFT or production claims.
- **Amendment 3 is a post-hoc change.** It replaced the promotion measurement screen after the exploratory pilots
  were inspected, and is disclosed as incident 2. Under the original frozen screen, these would not have been
  tested:
  - the execution conclusions (H23, H32, H33, H36);
  - H16 and H17.

  The original-screen decisions are reported beside the amended ones.
- **Realism gate** ([C09-GATE]). Every generator family is NO_ACCEPTABLE_MODEL for execution, realism, strategic and
  tail use, and PARTIAL for fills only. The gate was not weakened.
- **Reality ladder** ([C09-LADDER], [C09-GAP]).
  - The synthetic and hybrid rungs did not reproduce the replay effects. The single calibrated generator did not
    support C1, and the hybrid twin did not evaluate the conclusions.
  - H36's status follows its registered vault test, not survival of every rung.
- **Small confirmatory samples.** There are 8–21 days per test.
- **Integrity, not replication.** Verification establishes integrity and binding, not independent replication.

## Final hypothesis statuses

12 ESTABLISHED and 24 NOT_ESTABLISHED. Every NOT_ESTABLISHED candidate was not promoted and was never tested on
confirmatory data.

**ESTABLISHED on real days (E4):**

| H | Result |
|---|---|
| H2 [C09-H2] neural TPP vs classical point processes | 0.39 nats/event (0.32–0.46): ESTABLISHED |
| H5 [C09-H5] dynamic model averaging | 0.076 nats/event: ESTABLISHED |
| H6 [C09-H6] execution-aware fill calibration | Brier 0.18 better: ESTABLISHED |
| H7 [C09-H7] queue-survival models vs Kaplan–Meier | integrated Brier 0.094 better: ESTABLISHED |
| H16 [C09-H16] continual calibration vs naive fine-tuning | 0.016 nats/event less forgetting: ESTABLISHED |
| H17 [C09-H17] adaptive conformal coverage under drift | non-inferiority, margin 0.02: ESTABLISHED |

**ESTABLISHED on model rungs:**

| H | Rung | Result |
|---|---|---|
| H8 [C09-H8] calibrated fills reduce decision regret | hybrid twin | 0.47 bps: ESTABLISHED |
| H29 [C09-H29] synthetic ranking predicts hybrid ranking | model rungs only | agreement 0.92: ESTABLISHED |

**ESTABLISHED in bounded hypothetical replay (conservative bound):**

| H | Result |
|---|---|
| H23 [C09-H23] CVaR-MPC vs TWAP tail cost, medium mandate | 11.4 bps: ESTABLISHED |
| H32 [C09-H32] carried conclusions across time | ESTABLISHED |
| H33 [C09-H33] carried conclusions across instruments | ESTABLISHED |
| H36 [C09-H36] carried conclusions on the untouched 2025 vault | ESTABLISHED |

The carried conclusions are C1 (CVaR-MPC vs TWAP), C2 (TWAP vs sweep) and C3 (passive-deadline vs TWAP). On the
vault their conservative-bound means are 6.8, 7.3 and 1.3 bps.

**NOT_ESTABLISHED (not promoted):**

| Reason | Hypotheses |
|---|---|
| measurement screen | H1 [C09-H1], H3 [C09-H3], H30 [C09-H30], H31 [C09-H31] |
| effect too small | H4 [C09-H4], H9 [C09-H9], H10 [C09-H10], H11 [C09-H11], H12 [C09-H12], H13 [C09-H13], H14 [C09-H14], H18 [C09-H18], H19 [C09-H19], H20 [C09-H20], H21 [C09-H21], H22 [C09-H22], H24 [C09-H24], H25 [C09-H25], H26 [C09-H26], H27 [C09-H27], H28 [C09-H28], H35 [C09-H35] |
| pilot violated the non-inferiority margin | H15 [C09-H15] |
| no L1-identifiable conclusion carried | H34 [C09-H34] |

## What is included (M0–M89)

See [v09-requirement-coverage.md](v09-requirement-coverage.md). Highlights:
- **Data and reconstruction.** Bybit linear and spot full-depth L2 reconstruction with update-id checks, Binance L1,
  a quality engine with registered backups, and hash-ledgered day caches.
- **Fills.** Queue-clearance events as bounded fill conditions; survival models (Cox, Weibull AFT, boosted, neural
  hazard).
- **Generators.** Classical, neural point-process, sequence, latent and strategic generators, with a
  capability-specific realism gate.
- **Execution.** The hybrid twin, multi-fidelity agreement, and bounded counterfactual replay.
- **Studies.** Conformal, continual and meta adaptation; offline RL with OPE gates in synthetic worlds only; DRO,
  MPC and rare-event studies.
- **Discipline.** A promotion gate with a feasibility report; sealed designs; dry runs; a fail-closed batch runner;
  outcome-blind run checks; a confirmatory data audit; the claim graph and wording audit.

## Public evidence bundle

`examples/studies/v09` holds:
- the frozen configs, designs and ledger;
- every sealed evidence run's small JSON/Markdown files and figures;
- the report, cards and datasheets.

Raw provider data, day caches, checkpoints and engineering dry runs are excluded. Verify it offline with:

```bash
cleo verify-v09 --bundle examples/studies/v09
cleo claim-audit-v09 --bundle examples/studies/v09
```

## Data, amendments and incidents

- **Data** ([C09-DATA]). 35 confirm, 17 late and 36 vault days were read. 8 confirm-phase quality rejections and 4
  vault provider failures (HTTP 404) were replaced by registered backups in frozen order.
- **Amendments and incidents** ([C09-INCIDENTS]). There are three amendments and five incidents:
  - incident 1: a quality-rule bug, found before any outcome;
  - incident 2: Amendment 3;
  - incident 3: concurrent queues on descriptive studies, with no overwrite;
  - incident 4: the H16 unit order in the final analysis, found after close, fixed to the registered definition,
    with no status change;
  - incident 5: the vault backup rounds, with no data read.

  Every failed attempt is retained in the ledger.

## Documentation

- [v0.9 report](v09-paper.md)
- [final report](v09-final-report.md)
- [protocol](v09-protocol.md)
- [data](v09-data.md)
- [methods](v09-methods.md)
- [reproduction](v09-reproduction.md)
- [model cards](v09-model-cards.md), [policy cards](v09-policy-cards.md) and [datasheets](v09-datasheets.md)
