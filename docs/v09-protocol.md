# v0.9 protocol, ledger and promotion

The v0.9 protocol is `configs/v09/protocol.json`. It was frozen in the consumption ledger
`configs/v09/consumption-ledger.jsonl` before any v0.9 dataset was read. `cleo protocol-v09 verify` replays the
ledger hash chain, checks every referenced config against its frozen hash (or a ledgered amendment) and checks the
pinned v0.8 history.

## Data roles and the firewall

Every dataset has one role:

| Role | Pool | Period | Who may read it |
|---|---|---|---|
| development | `dev` | 2023-07-15 (plus the 2025-09-20 spot format day) | any design, engineering only |
| explore | `explore` | 2023-08 to 2023-10 | any design after the freeze; never used for confirmatory decisions |
| confirm | `confirm` | 2023-11 to 2024-03 | only sealed designs, only after the promotion freeze |
| confirm_late | `confirm_late` | 2024-06 to 2024-12 | as `confirm` (time-transfer pool) |
| final | `final` | 2025-05 to 2025-08 | exactly one sealed design, once, after `close_confirmatory` |

The ledger enforces this firewall when an entry is appended. An access by a design that has not been sealed, or
that does not list the dataset, is refused. So is a confirm access before `promote`, and a vault access before
`close_confirmatory` or by a second reader. Seeds come from namespaces `SeedSequence([9, namespace, study hash,
index])`. The `confirm`, `holdout` and `final` namespaces must be opened through the ledger by a sealed design.

A failed slot (an HTTP error, a missing archive or QUALITY_REJECTED) is replaced only by the next unused backup slot
of the same stream, in registered order. Nothing else may substitute. When every registered option fails, the unit
is NOT_AVAILABLE.

## Lifecycle of a hypothesis

All 36 hypotheses (H1–H36) and the six execution conclusions (C1–C6) were registered as candidates before any data
access. Each has a fixed endpoint, estimator, unit, direction, threshold, family, fidelity, capabilities and
evidence-level target (`configs/v09/hypotheses-candidates.json`). Each then moves through these stages:

1. **Exploratory pilot** on explore days or synthetic worlds (`explore` seeds). A pilot estimates the effect, the
   between-unit SD, the split halves and the compute cost.
2. **Promotion gate** (`hypothesis-promotion-v09 run`, exactly once). It checks, in order:
   - data and capability;
   - measurement;
   - effect (the pilot effect shrunk by one SE must exceed the threshold);
   - power ≥ 0.80 at the planning α (family α ÷ family size) within the unit ceiling;
   - compute.

   The output is PROMOTED with the minimal confirmatory sample size, or a NOT_PROMOTED_* reason.
3. **Freeze** (`hypothesis-promotion-v09 freeze`) writes `configs/v09/hypotheses-confirmatory.json` and appends
   `promote`.
4. **Sealed confirmatory designs** (`lob.v09.confirm.seal_all`) are sealed in one ledger batch before any
   confirmatory byte is read. Each promoted hypothesis reads the first `confirm_units` registered primary slots of
   its pool. Training sets are explore days.
5. **Confirmatory runs**, then the transfer study, then `close_confirmatory`.
6. **The final vault**: one sealed design and one bounded-replay run over the 2025 SOLUSDT, BTCUSDT and spot
   BTCUSDT days.
7. **The final analysis** (`final-analysis-v09`) applies the registered decision rule of every promoted hypothesis,
   with Holm within each registered family.

## Amendments and incidents (all ledgered, none erased)

**Amendment 1** (entry 365) came before any explore access. It fixed:
- the quality thresholds;
- the day-cache semantics;
- the net-of-fee cost endpoints;
- the spot development day.

**Amendment 2** (entry 367) came before any explore study outcome. It added the observed-grid rule: if the archive's
price precision is coarser than the registered tick, the observed grid is used and the anomaly is recorded.

**Amendment 3** (entry 474) came after the exploratory pilots were inspected, and before promotion and any
confirmatory or vault access. It replaced the frozen measurement screen (Spearman–Brown split-half reliability ≥ 0.5)
with a split-half consistency screen. The new screen fails when |mean(odd − even)| > 2 SE, or when no paired halves
exist.
- The frozen statistic measures between-day rank stability, not the precision of the mean effect that the
  confirmatory test uses. It scored a homogeneous effect (C3, seven times its threshold) at −0.09.
- The change was made after outcomes were seen, so it is disclosed as **incident 2** (`posthoc_change`).
- The promotion run reports the decisions of the frozen screen beside the amended ones (sensitivity analysis).
- Confirmatory decisions use the frozen thresholds and rules unchanged.

**Incident 1** (entry 435, `engineering_bug`): the gap-duration quality rule was implemented as (number of gaps ×
5 s). Two Binance explore days were wrongly labelled AVAILABLE.
- The fix sums the gap durations, with regression test `test_long_gap_duration_is_summed_not_counted`.
- Every cache was requalified under the corrected rule.
- The two days were replaced by registered backups.
- No study outcome had been computed on them.

**Incident 3** (entry 483, `other`): a superseded bash queue for the explore descriptive studies survived its stop
and ran concurrently with the fail-closed runner. Write-once directories prevented any overwrite, the source-binding
guard refused the one run whose worktree changed mid-run, and every duplicate or aborted attempt is ledgered.

**Incident 4** (entry 619, `engineering_bug`): after confirmatory close, a read-only preview of the final analysis
showed H16 using 6 of 8 registered units. The final analysis ordered continual units by the later day instead of the
registered earlier day. It was fixed with a regression test. It was found after outcomes were visible, so it is
disclosed. H16 is ESTABLISHED with 6 or 8 units, and no other unit count was affected.

**Incident 5** (entry 690, `engineering_bug`): four vault primaries returned provider HTTP 404 and were ledgered
NOT_AVAILABLE. The driver did not re-apply the registered backup rule in later rounds, so rounds 2–8 re-requested
refused slots without reading anything. Fixed with a regression test; the registered backups filled the slots.

**Failed attempts kept on record** (ledger `attempt` events; directories kept where they exist):

| Attempt | Entry | Outcome | Cause and replacement |
|---|---|---|---|
| first explore data preparation | 384 | RESOURCE_EXHAUSTED | memory ceiling; rerun |
| first continual-calibration pilot | 476 | INVALID | JSON string keys in aggregation; rerun as `continual-pilot-2` |
| first fill-study pilot | 477 | INVALID | NaN refused at sealing; rerun as `fills-pilot-2` |
| descriptive generator-fit, adversarial and counterexample | 478–482 | INVALID / ABORTED | the concurrent queues of incident 3; reruns `generator-fit-2` and `counterexample-2` (adversarial: the stale queue's sealed run is the valid one) |
| dry runs of bounds, bounds-late and transfer | 492, 494, 495 | INVALID | a namespace check and a CLI phase, fixed before any confirmatory access |
| first report run | 699 | INVALID | claim-graph key; rerun as `report-2` |

Every rerun used the same registered design and seeds. Quality and provider failures of individual days are
ledgered as NOT_AVAILABLE and replaced by registered backups in frozen order.

## Run evidence

Every study writes a write-once run directory (run schema v3), sealed by `lob.v09.artifacts.finalize`:
- config, result, provenance (source hashes, git state, runtime fingerprint), resources and lineage;
- a binding: protocol hash, confirmatory-registry hash, ledger anchor, dataset identities with raw source hashes,
  config and result hashes, seeds and namespace;
- checksums.

`cleo verify-v09` re-checks:
- the bytes, bindings and lineage of every run;
- that the experiment DAG is acyclic;
- that every retained attempt directory still exists.

Verification establishes integrity and binding. It is not independent replication.
