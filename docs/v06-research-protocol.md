# v0.6 research protocol

The machine-readable protocol is [`configs/v06/protocol.json`](../configs/v06/protocol.json),
schema `cleolob-v06-protocol-1`. Its canonical SHA-256 is recorded by the `freeze` entry of
[`configs/v06/consumption-ledger.jsonl`](../configs/v06/consumption-ledger.jsonl).

Verify it with:

```
cleo protocol-v06 verify     # hash, ledger chain and rules, referenced configs, pinned v0.5 history
cleo protocol-v06 status     # adds sealed designs, holdout first-access indices and recorded attempts
```

`verify` reports `valid: true` only when all of the following hold:
- the protocol hash equals the ledger's current protocol hash (the freeze hash, or the
  latest amendment's);
- every ledger entry chains and hashes correctly;
- every rule replays without violation;
- `configs/v06/datasets.json` is unchanged since the freeze;
- the v0.5 protocol hash, ledger length and ledger head equal the values pinned in
  `v05_history`.

## Statuses

There are eight evidence statuses:
- `ESTABLISHED`
- `NOT_ESTABLISHED`
- `FAILED`
- `INVALID`
- `NOT_AVAILABLE`
- `INCONCLUSIVE`
- `EXPLORATORY`
- `ASSUMPTION_DEPENDENT`

Scorecard equivalence uses four statuses: `EQUIVALENT_WITHIN_MARGIN`, `NOT_ESTABLISHED`,
`FAILED_MARGIN` and `NOT_EVALUABLE`.

`NOT_ESTABLISHED` never means equivalence or invariance.

## Ledger rules (enforced by replay)

| Rule | Detects |
|---|---|
| v0.5 consumed datasets and calendar periods are imported before the freeze. A consumed id or period cannot be declared fresh. | relabelling consumed data as fresh |
| Each access names a dataset, an analysis and a *use*. Allowed uses depend on role: development→`develop`, selection→`select`, retrospective→`retrospective`, holdouts→`download`/`evaluate`/`evaluate_posthoc`. | dataset-role violations |
| A holdout access requires a sealed design that lists it in `reads`, sealed before the holdout's first access. | final-holdout misuse |
| After first access, a new design that reads a holdout must be `posthoc`, and its accesses must use `evaluate_posthoc`. | reuse of consumed holdouts as fresh evidence |
| Source hashes may not change after first access. | silent data substitution |
| `attempt` events record failed, invalid or not-available attempts, with their retained directory. | hidden failures |
| Amendments carry the new protocol hash and may not affect accessed holdouts. | post-hoc protocol edits |
| Each entry hash-chains to its predecessor (index, `prev_sha256`, `sha256`). | edits, truncation (via anchors), reordering, malformed or blank lines |

The ledger is not an external timestamp authority. It cannot prove that nobody looked at
data outside this workflow.

## Chronology

1. **Development** (ETH 2020-04-01): fitting, observable binning, scales and thresholds.
2. **Selection** (ETH 2020-05-01): model selection, the near-optimal region and the ensemble.
3. **Sealed designs** before any holdout access, covering:
   - observables and margins;
   - calibration selection;
   - environment freeze;
   - policy registration;
   - domain gap, regimes and transfer.
4. **Retrospective** data (June, July ETH, July BTC, August ETH), consumed by v0.4/v0.5.
   Its results are labelled retrospective.
5. **Fresh holdouts:** ETH 2020-09-01 (same instrument), BTC 2020-09-01 (cross-instrument)
   and ETH 2020-10-01 (historical execution transfer, read only after the policy evaluation
   lock).

## Hypotheses and families

H1–H12 are listed in the protocol with these fields:
- estimator;
- datasets;
- sample unit;
- uncertainty method;
- multiplicity family;
- threshold;
- failure semantics.

Family sizes and adjusted alphas are stored explicitly. Families whose size depends on the
ensemble size K are sealed numerically before evaluation. Confirmatory families use
Bonferroni or Holm. FDR is never used in a confirmatory family.

| Family | Kind | Members | Size | Correction |
|---|---|---|---|---|
| F1 calibration | confirmatory | H1 (retrospective), H2, H3 | 3 | Bonferroni |
| F2 identifiability | descriptive | H4 | 1 | none (count rule) |
| F3 equifinality | confirmatory | H5 | 4 × C(K,2) | Bonferroni |
| F4 rank stability | confirmatory | H6 | 15 × K | Bonferroni |
| F5 sensitivity | confirmatory | H7 | 9 | Holm |
| F6 domain gap | confirmatory | H8 | 1 | — |
| F6b domain gap secondary | exploratory | — | 4 | Holm |
| F7 regime | confirmatory | H9, H10 | 4 | Bonferroni |
| F8 transfer learning | confirmatory | H11 | 4 | Bonferroni |
| F9 fill semantics | confirmatory | H12 | 30 | Bonferroni |
| F10 scorecard equivalence | descriptive | — | 9 per dataset × model | Bonferroni |
| F11 transfer sources | descriptive | — | 15 × 4 | Bonferroni |

## Invalidation and retention

These invalidation reasons are accepted:
- a holdout download that fails after one retry is `NOT_AVAILABLE`, with no substitution;
- a source integrity failure is `INVALID`;
- a binding or seal mismatch is `INVALID`;
- more than 5% `INVALID` episodes in a cell makes the cell `INVALID`;
- an implementation bug found after a registered run keeps that attempt. The fix is sealed
  as an implementation revision, and any rerun after outcomes were seen is `EXPLORATORY`.

## Evidence binding

Each sealed run in `results/v06/<analysis>` binds:
- the protocol hash;
- the ledger anchor;
- dataset identities and roles;
- normalized config and result hashes;
- seeds;
- source hashes, git commit and package version.

`cleo verify-v06 <dir>` re-derives the binding and checks every byte. Verification means
byte integrity plus binding consistency. It is **not** independent scientific replication.
