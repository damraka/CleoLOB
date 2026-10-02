# v0.6 reproduction

**Two different things can be checked:**
- **Verification:** byte integrity plus protocol, ledger, dataset, config and result
  binding. It needs only the repository, and the public bundle for the bundle check.
- **Re-execution:** rerunning the registered workflows. It needs the restricted Tardis files
  locally, and it appends new ledger entries.

Neither is independent scientific replication.

## Verification (no restricted data needed)

```
python -m pip install -e ".[dev,rl]"
cleo protocol-v06 verify                         # protocol hash, ledger chain/rules, pinned v0.5 history
cleo protocol-v06 status                         # holdout first accesses, sealed designs, recorded attempts
cleo verify-artifact examples/studies/v06/evidence   # public bundle bytes (independent checksums)
cleo claim-audit-v06 --claims examples/studies/v06/claims.json --docs docs/v06-final-report.md
cleo protocol verify                             # v0.5 history still valid and unchanged
```

With the local run directories present:

```
cleo verify-v06 results/v06                      # every sealed v0.6 run
cleo verify-v06 results/v06/m13/deribit-eth-perp-2020-09-01-registered
```

## Registered workflow, in ledger order

The commands below are the registered workflow as executed. Re-executing a phase writes a new
run directory and appends new ledger entries. A holdout can never become fresh again, so
re-execution is reproduction of a consumed analysis, not new confirmation.

```
cleo realism-study design --out results/v06/m1/design                       # seals m1-observables-design
cleo calibration-v3 develop --out results/v06/m6/develop
cleo calibration-v3 select --from results/v06/m6/develop --out results/v06/m6/select   # seals selection
cleo regime-calibration --out results/v06/m12/regime                        # seals regime models
cleo execution-study freeze --regime results/v06/m12/regime                 # seals environment-freeze
cleo execution-study train --out results/v06/m10/policies                   # seals policy-registration
cleo execution-study evaluate --policies results/v06/m10/policies --out results/v06/m10/evaluation
cleo holdout-study bank --regime results/v06/m12/regime --out results/v06/m13/bank-2
cleo holdout-study seal --bank results/v06/m13/bank-2 --regime results/v06/m12/regime
cleo holdout-study evaluate --bank results/v06/m13/bank-2 --dataset deribit-eth-perp-2020-06-01 --out ...
cleo holdout-study evaluate --bank results/v06/m13/bank-2 --dataset deribit-eth-perp-2020-09-01 --out ...-registered
cleo holdout-study evaluate --bank results/v06/m13/bank-2 --dataset deribit-btc-perp-2020-09-01 --out ...
cleo identifiability-study --out results/v06/m7/identifiability
cleo transfer-study-v06 seal
cleo transfer-study-v06 evaluate --dataset deribit-eth-perp-2020-10-01 --out results/v06/m14/deribit-eth-perp-2020-10-01
cleo benchmark-v06 --out results/v06/m16/benchmarks-2
cleo report-v06 --out examples/studies/v06                                  # figures + claims.json
```

**Determinism.**
- Every stochastic step takes its seed from the protocol seed table or a sealed design.
- Worker count changes speed, not results. Candidates of a search phase are drawn before
  evaluation.
- A pilot and the registered observable design produced byte-identical margins.

**Pilots.** Pilot runs (`results/v06/pilot/*`) use reduced budgets. They are labelled, never
sealed in the ledger and never used as evidence.

## Provenance line endings

Run `provenance.json` files hash source bytes as they were in the Windows working copy. In
14 of the 17 registered runs, 1–3 `lob/v06` files were hashed as CRLF bytes, and 15 runs
record `git_dirty: true` because the append-only ledger was uncommitted mid-run.

- **Same code.** Converting the committed LF files to CRLF reproduces every recorded hash
  exactly, so each run's code matches its recorded commit.
- **Seals and bindings unaffected.** They use normalized or document hashes and remain valid.
- **Fresh checkouts.** A byte-level source check on a fresh LF checkout may report these files
  as different unless line endings are normalized.

This is provenance-format debt, not a difference in any result. See
[v06-final-report.md](v06-final-report.md).

## Retained attempts

| Directory | Outcome | Why |
|---|---|---|
| `results/v06/m13/bank` | INVALID | NaN transition probabilities refused at sealing; simulation-only, before any holdout access |
| `results/v06/m13/deribit-eth-perp-2020-09-01` | INVALID | ledger rule bug (posthoc flag) refused the evaluate access after the recorded download; no holdout byte was read and no outcome produced; fix sealed as `m13-holdout-implementation-revision-1` before the registered rerun |
| none (intended `results/v06/m16/benchmarks`) | INVALID | synthetic benchmark world lacked depleting events. It failed before a run directory was created, so the attempt is retained only as ledger entry 79 and in this chronology. |
| `results/v06/pilot/m1-design-1`, `pilot/m6-*-1` | pilot | first pilots, including a NaN serialization failure |
