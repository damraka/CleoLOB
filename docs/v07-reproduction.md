# v0.7 reproduction

## Reproduction tiers (workstream 71)

| Tier | Meaning | Command |
|---|---|---|
| 1 — byte integrity | every file of a run or bundle matches its checksum; bindings match the ledger | `cleo verify-v07` (runs), `cleo verify-v07 --bundle examples/studies/v07/evidence` |
| 2 — artifact regeneration | registry, hypothesis table, claim graph, figures (with CSV sources) and explorer are regenerated from sealed run files | `cleo report-v07 --out results/v07/report` |
| 3 — public-data reproduction | the public replication subset reruns from code and synthetic fixtures only, and must match the stored digest | `cleo benchmark-v07 public` |
| 4 — full reproduction | requires the restricted Tardis.dev sample files and the self-recorded Bitstamp captures; rerun every registered study from the protocol | see "Full reproduction" below |

**Scope of the tiers.** Tier 1 and tier 2 are integrity and regeneration checks. They are never
independent scientific replication.

## Public replication subset (workstream 93)

The command `cleo benchmark-v07 public` recomputes the following and compares one SHA-256 digest
with `examples/studies/v07/public-subset-expected.json`:
- queue-model bound ordering on 500 random level histories (all 500 ordered)
- SMC-ABC recovery on an analytic 14-dimensional problem
- the decision-certification truth table
- matching-engine quantity conservation on 400 random orders
- the transfer-hierarchy variance decomposition
- identifiability rank of a known rank-deficient model
- equality of the v0.6 and v0.7 tape builders, with the replay hash chain, on a synthetic file

No restricted data is involved. The digest excludes the timing fields.

## Environment capture and line endings (workstream 72)

Every v0.7 run records the following in `provenance.json`:
- Python
- platform
- packages
- CPU
- logical CPU count
- git commit and dirty flag
- a source manifest hashed **after normalizing CRLF to LF**

This fixes the v0.6 provenance debt, where a checkout's line-ending conversion changed recorded
hashes. `lob.v07.benchmark.compute.environment()` adds:
- GPU status (NOT_AVAILABLE on the development machine)
- relevant environment variables
- the deterministic flags in use

## Full reproduction (tier 4)

1. Obtain the Tardis.dev free first-day-of-month samples for the dataset IDs in
   `configs/v07/protocol.json`.
2. Place them under `data/v07/tardis` (or `data/public`, `data/v06/tardis`).
3. Run the registered commands in order:

   ```
   cleo generator-study-v07 --out results/v07/m5/fit
   cleo posterior-study-v07 recovery --out results/v07/m6/recovery-2
   cleo posterior-study-v07 run --out results/v07/m6/posterior-2
   cleo posterior-study-v07 select --out results/v07/m6/select
   cleo calibration-v4 --out results/v07/m7/execution-aware
   cleo realism-v2 bank --out results/v07/m10/bank
   cleo realism-v2 seal --out unused
   cleo realism-v2 evaluate --dataset <id> --out results/v07/m17/<id>
   cleo execution-study-v07 --out results/v07/m15/execution
   cleo robust-policy-study-v07 run --out results/v07/m14/policies
   cleo robust-policy-study-v07 predict --out results/v07/m16/predictions
   cleo transfer-study-v07 seal --out unused
   cleo transfer-study-v07 run --out results/v07/m16/transfer
   ```

**Ledger and seed caveats.**
- The ledger is append-only. A full reproduction on a fresh clone must use a new ledger
  (`protocol-v07` initialization in a fresh root).
- In that new ledger, the holdouts consumed by this study are no longer fresh evidence for this
  team's claims.
- Seeds are fixed in the protocol and in sealed designs.
- Worker counts do not change results: tasks carry their own seeds, and results are collected
  in task order.
