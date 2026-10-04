# CleoLOB v0.7 public evidence

`evidence/` is the compact public bundle of the v0.7 research branch (`0.7.0.dev0`, not released).

**Contents.**
- the frozen protocol and its registration documents
- the append-only consumption ledger
- the sealed designs
- every registered run's config, result, evidence binding and provenance
- the result registry, hypothesis table, claim graph and negative-results list
- figure source tables (CSV) with the figure manifest
- the requirement-coverage table and the final report

**Not included.** Restricted provider data and anything derived from it at the window or row
level are excluded:
- simulation banks
- discriminator scores
- replay rows
- posterior particle files
- policy checkpoints

Each excluded file is listed in `export.json` with its SHA-256 and size.

**Verification (tier 1).**

```
cleo verify-v07 --bundle examples/studies/v07/evidence
```

This checks byte integrity and binding consistency. It is not independent scientific
replication.

**Public reproduction (tier 3).**

```
cleo benchmark-v07 public
```

This reruns the public replication subset from code and synthetic fixtures. It must reproduce
the digest in `public-subset-expected.json`. See `docs/v07-reproduction.md` for all four tiers.
