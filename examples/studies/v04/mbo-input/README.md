# Synthetic MBO ingestion inputs

These project-authored events and hand-computed aggregate reference states are
synthetic regression fixtures under the repository license. They are not native
exchange records or historical market evidence. The lifecycle extends the
existing canonical fixture into the v0.4 adapter/validation interface.

`adapter-manifest.json` freezes source identity, byte digest, units and supported
semantics. Prices and quantities use arbitrary ticks and lots (unit size 1).
`synthetic-aggregate.jsonl` contains five explicitly selected observation
boundaries and full-book integer levels. No source sequence continuity is
claimed: contiguous synthetic canonical indices test rejection mechanics only.

Run the command in `docs/v04-data-contracts.md` into a fresh output directory.
The empirical milestone remains `REAL_HISTORICAL_MBO_VALIDATION=NOT_AVAILABLE`.
