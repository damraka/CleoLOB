# v0.5 studies: template and public evidence

`evidence/` is the compact public bundle of every registered v0.5 run. It holds
summaries, bindings (protocol hash, ledger anchor, dataset identity and source hashes,
config/result/implementation hashes, git commit) and the ledger itself. Raw provider data,
per-order and per-episode rows, tapes, fitted historical parameters and checkpoints are
excluded; each excluded file is listed with its SHA-256 in `evidence/export.json`.

```sh
cleo verify-artifact examples/studies/v05/evidence   # bytes of the bundle
cleo protocol verify                                  # protocol, configs and ledger chain
```

A local run directory (not in the bundle) is verified against the ledger with
`cleo verify-v05 results/v05/<milestone>/<run>`.

## Template for a new registered study

v0.5 studies follow one pattern. Reuse it for a new dataset or question:

1. **Declare** the dataset in `configs/v05/datasets.json` and as fresh in a protocol
   amendment *before* any access. A consumed period can never be declared fresh again.
2. **Freeze** the design as a JSON file under `configs/v05/`: datasets and roles, fitted
   quantities and the data they come from, endpoints, statistical family, correction,
   interval method, gate or classification rule, seeds and budgets.
3. **Seal** it in the ledger (`seal_design`) before the holdout is read. Access to a holdout
   is refused until a sealed design lists it, and every access is recorded.
4. **Develop and pilot** on development data only; label pilot runs as such.
5. **Run once** into a new write-once directory. `lob.v05_evidence.finalize` binds the result
   and refuses to seal if the implementation changed during the run.
6. **Retain every outcome.** An execution failure is `INVALID`; a fix is recorded in the
   ledger before a new attempt, and an attempt after any outcome was inspected is labelled
   `EXPLORATORY_POST_HOC` and cannot replace the registered one (see
   [M1](../../../docs/v05-mbo.md) and [M8](../../../docs/v05-transfer.md)).
7. **Export** summaries with `tools/export_v05_evidence.py`.

The commands for every v0.5 study are in [v05-reproduction.md](../../../docs/v05-reproduction.md).
A new order-level source starts from
[`examples/adapters/custom_mbo_adapter.py`](../../adapters/custom_mbo_adapter.py).
