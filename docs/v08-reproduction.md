# Reproducing v0.8

The following steps are all offline. None of them needs restricted data.

```
python -m pip install -e ".[dev]"
cleo protocol-v08 verify                      # protocol, ledger, pinned v0.7 history
cleo smoke-v08 --out /tmp/cleo-smoke-v08      # generated-data smoke (isolated ledger)
cleo verify-v08 --bundle examples/studies/v08 # public evidence bundle: bytes, bindings, ledger chain
cleo claim-audit-v08 --bundle examples/studies/v08
python -m pytest -q tests/test_v08_*.py
```

## Rerunning studies

The full chain needs the consumed Tardis samples in `data/` (they are not redistributed). The order is:

1. `cleo inference-v08 posterior --out results/v08/inference/posterior-2 --budget medium`
2. `cleo inference-v08 worlds --out results/v08/inference/worlds --input posterior=...`
3. The confirmatory studies, with the sealed designs in `configs/v08/designs/`, for example:
   `cleo mpc-study-v08 --namespace confirm --design confirm-mpc --budget medium --input worlds=...`

   A fresh rerun needs a new design name. The `confirm` namespace is opened through the ledger and recorded,
   so a rerun never silently reuses or replaces sealed evidence.
4. The exploratory studies, in the `dev` namespace.
5. `cleo final-analysis-v08`, then `cleo report-v08` and `cleo evidence-export-v08`. The final analysis takes one
   `--input KEY=RUN` per sealed confirmatory run, plus `worlds` (needed by the H21 synthetic holdout),
   `transfer_fresh` and `ranking`. The sealed final analysis is `results/v08/final-2`; `results/v08/final` is a
   retained INVALID attempt. Pass `--final results/v08/final-2` to `report-v08` and `evidence-export-v08`.

## Long runs

Long runs were executed from git worktrees pinned to a committed revision:
- `61f03a1`: the registered posterior;
- `e5a40a9`: the confirmatory studies;
- `aeccdf0`: the exploratory studies.

Each run's `provenance.json` records the line-ending-normalized hashes of the code that actually executed. A
source change during a run refuses sealing. One such attempt is retained and ledgered.

## Determinism and tolerances

- Seeds are derived from `(namespace, study, index)` and never from scheduling.
- Serial and parallel runs give identical results (`tests/test_v08_statistics.py`).
- Byte-identical artifacts are expected on the same platform and numpy version.
- Across platforms, floating-point summation may differ in the last digits. Comparisons should use relative
  tolerances of 1e-9 for statistics.
