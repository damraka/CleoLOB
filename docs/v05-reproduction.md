# Reproducing v0.5

## Prerequisites

- Python 3.11+ and `python -m pip install -e ".[dev,rl]"`.
- The ledger enforces order: holdout access is refused unless the designs that read it are
  sealed. Every run directory is write-once, so each attempt needs a new `--out`.
- Raw provider data are downloaded into ignored `data/v05/` directories. The Tardis public
  first-of-month files need no credentials. The Bitstamp captures are one-off live recordings:
  new captures are new datasets and must be declared in a protocol amendment before recording.

## Verify the registration

```sh
cleo protocol verify           # protocol hash, referenced configs, ledger chain, freshness
cleo dataset-registry          # all registry fields; freshness derived from the ledger
```

## Commands in dependency order

```sh
# M1/M2 order-level validation on an existing capture (frozen; --posthoc is exploratory only)
cleo validate-mbo-source data/v05/bitstamp/validation-1800s-retry1 --dataset-id bitstamp-btcusd-mbo-validation --out results/v05/m1/<new>

# M2 aggregate-L2 bounded execution
cleo fill-bounds deribit-eth-perp-2020-04-01 --out results/v05/m2/<new>

# M3 calibration v2: develop -> select (appends the m3-selection ledger seal) -> evaluate holdouts
cleo calibration-v2 develop --out results/v05/m3/<develop>
cleo calibration-v2 select --from results/v05/m3/<develop> --out results/v05/m3/<select>
cleo calibration-v2 evaluate --from results/v05/m3/<select> --out results/v05/m3/<evaluate>

# M4 impact/resilience: development run freezes bucket thresholds; holdouts reuse them
cleo impact-study deribit-eth-perp-2020-04-01 --select results/v05/m3/<select> --out results/v05/m4/<dev>
cleo impact-study deribit-eth-perp-2020-07-01 --select results/v05/m3/<select> --thresholds-from results/v05/m4/<dev> --out results/v05/m4/<ext>

# M6 regimes and external validity
cleo regime-study thresholds --select results/v05/m3/<develop> --out results/v05/m6/<thresholds>
cleo regime-study evaluate --dataset deribit-eth-perp-2020-07-01 --select results/v05/m3/<select> --m4-dev results/v05/m4/<dev> --m6 results/v05/m6/<thresholds> --out results/v05/m6/<ext>

# M7 registered policy study (requires the sealed environment freeze)
cleo policy-study-v05 register --out results/v05/m7/<study>
cleo policy-study-v05 train --out results/v05/m7/<study>
cleo policy-study-v05 evaluate --out results/v05/m7/<study>
cleo policy-study-v05 verify --out results/v05/m7/<study>

# M8 transfer on the reserved August holdout and the July periods
cleo transfer-study --m7 results/v05/m7/<study> --m3-develop results/v05/m3/<develop> --out results/v05/m8/<new>
cleo transfer-regimes --m8 results/v05/m8/<run> --m7 results/v05/m7/<study> --m3-select results/v05/m3/<select> --m6 results/v05/m6/<thresholds> --out results/v05/m8/<run>-regimes

# M11 benchmarks, then verify any v0.5 run directory
cleo v05-benchmark --out results/v05/m11/<new>
cleo verify-v05 results/v05/<milestone>/<run>
```

Each module is also runnable with `python -m lob.<module>`.

## Public evidence bundle

```sh
python tools/export_v05_evidence.py --runs <run directories> --policy-study results/v05/m7/<study> --out <new directory>
cleo verify-artifact examples/studies/v05/evidence
```

The bundle excludes:
- raw data;
- per-order and per-episode rows;
- tapes and detailed historical summaries;
- fitted historical model parameters;
- checkpoints.

Each exclusion is listed with its original SHA-256.

## What byte verification does and does not establish

Checksums, bindings and ledger anchors detect accidental or silent changes to:
- data, configs, results, registrations, checkpoints and normalization;
- the implementation source and the commit.

They are not an external timestamp authority, and they do not prove that nobody looked at
data outside this workflow. A matching hash is not an independent scientific replication.
