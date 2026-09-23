# Reproduce the v0.3 research workflows

## Linux-first installation

Requires Python 3.11 or later. CI covers 3.11–3.14 on Linux and Windows.
From a source checkout:

```sh
python tools/bootstrap.py                 # core + tests + wheel tools
. .venv/bin/activate                      # PowerShell: .venv/Scripts/Activate.ps1
python -m pip check
python -m pip wheel . --no-deps -w dist
cleo smoke --out results/smoke/first
cleo verify-artifact results/smoke/first
cleo verify-artifact examples/studies/v03
```

Use a new output directory for each execution: completed evidence is never
overwritten. `python -m lob.cli` is equivalent to `cleo`. Smoke creates small
synthetic L2/MBO inputs, tests identity refusal, observes an actual fixture maker
fill, selects an observable model chronologically, evaluates its shifted later
fixture and executes all six classical controls on two common seeds. It needs
no exchange credentials, downloaded data or RL dependency. A calibration FAIL
is an expected scientific possibility and remains visible in `result.json`.

For isolated-wheel testing, create another virtual environment, install the
wheel there, change directory outside the checkout and run the same smoke.
The smoke fixture generator ships in the wheel. This checks packaging rather
than accidentally importing the checkout.

## Separate registered workflows

```sh
cleo mbo-replay examples/data/mbo-events.jsonl
cleo calibration-study --config configs/v03-calibration.json --out results/v03/calibration-new
python -m lob.generalization --verify --out results/v03/calibration-new
cleo benchmark-suite --operations 200 --repeats 3 --warmup 20 --out results/v03/benchmark-new
cleo verify-artifact results/v03/benchmark-new
```

The MBO fixture is fabricated and redistributable under this repository's
license. It validates mechanics only. The benchmark's 10/100/1,000 levels per
side are artificial books. It profiles before timing, warms each workload,
reports per-call p50/p95/p99 and batch throughput, separately instruments Python
allocations, and records CPU, OS, Python, dependency versions and source hashes.
Timer overhead, OS scheduling and fixed synthetic workloads limit interpretation.
Replay/episode percentiles from three repeats are descriptive, coarse estimates.
No production execution or colocation latency is measured.

To train PPO and DQN:

```sh
python tools/bootstrap.py --rl
cleo policy-study register --config configs/v03-policy-study.json --out results/v03/policy-new
cleo policy-study train --out results/v03/policy-new
cleo policy-study evaluate --out results/v03/policy-new
cleo policy-study verify --out results/v03/policy-new
```

Finish source changes before registration. Training and evaluation reject a
changed registered source. Keep every seed's checkpoint and failure metadata
locally. The checked-in 1,024-step-per-model design is a workflow smoke; it does
not certify convergence or policy superiority. It compares both algorithms with
TWAP, synthetic-profile VWAP, POV, AC, heuristic and random controls across three
regimes, with three optimizer seeds and eight common evaluation market seeds.
The finite observation and terminal-penalty ablations are in the same plan.
See [the RL protocol](rl-v03.md) for metrics and multiplicity.

## Historical evidence and required data

The retained April/May 2020 study needs the Deribit ETH-PERPETUAL Tardis
`incremental_book_L2`, `book_snapshot_5` and `trades` first-day files. Provider
URLs, source hashes, periods, unit caveats and exact assessment commands are in
[public market data](public-market-data.md) and
[the historical evidence](../examples/studies/historical/README.md).
The July/August/September 2026 simulator study uses BTC-PERPETUAL and
ETH-PERPETUAL top-five snapshots; exact identities and download budgets are in
[`DATA_PLAN.json`](../examples/studies/core/DATA_PLAN.json),
[`download_core_data.py`](../tools/download_core_data.py), and
[the original protocol](core-research-protocol.md).

Those prior later-date holdouts have already been consumed. They are historical
failure evidence, not fresh confirmation for a newly selected model. The new
calibration workflow accepts hash-declared, portable feature CSVs, but no fresh
historical holdout is supplied or claimed. Genuine MBO requires a separately
licensed source with real order IDs, complete sequence coverage and documented
snapshot/modify priority semantics. No suitable MBO file was available here.

Compact summaries do not contain old model checkpoints/raw datasets. Re-running
old studies may require downloading permitted source data and retraining; byte
verification alone is not reproduction or independent validation. Do not execute
archived source automatically or silently repair the retained failed outcomes.

## Container

```sh
docker build -t cleolob .
docker run --rm -v "$PWD/results:/work/results" cleolob smoke --out results/container-smoke
docker run --rm -v "$PWD/results:/work/results" cleolob verify-artifact results/container-smoke
```

The minimal image includes core dependencies and the installed package, with no
market data, Torch, checkpoints or desktop UI. It uses Python 3.12 slim; package
versions are recorded per run rather than claiming a permanently pinned image.

## Evidence integrity and review

`verify-artifact` rejects missing, changed, added and unsafe-path files. SHA-256
detects changed bytes; it cannot prove that an author generated truthful results.
Committed v0.3 evidence is a compact export with source identities and original
full-run manifests; large checkpoints, traces and source snapshots remain in
ignored local results. Use [the reproduction template](reproduction-report-template.md)
to report success or failure, including the exact command, commit, config hash,
dataset hash, environment and expected/observed outcome.
