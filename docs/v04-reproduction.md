# Reproduce v0.4 development evidence

Use Python 3.11+ and the implementation commit recorded in each registration.
This is `0.4.0.dev0`; the published stable release remains separate. Install from
the review branch with `python -m pip install -e '.[dev,rl]'`. Run commands from
the checkout, with a new output directory for every attempt. `cleo` and
`python -m lob.cli` are equivalent. No command below publishes a release.

```sh
python -m pytest -q
python -m ruff check lob tests tools train_rl.py server.py
python -m compileall -q lob tools train_rl.py server.py
python -m pip check
python -m build --outdir results/v04/reproduced-distributions

cleo validate-mbo examples/studies/v04/mbo-input/synthetic-events.jsonl --manifest examples/studies/v04/mbo-input/adapter-manifest.json --references examples/studies/v04/mbo-input/synthetic-aggregate.jsonl --out results/v04/mbo-reproduced
cleo verify-artifact results/v04/mbo-reproduced
cleo multiperiod-study --config configs/v04-multiperiod-smoke.json --out results/v04/multiperiod-reproduced
cleo multiperiod-study --out results/v04/multiperiod-reproduced --verify
python -m lob.historical_generalization --v03-study examples/studies/v03/calibration-holdout-fix --out results/v04/v03-diagnostics-reproduced

cleo policy-study register --config configs/v04-policy-study.json --out results/v04/policy-reproduced
cleo policy-study train --out results/v04/policy-reproduced
cleo policy-study evaluate --out results/v04/policy-reproduced
cleo policy-study verify --out results/v04/policy-reproduced

# Run measured performance after training and tests finish.
cleo scaling-study --out results/v04/scaling-reproduced
cleo verify-evidence results/v04/scaling-reproduced
cleo benchmark-suite --operations 200 --repeats 3 --warmup 20 --out results/v04/calls-reproduced
cleo verify-artifact results/v04/calls-reproduced
cleo smoke --out results/v04/smoke-reproduced
cleo verify-artifact results/v04/smoke-reproduced
cleo verify-artifact examples/studies/v03
```

MBO inputs and multi-period fixtures here are explicitly synthetic. A failed
calibration gate is a retained scientific result, not a command failure. Failed
input integrity is different and stops evaluation. The 24-model RL run requires
the optional RL dependencies and is a bounded study, not adequately powered
confirmation of superiority. All 1,080 cells and final checkpoints must remain.

The historical command is separate:

```sh
python -m lob.historical_generalization --config configs/v04-calibration-diagnostics.json --out results/v04/historical-diagnostics-new
```

It requires the exact old model/result and April/May source hashes declared by
the config. It uses the frozen old model without refitting, declares June 2020
before access, then downloads through the bounded existing provider client.
Detailed output stays local under current provider terms. A local consumption
marker prevents a previously accessed June file from being described as fresh.
Rechecking an already consumed dataset is a replication/retrospective task and
needs a separately labeled protocol; deleting the marker does not restore
freshness. No substituted date or favorable rerun is allowed.

To verify installation, create a new virtual environment, install the built
wheel, change to a directory outside the checkout, and run `cleo --help`,
`cleo smoke --out smoke`, `cleo verify-artifact smoke`, and `python -m pip check`
using that environment's executables. The wheel contains the offline generator;
the smoke does not depend on checkout fixtures/configuration files. Recorded
source copies are for inspection; the verifier never executes them.

Implementation files use LF checkout rules so their byte hashes survive
Windows/Linux checkout. Archived evidence retains its original bytes. Runtime
version changes can still change numerical results; byte verification alone is
not an independent scientific reproduction.
