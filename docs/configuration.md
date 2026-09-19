# Research configuration and CLI

Install `python -m pip install -e '.[dev]'` (or `.[dev,rl,web]` for the full
existing app). `cleo` and `python -m lob.cli` are equivalent. `python -m lob`
still runs the original engine demonstration.

```sh
cleo config validate configs/research.yaml
cleo config inspect configs/research.yaml
cleo config schema
cleo config diff configs/base/default.yaml configs/market/thin.yaml
cleo evaluate --config configs/research.yaml
cleo simulate --config configs/base/default.yaml --set execution.side=buy
cleo design --config configs/research.yaml --factors configs/experiments/liquidity_latency.json --budget tiny
cleo validate-data examples/data/canonical-events.jsonl
cleo reconstruct examples/data/canonical-events.jsonl
cleo replay examples/data/canonical-events.jsonl
cleo benchmark --pairs 2000 --repeats 3
cleo verify results/research/EXPERIMENT_ID
cleo report results/research/EXPERIMENT_ID
cleo reproduce results/research/EXPERIMENT_ID
cleo experiment diff results/research/EXP_A results/research/EXP_B
```

`simulate` and `evaluate` both run the configured synthetic execution study and
write a report. `reconstruct` and `replay` both apply all recorded events, without
wall-clock pacing. Replay pause/step/reset are available through the Python API.
`design` only previews selections: **no simulations are launched**. `report`
checks integrity and returns the existing offline HTML artifact; it never
overwrites a sealed run. Commands not listed in `cleo --help` are not implemented.

`download-sample` and `assess-l2` add a separate public aggregate-data workflow.
See [public market data](public-market-data.md) for complete commands and bounds.
These reports are feed-mechanics assessments, not synthetic experiment records.

## Schema and precedence

The frozen Pydantic schema rejects unknown keys, coercion of string quantities,
booleans as quantities, nonfinite values, invalid lots, invalid probabilities,
incomplete agent lists, repeated seeds and excessive workloads. The emitted
JSON Schema contains defaults, descriptions, types, enumerated choices and
numeric bounds. Cross-field validation additionally requires:

- Parent and seeded level quantities align with `market.lot_size`.
- Initial midpoint exceeds the 20-tick seeded ladder.
- Decision interval does not exceed the horizon.
- The reference is among the distinct configured agents.
- PPO requires an explicit local model path. A failed/missing checkpoint cannot
  be replaced by the heuristic in research commands.
- Episodes, estimated work and decisions fit the resource limits.

Composition uses `extends: relative/path.yaml` or an ordered list of files.
Parent files merge recursively left to right, then current fields override.
Paths resolve relative to the referencing file. YAML aliases/object tags,
duplicate YAML/JSON keys, inheritance cycles, excessive nesting and files larger
than 1 MiB are rejected. At most 64 files may be read in one composition.

Environment keys use `CLEO__EXECUTION__QUANTITY=1000`. `--set
execution.quantity=2000` has final precedence. Override values use safe YAML
scalars/lists. Arrays replace rather than concatenate. `extends` disappears
from the resolved settings; output files contain the effective values.
`reproduce` reads stored settings directly and ignores current environment
overrides and inheritance files.

Every nested config object is frozen; sequence fields are tuples. At run start
settings are revalidated, normalized as sorted JSON and SHA-256 hashed. The run
contains that exact JSON, independent of the source file's formatting.

## Fees, risk and bounds

Research presets charge 1 bps taker fees; the old UI/Python defaults remain zero
for compatibility. Maker bps may be negative for rebates. `per_share` is an
additional currency charge per filled unit. There is no minimum fee or tiered
venue schedule in this batch.

Parent reservations always cover both resting and in-flight orders. Configurable
child quantity, estimated notional, position, gross exposure, loss and kill-switch
limits are enforced by the execution controller. Monetary checks occur before
submission at visible prices; later marks may breach these estimates. These
limits are not a portfolio margin system. Cancels remain subject to latency and
can lose to fills. A loss limit stops new orders and requests cancellation;
automatic flattening is not implemented.

After decisions stop, `execution.settlement_timeout` and
`execution.settlement_poll_dt` control cancellation/fill draining. Their ratio
cannot exceed 100,000. Late fills/fees enter actual accounting; settlement timeout
makes economic metrics INVALID/null. The event-work estimate includes settlement.
The separate [portfolio workflow](validation-and-risk.md) supplies linear
multi-currency margin/exposure controls through its own strict JSON configuration.

`max_episodes` and `max_decisions_per_episode` reject excessive designs before
launch. `max_events_per_episode` stops exchange processing at its hard event cap
(also applied to each synthetic VWAP calibration path). `max_estimated_events`
is an admission estimate rather than a memory guarantee. `max_runtime_seconds`
is checked **between episodes**, not inside matching or model loading. An episode
can overrun that wall-time budget; remaining episodes are then marked PARTIAL.
Data replay has separate byte/row/event bounds documented in `historical-replay.md`.

## Output and integrity

Each new directory has a timestamp plus random unique ID. Files include
`resolved_config.json`, `metadata.json`, `episodes.jsonl`, `episodes.csv`,
`summary.csv`, `paired.csv`, `result.json`, `report.html`, `logs/` and `source/`.
Every planned episode gets an outcome, including failures and episodes not run.
JSONL is flushed after each episode. A hard process kill can leave an unsealed
partial directory; that directory fails verification and is not valid evidence.

The final manifest hashes every artifact. Metadata records full Git commit,
dirty status, actual source-file hashes (including untracked implementation),
runtime/package versions and checkpoint hash. The source snapshot is an audit
artifact, never automatically executed. Exact reproduction requires matching
current code, runtime versions and checkpoint bytes and writes a **new** run.
It compares full episode JSONL bytes. Checksums detect accidental changes but do
not prevent a person with write access from forging both results and checksums.

CSV/JSON are suitable for this small batch. Parquet, DuckDB/SQLite indexing,
parallel registered runs and a model registry remain pending.

Calibration, portfolio and stress-family roots use `plan.json`, `provenance.json`,
`source/`, `result.json` and a complete recursive manifest. `verify/report` detect
this format and include nested child manifests. See `validation-and-risk.md` for
`calibrate`, `stress` and `portfolio` commands and their inference boundaries.
