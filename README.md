# CleoLOB — Execution research

Fixed-budget PPO reduced mean net execution cost relative to simulator-fitted,
risk-neutral Almgren–Chriss by **1.605 bps: PPO minus AC = −1.605 bps, 95% CI
[−2.095, −1.015]**, across five optimizer seeds and 32 common market seeds.
The primary policy actually filled **86.90%** of the parent order on average;
the endpoint includes hypothetical visible-book liquidation of the remaining
13.10%. The simulator failed **all six external real-data calibration gates**,
so this is a conditional synthetic result, not demonstrated market performance.

CleoLOB combines a FIFO exchange, explicit execution accounting and a recorded
PPO comparison. The active scope is this research question. Portfolio/FX, new
interfaces and additional model families are frozen.

## Final result

All **20 PPO models**, **163,840 training steps** and **704 final evaluation
episodes** completed. Training and final evaluation contain **zero INVALID
episodes**. Mean net cost was **0.583 bps** for primary PPO and **2.188 bps** for
risk-neutral AC. The primary comparison against the fixed risk-sensitive AC
comparator was **−1.658 bps, 99% CI [−2.229, −0.877]**.

| Training completion penalty | PPO minus risk-neutral AC (bps) | Family-adjusted 99% CI | Mean actual fill |
|---|---:|---:|---:|
| **0 bps — primary** | **−1.605** | **[−2.213, −0.810]** | **86.90%** |
| 5 bps | −1.839 | [−2.256, −1.343] | 98.42% |
| 25 bps | −1.745 | [−2.164, −1.237] | 98.67% |
| 100 bps | −1.675 | [−2.160, −1.126] | 100.00% |

The primary remains the preregistered 0 bps arm. The 100 bps arm completed every
evaluated parent order; its mean economic cost was 0.513 bps. Training completion
penalties accounted for 0%, 1.11%, 1.18% and 4.60% of absolute reward components
on average across the five optimizer seeds in the respective arms. They did
not dominate the recorded training objectives. All final outcomes were priced,
so the 100/500 bps invalid-residual sensitivities reproduce the raw comparisons.

Read the [result summary](examples/studies/core/reports/summary.md), inspect the
[sealed result](examples/studies/core/ppo-final-20260922/result.json), or download
and open the [standalone Plotly report index](examples/studies/core/reports/index.html)
with its neighboring HTML files. Reports include confidence intervals, training
curves, reward decomposition and actual/hypothetical execution-cost components.

## Study design and evidence

The [frozen configuration](configs/core-study.json) sells **1,714 synthetic lots
over 240 seconds**, with a 30-second warmup, 5-second policy decisions, zero maker
fees and 1 bps taker fees. Five optimizer seeds are trained separately at each
completion penalty: **0, 5, 25 and 100 bps**, using **8,192 steps per model**.
The 163,840-step budget is fixed; it is not evidence of policy convergence.

Economic cost includes actual fill shortfall and fees plus hypothetical
visible-book liquidation of any priceable residual. The completion penalty
affects training, but is excluded from the economic endpoint. Actual fill
fractions remain separate from hypothetical terminal valuation. Unpriced
residuals remain INVALID; they are not counted as zero-cost outcomes.

The primary comparator is risk-neutral AC, whose analytical schedule equals
TWAP. The fixed risk-sensitive comparator uses κT=1, with risk aversion
4.98329×10⁻⁵. Independent no-parent paths fitted temporary impact
η=5.51834×10⁻⁵ and volatility σ=0.00438465 at the AC schedule's 12-second child
interval. The [fit](examples/studies/core/execution-controls/execution-amendment/final-ac-fit/ac-fit.json)
passes its gate: R²=0.5013 and 92.94% probe coverage. It estimates a local
visible-depth cost approximation, not causal or permanent market impact.

| Evidence | Outcome |
|---|---|
| [Original positive-control grid](examples/studies/core/execution-controls/compute-continuation/positive-control/result.json) | FAIL; retained |
| [Registered execution amendment](examples/studies/core/execution-controls/execution-amendment/plan.json) | Finite ordered search; physical simulator unchanged |
| [Independent control confirmation](examples/studies/core/execution-controls/execution-amendment/result.json) | PASS; Random−TWAP −0.9591 bps, 95% CI [−1.1953, −0.7155]; 64 seeds and valid capacity controls |
| Additional capacity preflight | 384 fixed-action/random episodes; no invalid outcomes in the selected setting |
| [Final AC parameter fit](examples/studies/core/execution-controls/execution-amendment/final-ac-fit/ac-fit.json) | PASS; separate seeds 46000–46015 |
| [PPO registration and sealed result](examples/studies/core/ppo-final-20260922/preregistration.json) | 20 models; 704/704 final episodes; integrity verification PASS |

Training completed **4,957 episodes and 163,840 steps with zero INVALID episodes**.
Diagnostic seeds 61000–61019 locked **32 final market seeds, 71000–71031**, before
this run's final evaluation. Estimated power is **81.48%** for a 0.5 bps primary
effect, against an 80% target; the 256-market cap was not binding. The pilot
variance estimate and five optimizer seeds limit the precision of that estimate.
Training markets use [1000000, 2000000). Crossed bootstrap intervals independently resample
optimizer and common market seeds. The primary interval is 95%; the five-member
family of penalty arms and fixed AC sensitivity uses 99% intervals. A single
invalid outcome does not suppress unrelated comparisons; incomplete-pair
intervals and 100/500 bps residual-price sensitivities are explicitly labeled.

This local computation is a **computational replication**, not a fresh test
holdout: two earlier Linux CI runs already executed the fixed design. Their
results are not pooled as extra independent seeds. An interrupted local serial
attempt remains preserved; the retained run uses isolated worker processes
under a new registration, without changing scientific settings. See the
[protocol and chronology](docs/core-research-protocol.md),
[computational amendment](examples/studies/core/PARALLEL_REPLICATION.json) and
[prior CI evidence](examples/studies/core/ci-prior-runs/export-provenance.json).

## Real-data calibration failed

Public Deribit BTC and ETH perpetual top-five samples cover July 1, August 1
and September 1, 2026. July data fitted the physical simulator. Frozen ETH, BTC
and pooled fits each failed August validation and the previously unopened
September holdout: **6/6 FAIL**. The four distinct external files contain
**7,461,728 snapshots**, producing **345,595 causal one-second observations**.
Pooled results reuse those observations and are not additional independent data.

All September cohorts also failed the unchanged 95% valid-observation gate.
For ETH, September spread and visible-impact errors were 1.0534 and 1.3693,
against a maximum allowed error of 0.60. Models were not retuned after these
failures. September is now consumed holdout evidence.

The [full assessment](examples/studies/core/calibration/EXTERNAL_RESULTS.md)
retains every failed family and source/model hash. L2 snapshots do not identify
FIFO queues, hidden liquidity or historical counterfactual fills. Synthetic
normalized lots are not funded inverse-perpetual positions. This study does not
establish live alpha or historical strategy performance.

## Run and verify

Python 3.11+ is supported. The retained local runtime is Python 3.14.6 on Windows;
Linux correctness CI covers Python 3.11 and 3.13.

```sh
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e '.[dev,rl,plots]'

python -m pytest -q
python -m ruff check lob tests tools legacy/evaluate.py train_rl.py server.py
cleo config validate configs/core-study.json
python train_rl.py verify --study examples/studies/core/ppo-final-20260922
python tools/report_core_study.py examples/studies/core/ppo-final-20260922 --out results/retained-reports
```

The full local suite passes **573 tests**. Two existing Gymnasium warnings concern
the unbounded observation Box. Linux CI also checks saved study integrity,
configuration/data validation and a small matching benchmark. Full retraining is
an explicit workflow option that uploads completed or failed evidence.
See [Execution research CI](https://github.com/damraka/CleoLOB/actions/workflows/research.yml).

To repeat the fixed design in a new directory:

```sh
python train_rl.py register --config configs/core-study.json --out results/replication/study
python train_rl.py train --study results/replication/study --workers 4
python train_rl.py evaluate --study results/replication/study
python train_rl.py verify --study results/replication/study
python tools/report_core_study.py results/replication/study --out results/replication/reports
```

Registration freezes configuration and source hashes. Training aborts at the
first invalid terminal economic outcome and preserves the attempt. Completed
checkpoints can be reused only after hash verification. Reports are generated
outside the sealed study directory and do not rerun policies. Reusing this
design's seeds is a replication, not a new independent experiment.

## Scope and earlier evidence

The engine uses integer ticks/lots, FIFO priority, persistent event clocks,
separate random streams, delayed messages, order-state tracking and in-flight
reservations. Settlement reconciles late fills and fees before final valuation.
Research artifacts retain configurations, source snapshots, checkpoints,
training traces, every evaluation outcome and integrity manifests.

The earlier [300-episode stress study](examples/studies/validation/README.md)
retains 93 INVALID episodes. Its separately labeled
[residual-price sensitivity](examples/studies/core/stress-sensitivity/20260919T092238-bbe58e0a27dd/report.md)
recovers unaffected raw comparisons and reports explicit adverse-price
assumptions instead of removing failures. Earlier
[historical replay](examples/studies/historical/README.md) and
[10-seed foundation](examples/studies/foundation/README.md) records are historical
evidence, not substitutes for the present study.

Old Streamlit, flat training/evaluation scripts and 100-seed artifacts are in
[`legacy/`](legacy/README.md). The existing FastAPI/Three.js exploration UI can be
started with `python server.py --no-browser` after installing `.[web]`.
No broker connection or real-money order submission is included.

- [Execution protocol](docs/core-research-protocol.md)
- [Configuration and provenance](docs/configuration.md)
- [Public data commands and limitations](docs/public-market-data.md)
- [Architecture](docs/architecture.md)
- [Full platform checklist and historical batches](docs/implementation-status.md)

The bounded execution study does not complete every item in the broader platform
brief. Historical counterfactual execution, exchange-native feed adapters,
additional RL families and advanced market dynamics remain outside this release.
