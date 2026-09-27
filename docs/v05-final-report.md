# v0.5 final report: real-market execution validation

Branch `research/v0.5-real-market-validation`, from base `224415d`. Package version
`0.5.0.dev0`. **This is not a release:** there is no tag, no PyPI upload and no GitHub
release, and nothing is merged to `main`.

The scientific narrative is in [v05-paper.md](v05-paper.md). This report records status,
evidence locations and the release-gate audit.

## Milestone status

| Milestone | Software | Empirical outcome |
|---|---|---|
| M0 protocol freeze | done: protocol hash `feeec468…`, hash-chained consumption ledger (80 entries) | — |
| M1 order-level (MBO) validation | done: Bitstamp adapter, lifecycle replay, reference comparison | registered **INVALID**; post hoc would be **FAILED** (E1 0.867); vendor archive **NOT_AVAILABLE** |
| M2 bounded historical execution | done: nested fill bounds, frozen class vocabulary | descriptive bounds on 5 periods plus Bitstamp; bounds never falsified by 2,702 real orders (post hoc) |
| M3 calibration v2 | done: 6 extensions, one observable operator, sealed chronology | improvement **ESTABLISHED** on June and July ETH, **NOT_ESTABLISHED** on BTC; every gate **FAILED** |
| M4 impact and resilience | done | H4 **NOT_ESTABLISHED** everywhere |
| M5 strict completion | done: mandate block for every policy | within-horizon and settlement completion reported separately |
| M6 external validity | done: development-only regime thresholds, dataset registry | only the relative M3 improvement is regime-robust (ETH); cross-venue comparison **refused** |
| Environment freeze | done (`configs/v05/environment-freeze.json`, ledger 56) | the M8-only replay module was revised twice afterwards to fix crashes (ledger 62, 70); M7 never uses it |
| M7 policy study | done: 48 models, 34,560 episodes | 5/64 joint gates pass (DQN only); stress and calibrated costs withheld (INVALID economics) |
| M8 transfer | done | registered attempt 2: 6 reverses, 5 indeterminate, 1 agrees, 27 not_evaluable; attempt 1 **INVALID**; attempt 3 **EXPLORATORY_POST_HOC** |
| M9 report | done: [v05-paper.md](v05-paper.md) plus supporting docs | — |
| M10 reproducibility | done: bindings, tamper tests, package version in provenance, public bundle | every run verifies |
| M11 performance | done: [v05-performance.md](v05-performance.md) | local research workloads only |
| M12 packaging and UX | done: v0.5 CLI, adapter and study templates, missing-dependency errors | wheel smoke passes |

## Datasets

**Used:**
- Deribit ETH-PERPETUAL on 1 April, 1 May, 1 June, 1 July and 1 August 2020, and BTC-PERPETUAL
  on 1 July 2020 (Tardis public samples).
- Bitstamp btcusd live order-level captures: 900 s development, a 1,800 s validation retry,
  and the retained GAP attempt.

**Not available:**
- LOBSTER, which requires registration and terms acceptance.
- Databento MBO, which requires credentials.
- Authenticated Coinbase and Kraken L3.
- Any second venue at the same capability level.

**Freshness.** July ETH, July BTC and August ETH were fresh at freeze. The July periods were
consumed by M2–M6 before M8 read them. Nothing was relabelled fresh.

## Negative, null and invalid results (all retained)

- **M1:**
  - the registered validation is INVALID;
  - the post-hoc E1 of 0.867 would FAIL;
  - rules from a 15-minute development window missed later message types.
- **M2:** 30–41% of hypothetical passive orders stay undetermined from aggregate L2, and the
  BTC bounds span 52% of order size.
- **M3:**
  - every absolute gate FAILED on every holdout for both models;
  - the BTC improvement is NOT_ESTABLISHED;
  - seven candidates hit the event cap;
  - the selection seal followed the first July access by M2 (disclosed).
- **M4:** H4 is NOT_ESTABLISHED on all 4 datasets. Simulated persistence is 0.85, against
  1.10–2.72 historically.
- **M6:** no absolute calibration or impact conclusion survives a regime or transfers to BTC.
- **M7:**
  - PPO passes no contrast;
  - DQN fails against the heuristic and random controls;
  - 109 stress and 5,477 calibrated episodes had INVALID economics, so those costs are withheld;
  - the `no_terminal` ablation is vacuous.
- **M8:**
  - attempt 1 was INVALID (2,185 crashed rows);
  - attempt 2 is registered, but its July comparisons are not_evaluable (96 crashed rows);
  - learned policies are costlier than TWAP, VWAP and AC on the fresh August holdout;
  - the M7 original-regime gate pass is indeterminate;
  - the historical and synthetic rankings are unrelated (τ = −0.14);
  - post hoc, no learned advantage survives the conservative bound on any dataset;
  - the design's regime-sensitivity report was omitted from the run and added afterwards
    (descriptive).

## Evidence locations

- **Public bundle:** `examples/studies/v05/evidence`.
  - 89 files: summaries, bindings, portable provenance, the M7 plan, seals, locks,
    normalization and result, and the protocol and ledger.
  - Every excluded file is listed with its SHA-256.
- **Local sealed runs** (ignored; restricted data): `results/v05/{m1,m2,m3,m4,m6,m7,m8,m11}`.
- **Raw data and captures:** `data/v05/` (ignored, never committed).
- **Registration:** `configs/v05/*.json`, `configs/v05/consumption-ledger.jsonl`.

## Release-gate audit (local, Windows 11, CPython 3.14.6)

| Item | Result |
|---|---|
| Full test suite | **974 passed**, 0 failed (2 third-party warnings), 529 s |
| Ruff | all checks passed |
| compileall (`lob`, `tools`, `examples`, `tests`) | passed |
| pip check | no broken requirements |
| Build | `cleolob-0.5.0.dev0` wheel and sdist built. The wheel contains all 76 `lob` modules. Neither contains restricted data, results or local notes. |
| Clean-wheel smoke (new venv, outside the checkout) | `cleo --help` lists every v0.5 command; `cleo smoke` and `verify-artifact smoke` pass; pip check is clean; misuse is refused |
| `cleo protocol verify` | valid; 80 ledger entries; no fresh dataset remains |
| `cleo verify-v05`, all 26 sealed runs | all valid |
| `cleo policy-study-v05 verify` (M7) | valid |
| `verify-artifact`: v0.3 (77 files), v0.4 (58), v0.5 (89) bundles | all valid |
| v0.3/v0.4 evidence changed since `224415d` | none |
| Linux/Windows GitHub CI | **not run.** The branch has not been pushed (22 commits ahead of `origin`). This release-gate item is open. |

## README claim audit

The README now states only evidence-backed v0.5 claims:
- the INVALID and post-hoc-FAILED MBO validation;
- fill bounds as ranges;
- calibration improvement established while absolute gates fail;
- impact agreement not established;
- settlement-only completion in stress;
- 5/64 DQN-only synthetic gates;
- non-transfer on the August holdout.

v0.3 and v0.4 statements are kept. Limitations now include:
- impact agreement;
- simulator-to-history transfer.

No profitability, alpha, production-HFT, universal-generalization or learned-superiority claim
is made.

## Remaining blockers and NOT_AVAILABLE items

1. **Genuine historical vendor MBO: NOT_AVAILABLE.** It needs licensed or credentialed access.
2. **M1 registered outcome: INVALID.** A new, separately registered capture would be needed.
   It can never replace this outcome.
3. **M8 July comparisons: not_evaluable in the registered run.** Only post-hoc evidence exists.
4. **Linux/Windows CI:** requires pushing the branch.
5. **Process RSS in M11: NOT_AVAILABLE.**
