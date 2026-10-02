# v0.6 final report: market realism, calibration uncertainty and model risk

**Branch and version.** Branch `research/v0.6-market-realism`, package version `0.6.0.dev0`.
**This is not a release:** there is no tag, no PyPI upload and no GitHub release.

**Protocol.**
- `configs/v06/protocol.json`, frozen 2026-10-01, with ledger amendments 1–2 recorded before
  any holdout access.
- Ledger: `configs/v06/consumption-ledger.jsonl`.

The narrative is in [v06-paper.md](v06-paper.md). The machine-readable claim graph is
`examples/studies/v06/claims.json`. `cleo claim-audit-v06` checks this table against it.

## Claim table

Statuses are copied by the audit from the sealed result files.

| Claim | Status | Evidence level | Estimate | Run |
|---|---|---|---|---|
| H1 | ESTABLISHED | retrospective | objective(v3) − objective(v0.5) = −1.76 [−1.87, −1.63] on ETH Jun | `m13/deribit-eth-perp-2020-06-01` |
| H2 | ESTABLISHED | confirmatory (fresh) | −1.02 [−1.18, −0.88] on ETH Sep | `m13/deribit-eth-perp-2020-09-01-registered` |
| H3 | ESTABLISHED | confirmatory (fresh, cross-instrument) | −1.17 [−1.27, −1.10] on BTC Sep | `m13/deribit-btc-perp-2020-09-01` |
| H4 | ESTABLISHED | descriptive | 2 materially distinct vectors within the 10% near-optimal tolerance (L∞ 0.66), from a 32-candidate selection pool | `m7/identifiability` |
| H5 | NOT_ESTABLISHED | confirmatory (simulation) | max \|member difference\| 1.00 bps; intervals contain 0; MDE 1.6–2.9 bps exceeds the 1 bps margin | `m10/evaluation` |
| H6 | NOT_ESTABLISHED | confirmatory (simulation) | no certified reversal; 2 of 30 cells determinate; MDE 1.5–2.6 bps; τ 0.07 not interpretable below resolution | `m10/evaluation` |
| H7 | NOT_ESTABLISHED | confirmatory (simulation) | max ρ 0.44 (temporal), Holm p ≥ 0.61 | `m10/evaluation` |
| H8 | ESTABLISHED | confirmatory (fresh) | logistic AUC 1.000 (lower bound 1.000) on ETH Sep | `m13/deribit-eth-perp-2020-09-01-registered` |
| H9 | FAILED | confirmatory (fresh) | high regime −0.32 [−0.44, −0.18]; low regime +0.96 [+0.78, +1.14] | same |
| H10 | FAILED | confirmatory (fresh) | off-regime +1.51 [+1.32, +1.71] and +0.23 [+0.14, +0.34] vs δ 0.244 | same |
| H11-ppo | NOT_ESTABLISHED | confirmatory (fresh) | −0.58 [−1.78, 1.16] conservative; −0.71 [−1.85, 1.11] optimistic | `m14/deribit-eth-perp-2020-10-01` |
| H11-dqn | NOT_ESTABLISHED | confirmatory (fresh) | +0.41 [−1.36, 2.11] conservative; +0.38 [−1.37, 1.99] optimistic | same |
| H12 | ESTABLISHED | confirmatory (fresh) | vacuous under the registered rule: all 15 pairs indeterminate in both fill modes; no stability or equivalence conclusion | same |
| S-support-eth-sep | ESTABLISHED | descriptive | 100% of fresh ETH windows out of simulator support (NO_CLAIM) | `m13/deribit-eth-perp-2020-09-01-registered` |

## Results by status

**ESTABLISHED:**
- H1 (retrospective), H2, H3: relative improvement over the v0.5 control.
- H4: calibration is not uniquely identified. Two materially distinct vectors fall within the
  preregistered 10% tolerance. They are not statistically equal fits, and the count of two is
  bounded by the frozen tolerance, distinctness rule and search budget.
- H8: real vs synthetic windows are distinguishable.
- H12: registered status kept, but vacuous. Every pair was indeterminate in both fill modes,
  so it supports no substantive stability or equivalence conclusion.
- Out-of-support labels on every evaluated dataset.

**NOT_ESTABLISHED:**
- H5 and H6: underpowered relative to the 1 bps margin (MDE about 1.6–2.9 bps and 1.5–2.6 bps).
  This implies neither equivalence of worlds nor stable rankings.
- H7.
- H11 for PPO and for DQN.

**FAILED:**
- H9: the low-volatility regime model is worse within its own regime.
- H10: regime models do not transfer outside their regime.
- On every holdout, every realism family of both models misses its equivalence margin
  (`FAILED_MARGIN`), except:
  - returns: `EQUIVALENT_WITHIN_MARGIN` for v3 on June; `NOT_ESTABLISHED` for v3 on ETH
    July, August and September and on BTC September; `FAILED_MARGIN` on BTC July;
  - event process: `NOT_ESTABLISHED` for v3 on June and July ETH;
  - event activity: `EQUIVALENT_WITHIN_MARGIN` for the v0.5 control on July ETH and BTC, and
    `NOT_ESTABLISHED` for it on June.

**INVALID (retained, never overwritten):**
- simulation bank attempt 1;
- fresh ETH evaluation attempt 1: ledger rule bug before any byte was read;
- benchmark attempt 1. It failed before a run directory was created, so it is retained in the
  consumption ledger (entry 79) and the research chronology, not as a directory.

See [v06-reproduction.md](v06-reproduction.md).

**NOT_AVAILABLE:**
- Order-level observables (order size, modify intensity, queue position, hidden liquidity).
- Vendor historical MBO.
- ROC curves: per-sample discriminator scores were not stored; only AUCs and intervals exist.
  Producing ROC curves would mean reading consumed holdouts again (posthoc, EXPLORATORY), so
  none were produced.
- Historical fill-bound component of the model-risk table in simulation: it is reported from
  the transfer runs instead.

**EXPLORATORY:**
- Morris elementary effects.
- The secondary domain-gap discriminators (family F6b).
- All `results/v06/pilot/*` runs.

**ASSUMPTION_DEPENDENT:** none. No transfer conclusion differs between fill modes.

## Datasets and freshness

| Dataset | v0.6 role | Status after v0.6 |
|---|---|---|
| ETH 2020-04-01 | development | consumed (since v0.3) |
| ETH 2020-05-01 | selection | consumed (since v0.4) |
| ETH 2020-06/07/08-01, BTC 2020-07-01 | retrospective | consumed (v0.4/v0.5); results labelled retrospective |
| ETH 2020-09-01 | fresh_external | consumed by v0.6 (download at ledger index 65, after the design seal at 63) |
| BTC 2020-09-01 | cross_instrument_external | consumed by v0.6 |
| ETH 2020-10-01 | transfer_holdout | consumed by v0.6 (after the transfer seal) |

## Deviations and incidents (disclosed)

1. **Protocol amendment 1** (before calibration): the two inert size means were removed from
   the search, and the impact interventions scale the size table instead.
2. **Protocol amendment 2** (before calibration): the absolute event cap became a rate cap of
   500 events per simulated second.
3. **Domain-gap resampling unit.** Simulated windows are resampled in 10-window blocks within
   seeds instead of whole seeds. This was fixed before the domain-gap design was sealed, after
   synthetic null calibration (8% → 5.0%).
4. **Ledger rule bug.** The bug refused the fresh-ETH evaluate access after its recorded
   download. No byte was read. The fix was sealed as `m13-holdout-implementation-revision-1`
   before the registered rerun, and the empty attempt directory is retained.
5. **OOD standardization.** The protocol says "development statistics"; the implementation
   uses the simulated windows. This is disclosed in [v06-domain-gap.md](v06-domain-gap.md),
   with no post-hoc rerun.
6. **Inert intervention.** `spread_distortion` is inert because the selected `offset_p`
   already sits at the bound. It is retained and flagged.
7. **AC identification** failed for the v0.5 control world, which falls back to default AC
   parameters.
8. **Low-volatility regime model** was chosen on only 23 selection blocks.
9. **v0.5-era check.** v0.5 `finalize` compared the source manifest with itself. v0.6 run
   directories capture it at creation instead, so a source edit during a run refuses sealing.
10. **Pre-existing v0.5 condition, not changed by v0.6.** `lob.policy_study_v05.read_freeze()`
    fails already at the `v0.5.0` tag, for these reasons:
    - The v0.5 environment freeze records `lob/historical_sim.py` at `dca9622e…`.
    - The v0.5 ledger then recorded two M8 revisions (entries 62 and 70) ending at
      `98cbff45…`, which is the tagged file.
    - The file is byte-identical between `v0.5.0` and this branch.

    `cleo policy-study-v05 verify` is unaffected and still valid. Re-running v0.5 M7
    training or evaluation would refuse. v0.5 history was not edited.

## Evidence and verification

- **Sealed runs.** Every sealed run in `results/v06` verifies with `cleo verify-v06`. Those
  runs cover:
  - the design;
  - calibration development and selection;
  - regime calibration and identifiability;
  - policies and evaluation;
  - the bank;
  - six holdout and two transfer datasets;
  - benchmarks.
- **Public bundle.** `examples/studies/v06/evidence` holds the protocol, ledger, registration
  artifacts and per-run config/result/binding/provenance files. Excluded files are listed
  with their SHA-256 values, and the bundle is sealed with independent checksums.
- **Meaning of verification:** byte integrity plus binding consistency. It is not independent
  scientific replication.
- **v0.5 history is unchanged:**
  - the v0.5 protocol verifies, and its 80-entry ledger head is pinned by the v0.6 protocol;
  - no v0.5 file, result or tag was modified.
- **Provenance-format debt (line endings).** This is not a difference in any scientific
  result.
  - **Dirty flag.** 15 of the 17 registered runs record `git_dirty: true`; `m13/bank-2` and
    `m16/benchmarks-2` record `false`. Runs append to the ledger while they execute, so the
    ledger was uncommitted, and some working-copy files had different line endings.
  - **Affected runs.** 14 of the 17 registered runs record source hashes over CRLF
    working-copy bytes for 1–3 `lob/v06` files:
    - `m10/evaluation`, `m10/policies`, `m12/regime`, `m7/identifiability`;
    - `m13/bank-2`, `m16/benchmarks-2`;
    - all six `m13` holdout runs;
    - both `m14` transfer runs.

    `m1/design`, `m6/develop` and `m6/select` are unaffected. The other 7 of the 24 sealed
    runs are labelled pilots and are not evidence.
  - **Same code.** Converting the committed LF files to CRLF reproduces those recorded hashes
    exactly, so the code content each run used matches the committed source at its recorded
    commit.
  - **Seals unaffected.** The protocol, ledger, observable-design seal, environment-freeze
    seal and evidence bindings use line-ending-normalized or document hashes, and they all
    remain valid. The sealed provenance files were not rewritten.
  - **Future checks.** A byte-level source check (for example `check_source=True`) on a fresh
    LF checkout may report a mismatch for these files unless line endings are normalized
    first.

## Release-gate audit (local, Windows 11, CPython 3.14.6)

| Item | Result |
|---|---|
| Full test suite | **1,090 passed**: 974 baseline + 116 v0.6. The 2 gymnasium warnings are third-party and pre-existing. |
| Ruff, compileall (`lob`, `tools`, `examples`, `tests`), pip check | all pass |
| Build | `cleolob-0.6.0.dev0` wheel (107 `lob` modules, 31 in `lob.v06`) and sdist (182 entries). Neither contains `data/`, `results/`, `.csv.gz`, `.pkl` or `.npz` files. |
| Isolated wheel install (new venv outside the checkout) | imports from `site-packages/lob`; pip check clean; all 13 v0.6 commands answer `--help`; `cleo smoke` and `verify-artifact` pass; misuse exits with a clear `INVALID` |
| `cleo protocol-v06 verify` | valid; 80 ledger entries; no fresh dataset remains |
| `cleo verify-v06 results/v06` | 24 sealed runs (17 registered + 7 labelled pilots), all valid |
| Public bundle | `verify-artifact` valid. 76 files were independently re-hashed with Python `hashlib`: 0 mismatches. |
| `cleo claim-audit-v06` | valid: 14 claims, 0 issues. The flagged "risky" lines are negated list items under "not claimed" and were reviewed. |
| v0.5 history | `cleo protocol verify` valid (80 entries); v0.5 bundle valid; all v0.5 sealed runs valid; `policy-study-v05 verify` valid. No v0.3–v0.5 config, evidence or frozen module differs from `v0.5.0` (but see item 10 above). |
| Commits | 28 on the branch since `50ed08b`, all by the configured author. No AI attribution trailers. `git diff --check` is clean. |
| Tags and release | none created; no push |

## Known limitations

- **Data.** Aggregate L2 and trade prints from one venue (Deribit) and one era (2020). There is
  no order-level data.
- **Ensemble size.** The ensemble has only two members. The pipeline was:
  - 2,304 search candidates;
  - the top 64 re-scored;
  - the top 32 scored on the selection day;
  - 4 within the frozen near-optimal tolerance;
  - 2 materially distinct.

  The limit of 8 was not binding. Two members reflect the frozen tolerance, distinctness rule
  and finite budget, not evidence that only two plausible configurations exist. It is a coarse
  view of calibration uncertainty.
- **Resolution.** At the tests' own α and 80% power, the minimum detectable effect is about
  1.6–2.9 bps for H5 (independent worlds) and 1.5–2.6 bps for H6 (paired within world), with
  200 market seeds. Smaller differences are not detectable.
- **Identifiability diagnostics** work at the family level. The 9 × 14 sensitivity matrix is
  rank-capped at 9, and its effective rank is about 5–7 under the audit tolerances. Many
  profile draws hit the event cap. The parameters are not identified; only non-uniqueness is
  established.
- **Model-risk components** are standard deviations over very small world counts (2, 3 and
  14). They are unstable.
- **Historical replay** has no market impact of the hypothetical parent. Fills are bounded,
  never exact.
- **The simulators are far from history:**
  - discriminator AUC is about 1;
  - essentially no historical window is in support;
  - so absolute realism claims are withheld.

## Remaining blockers

- No CI has run on this branch; it has not been pushed.
- No v0.6.0 release, by instruction.
