# Reproducing v0.9

## Verify the committed evidence (offline, minutes)

```bash
python -m pip install -e '.[dev]'
cleo protocol-v09 verify                               # protocol, ledger chain, frozen configs, v0.8 pins
cleo verify-v09 --bundle examples/studies/v09          # bundle bytes, run bindings, lineage, ledger chain
cleo claim-audit-v09 --bundle examples/studies/v09     # claim graph and wording audit of README and docs/v09-*
cleo smoke-v09 --out /tmp/smoke-v09                    # generated-data functionality smoke (isolated ledger)
```

Verification checks byte integrity and binding. It does not replicate the science independently.

## Re-run the studies (hours to days; needs the public archives)

The commands below repeat the registered pipeline. Each study takes `--root` (repository), `--out` (a write-once
run directory under `results/v09`), `--namespace` and, for guarded namespaces, `--design`. Long studies checkpoint
after every task and resume with `--resume`. Use at most one heavy study family at a time, and choose `--workers` to
stay within the registered 12 GB per-study ceiling. The pilots in this release used 2 to 6 workers on a 32 GB, 16-thread
CPU machine. PyTorch runs on the CPU only, with threads capped by `CLEOLOB_TORCH_THREADS` (default 2).

1. **Explore day caches:**
   `cleo validate-market-data-v09 --namespace explore --dataset slots:explore:bybit-linear:BTCUSDT:8 ...`
   Downloads are ledgered with their source hashes, and failures move to registered backups.
2. **Exploratory pilots** (`--namespace explore`), for example `cleo counterfactual-bounds-v09 pilot`,
   `cleo hybrid-twin-v09 pilot`, `cleo multifidelity-agreement-v09 --input bounds=...`, `cleo offline-rl-v09`, and
   the others listed in `docs/v09-final-report.md`.
3. **Feasibility and promotion:**
   - `cleo hypothesis-promotion-v09 feasibility --out ... --input KEY=RUN ...`
   - `cleo hypothesis-promotion-v09 run --out ... --input KEY=RUN ...` (exactly once)
   - `cleo hypothesis-promotion-v09 freeze --run ...`
4. **Seal the confirmatory designs:** `python -c "from pathlib import Path; from lob.v09 import confirm;
   print(confirm.seal_all(Path('.')))"`. This seals every design and opens the confirm namespace before any
   confirmatory access.
5. **Confirmatory data and studies:**
   - `cleo validate-market-data-v09 --namespace confirm --design confirm-data --dataset ...`
   - then each study with `--namespace confirm --design confirm-<study>` (phase `confirm` where the command has
     phases)
   - then `cleo transfer-study-v09 --namespace confirm --design confirm-bounds --input bounds=... [--input late=...]`
6. **Close and the vault:**
   - `lob.v09.confirm.close(root, {design: run})`
   - seal `final-vault` from `lob.v09.confirm.final_design`
   - open the final namespace
   - prepare the vault days and run `cleo counterfactual-bounds-v09 confirm --namespace final --design final-vault`
     (once)
7. **Final analysis and reporting:**
   - `cleo final-analysis-v09 --namespace final --design final-vault --input vault=... --input gate=... --input
     counterexample=...`
   - then `cleo report-v09`, `cleo evidence-export-v09` and `cleo claim-audit-v09`

**The driver used for this release.** Steps 4–7 were executed through `tools/v09_pipeline.py`, from a clean
worktree at the committed code. It runs one study at a time through the fail-closed batch runner
(`lob/v09/batch.py`), with workers = 2 and an 8 GB free-memory guard. The steps, in order:

```bash
python tools/v09_pipeline.py --root ../CleoLOB dryrun     # confirm branches on explore days; must all pass
python tools/v09_pipeline.py --root ../CleoLOB seal       # seal every confirmatory design
python tools/v09_pipeline.py --root ../CleoLOB data       # confirmatory day caches (registered backups only)
python tools/v09_data_audit.py --root ../CleoLOB          # read-only audit of slots, failures and substitutions
python tools/v09_pipeline.py --root ../CleoLOB studies    # confirmatory studies in registered order
python tools/v09_pipeline.py --root ../CleoLOB transfer
python tools/v09_pipeline.py --root ../CleoLOB close
python tools/v09_pipeline.py --root ../CleoLOB vault      # seals final-vault, opens it once, runs the vault replay
python tools/v09_pipeline.py --root ../CleoLOB --input gate=results/v09/explore/realism-gate \
    --input counterexample=results/v09/descriptive/counterexample-2 final
python tools/v09_runcheck.py --root ../CleoLOB results/v09/confirm/confirm-*   # outcome-blind seal/binding check
```

The guarded steps (4–6) can be executed only once per ledger. A fresh replication needs a fresh protocol, ledger and
data chronology; the consumed v0.9 days are never fresh again.
