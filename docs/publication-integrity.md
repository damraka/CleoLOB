# Publishing sealed study evidence

The 2026-09-21 publication audit checked the 15 existing `manifest.json` and
`checksums.json` packages under `examples/studies/`. All 932 distinct referenced
artifacts matched their recorded SHA-256 hashes in the working tree. The three
amended calibration model files also matched the file hashes recorded before
external evaluation. No data, results, expected hashes, or source snapshots were
rewritten.

The previous commit (`dbbd5cf`) contained 61 of those artifacts with Git-normalized
LF newlines instead of their authenticated bytes, and omitted 420 synthetic
execution logs because the repository-wide `logs/` ignore rule applied to study
archives too. The omitted files totalled 6,018,906 bytes. The other 451 referenced
artifacts already matched the committed blobs. These were packaging defects, not
recomputed experiments.

`.gitattributes` now disables text conversion for `examples/studies/**` and the
frozen `configs/core-study*.json` files. A narrow `.gitignore` exception retains
archived synthetic execution logs. Restaging the authenticated files preserves
their original bytes and unchanged seals. Existing CRLF artifacts remain CRLF;
existing LF artifacts remain LF.

A second narrow exception includes the final study's generated PPO checkpoints
and their training traces under `examples/studies/core/ppo-final-20260922/models/`,
plus the interrupted earlier attempt under `ppo-final-20260921/models/`.
Other model directories and downloaded raw market data retain their existing
ignore rules.

The Linux correctness workflow verifies the published final PPO study with
`python train_rl.py verify --study examples/studies/core/ppo-final-20260922`.
That check validates the artifact set, all hashes, model/training metadata,
registration, and completed evaluation design without retraining the models.

The 2026-09-22 follow-up audit at commit `c459dc6` found the 420 omitted logs had
been included. All 932 local artifacts still matched their original seals;
871 corresponding Git blobs matched, while the 61 newline conversions still
required restaging with text conversion disabled.

After restaging on 2026-09-22, all 932 referenced artifacts matched their original
SHA-256 hashes in Git's index. All 94 staged archive files differed from the
previous commit only by newline representation; no scientific content, recorded
hash, or file identity changed.

Full research reruns are available through the workflow's explicit
`run_full_study` dispatch input. Ordinary pushes run correctness and PPO contract
checks and verify the published archive. A dispatched reproduction saves model
files, raw episodes, logs, and Plotly reports as one GitHub Actions artifact even
when a training or evaluation step fails. Reusing the frozen seed blocks is a
reproduction and does not create an independent confirmatory sample.
