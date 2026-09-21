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
and their training traces under `examples/studies/core/ppo-final-20260921/models/`.
Other model directories and downloaded raw market data retain their existing
ignore rules.

The Linux correctness workflow verifies the published final PPO study with
`python train_rl.py verify --study examples/studies/core/ppo-final-20260921`.
That check validates the artifact set, all hashes, model/training metadata,
registration, and completed evaluation design without retraining the models.
