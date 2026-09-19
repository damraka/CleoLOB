# Chronological calibration study

A fitted observable model was frozen before later-date evaluation. Drift and coverage are diagnostics; a completed study does not mean the model generalizes.

Training, validation and test use disjoint ordered rows with a 60-second embargo plus a one-second outcome horizon. Four purged expanding folds exercise within-training-day stability. No parameters are updated after validation.

Detailed drift/coverage scorecards and all provenance are in result.json, plan.json and model.json.
