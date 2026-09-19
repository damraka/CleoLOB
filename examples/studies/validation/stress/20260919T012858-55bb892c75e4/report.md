# Registered execution stress test

Status: COMPLETE
Episodes: 300 / 300; family inference: WITHHELD

| Scenario | Outcome counts |
|---|---|
| reference | {'VALID': 50} |
| thin_depth | {'VALID': 50} |
| aggressive_flow | {'VALID': 50} |
| slow_messages | {'VALID': 50} |
| combined | {'INVALID': 43, 'VALID': 4, 'WARNING': 3} |
| liquidity_exhaustion | {'INVALID': 50} |

Stress controls and economic failures are reported for every frozen scenario. Unavailable liquidation or unsettled orders invalidate outcomes. No robust alpha claim.

See episodes.csv and child audit logs for every planned result.
