# Parking-State Features & Availability Baseline

Module: `sf_parking.features` · CLI: `scripts/meter_features.py`

Everything in this layer is a **deterministic function of database state plus
explicit parameters** (`window_days`, `at`/`now`). No randomness, no hidden
clocks, no fitted parameters: running twice against the same data yields
identical output, which the test suite asserts directly.

## Features (`meter_features`)

Per meter, over completed sessions whose `session_start` falls inside the
window (default trailing 28 days):

| Feature | Definition |
|---|---|
| `session_count` | number of completed sessions starting in the window |
| `active_days` | distinct **America/Los_Angeles calendar dates** with a session start |
| `total_paid_minutes` | sum of stored `duration_minutes` |
| `mean_session_minutes` | arithmetic mean session duration |
| `median_session_minutes` | SQL `PERCENTILE_CONT(0.5)` of durations |
| `last_session_at` | latest absolute session end |
| `sessions_per_active_day` | `session_count / active_days` (property) |

Durations always come from absolute instants, so sessions crossing DST
transitions count real elapsed time (a 00:30–04:30 PT session on fall-back
day is 300 minutes, not 240).

Meters with no completed sessions in the window are omitted — absence is not
modelled as zero activity.

## Baseline availability (`availability_baseline`)

    method = "deterministic_v0"

For meter *m* and instant *T*:

1. Take the America/Los_Angeles clock hour containing *T* (e.g. 13:00–14:00).
2. Sum every historical session's overlap with that clock hour across all its
   occurrences in the observation span (on fall-back days the repeated local
   hour counts both absolute occurrences; nonexistent spring-forward hours
   contribute nothing).
3. Divide by `evidence_days × 60`, where `evidence_days` spans the first to
   last session date of that meter inside the window.
4. `score = clamp(1 − occupied / possible, 0, 1)`, rounded to 3 decimals.

If the meter has **no completed sessions** in the window, `score is None`
(`insufficient history`) rather than a fabricated number.

## Assumptions & caveats — read before using scores

* **Not a calibrated probability.** The score is a transparent occupancy
  ratio. It has no probabilistic interpretation and has not been validated
  against ground-truth occupancy.
* Paid transactions **under-count occupancy**: unpaid windows, expired-but-
  parked time, and non-payment are invisible to transaction data, so true
  availability is systematically *lower* than scored.
* Observability assumption: a meter is treated as observable from its first
  session in the window onward. Days before that first session are excluded
  from the denominator; a meter installed mid-window is handled, but meters
  that stopped reporting look increasingly "available".
* Regulation schedules are **not** factored in yet: during unregulated hours
  real availability is ~1 regardless of the score. Combining with
  `meter_policies` (once SFMTA republishes that dataset) is future work.
* Sessions are attributed to their start hour; a session spanning several
  hours contributes occupancy to each hour it overlaps.
* All wall-clock logic uses `America/Los_Angeles` per the established SFMTA
  timestamp semantics (docs/ROADMAP.md); all elapsed-time math uses absolute
  UTC instants.

## Reproducibility contract

Same database contents + same parameters ⇒ identical results, including the
`method` tag. Any change to the formula must bump `BASELINE_METHOD`.
