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

## Blockface-pooled availability (`blockface_hourly`)

    method = "blockface_hourly"

For meter *m* on blockface *b* at local clock hour *h*:

### Blockface score

Pool occupancy across **all meters** on blockface *b* for hour *h*:

    bf_occupied = Σ_meters  overlap_minutes(m, h)
    bf_possible = Σ_meters  evidence_days(m) × 60
    bf_score    = clamp(1 − bf_occupied / bf_possible, 0, 1)

The denominator is the **sum of each meter's individual observable
meter-hours**, NOT a shared constant. A blockface with 10 meters
observed for 7 days each has `bf_possible = 10 × 7 × 60 = 4200`
minutes; one with 3 meters observed for 3 days has `3 × 3 × 60 = 540`.

### SS vs MS treatment

* **SS (single-space)** meters: `post_id` = one parking space (1:1).
  At most one concurrent session per post_id. Occupied minutes per
  hour ∈ [0, 60].
* **MS (multi-space)** meters: `post_id` = one pay station covering
  potentially many spaces. Concurrent sessions at the same post_id
  represent different spaces. Occupied minutes per hour can exceed 60.
  The blockface denominator does NOT attempt to count physical spaces
  for MS meters (the count is unknown without `ms_id`/`space_num` from
  the source data); it counts observable meter-hours, which is a
  conservative lower bound on true space-hours.

### Blending

The per-meter evidence controls the blend between the per-meter
estimate and the blockface prior:

    w(m)    = min(evidence_days(m) / evidence_halflife, 1.0)
    blended = w × per_meter_score + (1 − w) × bf_score

With `evidence_halflife = 14` (default):

| Evidence days | Weight *w* | Interpretation |
|---|---|---|
| 0 | 0.0 | Pure blockface prior |
| 7 | 0.5 | Equal blend |
| 14+ | 1.0 | Pure per-meter |

### Fallbacks

* No blockface assignment → per-meter score only (`deterministic_v0`).
* Blockface has no sessions → per-meter score only.
* Neither has data → `None` (insufficient history).

### Design rationale

With ~1 week of transaction history, per-meter hourly estimates rest
on at most ~7 observations — far too few for stable ratios. A
blockface with 10 meters gains ~70 observations per hour cell. The
blending weight naturally handles the cold-start / shallow-history
problem: meters with little data lean on the blockface average; meters
with abundant data use their own estimate.
