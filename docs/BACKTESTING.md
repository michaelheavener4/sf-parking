# Backtesting & Evaluation Framework

Module: `sf_parking.backtest` · CLI: `python3 -m sf_parking.backtest --help`

## The question this answers

> "If the system had made a parking prediction at time T using only
> information that would actually have been available at T, how accurate
> would that prediction have been?"

This is strictly a **measurement** milestone: the V0 baseline formula is not
modified here, and nothing in this framework is a calibrated probability.

## Target definition — three explicitly distinct levels

| Level | What it is | Status |
|---|---|---|
| **Observed paid sessions** | Transactions in `meter_transactions`: when a customer paid at a meter. This is the raw observation mechanism. | Available |
| **Inferred occupancy proxy** | Derived outcome: minutes of a target hour overlapped by paid sessions (`proxy_occupied_minutes`). Binary form: `proxy_availability = 1` iff no paid session overlapped the slot. | Computed by the harness; the ONLY evaluation target |
| **True occupancy** | Whether a car was physically at the curb, paid or not. | **Unavailable.** No sensor for it exists in our data |

Every metric in every report is computed against level 2 and labelled as a
proxy. Because unpaid occupancy (expired-but-parked, non-payment, broken
meters) is invisible to transactions, the proxy **overstates availability**;
real accuracy against truth would be worse. Any consumer of these reports
must treat numbers as upper bounds on model quality.

## Observation generation

For each meter with at least one completed session, observations are generated
for every local (America/Los_Angeles) calendar date from the meter's first
session date through the evaluation window end, crossed with the configured
local clock hours (default all 24). Each observation is one absolute
clock-hour slot `[T, T+1h)`.

Slots before a meter's first-ever session are skipped (`skipped_no_history`)
rather than scored "free" — absence of history is never interpreted as
absence of cars.

## Leakage rules (enforced, tested)

A prediction for slot starting at instant T may use:

- transactions with `session_start < T`;
- those sessions' elapsed portions only: ends are truncated at `min(end, T)`
  because an ongoing session's eventual end is unknowable at T;
- meter geometry/blockface exclusively from `meter_placements` rows whose
  validity range contains T (point-in-time lookup against the canonical
  temporal table) — later inventory snapshots cannot leak backwards;
- static attributes recorded before T (meter_type).

It may never use:

- transactions occurring at or after T (including ones *inside* the predicted
  hour — they legitimately appear in the outcome, never in the prediction);
- untruncated session ends;
- placement/inventory rows valid only after T;
- any feature derived from future data.

Each of these rules has a dedicated regression test with fixtures where a
future transaction would dramatically change the prediction if leaked.

## Baseline model

`deterministic_v0` (`DeterministicV0Baseline`), unchanged from
docs/PARKING_STATE.md:

    score = clamp(1 − Σ overlap(prior paid sessions, this local clock hour)
                      / (evidence_days × 60 minutes), 0, 1)

where the sum runs over every occurrence of the target local clock hour
within the meter's evidence span (first→last truncated session date) inside
the 28-day lookback window.

Models implement the `BaselineModel` protocol (`method` + `predict()`) and
register in `MODELS`; the harness is model-agnostic so future baselines
(transaction-rate, schedule-aware, spatial-neighbour, weather/event-aware,
ML) can be compared without touching measurement code.

## Metrics

Per cell (overall + breakdowns): `n`, `mean_score`, `proxy_availability_rate`,
**MAE** and **RMSE** against the continuous proxy free-fraction
(`1 − occupied_minutes/60`), and **Brier score** against the binary proxy.
Plus a calibration table bucketing predicted scores (0.0–0.1 … 0.9–1.0)
against observed proxy availability rates.

Breakdowns: local hour, weekday, meter type, point-in-time blockface, and
evidence-depth buckets (`1`, `2–3`, `4–7`, `8–14`, `15+` days).

Cells with fewer than `--min-samples` observations (default 30) report only
their count with `"suppressed": true` — no metrics are calculated on thin
slices.

## DST / timezone semantics

All wall-clock logic is America/Los_Angeles (established SFMTA semantics);
all elapsed-time math is absolute UTC instants:

- fall-back: the repeated local hour yields **two** absolute slot
  occurrences, both observed independently;
- spring-forward: the nonexistent local hour yields no slots;
- durations always reflect real elapsed time across transitions.

## Reproducibility

Same database state + same parameters ⇒ byte-identical serialized JSON
(`report.to_dict()` with sorted keys; asserted in tests). Observations carry
`method_version`, `cutoff`, evidence counts, and provenance sufficient to
recompute them.

## Limitations — read before quoting numbers

1. Proxy ≠ ground truth (see above); scores are upper bounds on quality.
2. The baseline is not a calibrated probability; calibration tables describe
   the proxy relationship only.
3. Production history is currently shallow (~one week of transactions),
   limiting evidence depth and making most breakdown cells suppressed at
   default thresholds. Depth grows as ingestion continues.
4. Meters that stopped reporting look increasingly "available" (observability
   assumption documented in docs/PARKING_STATE.md).
5. Regulation schedules are not yet modeled (blocked upstream: DataSF emptied
   `qq7v-hds4` on 2026-08-24).
6. Only credit-card-style completed sessions are used; NULL-ended sessions
   are excluded everywhere.
