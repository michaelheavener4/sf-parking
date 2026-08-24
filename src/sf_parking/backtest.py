"""Point-in-time-safe historical backtesting for parking-state baselines.

Answers: "If the system had made a parking prediction at time T using only
information actually available at T, how accurate would it have been?"

Protocol
--------
1. Observations are (meter, absolute clock-hour slot) pairs generated for
   every local date a meter was observed within the evaluation window.
2. A prediction for a slot uses ONLY information available at its start:
   * transactions started strictly before the slot,
   * session ends truncated at the slot (an ongoing session's eventual end
     is unknowable at prediction time),
   * meter location/blockface taken from ``meter_placements`` rows whose
     validity range contains the slot (never later inventory state).
3. Outcome: an *observable proxy*, explicitly not ground truth - the paid-
   occupancy minutes overlapped by real transactions (untruncated). The
   dataset observes paid sessions, not occupancy; unpaid parking is
   invisible, so true availability is systematically lower than the proxy
   implies. See docs/BACKTESTING.md for the three-level distinction between
   observed paid sessions, inferred occupancy, and (unavailable) truth.

The V0 baseline formula lives in ``sf_parking.features`` and is not modified
here; this module feeds it honest inputs and measures it. New models can be
added by implementing :class:`BaselineModel` without touching the harness.

Run as a CLI: ``python3 -m sf_parking.backtest --help``.
"""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

import pg8000.native

from .features import SF_TZ, _candidate_slot_starts, _overlap_seconds

BASELINE_METHOD = "deterministic_v0"
SLOT_MINUTES = 60
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Prediction:
    score: float
    evidence_days: int
    evidence_sessions: int


class BaselineModel(Protocol):
    """Contract for scorers evaluated by the harness.

    Implementations receive only cutoff-truncated history and must be
    deterministic functions of it plus their own parameters.
    """

    method: str

    def predict(
        self,
        sessions: list[tuple[datetime, datetime]],
        slot_start: datetime,
        *,
        history_window_days: int,
    ) -> Prediction | None: ...


def score_v0(occupied_minutes: float, possible_minutes: float) -> float:
    """The V0 scoring formula (shared with sf_parking.features), verbatim."""
    if possible_minutes <= 0:
        raise ValueError("possible_minutes must be positive")
    return round(max(0.0, min(1.0, 1.0 - occupied_minutes / possible_minutes)), 3)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class DeterministicV0Baseline:
    """Historical clock-hour occupancy ratio for the same meter.

    score = clamp(1 - paid_occupied_minutes / (evidence_days * 60), 0, 1)

    where occupied minutes come from prior sessions overlapping this local
    clock hour across the meter's evidence span inside the lookback window.
    """

    method = BASELINE_METHOD

    def predict(
        self,
        sessions: list[tuple[datetime, datetime]],
        slot_start: datetime,
        *,
        history_window_days: int,
    ) -> Prediction | None:
        window_start = slot_start - timedelta(days=history_window_days)

        # Point-in-time history: sessions started before the cutoff, ends
        # truncated at the cutoff, restricted to the lookback window.
        hi = bisect_left([s for s, _ in sessions], slot_start)
        history = [
            (s, min(e, slot_start))
            for s, e in sessions[:hi]
            if min(e, slot_start) > window_start
        ]
        if not history:
            return None

        hist_starts = [s for s, _ in history]
        n_hist = len(history)
        local_hour = slot_start.astimezone(SF_TZ).hour
        first_day = history[0][0].astimezone(SF_TZ).date()
        last_day = history[-1][1].astimezone(SF_TZ).date()
        evidence_days = max((last_day - first_day).days + 1, 1)

        occupied = 0.0
        day = first_day
        while day <= last_day:
            for lo in _candidate_slot_starts(day, local_hour):
                if lo < window_start or lo >= slot_start:
                    continue  # never score the slot itself or stale windows
                j = max(bisect_left(hist_starts, lo - timedelta(days=2)) - 1, 0)
                while j < n_hist:
                    s, e = history[j]
                    if s >= lo + timedelta(hours=1):
                        break
                    occupied += _overlap_seconds(s, e, lo, lo + timedelta(hours=1))
                    j += 1
            day += timedelta(days=1)

        return Prediction(
            score=score_v0(occupied / 60.0, evidence_days * SLOT_MINUTES),
            evidence_days=evidence_days,
            evidence_sessions=n_hist,
        )


MODELS: dict[str, BaselineModel] = {
    BASELINE_METHOD: DeterministicV0Baseline(),
}


# ---------------------------------------------------------------------------
# Observation / result models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BacktestObservation:
    """One out-of-sample prediction with its observable proxy outcome."""

    post_id: str
    #: The slot is both the prediction cutoff (features may not see past it)
    #: and the target hour being predicted.
    prediction_time: datetime
    cutoff: datetime
    target_hour_start: datetime
    local_date: date
    local_hour: int
    weekday: str
    #: Canonical-spatial context, resolved point-in-time.
    post_blockface_id: str | None
    latitude: float | None
    longitude: float | None
    location_source: str  # "placement_at_t" or "unresolved"
    predicted_score: float
    #: Observable proxy outcome (paid-session overlap), NOT ground truth.
    proxy_occupied_minutes: float
    proxy_availability: int  # 1 iff no paid session overlapped the slot
    #: signed error vs continuous proxy free-fraction
    prediction_error: float
    method_version: str
    evidence_days: int
    evidence_sessions: int

    @property
    def proxy_free_fraction(self) -> float:
        return 1.0 - self.proxy_occupied_minutes / SLOT_MINUTES

    def as_dict(self) -> dict[str, Any]:
        return {
            "post_id": self.post_id,
            "prediction_time": self.prediction_time.isoformat(),
            "cutoff": self.cutoff.isoformat(),
            "target_hour_start": self.target_hour_start.isoformat(),
            "local_date": self.local_date.isoformat(),
            "local_hour": self.local_hour,
            "weekday": self.weekday,
            "blockface_id": self.post_blockface_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "location_source": self.location_source,
            "predicted_score": self.predicted_score,
            "proxy_occupied_minutes": self.proxy_occupied_minutes,
            "proxy_availability": self.proxy_availability,
            "prediction_error": self.prediction_error,
            "method_version": self.method_version,
            "evidence_days": self.evidence_days,
            "evidence_sessions": self.evidence_sessions,
        }


#: Historical alias used by earlier revisions/tests.
Observation = BacktestObservation


def compute_metrics(observations: Iterable[BacktestObservation]) -> MetricSet:
    """MAE/RMSE vs the continuous proxy, Brier vs the binary proxy."""
    obs = list(observations)
    n = len(obs)
    if n == 0:
        return MetricSet(0)
    mean_score = sum(o.predicted_score for o in obs) / n
    avail_rate = sum(o.proxy_availability for o in obs) / n
    errors_free = [
        o.predicted_score - o.proxy_free_fraction for o in obs
    ]
    mae = sum(abs(e) for e in errors_free) / n
    rmse = math.sqrt(sum(e * e for e in errors_free) / n)
    brier = sum(
        (o.predicted_score - o.proxy_availability) ** 2 for o in obs
    ) / n
    return MetricSet(
        n=n,
        mean_score=round(mean_score, 4),
        proxy_availability_rate=round(avail_rate, 4),
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        brier=round(brier, 4),
    )


@dataclass(frozen=True, slots=True)
class MetricSet:
    n: int
    mean_score: float | None = None
    proxy_availability_rate: float | None = None
    mae: float | None = None
    rmse: float | None = None
    brier: float | None = None
    suppressed: bool = False

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"n": self.n}
        if self.suppressed:
            d["suppressed"] = True
            return d
        d.update(
            mean_score=self.mean_score,
            proxy_availability_rate=self.proxy_availability_rate,
            mae=self.mae,
            rmse=self.rmse,
            brier=self.brier,
        )
        return d


@dataclass(frozen=True, slots=True)
class BacktestReport:
    method: str
    eval_days: int
    history_window_days: int
    until: datetime
    min_samples: int
    observations_generated: int
    predictions_made: int
    skipped_no_history: int
    overall: MetricSet
    by_hour: dict[int, MetricSet] = field(default_factory=dict)
    by_weekday: dict[str, MetricSet] = field(default_factory=dict)
    by_meter_type: dict[str, MetricSet] = field(default_factory=dict)
    by_blockface: dict[str, MetricSet] = field(default_factory=dict)
    by_evidence_days_bucket: dict[str, MetricSet] = field(default_factory=dict)
    calibration: list[dict[str, Any]] = field(default_factory=list)
    observations: list[BacktestObservation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "method": self.method,
            "eval_days": self.eval_days,
            "history_window_days": self.history_window_days,
            "until": self.until.isoformat(),
            "min_samples": self.min_samples,
            "observations_generated": self.observations_generated,
            "predictions_made": self.predictions_made,
            "skipped_no_history": self.skipped_no_history,
            "overall": self.overall.as_dict(),
            "by_hour": {str(k): v.as_dict() for k, v in self.by_hour.items()},
            "by_weekday": {k: v.as_dict() for k, v in self.by_weekday.items()},
            "by_meter_type": {
                k: v.as_dict() for k, v in self.by_meter_type.items()
            },
            "by_blockface": {
                k: v.as_dict() for k, v in self.by_blockface.items()
            },
            "by_evidence_days_bucket": {
                k: v.as_dict() for k, v in self.by_evidence_days_bucket.items()
            },
            "calibration": self.calibration,
            "target_definition": (
                "paid-session overlap proxy (not ground-truth occupancy)"
            ),
        }
        if self.observations:
            d["observations"] = [obs.as_dict() for obs in self.observations]
        return d



# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _bucket(evidence_days: int) -> str:
    if evidence_days <= 1:
        return "1"
    if evidence_days <= 3:
        return "2-3"
    if evidence_days <= 7:
        return "4-7"
    if evidence_days <= 14:
        return "8-14"
    return "15+"


def _group_by(
    observations: list[BacktestObservation],
    key,
    *,
    min_samples: int,
) -> dict[Any, MetricSet]:
    groups: dict[Any, list[BacktestObservation]] = {}
    for o in observations:
        groups.setdefault(key(o), []).append(o)
    out: dict[Any, MetricSet] = {}
    for k, group in sorted(
        groups.items(), key=lambda kv: (isinstance(kv[0], str), kv[0])
    ):
        if len(group) >= min_samples:
            out[k] = compute_metrics(group)
        else:
            out[k] = MetricSet(n=len(group), suppressed=True)
    return out


def calibration_table(
    observations: list[BacktestObservation],
    *,
    buckets: int = 10,
    min_samples: int = 30,
) -> list[dict[str, Any]]:
    """Score-bucket calibration vs the binary proxy availability outcome."""
    width = 1.0 / buckets
    grouped: dict[int, list[BacktestObservation]] = {}
    for o in observations:
        idx = min(int(o.predicted_score / width), buckets - 1)
        grouped.setdefault(idx, []).append(o)
    table = []
    for idx in range(buckets):
        group = grouped.get(idx, [])
        lo, hi_b = round(idx * width, 2), round((idx + 1) * width, 2)
        entry: dict[str, Any] = {
            "score_bucket": f"{lo:.1f}-{hi_b:.1f}",
            "n": len(group),
        }
        if len(group) >= min_samples:
            entry["mean_score"] = round(
                sum(o.predicted_score for o in group) / len(group), 4
            )
            entry["proxy_availability_rate"] = round(
                sum(o.proxy_availability for o in group) / len(group), 4
            )
        table.append(entry)
    return table


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _load_sessions(
    conn: pg8000.native.Connection,
    *,
    load_since: datetime,
    until: datetime,
    post_ids: list[str] | None,
) -> dict[str, list[tuple[datetime, datetime]]]:
    clause = ""
    params: dict[str, Any] = {"since": load_since, "until": until}
    if post_ids is not None:
        clause = "AND post_id = ANY(:post_ids)"
        params["post_ids"] = post_ids
    rows = conn.run(
        "SELECT post_id, session_start, session_end FROM meter_transactions "
        f"WHERE session_end IS NOT NULL AND session_start < :until "
        f"AND session_end >= :since {clause} "
        "ORDER BY post_id, session_start",
        **params,
    )
    sessions: dict[str, list[tuple[datetime, datetime]]] = {}
    for post_id, start, end in rows:
        sessions.setdefault(post_id, []).append((start, end))
    return sessions


@dataclass(frozen=True, slots=True)
class PlacementSpan:
    valid_from: float  # epoch seconds; -inf allowed
    valid_until: float  # epoch seconds; +inf when open
    latitude: float
    longitude: float
    blockface_id: str | None


def _load_placements(
    conn: pg8000.native.Connection,
    post_ids: set[str],
) -> dict[str, list[PlacementSpan]]:
    """Point-in-time meter geometry from the canonical temporal table."""
    if not post_ids:
        return {}
    rows = conn.run(
        "SELECT m.post_id, "
        "EXTRACT(EPOCH FROM p.valid_from)::float8, "
        "EXTRACT(EPOCH FROM p.valid_until)::float8, "
        "p.latitude, p.longitude, b.source_blockface_id "
        "FROM meter_placements p "
        "JOIN meters m ON m.meter_id = p.meter_id "
        "LEFT JOIN blockfaces b ON b.blockface_id = p.blockface_id "
        "WHERE m.post_id = ANY(:posts)",
        posts=sorted(post_ids),
    )
    spans: dict[str, list[PlacementSpan]] = {}
    for post_id, vf, vu, lat, lon, bf in rows:
        span = PlacementSpan(
            valid_from=vf,
            valid_until=float("inf") if vu is None else vu,
            latitude=lat,
            longitude=lon,
            blockface_id=bf,
        )
        spans.setdefault(post_id, []).append(span)
    for spans_list in spans.values():
        spans_list.sort(key=lambda sp: sp.valid_from)
    return spans


def _placement_at(spans: list[PlacementSpan], instant: datetime) -> PlacementSpan | None:
    epoch = instant.timestamp()
    starts = [sp.valid_from for sp in spans]
    i = bisect_left(starts, epoch)
    if i < len(starts) and starts[i] == epoch:
        return spans[i]
    if i == 0:
        return None
    candidate = spans[i - 1]
    return candidate if candidate.valid_until > epoch else None


def _meter_types(conn: pg8000.native.Connection) -> dict[str, str | None]:
    return {pid: mtype for pid, mtype in conn.run(
        "SELECT post_id, meter_type FROM parking_meters"
    )}


def run_backtest(
    conn: pg8000.native.Connection,
    *,
    until: datetime | None = None,
    eval_days: int = 7,
    history_window_days: int = 28,
    hours: tuple[int, ...] | None = None,
    post_ids: list[str] | None = None,
    max_meters: int | None = None,
    include_observations: bool = False,
    min_samples: int = 30,
    model: BaselineModel | None = None,
) -> BacktestReport:
    """Run the out-of-sample evaluation. Deterministic given DB state."""
    model = model or MODELS[BASELINE_METHOD]
    if until is None:
        until = datetime.now(UTC)
    elif until.tzinfo is None:
        until = until.replace(tzinfo=UTC)
    until = until.astimezone(UTC)

    target_hours = set(hours) if hours is not None else set(range(24))

    sessions = _load_sessions(
        conn,
        load_since=until - timedelta(days=eval_days + history_window_days + 2),
        until=until,
        post_ids=post_ids,
    )

    placements = _load_placements(
        conn, set(sessions) if post_ids is None else set(post_ids)
    )
    meter_types = _meter_types(conn)

    eval_start_local = (until - timedelta(days=eval_days)).astimezone(SF_TZ).date()
    # Slots on until's own date are kept; anything at/after ``until`` is
    # skipped per-slot below.
    eval_end_local = until.astimezone(SF_TZ).date()

    observations: list[BacktestObservation] = []
    generated = 0
    skipped_no_history = 0
    meters = sorted(sessions) if max_meters is None else sorted(sessions)[:max_meters]

    for post_id in meters:
        meter_sessions = sessions[post_id]
        meter_starts = [s for s, _ in meter_sessions]
        spans = placements.get(post_id, [])
        first_day = meter_sessions[0][0].astimezone(SF_TZ).date()
        day = max(first_day, eval_start_local)
        while day <= eval_end_local:
            for hour in sorted(target_hours):
                for lo in _candidate_slot_starts(day, hour):
                    if lo >= until:
                        continue
                    generated += 1

                    # Outcome proxy: FULL transactions overlapping the slot.
                    outcome_occupied = 0.0
                    j = max(bisect_left(meter_starts, lo - timedelta(days=2)) - 1, 0)
                    while j < len(meter_sessions):
                        s, e = meter_sessions[j]
                        if s >= lo + timedelta(hours=1):
                            break
                        outcome_occupied += _overlap_seconds(
                            s, e, lo, lo + timedelta(hours=1)
                        )
                        j += 1

                    pred = model.predict(
                        meter_sessions,
                        lo,
                        history_window_days=history_window_days,
                    )
                    if pred is None:
                        skipped_no_history += 1
                        continue

                    placement = _placement_at(spans, lo)
                    occupied_minutes = outcome_occupied / 60.0
                    observations.append(
                        BacktestObservation(
                            post_id=post_id,
                            prediction_time=lo,
                            cutoff=lo,
                            target_hour_start=lo,
                            local_date=day,
                            local_hour=lo.astimezone(SF_TZ).hour,
                            weekday=WEEKDAYS[lo.astimezone(SF_TZ).weekday()],
                            post_blockface_id=(
                                placement.blockface_id if placement else None
                            ),
                            latitude=placement.latitude if placement else None,
                            longitude=placement.longitude if placement else None,
                            location_source=(
                                "placement_at_t" if placement else "unresolved"
                            ),
                            predicted_score=pred.score,
                            proxy_occupied_minutes=round(occupied_minutes, 3),
                            proxy_availability=1 if outcome_occupied == 0.0 else 0,
                            prediction_error=round(
                                pred.score - (1.0 - occupied_minutes), 4
                            ),
                            method_version=model.method,
                            evidence_days=pred.evidence_days,
                            evidence_sessions=pred.evidence_sessions,
                        )
                    )
            day += timedelta(days=1)

    return BacktestReport(
        method=model.method,
        eval_days=eval_days,
        history_window_days=history_window_days,
        until=until,
        min_samples=min_samples,
        observations_generated=generated,
        predictions_made=len(observations),
        skipped_no_history=skipped_no_history,
        overall=compute_metrics(observations),
        by_hour=_group_by(observations, lambda o: o.local_hour, min_samples=min_samples),
        by_weekday=_group_by(observations, lambda o: o.weekday, min_samples=min_samples),
        by_meter_type=_group_by(
            observations,
            lambda o: (meter_types.get(o.post_id) or "unknown"),
            min_samples=min_samples,
        ),
        by_blockface=_group_by(
            observations,
            lambda o: o.post_blockface_id or "unresolved",
            min_samples=min_samples,
        ),
        by_evidence_days_bucket=_group_by(
            observations, lambda o: _bucket(o.evidence_days), min_samples=min_samples
        ),
        calibration=calibration_table(observations, min_samples=min_samples),
        observations=observations if include_observations else [],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def format_summary(report: BacktestReport) -> str:
    lines = [
        (
            f"Backtest — method={report.method} eval_days={report.eval_days} "
            f"history_window={report.history_window_days}d until={report.until.isoformat()}"
        ),
        (
            f"observations generated={report.observations_generated} "
            f"predicted={report.predictions_made} "
            f"skipped(no history)={report.skipped_no_history}"
        ),
    ]
    o = report.overall
    if o.n and not o.suppressed:
        lines.append(
            f"overall: n={o.n} MAE={o.mae} RMSE={o.rmse} Brier={o.brier} "
            f"mean_score={o.mean_score} proxy_avail_rate={o.proxy_availability_rate}"
        )
    else:
        lines.append(f"overall: n={o.n} (insufficient observations for metrics)")
    for name, breakdown in (
        ("by_hour", report.by_hour),
        ("by_weekday", report.by_weekday),
        ("by_meter_type", report.by_meter_type),
        ("by_evidence_days", report.by_evidence_days_bucket),
    ):
        rendered = ", ".join(
            f"{k}: n={m.n}" + ("" if m.suppressed else f" MAE={m.mae}")
            for k, m in breakdown.items()
        )
        if rendered:
            lines.append(f"{name}: {rendered}")
    cal_cells = [
        f"{c['score_bucket']}:n={c['n']}"
        + (f":obs={c['proxy_availability_rate']}" if "proxy_availability_rate" in c else "")
        for c in report.calibration
        if c["n"]
    ]
    if cal_cells:
        lines.append("calibration(vs binary proxy): " + " ".join(cal_cells))
    lines.append(
        "NOTE: outcomes are a paid-session overlap PROXY, not ground-truth "
        "occupancy; scores are not calibrated probabilities."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sf_parking.backtest",
        description="Point-in-time-safe backtesting of parking-state baselines.",
    )
    parser.add_argument("--start", help="eval window start, YYYY-MM-DD (local PT)")
    parser.add_argument("--end", help="eval window end (inclusive), YYYY-MM-DD (local PT)")
    parser.add_argument("--until", help="absolute cutoff instant, ISO-8601")
    parser.add_argument("--eval-days", type=int, default=7)
    parser.add_argument("--window-days", type=int, default=28,
                        help="baseline lookback window")
    parser.add_argument("--hour", type=int, action="append",
                        help="restrict to local hour(s); repeatable")
    parser.add_argument("--post-id", action="append", dest="post_ids",
                        help="restrict to meter(s); repeatable")
    parser.add_argument("--near", metavar="LAT,LON",
                        help="restrict to meters within RADIUS_METERS of point")
    parser.add_argument("--radius-meters", type=float, default=250.0)
    parser.add_argument("--max-meters", type=int)
    parser.add_argument("--min-samples", type=int, default=30,
                        help="minimum observations per breakdown cell")
    parser.add_argument("--model", choices=sorted(MODELS), default=BASELINE_METHOD)
    parser.add_argument("--include-observations", action="store_true")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON instead of summary")
    args = parser.parse_args(argv)

    until: datetime | None = None
    eval_days = args.eval_days
    if args.start and args.end:
        start_pt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=SF_TZ)
        end_pt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=SF_TZ)
        until = (end_pt + timedelta(days=1)).astimezone(UTC)
        eval_days = max((end_pt.date() - start_pt.date()).days + 1, 1)
    elif args.until:
        until = datetime.fromisoformat(args.until)
    if until is not None and until.tzinfo is None:
        until = until.replace(tzinfo=UTC)

    scoped_post_ids = list(args.post_ids) if args.post_ids else None
    near_clause = ""
    near_params: dict[str, Any] = {}
    if args.near:
        lat_s, lon_s = args.near.split(",")
        near_clause = (
            "AND ST_DWithin(location, "
            "ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography, :radius)"
        )
        near_params = {"lat": float(lat_s), "lon": float(lon_s),
                       "radius": args.radius_meters}

    conn = pg8000.native.Connection(**_conn_kwargs())
    try:
        if scoped_post_ids is None and args.near:
            scoped_post_ids = [
                r[0]
                for r in conn.run(
                    "SELECT post_id FROM parking_meters WHERE true " + near_clause,
                    **near_params,
                )
            ]
            if not scoped_post_ids:
                print(json.dumps({"error": "no meters within radius"}))
                return 1

        report = run_backtest(
            conn,
            until=until,
            eval_days=eval_days,
            history_window_days=args.window_days,
            hours=tuple(args.hour) if args.hour else None,
            post_ids=scoped_post_ids,
            max_meters=args.max_meters,
            include_observations=args.include_observations,
            min_samples=args.min_samples,
            model=MODELS[args.model],
        )
    finally:
        conn.close()

    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(format_summary(report))
        print("use --json for machine-readable output")
    return 0


def _conn_kwargs() -> dict[str, Any]:
    import os
    from urllib.parse import urlparse

    from .database import database_url_from_env

    parsed = urlparse(database_url_from_env())
    password = parsed.password or os.environ.get("PGPASSWORD", "postgres")
    return {
        "user": parsed.username or "postgres",
        "password": password,
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "database": (parsed.path or "/").lstrip("/") or "sf_parking",
        "timeout": 60,
    }


if __name__ == "__main__":
    raise SystemExit(main())
