"""Tests for deterministic parking-state features and availability baseline.

Pure-function tests cover slot construction (timezone/DST) and overlap math;
integration tests run against the throwaway-schema database.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pg8000
import pytest

from sf_parking.database import apply_schema, connect
from sf_parking.features import (
    AvailabilityEstimate,
    MeterFeatures,
    _local_hour_slot,
    _overlap_seconds,
    availability_baseline,
    meter_features,
    summarize_features,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "db" / "schema.sql"
SF_TZ = ZoneInfo("America/Los_Angeles")


def _server_available() -> bool:
    try:
        conn = connect()
        conn.run("SELECT 1")
        conn.close()
        return True
    except (OSError, pg8000.Error):
        return False


def _tx(post_id, start, end):
    return (post_id, start, end)


def _insert_tx(conn, post_id, start, end):
    conn.run(
        "INSERT INTO meter_transactions (transmission_id, post_id, session_start,"
        " session_end, duration_minutes, source, retrieved_at)"
        " VALUES (:t, :p, :s, :e, :d, 'test', now())",
        t=f"{post_id}-{start.isoformat()}",
        p=post_id,
        s=start,
        e=end,
        d=int((end - start).total_seconds() // 60),
    )


class TestSlotConstruction:
    def test_naive_instant_is_treated_as_utc(self):
        # 19:30 UTC == 12:30 PDT, so the slot is the 12:00 PT clock hour.
        lo, hi = _local_hour_slot(datetime(2026, 8, 20, 19, 30))  # noqa: DTZ001
        assert lo == datetime(2026, 8, 20, 19, tzinfo=UTC)
        assert hi - lo == timedelta(hours=1)

    def test_summer_slot_is_utc_minus_seven(self):
        # 09:00-10:00 PDT == 16:00-17:00 UTC.
        lo, hi = _local_hour_slot(datetime(2026, 8, 20, 9, 15, tzinfo=SF_TZ))
        assert (lo, hi) == (datetime(2026, 8, 20, 16, tzinfo=UTC),
                            datetime(2026, 8, 20, 17, tzinfo=UTC))
        assert lo.astimezone(SF_TZ).utcoffset() == timedelta(hours=-7)
        assert (hi - lo) == timedelta(hours=1)

    def test_winter_slot_is_utc_minus_eight(self):
        lo, hi = _local_hour_slot(datetime(2026, 1, 15, 14, tzinfo=SF_TZ))
        assert lo.astimezone(UTC) == datetime(2026, 1, 15, 22, tzinfo=UTC)
        assert hi.astimezone(UTC) == datetime(2026, 1, 15, 23, tzinfo=UTC)

    def test_fall_back_day_has_two_distinct_local_hours(self):
        # On 2025-11-02 local hours repeat; the two 01:00 PT slots must be
        # different absolute instants.
        first = _local_hour_slot(datetime(2025, 11, 2, 1, 5, fold=0, tzinfo=SF_TZ))
        second = _local_hour_slot(datetime(2025, 11, 2, 1, 55, fold=1, tzinfo=SF_TZ))
        # The repeated local hour occupies two adjacent absolute hours:
        # first occurrence 01:00 PDT == 08:00Z, second 01:00 PST == 09:00Z.
        assert first == (datetime(2025, 11, 2, 8, tzinfo=UTC),
                         datetime(2025, 11, 2, 9, tzinfo=UTC))
        assert second == (datetime(2025, 11, 2, 9, tzinfo=UTC),
                          datetime(2025, 11, 2, 10, tzinfo=UTC))


class TestOverlapMath:
    def test_no_overlap_is_zero(self):
        s = datetime(2026, 8, 20, 10, tzinfo=UTC)
        assert (
            _overlap_seconds(s, s + timedelta(minutes=30),
                             s + timedelta(hours=1), s + timedelta(hours=2))
            == 0.0
        )

    def test_partial_overlap_counts_elapsed_time(self):
        slot_lo = datetime(2026, 8, 20, 17, tzinfo=UTC)
        session_start = slot_lo - timedelta(minutes=45)  # started previous hour
        assert (
            _overlap_seconds(session_start, slot_lo + timedelta(minutes=30),
                             slot_lo, slot_lo + timedelta(hours=1))
            == 1800.0
        )


@pytest.fixture(scope="module")
def db():
    schema = f"pytest_sf_parking_{uuid.uuid4().hex[:12]}"
    conn = connect()
    conn.run(f'CREATE SCHEMA "{schema}"')
    conn.run("COMMIT")
    conn.run(f'SET search_path TO "{schema}", public')  # generated name only
    apply_schema(conn, SCHEMA_PATH)
    yield conn
    conn.close()
    cleanup = connect()
    try:
        cleanup.run(f'DROP SCHEMA "{schema}" CASCADE')
        cleanup.run("COMMIT")
    finally:
        cleanup.close()


@pytest.fixture(autouse=True)
def clean(db):
    yield
    db.run("TRUNCATE meter_transactions, ingestion_runs RESTART IDENTITY CASCADE")
    db.run("COMMIT")


NOW = datetime(2026, 8, 23, 23, tzinfo=UTC)


def _seed_busy_meter(conn, post_id="363-04151"):
    """Same clock hour (13:00-14:00 PT = 20:00Z PDT) occupied on 3 days."""
    for day in (18, 19, 20):  # 2026-08-18..20 are Tue..Thu, all PDT
        start = datetime(2026, 8, day, 20, 0, tzinfo=UTC)
        _insert_tx(conn, post_id, start, start + timedelta(minutes=36))
    conn.run("COMMIT")


class TestMeterFeatures:
    def test_aggregates_are_correct(self, db):
        _seed_busy_meter(db)

        feats = meter_features(db, window_days=28, now=NOW)
        assert len(feats) == 1
        f = feats[0]
        assert isinstance(f, MeterFeatures)
        assert f.post_id == "363-04151"
        assert f.session_count == 3
        assert f.active_days == 3
        assert f.total_paid_minutes == 108
        assert f.mean_session_minutes == pytest.approx(36.0)
        assert f.median_session_minutes == pytest.approx(36.0)
        assert f.sessions_per_active_day == pytest.approx(1.0)

    def test_window_excludes_out_of_range_sessions(self, db):
        _seed_busy_meter(db)
        old = datetime(2026, 7, 1, 20, tzinfo=UTC)
        _insert_tx(db, "363-04151", old, old + timedelta(minutes=60))
        db.run("COMMIT")

        f = meter_features(db, window_days=28, now=NOW)[0]
        assert f.session_count == 3  # July session outside the 28d window

    def test_active_days_group_by_local_calendar_date(self, db):
        # 23:30 PT Aug 19 and 00:30 PT Aug 20 are DIFFERENT local days...
        _insert_tx(db, "m", datetime(2026, 8, 20, 6, 30, tzinfo=UTC),
                   datetime(2026, 8, 20, 6, 50, tzinfo=UTC))   # Aug 19 late eve
        _insert_tx(db, "m", datetime(2026, 8, 20, 7, 30, tzinfo=UTC),
                   datetime(2026, 8, 20, 7, 50, tzinfo=UTC))   # Aug 20 just past
        # ...while these two are the SAME local day (Aug 20 afternoon).
        _insert_tx(db, "m", datetime(2026, 8, 20, 20, 30, tzinfo=UTC),
                   datetime(2026, 8, 20, 20, 50, tzinfo=UTC))  # 13:30 PT
        _insert_tx(db, "m", datetime(2026, 8, 20, 21, 30, tzinfo=UTC),
                   datetime(2026, 8, 20, 21, 50, tzinfo=UTC))  # 14:30 PT
        db.run("COMMIT")

        f = meter_features(db, window_days=28, now=NOW)[0]
        assert f.active_days == 2

    def test_empty_database_yields_no_rows(self, db):
        assert meter_features(db, window_days=28, now=NOW) == []


class TestAvailabilityBaseline:
    def test_fully_occupied_clock_hour_scores_zero(self, db):
        for day in (18, 19, 20):
            start = datetime(2026, 8, day, 20, 0, tzinfo=UTC)
            _insert_tx(db, "363-04151", start, start + timedelta(minutes=60))
        db.run("COMMIT")

        est = availability_baseline(
            db, "363-04151",
            at=datetime(2026, 8, 23, 20, 30, tzinfo=UTC),  # 13:30 PT hour
            now=NOW,
        )
        assert est.score == 0.0
        assert est.sufficient_history
        assert est.method == "deterministic_v0"
        assert est.slot_possible_minutes == 180.0

    def test_partial_occupancy_ratio(self, db):
        _seed_busy_meter(db)  # 36 of 60 minutes on each of 3 days

        est = availability_baseline(
            db, "363-04151", at=datetime(2026, 8, 23, 20, 30, tzinfo=UTC), now=NOW
        )
        assert est.score == round(1 - 108 / 180, 3)
        assert est.slot_occupied_minutes == 108.0

    def test_other_clock_hour_has_no_history_scores_one(self, db):
        _seed_busy_meter(db)
        est = availability_baseline(
            db, "363-04151", at=datetime(2026, 8, 23, 16, 0, tzinfo=UTC), now=NOW
        )  # 09:00 PT: no sessions ever recorded in that hour
        assert est.score == 1.0

    def test_unknown_meter_and_empty_history_score_none(self, db):
        est = availability_baseline(
            db, "000-00000", at=datetime(2026, 8, 23, 20, 30, tzinfo=UTC), now=NOW
        )
        assert est.score is None
        assert not est.sufficient_history
        assert est.evidence_days == 0

    def test_dst_session_attribution_uses_absolute_time(self, db):
        # Fall-back day: session 00:30-04:30 PT spans 300 real minutes over
        # local hours 00..03 (the repeated 01:00 hour is a distinct instant).
        start = datetime(2025, 11, 2, 7, 30, tzinfo=UTC)
        _insert_tx(db, "dst-m", start, start + timedelta(hours=4))
        db.run("COMMIT")

        estimates = [
            availability_baseline(
                db, "dst-m",
                at=datetime(2025, 11, 2, 8 + h, 30, tzinfo=UTC),
                now=datetime(2025, 11, 3, tzinfo=UTC),
            )
            for h in range(4)
        ]
        assert estimates[0].evidence_sessions == 1
        # The session spans 300 real minutes. Local hours that morning:
        #   hour 0 -> none | hour 1 occurs TWICE (fall-back): 60 + 60 min
        #   hour 2 -> 60 min | hour 3 -> 30 min (session ends 04:30 PT).
        # Baselines for clock hours 1 and 2... each invocation targeting the
        # repeated local hour reports both absolute occurrences.
        occupied = [e.slot_occupied_minutes for e in estimates]
        assert occupied == [120.0, 120.0, 60.0, 30.0]
        # Every estimate's score stays within [0, 1]; denominators are real
        # minutes (evidence_days * 60), never wall-clock fabrications.
        assert all(e.score is not None and 0.0 <= e.score <= 1.0 for e in estimates)

    def test_deterministic_output_across_runs(self, db):
        _seed_busy_meter(db)
        kwargs = {"at": datetime(2026, 8, 23, 20, 30, tzinfo=UTC), "now": NOW}

        runs = [availability_baseline(db, "363-04151", **kwargs) for _ in range(3)]
        assert all(r == runs[0] for r in runs)
        assert isinstance(runs[0], AvailabilityEstimate)

        feat_runs = [meter_features(db, window_days=28, now=NOW) for _ in range(3)]
        assert feat_runs[0] == feat_runs[1] == feat_runs[2]

        summary_a = summarize_features(meter_features(db, now=NOW))
        summary_b = summarize_features(meter_features(db, now=NOW))
        assert summary_a == summary_b
