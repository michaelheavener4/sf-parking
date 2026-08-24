"""Tests for the deterministic_v0 backtesting/evaluation framework.

Pure-function tests cover the V0 formula and metric aggregation; integration
tests run against the throwaway-schema database and focus on temporal cutoff
enforcement, empty history, DST boundaries, determinism, and proxy outcomes.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pg8000
import pytest

from sf_parking.backtest import (
    BacktestObservation,
    DeterministicV0Baseline,
    Prediction,
    compute_metrics,
    run_backtest,
    score_v0,
)

_V0 = DeterministicV0Baseline()
_predict = _V0.predict
from sf_parking.database import apply_schema, connect

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "db" / "schema.sql"
PT = timedelta(hours=-7)  # PDT throughout these tests (August 2026)


def _server_available() -> bool:
    try:
        conn = connect()
        conn.run("SELECT 1")
        conn.close()
        return True
    except (OSError, pg8000.Error):
        return False


def pt(day: int, hour: int, minute: int = 0) -> datetime:
    """August 2026 day-of-month at a Pacific wall time, as absolute instant.

    PDT = UTC-7, so UTC hour = PT hour + 7.
    """
    return datetime(2026, 8, day, hour + 7, minute, tzinfo=UTC)


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
    db.run("TRUNCATE meter_transactions, parking_meters, ingestion_runs "
           "RESTART IDENTITY CASCADE")
    db.run("COMMIT")


def _tx(conn, post_id, start, end):
    conn.run(
        "INSERT INTO meter_transactions (transmission_id, post_id, session_start,"
        " session_end, duration_minutes, source, retrieved_at)"
        " VALUES (:t, :p, :s, :e, :d, 'backtest-test', now())",
        t=f"{post_id}-{start.isoformat()}-{end.isoformat()}",
        p=post_id,
        s=start,
        e=end,
        d=int((end - start).total_seconds() // 60),
    )


def _meter(conn, post_id, meter_type="SS"):
    conn.run(
        "INSERT INTO parking_meters (post_id, latitude, longitude, active, meter_type)"
        " VALUES (:p, 37.79, -122.4, true, :m)",
        p=post_id,
        m=meter_type,
    )
    conn.run("COMMIT")


UNTIL = datetime(2026, 8, 21, tzinfo=UTC)  # Aug 20 17:00 PT


class TestScoreAndMetrics:
    def test_v0_formula_clamps_and_rounds(self):
        assert score_v0(0.0, 60.0) == 1.0
        assert score_v0(30.0, 60.0) == 0.5
        assert score_v0(60.0, 60.0) == 0.0
        assert score_v0(600.0, 60.0) == 0.0  # clamped, never negative
        assert score_v0(1.0, 3.0) == round(1 - 1 / 3, 3)

    def test_metric_calculations_hand_computed(self):
        def obs(score, avail, occ):
            slot = datetime(2026, 8, 20, 20, tzinfo=UTC)
            return BacktestObservation(
                post_id="m", prediction_time=slot, cutoff=slot,
                target_hour_start=slot, local_date=slot.date(),
                local_hour=13, weekday="Thu", post_blockface_id=None,
                latitude=None, longitude=None, location_source="unresolved",
                predicted_score=score, proxy_occupied_minutes=occ,
                proxy_availability=avail,
                prediction_error=round(score - (1 - occ / 60.0), 4),
                method_version="deterministic_v0", evidence_days=1,
                evidence_sessions=1,
            )

        obs = [obs(1.0, 1, 0.0), obs(0.0, 0, 60.0), obs(0.5, 1, 0.0)]
        m = compute_metrics(obs)
        assert m.n == 3
        assert m.mean_score == pytest.approx(0.5)
        assert m.proxy_availability_rate == pytest.approx(2 / 3, abs=1e-4)
        # Brier vs binary proxy: (0)^2 + (0-0)^2... scores vs availability:
        # (1-1)^2 + (0-0)^2 + (0.5-1)^2 => (0 + 0 + 0.25)/3
        assert m.brier == pytest.approx(0.25 / 3, abs=1e-4)
        # MAE vs continuous free-fraction: |1-1| + |0-0| + |0.5-1| over 3
        assert m.mae == pytest.approx(0.5 / 3, abs=1e-4)

    def test_empty_observations_give_zeroed_metrics(self):
        m = compute_metrics([])
        assert m.n == 0 and m.brier is None and m.mean_score is None


class TestCutoffEnforcement:
    def test_future_session_inside_predicted_slot_does_not_leak(self, db):
        # History: one fully-paid hour the previous day.
        _meter(db, "363-04151")
        _tx(db, "363-04151", pt(19, 13), pt(19, 14))
        # Trap: paid session INSIDE the predicted slot (after its cutoff).
        # If this leaked into prediction, the score would not be exactly 0.
        _tx(db, "363-04151", pt(20, 13, 10), pt(20, 13, 40))
        db.run("COMMIT")

        report = run_backtest(
            db, until=UNTIL, eval_days=3, include_observations=True
        )
        slot_obs = [
            o for o in report.observations
            if o.target_hour_start == datetime(2026, 8, 20, 20, tzinfo=UTC)
        ]
        assert len(slot_obs) == 1
        o = slot_obs[0]
        assert o.predicted_score == 0.0  # history says hour 13 is always taken
        assert o.evidence_days == 1
        # The trap session IS visible to the outcome proxy (it happened):
        assert o.proxy_availability == 0
        assert o.proxy_occupied_minutes == pytest.approx(30.0)  # minutes

    def test_ongoing_session_is_truncated_at_cutoff(self, db):
        # A session spanning from before the cutoff until after it must count
        # only its elapsed portion: its end time is unknowable at prediction.
        _meter(db, "m-long")
        _tx(db, "m-long", pt(19, 13), pt(19, 14))          # prior-day evidence
        _tx(db, "m-long",
            datetime(2026, 8, 19, 21, 30, tzinfo=UTC),   # Aug 19 14:30 PT
            datetime(2026, 8, 22, 8, tzinfo=UTC))        # ends AFTER the cutoff
        db.run("COMMIT")

        sessions = [
            (datetime(2026, 8, 19, 20, tzinfo=UTC), datetime(2026, 8, 19, 21, tzinfo=UTC)),
            (datetime(2026, 8, 19, 21, 30, tzinfo=UTC), datetime(2026, 8, 22, 8, tzinfo=UTC)),
        ]
        cutoff_slot = datetime(2026, 8, 20, 20, tzinfo=UTC)  # 13:00 PT Thu
        result = _predict(sessions, cutoff_slot, history_window_days=28)

        # Truncated history ends Aug 20 13:00 PT -> evidence_days = 2, so
        # possible = 120 min; occupied = full overlap with Aug 19's hour 13.
        # An UNTRUNCATED end (Aug 22) would inflate evidence_days to 3+.
        assert isinstance(result, Prediction)
        assert result.evidence_days == 2
        assert result.evidence_sessions == 2
        assert result.score == pytest.approx(score_v0(60, 120))

    def test_session_starting_after_cutoff_never_influences_prediction(
        self, db
    ):
        _meter(db, "m-future")
        _tx(db, "m-future", pt(18, 9), pt(18, 10))  # only pre-cutoff history
        db.run("COMMIT")

        kwargs = {
            "until": datetime(2026, 8, 19, 20, tzinfo=UTC),
            "eval_days": 2,
            "include_observations": True,
        }
        baseline = run_backtest(db, **kwargs)
        baseline_scores = {
            (o.post_id, o.target_hour_start): o.predicted_score for o in baseline.observations
        }
        assert baseline_scores

        # Insert data starting AFTER every predicted slot, then re-run the
        # same evaluation window: predictions must be identical.
        _tx(db, "m-future", pt(20, 9), pt(20, 10))
        _tx(db, "m-future", pt(25, 12), pt(25, 13))
        db.run("COMMIT")
        after = run_backtest(db, **kwargs)
        assert {
            (o.post_id, o.target_hour_start): o.predicted_score for o in after.observations
        } == baseline_scores
        # The outcome proxy, by contrast, legitimately sees nothing new for
        # these slots either - future sessions land outside the window.
        assert after.predictions_made == baseline.predictions_made


class TestEmptyHistory:
    def test_slots_before_first_session_are_skipped_not_fabricated(self, db):
        _meter(db, "m-late")
        _tx(db, "m-late", pt(20, 15), pt(20, 16))  # first ever: Aug 20 15:00 PT
        db.run("COMMIT")

        report = run_backtest(
            db, until=UNTIL, eval_days=4, include_observations=True
        )
        assert report.predictions_made > 0
        # Every observation slot before the meter's first-ever session must
        # have been skipped (no basis for a number), never scored as free.
        first_slot = datetime(2026, 8, 20, 22, tzinfo=UTC)  # 15:00 PT
        early = [
            o for o in report.observations if o.target_hour_start < first_slot
        ]
        assert early == []
        assert report.skipped_no_history > 0

    def test_meter_without_transactions_generates_nothing(self, db):
        _meter(db, "m-silent")  # inventory row, zero transactions
        db.run("COMMIT")
        report = run_backtest(db, until=UNTIL, eval_days=7)
        assert report.observations_generated == 0
        assert report.predictions_made == 0
        assert report.overall.n == 0


class TestDSTBoundaries:
    def test_fall_back_repeated_hour_predictions(self, db):
        # History covering both occurrences of the repeated local hour on
        # different days around the Nov 2 2025 fall-back transition.
        _meter(db, "dst-m")
        _tx(db, "dst-m",
            datetime(2025, 10, 31, 8, tzinfo=UTC),   # 01:00 PDT (first pass)
            datetime(2025, 10, 31, 9, tzinfo=UTC))
        _tx(db, "dst-m",
            datetime(2025, 11, 1, 9, tzinfo=UTC),    # 01:00 PST (second pass)
            datetime(2025, 11, 1, 10, tzinfo=UTC))
        db.run("COMMIT")

        kwargs = {
            "until": datetime(2025, 11, 3, 12, tzinfo=UTC),
            "eval_days": 3,
            "include_observations": True,
        }
        report = run_backtest(db, **kwargs)

        hour1 = [o for o in report.observations if o.local_hour == 1]
        # Both absolute occurrences of the repeated hour are observed.
        starts = sorted(o.target_hour_start for o in hour1)
        assert datetime(2025, 11, 2, 8, tzinfo=UTC) in starts   # 01:00 PDT
        assert datetime(2025, 11, 2, 9, tzinfo=UTC) in starts   # 01:00 PST
        assert all(0.0 <= o.predicted_score <= 1.0 for o in report.observations)

        rerun = run_backtest(db, **kwargs)
        assert rerun == report  # deterministic across DST too


class TestDeterminism:
    def test_reports_identical_across_runs(self, db):
        _meter(db, "363-04151")
        _meter(db, "102-02990", meter_type="MS")
        for day in (17, 18, 19, 20):
            _tx(db, "363-04151", pt(day, 9), pt(day, 9, 45))
            _tx(db, "102-02990", pt(day, 13), pt(day, 14))
        db.run("COMMIT")

        r1 = run_backtest(db, until=UNTIL, eval_days=4)
        r2 = run_backtest(db, until=UNTIL, eval_days=4)
        assert r1 == r2


class TestBreakdowns:
    def test_hour_weekday_and_meter_type_breakdowns(self, db):
        _meter(db, "ss-meter", meter_type="SS")
        _meter(db, "ms-meter", meter_type="MS")
        for day in (18, 19, 20):  # Tue..Thu
            _tx(db, "ss-meter", pt(day, 9), pt(day, 10))     # 09:00-10:00 PT
            _tx(db, "ms-meter", pt(day, 14), pt(day, 15))    # 14:00-15:00 PT
        db.run("COMMIT")

        report = run_backtest(db, until=UNTIL, eval_days=3)
        assert set(report.by_meter_type) == {"MS", "SS"}
        assert {int(h) for h in report.by_hour} >= {9, 14}
        assert {"Tue", "Wed", "Thu"} <= set(report.by_weekday)
        # Sample-size buckets exist and every bucket carries metrics.
        assert all(b.n > 0 for b in report.by_evidence_days_bucket.values())
        assert sum(b.n for b in report.by_evidence_days_bucket.values()) \
            == report.predictions_made


class TestPointInTimeSpatial:
    def test_location_resolved_from_placement_valid_at_t(self, db):
        # Canonical chain: meter + blockface + two placements over time.
        db.run(
            "INSERT INTO meters (post_id, source, retrieved_at)"
            " VALUES ('m-pit', 'test', now())"
        )
        db.run(
            "INSERT INTO blockfaces (source_blockface_id, source, retrieved_at)"
            " VALUES ('363041', 'test', now())"
        )
        db.run("COMMIT")
        db.run(
            "INSERT INTO meter_placements (meter_id, blockface_id, active,"
            " source_post_id, latitude, longitude, valid_from,"
            " valid_until, source, retrieved_at)"
            " SELECT meter_id, blockface_id, true, 'm-pit', 37.10, -122.10,"
            " :vf, :vu, 'test', now() FROM meters, blockfaces",
            vf=datetime(2026, 8, 1, tzinfo=UTC),
            vu=datetime(2026, 8, 19, 20, tzinfo=UTC),
        )
        db.run(
            "INSERT INTO meter_placements (meter_id, active, source_post_id,"
            " latitude, longitude, valid_from, source, retrieved_at)"
            " SELECT meter_id, true, 'm-pit', 37.20, -122.20,"
            " :vf, 'test', now() FROM meters",
            vf=datetime(2026, 8, 19, 20, tzinfo=UTC),
        )
        _tx(db, "m-pit", pt(18, 9), pt(18, 10))
        _tx(db, "m-pit", pt(19, 9), pt(19, 10))
        db.run("COMMIT")

        report = run_backtest(
            db,
            until=datetime(2026, 8, 21, tzinfo=UTC),
            eval_days=4,
            include_observations=True,
        )
        before = [o for o in report.observations
                  if o.target_hour_start < datetime(2026, 8, 19, 20, tzinfo=UTC)]
        after = [o for o in report.observations
                 if o.target_hour_start >= datetime(2026, 8, 19, 20, tzinfo=UTC)]
        assert all(
            o.location_source == "placement_at_t"
            and (o.latitude, o.longitude, o.post_blockface_id)
            == (37.10, -122.10, "363041")
            for o in before
        )
        assert all(
            o.location_source == "placement_at_t"
            and (o.latitude, o.longitude) == (37.20, -122.20)
            for o in after
        )
        # The spatial-area breakdown keys off the point-in-time blockface.
        assert set(report.by_blockface) <= {"363041", "unresolved"}

    def test_no_placement_means_unresolved_not_invented(self, db):
        _meter(db, "m-noplace")
        _tx(db, "m-noplace", pt(18, 9), pt(18, 10))
        db.run("COMMIT")
        report = run_backtest(
            db, until=UNTIL, eval_days=2, include_observations=True
        )
        assert report.observations
        assert all(
            o.location_source == "unresolved" and o.latitude is None
            for o in report.observations
        )


class TestProvenanceAndCalibration:
    def test_observation_provenance_is_complete(self, db):
        _meter(db, "prov-m")
        _tx(db, "prov-m", pt(18, 13), pt(18, 14))
        db.run("COMMIT")
        report = run_backtest(db, until=UNTIL, eval_days=3,
                              include_observations=True)
        for o in report.observations:
            assert o.method_version == "deterministic_v0"
            assert o.cutoff == o.target_hour_start == o.prediction_time
            expected_error = round(
                o.predicted_score
                - (1 - o.proxy_occupied_minutes / 60.0), 4
            )
            assert o.prediction_error == expected_error

    def test_min_samples_suppresses_sparse_cells(self, db):
        _meter(db, "sparse-m")
        _tx(db, "sparse-m", pt(18, 13), pt(18, 14))
        db.run("COMMIT")
        report = run_backtest(db, until=UNTIL, eval_days=2, min_samples=100)
        assert report.predictions_made > 0
        assert report.overall.suppressed or report.overall.n < 100
        for cell in report.by_hour.values():
            assert cell.n < 100
            if cell.suppressed:
                assert cell.mae is None

    def test_calibration_buckets_report_counts_and_rates(self, db):
        _meter(db, "cal-m")
        for day in (17, 18, 19, 20):
            _tx(db, "cal-m", pt(day, 13), pt(day, 14))  # hour always occupied
        db.run("COMMIT")
        report = run_backtest(db, until=UNTIL, eval_days=4, min_samples=1)
        nonempty = [c for c in report.calibration if c["n"]]
        assert any(c["score_bucket"] == "0.0-0.1" for c in nonempty)
        for c in nonempty:
            if "proxy_availability_rate" in c:
                assert 0.0 <= c["proxy_availability_rate"] <= 1.0


class TestByteIdenticalSerialization:
    def test_same_state_and_params_give_identical_json(self, db):
        _meter(db, "det-m")
        for day in (17, 18, 19, 20):
            _tx(db, "det-m", pt(day, 9), pt(day, 9, 30))
            _tx(db, "det-m", pt(day, 14), pt(day, 15))
        db.run("COMMIT")
        kwargs = {
            "until": UNTIL,
            "eval_days": 4,
            "include_observations": True,
        }
        dumps = [
            json.dumps(run_backtest(db, **kwargs).to_dict(),
                       sort_keys=True, separators=(",", ":"))
            for _ in range(3)
        ]
        assert dumps[0] == dumps[1] == dumps[2]
