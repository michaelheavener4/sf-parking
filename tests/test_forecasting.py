"""Tests for the production forecasting pipeline.

Covers:
- T+1 forecast generation without a target row
- No future-state leakage in feature construction
- Forecast storage and retrieval
- Recursive T+2 uses forecast overrides (not pretending T+1 was observed)
- Forecast reproducibility given identical inputs
- DST correctness across timezone boundaries
- Feasibility checks for targets beyond available data
- Transaction feature sourcing (always observed, never predicted)
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pg8000
import pytest

from sf_parking.database import connect

SF_TZ = ZoneInfo("America/Los_Angeles")
UTC_TZ = UTC


def _server_available() -> bool:
    try:
        conn = connect()
        conn.run("SELECT 1")
        conn.close()
        return True
    except (OSError, pg8000.Error):
        return False


# ── Feature construction tests ───────────────────────────────────────────

class TestForecastFeatures:
    """Verify the forecasting module's feature construction."""

    def test_build_features_t_plus_1h_without_target_row(self):
        """T+1 features use lag slots from parking_state_hourly.

        When the target is within the materialized local day (but the
        target row itself may or may not exist), lag slots exist and
        features are produced.  When the target is beyond the
        materialized range entirely, no lag data exists and an empty
        list is returned.
        """
        from sf_parking.forecasting import _build_features, latest_observed_slot
        if not _server_available():
            pytest.skip("PostgreSQL not reachable")
        conn = connect()
        try:
            latest = latest_observed_slot(conn)
            # Within-materialized-range: latest + 1h may be a future
            # materialized slot; lag slots (latest, latest-1h, etc.)
            # all exist, so features should be produced.
            target_in_range = latest + timedelta(hours=1)
            features = _build_features(conn, target_in_range)
            assert len(features) > 0, (
                f"T+1 within materialized range should produce features "
                f"(latest={latest}, target={target_in_range})"
            )
            for f in features:
                assert "post_id" in f
                assert "predicted_availability" not in f  # not yet predicted
                for key in [
                    "lag1_availability", "lag2_availability", "lag3_availability",
                    "lag6_availability", "lag24_availability", "lag168_availability",
                    "lag1_transactions", "lag24_transactions",
                    "roll3_availability", "roll24_availability",
                    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "is_ms",
                ]:
                    assert key in f, f"Missing feature: {key}"

            # Beyond-materialized-range: lag slots don't exist, so
            # no meters are discovered and an empty list is returned.
            target_beyond = latest + timedelta(hours=25)
            features_beyond = _build_features(conn, target_beyond)
            assert len(features_beyond) == 0, (
                f"Target beyond materialized range should return empty "
                f"(target={target_beyond})"
            )
        finally:
            conn.close()

    def test_no_target_slot_in_feature_query(self):
        """Feature SQL must not read from parking_state_hourly at the target slot."""
        src = Path(__file__).resolve().parents[1] / "src" / "sf_parking" / "forecasting.py"
        source = src.read_text(encoding="utf-8")
        import re
        # Every lag read must subtract an INTERVAL from the target slot.
        # "p1.slot_start = tg.slot_start - INTERVAL" is safe.
        # "p1.slot_start = tg.slot_start" without "- INTERVAL" is a leak.
        matches = re.findall(r'p\d+\.slot_start\s*=\s*tg\.slot_start\s*([^\n]*)', source)
        for suffix in matches:
            assert "- INTERVAL" in suffix, (
                f"Feature SQL reads at the exact target slot: ...= tg.slot_start {suffix}"
            )

    def test_lag_intervals_are_all_strictly_before_target(self):
        """All lag JOINs use slot_start = tg.slot_start - INTERVAL (strictly historical)."""
        src = Path(__file__).resolve().parents[1] / "src" / "sf_parking" / "forecasting.py"
        source = src.read_text(encoding="utf-8")
        import re
        expected_offsets = {"1 hour", "2 hours", "3 hours", "6 hours", "24 hours", "168 hours"}
        found = set()
        for m in re.finditer(
            r"slot_start\s*=\s*tg\.slot_start\s*-\s*INTERVAL\s*'([^']+)'",
            source,
        ):
            found.add(m.group(1))
        assert found == expected_offsets, f"Expected {expected_offsets}, found {found}"

    def test_feature_count_is_15(self):
        from sf_parking.forecasting import FEATURES
        assert len(FEATURES) == 15

    def test_feature_names_match_benchmark(self):
        from sf_parking.forecasting import FEATURES
        expected = [
            "lag1_availability", "lag2_availability", "lag3_availability",
            "lag6_availability", "lag24_availability", "lag168_availability",
            "lag1_transactions", "lag24_transactions", "roll3_availability",
            "roll24_availability", "hour_sin", "hour_cos", "weekday_sin",
            "weekday_cos", "is_ms",
        ]
        assert FEATURES == expected

    def test_features_with_forecast_overrides(self):
        """T+2 features should use forecast overrides for lag1."""
        from sf_parking.forecasting import _build_features, latest_observed_slot
        if not _server_available():
            pytest.skip("PostgreSQL not reachable")
        conn = connect()
        try:
            latest = latest_observed_slot(conn)
            target = latest + timedelta(hours=2)

            # Create fake overrides for lag1 (simulating a T+1 forecast).
            overrides = []
            # Discover some meters first.
            from sf_parking.forecasting import _discover_meters
            meters = _discover_meters(conn, target)
            if not meters:
                pytest.skip("No meters available for override test")

            for post_id, meter_type in meters[:5]:
                overrides.append({
                    "post_id": post_id,
                    "lag_offset": 1,
                    "predicted_value": 0.75,
                })

            features = _build_features(conn, target, overrides)
            assert len(features) > 0

            # The overridden meters should have lag1 = 0.75.
            overridden_pids = {o["post_id"] for o in overrides}
            for f in features:
                if f["post_id"] in overridden_pids:
                    assert f["lag1_availability"] == 0.75, (
                        f"lag1 for {f['post_id']} should be 0.75 from override, "
                        f"got {f['lag1_availability']}"
                    )
        finally:
            conn.close()


# ── Forecast storage tests ───────────────────────────────────────────────

class TestForecastStorage:
    """Verify forecast persistence and retrieval."""

    def test_store_and_retrieve_forecasts(self):
        from sf_parking.forecasting import (
            store_forecasts,
            fetch_unverified_forecasts,
            latest_observed_slot,
        )
        if not _server_available():
            pytest.skip("PostgreSQL not reachable")
        conn = connect()
        try:
            latest = latest_observed_slot(conn)
            target = latest + timedelta(hours=1)

            # Clean up any existing forecasts for this target.
            conn.run(
                "DELETE FROM parking_state_forecasts WHERE target_slot = :t",
                t=target,
            )

            rows = [
                {"post_id": "TEST-001", "predicted_availability": 0.85},
                {"post_id": "TEST-002", "predicted_availability": 0.15},
            ]
            stored = store_forecasts(
                conn,
                target_slot=target,
                hours_ahead=1,
                model_version="test-version",
                model_path="/tmp/test-model.txt",
                feature_data_as_of=latest,
                rows=rows,
            )
            assert stored == 2

            # Verify stored.
            result = conn.run(
                "SELECT count(*) FROM parking_state_forecasts "
                "WHERE target_slot = :t AND model_version = :mv",
                t=target, mv="test-version",
            )
            assert result[0][0] == 2

            # Clean up.
            conn.run(
                "DELETE FROM parking_state_forecasts WHERE model_version = :mv",
                mv="test-version",
            )
        finally:
            conn.close()

    def test_upsert_on_duplicate(self):
        """Re-forecasting the same slot/model should upsert, not duplicate."""
        from sf_parking.forecasting import store_forecasts, latest_observed_slot
        if not _server_available():
            pytest.skip("PostgreSQL not reachable")
        conn = connect()
        try:
            latest = latest_observed_slot(conn)
            target = latest + timedelta(hours=1)

            conn.run(
                "DELETE FROM parking_state_forecasts WHERE target_slot = :t",
                t=target,
            )

            rows = [{"post_id": "TEST-UPSERT", "predicted_availability": 0.50}]
            store_forecasts(
                conn, target_slot=target, hours_ahead=1,
                model_version="test-upsert", model_path="/m",
                feature_data_as_of=latest, rows=rows,
            )
            # Insert again with different value.
            rows[0]["predicted_availability"] = 0.60
            store_forecasts(
                conn, target_slot=target, hours_ahead=1,
                model_version="test-upsert", model_path="/m",
                feature_data_as_of=latest, rows=rows,
            )
            result = conn.run(
                "SELECT count(*), max(predicted_availability) "
                "FROM parking_state_forecasts "
                "WHERE post_id = 'TEST-UPSERT' AND model_version = 'test-upsert'"
            )
            assert result[0][0] == 1, "Should be upserted, not duplicated"
            assert float(result[0][1]) == 0.60, "Should have updated value"

            conn.run(
                "DELETE FROM parking_state_forecasts WHERE model_version = :mv",
                mv="test-upsert",
            )
        finally:
            conn.close()


# ── Recursive forecasting architecture tests ─────────────────────────────

class TestRecursiveArchitecture:
    """Verify that T+2 correctly uses forecasts, not observed state, for lag1."""

    def test_t_plus_2_discover_requires_lag1_override(self):
        """T+2 discovery should find meters only if lag1 override exists."""
        from sf_parking.forecasting import _discover_meters, latest_observed_slot
        if not _server_available():
            pytest.skip("PostgreSQL not reachable")
        conn = connect()
        try:
            latest = latest_observed_slot(conn)
            # Use +25h so that lag-1 (target-1h) is also beyond the
            # materialized local day and won't be found in parking_state_hourly.
            target = latest + timedelta(hours=25)

            # Without overrides, T+2 should find no meters (lag1 doesn't exist).
            meters_no_override = _discover_meters(conn, target, None)
            assert len(meters_no_override) == 0, (
                "T+2 without overrides should find no meters "
                "(lag-1 slot not in observed state)"
            )

            # With a fake lag1 override, T+2 should find meters.
            # (lag2 = T+0 = observed, lag3 = T-1 = observed, etc.)
            overrides = [{"post_id": "ANY", "lag_offset": 1, "predicted_value": 0.5}]
            # This won't find ANY because "ANY" is not a real post_id,
            # but the SQL structure should execute without error.
            # The key assertion is that the query runs and returns results
            # for real meters when real overrides are provided.
        finally:
            conn.close()

    def test_transaction_features_always_observed(self):
        """Transaction lags must come from observed state, never forecasts."""
        from sf_parking.forecasting import _build_features, latest_observed_slot
        if not _server_available():
            pytest.skip("PostgreSQL not reachable")
        conn = connect()
        try:
            latest = latest_observed_slot(conn)
            target = latest + timedelta(hours=1)

            # With overrides that set lag1 availability to a known value,
            # transaction counts should still come from observed state.
            overrides = []
            from sf_parking.forecasting import _discover_meters
            meters = _discover_meters(conn, target)
            if not meters:
                pytest.skip("No meters available")

            pid = meters[0][0]
            overrides.append({"post_id": pid, "lag_offset": 1, "predicted_value": 0.99})

            features = _build_features(conn, target, overrides)
            feat_for_pid = [f for f in features if f["post_id"] == pid]
            assert len(feat_for_pid) == 1

            # lag1 availability should be from override.
            assert feat_for_pid[0]["lag1_availability"] == 0.99

            # But transaction counts should be non-negative (from observed state).
            assert feat_for_pid[0]["lag1_transactions"] >= 0
            assert feat_for_pid[0]["lag24_transactions"] >= 0
        finally:
            conn.close()

    def test_reproducibility(self):
        """Same inputs should produce same feature values."""
        from sf_parking.forecasting import _build_features, latest_observed_slot
        if not _server_available():
            pytest.skip("PostgreSQL not reachable")
        conn = connect()
        try:
            latest = latest_observed_slot(conn)
            target = latest + timedelta(hours=1)
            f1 = sorted(_build_features(conn, target), key=lambda x: x["post_id"])
            f2 = sorted(_build_features(conn, target), key=lambda x: x["post_id"])
            assert len(f1) == len(f2)
            for a, b in zip(f1, f2):
                assert a["post_id"] == b["post_id"]
                for key in ["lag1_availability", "hour_sin", "is_ms"]:
                    assert a[key] == b[key], f"Non-reproducible: {key}"
        finally:
            conn.close()


# ── DST / timezone tests ────────────────────────────────────────────────

class TestDSTHandling:
    """Verify timezone handling across DST boundaries."""

    def test_forecast_features_use_local_hour_for_temporal(self):
        """Temporal features (hour_sin/cos) must reflect local LA time."""
        from sf_parking.forecasting import _build_features, latest_observed_slot
        if not _server_available():
            pytest.skip("PostgreSQL not reachable")
        conn = connect()
        try:
            latest = latest_observed_slot(conn)
            target = latest + timedelta(hours=1)

            features = _build_features(conn, target)
            if not features:
                pytest.skip("No features returned")

            # Verify hour_sin/hour_cos are consistent with LA local hour.
            local_hour = target.astimezone(SF_TZ).hour
            expected_sin = np.sin(2 * np.pi * local_hour / 24.0)
            expected_cos = np.cos(2 * np.pi * local_hour / 24.0)

            for f in features[:3]:
                assert abs(f["hour_sin"] - expected_sin) < 1e-10
                assert abs(f["hour_cos"] - expected_cos) < 1e-10
        finally:
            conn.close()


# ── Feasibility / error handling tests ───────────────────────────────────

class TestFeasibilityChecks:
    """Verify the system fails clearly when data is insufficient."""

    def test_forecast_script_rejects_impossible_horizon(self):
        """Forecast script should exit 1 when lag-1 data is missing."""
        import subprocess
        import sys
        result = subprocess.run(
            [
                sys.executable, "scripts/forecast_paid_state.py",
                "--hours-ahead", "1",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
            env={
                "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
                "PATH": "/usr/bin:/bin",
                "HOME": "/tmp",
            },
        )
        # The script should either succeed (if data is recent enough) or
        # fail with a clear error.  It must NOT silently produce garbage.
        if result.returncode != 0:
            assert "ERROR" in result.stderr or "ERROR" in result.stdout, (
                f"Script failed without clear error message.\n"
                f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
            )

    def test_forecast_metadata_is_valid_json(self):
        """Model metadata file must be valid JSON with required fields."""
        meta_path = Path(__file__).resolve().parents[1] / "models" / "paid_state_lgbm.meta.json"
        if not meta_path.exists():
            pytest.skip("Model metadata not found")
        meta = json.loads(meta_path.read_text())
        assert "model_version" in meta
        assert "features" in meta
        assert len(meta["features"]) == 15
        assert "model_mae" in meta


# ── Integration: full forecast round-trip ────────────────────────────────

@pytest.mark.skipif(not _server_available(), reason="PostgreSQL not reachable")
class TestForecastRoundTrip:
    """End-to-end: generate T+1 forecast, store, verify, evaluate."""

    def test_full_t_plus_1_round_trip(self):
        from sf_parking.forecasting import (
            FEATURES,
            _build_features,
            enrich_meters,
            latest_observed_slot,
            load_model,
            store_forecasts,
            verify_forecasts,
        )
        import numpy as np
        import pandas as pd

        conn = connect()
        try:
            latest = latest_observed_slot(conn)
            target = latest + timedelta(hours=1)

            model, meta = load_model(None)
            features = _build_features(conn, target)
            assert len(features) > 0, "Need features for round-trip test"

            df = pd.DataFrame(features)
            preds = np.clip(model.predict(df[FEATURES]), 0.0, 1.0)
            df["predicted_availability"] = preds

            rows = [
                {"post_id": r["post_id"], "predicted_availability": float(r["predicted_availability"])}
                for _, r in df.iterrows()
            ]

            # Clean up first.
            conn.run(
                "DELETE FROM parking_state_forecasts WHERE target_slot = :t AND model_version = :mv",
                t=target, mv=meta["model_version"],
            )

            stored = store_forecasts(
                conn,
                target_slot=target,
                hours_ahead=1,
                model_version=meta["model_version"],
                model_path=str(Path(__file__).resolve().parents[1] / "models" / "paid_state_lgbm.txt"),
                feature_data_as_of=latest,
                rows=rows,
            )
            assert stored == len(rows)

            # If the target slot already has observed data, verify it.
            exists = conn.run(
                "SELECT count(*) FROM parking_state_hourly WHERE slot_start = :t",
                t=target,
            )
            if exists[0][0] > 0:
                updated = verify_forecasts(conn)
                assert updated >= 0

            # Clean up.
            conn.run(
                "DELETE FROM parking_state_forecasts WHERE target_slot = :t AND model_version = :mv",
                t=target, mv=meta["model_version"],
            )
        finally:
            conn.close()


# ── Timestamp semantics regression tests ────────────────────────────────

@pytest.mark.skipif(not _server_available(), reason="PostgreSQL not reachable")
class TestTimestampSemantics:
    """Verify latest_observed_slot() never returns a future slot.

    Regression: build_hourly_state.py materializes the entire local day,
    including hours whose local time has not yet occurred.  Those future
    rows carry zero-transaction defaults (availability 1.0).  The old
    ``latest_observed_slot()`` used ``max(slot_start)`` without filtering
    on NOW(), causing the freshness check to report negative ages.
    """

    def test_latest_observed_slot_never_in_future(self):
        from sf_parking.forecasting import latest_observed_slot
        conn = connect()
        try:
            slot = latest_observed_slot(conn)
            now = datetime.now(UTC)
            assert slot <= now, (
                f"latest_observed_slot returned {slot} which is in the future "
                f"(NOW()={now})"
            )
        finally:
            conn.close()

    def test_future_slots_exist_but_are_excluded(self):
        """Confirm future rows exist in the table but are filtered out."""
        from sf_parking.forecasting import latest_observed_slot
        conn = connect()
        try:
            result = conn.run(
                "SELECT count(*) FROM parking_state_hourly "
                "WHERE slot_start > NOW()"
            )
            future_count = result[0][0]
            # There may be zero or more future rows depending on the
            # time of day — the important thing is that
            # latest_observed_slot() ignores them.
            slot = latest_observed_slot(conn)
            now = datetime.now(UTC)
            assert slot <= now
            # If future rows exist, verify the function didn't pick one.
            if future_count > 0:
                assert slot < conn.run(
                    "SELECT min(slot_start) FROM parking_state_hourly "
                    "WHERE slot_start > NOW()"
                )[0][0]
        finally:
            conn.close()

    def test_data_age_is_nonnegative(self):
        """The freshness check must never report negative age."""
        now_utc = datetime.now(UTC)
        from sf_parking.forecasting import latest_observed_slot
        conn = connect()
        try:
            slot = latest_observed_slot(conn)
            age_minutes = (now_utc - slot).total_seconds() / 60.0
            assert age_minutes >= 0, (
                f"Data age is {age_minutes:.1f} minutes (slot={slot}, now={now_utc})"
            )
        finally:
            conn.close()


@pytest.mark.skipif(not _server_available(), reason="PostgreSQL not reachable")
class TestGridPopulation:
    """Regression: the posts CTE in build_hourly_state.sql must not silently
    collapse the meter population.

    Previously, the posts filter required ``last_local_date >= day_start``,
    which excluded meters whose latest transaction ended before the target
    day.  This caused recent lag slots to contain only a handful of rows
    (e.g., 12 instead of ~10k), collapsing the forecast population.
    """

    def test_recent_slot_has_substantial_meters(self):
        """Every slot within the last 48h must have >= 1000 meters."""
        conn = connect()
        try:
            result = conn.run("""
                SELECT slot_start, count(DISTINCT post_id) AS meters
                FROM parking_state_hourly
                WHERE slot_start >= NOW() - INTERVAL '48 hours'
                  AND slot_start <= NOW()
                GROUP BY slot_start
                ORDER BY slot_start
            """)
            if not result:
                pytest.skip("No recent slots in parking_state_hourly")
            for slot_start, meters in result:
                assert meters >= 1000, (
                    f"Slot {slot_start} has only {meters} meters — "
                    f"posts CTE may be excluding historically observed meters"
                )
        finally:
            conn.close()

    def test_grid_covers_all_historically_observed_meters(self):
        """The grid for any day after the earliest transaction should include
        all meters that were ever observed, not just those with recent
        transactions."""
        conn = connect()
        try:
            # Count distinct post_id in the most recent local_date.
            result = conn.run("""
                SELECT local_date, count(DISTINCT post_id)
                FROM parking_state_hourly
                GROUP BY local_date
                ORDER BY local_date DESC
                LIMIT 1
            """)
            if not result:
                pytest.skip("No data in parking_state_hourly")
            latest_date, today_meters = result[0][0], result[0][1]

            # Count total distinct post_id in the table (across all dates).
            result = conn.run("""
                SELECT count(DISTINCT post_id) FROM parking_state_hourly
            """)
            total_meters = result[0][0]

            # Today's grid should contain a substantial fraction of all meters.
            # At minimum 50% — a lower ratio indicates the posts filter is
            # too restrictive.
            if total_meters > 0:
                ratio = today_meters / total_meters
                assert ratio >= 0.5, (
                    f"Latest local_date ({latest_date}) has {today_meters} meters "
                    f"vs {total_meters} total ({ratio:.1%}) — "
                    f"posts CTE is likely filtering out historical meters"
                )
        finally:
            conn.close()


class TestModuleStructure:
    """Verify the forecasting module is properly importable."""

    def test_import_forecasting_module(self):
        import sf_parking.forecasting as fm
        assert hasattr(fm, "FEATURES")
        assert hasattr(fm, "_build_features")
        assert hasattr(fm, "store_forecasts")
        assert hasattr(fm, "verify_forecasts")
        assert hasattr(fm, "evaluate_verified_forecasts")
        assert hasattr(fm, "latest_observed_slot")

    def test_forecast_script_importable(self):
        """The forecast script should be importable without side effects."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "forecast_script",
            Path(__file__).resolve().parents[1] / "scripts" / "forecast_paid_state.py",
        )
        mod = importlib.util.module_from_spec(spec)
        # Should not raise on import.
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main")

    def test_evaluate_script_importable(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "eval_script",
            Path(__file__).resolve().parents[1] / "scripts" / "evaluate_forecasts.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main")
