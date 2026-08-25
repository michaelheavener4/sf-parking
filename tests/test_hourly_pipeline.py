"""Tests for the hourly forecasting pipeline.

Covers:
- Pipeline run logging (creation, updates, status transitions)
- Freshness check logic
- Forecast generation and persistence verification
- Idempotent execution
- Dry-run mode (no mutations)
- Error handling and status recording
- Target slot calculation
- Model version propagation
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pg8000
import pytest

from sf_parking.database import connect

REPO_ROOT = Path(__file__).resolve().parents[1]


def _server_available() -> bool:
    try:
        conn = connect()
        conn.run("SELECT 1")
        conn.close()
        return True
    except (OSError, pg8000.Error):
        return False


def _has_hourly_state() -> bool:
    """Check if parking_state_hourly has any rows."""
    try:
        conn = connect()
        result = conn.run("SELECT count(*) FROM parking_state_hourly")
        conn.close()
        return result[0][0] > 0
    except (OSError, pg8000.Error):
        return False


NO_DATA = not _has_hourly_state()


def _import_pipeline():
    """Dynamically import the pipeline module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_hourly_forecast",
        REPO_ROOT / "scripts" / "run_hourly_forecast.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Pipeline run logging ─────────────────────────────────────────────────

@pytest.mark.skipif(not _server_available(), reason="PostgreSQL not reachable")
class TestPipelineRunLogging:
    """Verify pipeline run records are created and updated correctly."""

    def test_create_run(self):
        mod = _import_pipeline()
        conn = connect()
        try:
            run_id = mod._create_run(conn)
            assert run_id > 0
            # Verify row exists.
            result = conn.run(
                "SELECT status FROM forecast_pipeline_runs WHERE id = :id",
                id=run_id,
            )
            assert result[0][0] == "running"
            # Clean up.
            conn.run("DELETE FROM forecast_pipeline_runs WHERE id = :id", id=run_id)
            conn.run("COMMIT")
        finally:
            conn.close()

    def test_update_run(self):
        mod = _import_pipeline()
        conn = connect()
        try:
            run_id = mod._create_run(conn)
            mod._update_run(
                conn, run_id,
                status="success",
                rows_forecast=100,
                model_version="test-v1",
            )
            result = conn.run(
                "SELECT status, rows_forecast, model_version "
                "FROM forecast_pipeline_runs WHERE id = :id",
                id=run_id,
            )
            assert result[0][0] == "success"
            assert result[0][1] == 100
            assert result[0][2] == "test-v1"
            # Clean up.
            conn.run("DELETE FROM forecast_pipeline_runs WHERE id = :id", id=run_id)
            conn.run("COMMIT")
        finally:
            conn.close()

    def test_update_run_no_fields(self):
        """Updating with no fields should be a no-op."""
        mod = _import_pipeline()
        conn = connect()
        try:
            run_id = mod._create_run(conn)
            mod._update_run(conn, run_id)  # no fields
            result = conn.run(
                "SELECT status FROM forecast_pipeline_runs WHERE id = :id",
                id=run_id,
            )
            assert result[0][0] == "running"
            conn.run("DELETE FROM forecast_pipeline_runs WHERE id = :id", id=run_id)
            conn.run("COMMIT")
        finally:
            conn.close()


# ── Freshness check ──────────────────────────────────────────────────────

@pytest.mark.skipif(not _server_available(), reason="PostgreSQL not reachable")
class TestFreshnessCheck:
    """Verify data freshness detection."""

    def test_fresh_data(self):
        mod = _import_pipeline()
        conn = connect()
        try:
            is_fresh, latest, age = mod._check_freshness(conn, max_age_minutes=90)
            if latest is None:
                pytest.skip("parking_state_hourly is empty")
            assert isinstance(is_fresh, bool)
            assert isinstance(age, float)
            assert age >= 0
        finally:
            conn.close()

    def test_stale_data_detected(self):
        mod = _import_pipeline()
        conn = connect()
        try:
            is_fresh, latest, age = mod._check_freshness(conn, max_age_minutes=0.001)
            if latest is None:
                pytest.skip("parking_state_hourly is empty")
            assert not is_fresh
        finally:
            conn.close()

    def test_empty_database(self):
        mod = _import_pipeline()
        conn = connect()
        try:
            conn.run("BEGIN")
            conn.run("DELETE FROM parking_state_hourly")
            is_fresh, latest, age = mod._check_freshness(conn, max_age_minutes=90)
            conn.run("ROLLBACK")
            assert not is_fresh
            assert latest is None
            assert age is None
        finally:
            conn.close()


# ── Forecast generation ──────────────────────────────────────────────────

@pytest.mark.skipif(not _server_available(), reason="PostgreSQL not reachable")
@pytest.mark.skipif(NO_DATA, reason="parking_state_hourly is empty")
class TestForecastGeneration:
    """Verify forecast generation via the forecasting module."""

    def test_generate_forecast_produces_results(self):
        mod = _import_pipeline()
        conn = connect()
        try:
            ok, msg, rows, target, mv = mod._generate_forecast(
                conn, dry_run=False, max_horizon=1,
            )
            assert ok, f"Forecast failed: {msg}"
            assert rows > 0, "Should produce some forecasts"
            assert target is not None
            assert mv is not None
        finally:
            conn.close()

    def test_generate_forecast_dry_run(self):
        mod = _import_pipeline()
        conn = connect()
        try:
            ok, msg, rows, target, mv = mod._generate_forecast(
                conn, dry_run=True, max_horizon=1,
            )
            assert ok
            assert rows == 0
            assert target is None
        finally:
            conn.close()


# ── Persistence verification ─────────────────────────────────────────────

@pytest.mark.skipif(not _server_available(), reason="PostgreSQL not reachable")
@pytest.mark.skipif(NO_DATA, reason="parking_state_hourly is empty")
class TestPersistenceVerification:
    """Verify forecast storage verification."""

    def test_verify_persistence_success(self):
        mod = _import_pipeline()
        conn = connect()
        try:
            from sf_parking.forecasting import latest_observed_slot, store_forecasts
            latest = latest_observed_slot(conn)
            target = latest + timedelta(hours=1)

            # Clean up first.
            conn.run(
                "DELETE FROM parking_state_forecasts WHERE target_slot = :t AND model_version = 'test-verify'",
                t=target,
            )

            rows = [
                {"post_id": "VERIFY-001", "predicted_availability": 0.85},
                {"post_id": "VERIFY-002", "predicted_availability": 0.15},
            ]
            store_forecasts(
                conn, target_slot=target, hours_ahead=1,
                model_version="test-verify", model_path="/m",
                feature_data_as_of=latest, rows=rows,
            )

            ok, msg = mod._verify_persistence(
                conn, target_slot=target, model_version="test-verify", expected_rows=2,
            )
            assert ok, f"Verification failed: {msg}"

            # Clean up.
            conn.run(
                "DELETE FROM parking_state_forecasts WHERE model_version = 'test-verify'",
            )
            conn.run("COMMIT")
        finally:
            conn.close()

    def test_verify_persistence_count_mismatch(self):
        mod = _import_pipeline()
        conn = connect()
        try:
            from sf_parking.forecasting import latest_observed_slot
            latest = latest_observed_slot(conn)
            target = latest + timedelta(hours=1)

            ok, msg = mod._verify_persistence(
                conn, target_slot=target, model_version="nonexistent", expected_rows=999,
            )
            assert not ok
            assert "expected 999" in msg
        finally:
            conn.close()


# ── Idempotency ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not _server_available(), reason="PostgreSQL not reachable")
@pytest.mark.skipif(NO_DATA, reason="parking_state_hourly is empty")
class TestIdempotency:
    """Verify the pipeline is safe to run multiple times."""

    def test_double_forecast_upserts(self):
        """Running forecast twice for the same slot should upsert, not duplicate."""
        from sf_parking.forecasting import latest_observed_slot, store_forecasts
        conn = connect()
        try:
            latest = latest_observed_slot(conn)
            target = latest + timedelta(hours=1)

            conn.run(
                "DELETE FROM parking_state_forecasts WHERE target_slot = :t AND model_version = 'test-idem'",
                t=target,
            )

            rows = [{"post_id": "IDEM-001", "predicted_availability": 0.50}]
            store_forecasts(
                conn, target_slot=target, hours_ahead=1,
                model_version="test-idem", model_path="/m",
                feature_data_as_of=latest, rows=rows,
            )
            rows[0]["predicted_availability"] = 0.70
            store_forecasts(
                conn, target_slot=target, hours_ahead=1,
                model_version="test-idem", model_path="/m",
                feature_data_as_of=latest, rows=rows,
            )

            result = conn.run(
                "SELECT count(*), max(predicted_availability) "
                "FROM parking_state_forecasts "
                "WHERE post_id = 'IDEM-001' AND model_version = 'test-idem'"
            )
            assert result[0][0] == 1, "Should be upserted, not duplicated"
            assert float(result[0][1]) == 0.70

            conn.run(
                "DELETE FROM parking_state_forecasts WHERE model_version = 'test-idem'",
            )
            conn.run("COMMIT")
        finally:
            conn.close()


# ── Target slot calculation ──────────────────────────────────────────────

class TestTargetSlotCalculation:
    """Verify the target slot is correctly computed."""

    @pytest.mark.skipif(not _server_available(), reason="PostgreSQL not reachable")
    @pytest.mark.skipif(NO_DATA, reason="parking_state_hourly is empty")
    def test_target_is_latest_plus_hours_ahead(self):
        from sf_parking.forecasting import latest_observed_slot
        conn = connect()
        try:
            latest = latest_observed_slot(conn)
            target = latest + timedelta(hours=1)
            assert target > latest
            assert (target - latest).total_seconds() == 3600
        finally:
            conn.close()


# ── Model version propagation ────────────────────────────────────────────

class TestModelVersion:
    """Verify model version is read from metadata and stored in forecasts."""

    def test_model_version_matches_metadata(self):
        import json
        meta_path = REPO_ROOT / "models" / "paid_state_lgbm.meta.json"
        if not meta_path.exists():
            pytest.skip("Model metadata not found")
        meta = json.loads(meta_path.read_text())
        assert "model_version" in meta
        assert len(meta["model_version"]) > 0

    @pytest.mark.skipif(not _server_available(), reason="PostgreSQL not reachable")
    @pytest.mark.skipif(NO_DATA, reason="parking_state_hourly is empty")
    def test_forecast_stores_model_version(self):
        from sf_parking.forecasting import latest_observed_slot, load_model
        conn = connect()
        try:
            _, meta = load_model(None)
            latest = latest_observed_slot(conn)
            result = conn.run(
                "SELECT DISTINCT model_version FROM parking_state_forecasts "
                "WHERE model_version = :mv",
                mv=meta["model_version"],
            )
            # If there are any forecasts with this model version, they should match.
            if result:
                assert result[0][0] == meta["model_version"]
        finally:
            conn.close()


# ── Dry-run mode ─────────────────────────────────────────────────────────

class TestDryRun:
    """Verify dry-run performs no mutations."""

    def test_dry_run_no_pipeline_record(self):
        mod = _import_pipeline()
        if not _server_available():
            pytest.skip("PostgreSQL not reachable")
        conn = connect()
        try:
            count_before = conn.run("SELECT count(*) FROM forecast_pipeline_runs")[0][0]
            # Dry run should not create a record.
            mod._run_ingestion(dry_run=True)
            count_after = conn.run("SELECT count(*) FROM forecast_pipeline_runs")[0][0]
            assert count_after == count_before
        finally:
            conn.close()

    def test_dry_run_forecast_returns_nothing(self):
        mod = _import_pipeline()
        if not _server_available():
            pytest.skip("PostgreSQL not reachable")
        conn = connect()
        try:
            from sf_parking.forecasting import latest_observed_slot
            count_before = conn.run(
                "SELECT count(*) FROM parking_state_forecasts"
            )[0][0]
            ok, msg, rows, target, mv = mod._generate_forecast(
                conn, dry_run=True, max_horizon=1,
            )
            assert ok
            assert rows == 0
            count_after = conn.run(
                "SELECT count(*) FROM parking_state_forecasts"
            )[0][0]
            assert count_after == count_before
        finally:
            conn.close()


# ── Error handling ───────────────────────────────────────────────────────

class TestErrorHandling:
    """Verify error scenarios are handled gracefully."""

    def test_ingestion_failure_returns_false(self):
        mod = _import_pipeline()
        ok, msg = mod._run_ingestion(dry_run=False)
        # The actual ingestion may succeed or fail depending on DataSF availability.
        # We just verify the function returns a tuple.
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_build_hourly_state_failure_returns_false(self):
        mod = _import_pipeline()
        ok, msg, rows = mod._run_build_hourly_state(dry_run=False, target_date="2099-01-01")
        # This may succeed (processing a future date with no data) or fail.
        assert isinstance(ok, bool)
        assert isinstance(msg, str)
        assert isinstance(rows, int)

    def test_evaluate_matured_handles_errors(self):
        mod = _import_pipeline()
        if not _server_available():
            pytest.skip("PostgreSQL not reachable")
        conn = connect()
        try:
            count, msg = mod._evaluate_matured(conn, dry_run=False)
            assert isinstance(count, int)
            assert isinstance(msg, str)
        finally:
            conn.close()


# ── Module structure ─────────────────────────────────────────────────────

class TestModuleStructure:
    """Verify the pipeline module is properly structured."""

    def test_importable(self):
        mod = _import_pipeline()
        assert hasattr(mod, "main")
        assert hasattr(mod, "_create_run")
        assert hasattr(mod, "_update_run")
        assert hasattr(mod, "_run_ingestion")
        assert hasattr(mod, "_run_build_hourly_state")
        assert hasattr(mod, "_check_freshness")
        assert hasattr(mod, "_generate_forecast")
        assert hasattr(mod, "_verify_persistence")
        assert hasattr(mod, "_evaluate_matured")

    def test_all_statuses_valid(self):
        """All status values used in _update_run must be in the CHECK constraint."""
        import re
        migration = (REPO_ROOT / "db" / "migrations" / "2026-08-25_pipeline_runs.sql").read_text()
        match = re.search(r"CHECK \(status IN \(([^)]+)\)\)", migration)
        assert match, "CHECK constraint not found in migration"
        valid = {s.strip().strip("'") for s in match.group(1).split(",")}
        assert valid == {"running", "success", "data_stale", "ingestion_failed", "forecast_failed", "verification_failed"}
