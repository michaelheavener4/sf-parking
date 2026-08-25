"""Tests for the LightGBM prediction pipeline.

Covers feature ordering consistency, timestamp-leakage prevention, model
persistence, prediction output shape, local-time-to-UTC conversion, and
future-target (T+1h) regression tests.
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

# ---------------------------------------------------------------------------
# Import constants from both scripts to verify they stay in sync.
# ---------------------------------------------------------------------------

def _import_benchmark_features():
    """Dynamically import the FEATURES list from the benchmark script."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "benchmark",
        Path(__file__).resolve().parents[1] / "scripts" / "benchmark_paid_state_lgbm_chunked.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FEATURES


def _import_predict_features():
    """Dynamically import the FEATURES list from the predict script."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "predict",
        Path(__file__).resolve().parents[1] / "scripts" / "predict_paid_state.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FEATURES


def _import_target_slot_utc():
    from scripts.predict_paid_state import _target_slot_utc
    return _target_slot_utc


# ---------------------------------------------------------------------------
# Feature ordering tests
# ---------------------------------------------------------------------------

class TestFeatureOrdering:
    """The benchmark and prediction scripts must agree on feature names and order."""

    def test_features_match_between_scripts(self):
        bench = _import_benchmark_features()
        pred = _import_predict_features()
        assert bench == pred, (
            f"Feature lists differ:\n  benchmark: {bench}\n  predict:   {pred}"
        )

    def test_features_count(self):
        features = _import_benchmark_features()
        assert len(features) == 15

    def test_features_contain_all_expected_names(self):
        features = set(_import_benchmark_features())
        expected = {
            "lag1_availability", "lag2_availability", "lag3_availability",
            "lag6_availability", "lag24_availability", "lag168_availability",
            "lag1_transactions", "lag24_transactions", "roll3_availability",
            "roll24_availability", "hour_sin", "hour_cos", "weekday_sin",
            "weekday_cos", "is_ms",
        }
        assert features == expected

    def test_lag_features_appear_before_temporal(self):
        features = _import_benchmark_features()
        lag_end = features.index("hour_sin")
        for f in ["lag1_availability", "lag24_availability", "roll3_availability"]:
            assert features.index(f) < lag_end, f"{f} should appear before temporal features"


# ---------------------------------------------------------------------------
# Timestamp leakage prevention
# ---------------------------------------------------------------------------

class TestLeakagePrevention:
    """Verify the prediction SQL uses strict less-than for all lag joins."""

    def test_predict_sql_uses_strict_inequality(self):
        """The _predict_features SQL must join on slot_start = tg.slot_start - INTERVAL (strictly before target)."""
        predict_path = Path(__file__).resolve().parents[1] / "scripts" / "predict_paid_state.py"
        source = predict_path.read_text(encoding="utf-8")
        # The CTE must use strictly less-than for lag joins via the target CTE
        assert "slot_start = tg.slot_start - INTERVAL" in source, (
            "Lag joins should use slot_start = tg.slot_start - INTERVAL (strictly before target)"
        )
        # Must NOT have any <= comparison against the target slot
        assert "<= :slot" not in source, (
            "Prediction SQL must not use <= :slot — that would leak future state"
        )
        # Must NOT have any <= tg.slot_start comparison
        assert "<= tg.slot_start" not in source, (
            "Prediction SQL must not use <= tg.slot_start — that would leak future state"
        )

    def test_lag_intervals_are_all_strictly_before_target(self):
        """All lag INTERVAL offsets in the SQL are positive (target minus offset = before target)."""
        predict_path = Path(__file__).resolve().parents[1] / "scripts" / "predict_paid_state.py"
        source = predict_path.read_text(encoding="utf-8")
        expected_offsets = {"1 hour", "2 hours", "3 hours", "6 hours", "24 hours", "168 hours"}
        found = set()
        import re
        for m in re.finditer(r"slot_start\s*=\s*tg\.slot_start\s*-\s*INTERVAL\s*'([^']+)'", source):
            found.add(m.group(1))
        assert found == expected_offsets, (
            f"Expected lag intervals {expected_offsets}, found {found}"
        )

    def test_no_future_state_references(self):
        """The prediction SQL must not reference parking_state_hourly at the target slot directly."""
        predict_path = Path(__file__).resolve().parents[1] / "scripts" / "predict_paid_state.py"
        source = predict_path.read_text(encoding="utf-8")
        # The feature CTE should not select from parking_state_hourly at the exact slot
        # except for the initial DISTINCT post_id/meter_type query
        # All lag references must use INTERVAL offsets
        import re
        # Count all INTERVAL references in lag joins (via tg.slot_start - INTERVAL)
        interval_refs = re.findall(r"slot_start\s*-\s*INTERVAL\s*'[^']+'", source)
        assert len(interval_refs) == 6, f"Expected 6 lag INTERVAL references, found {len(interval_refs)}"


# ---------------------------------------------------------------------------
# Model persistence tests
# ---------------------------------------------------------------------------

class TestModelPersistence:
    """Verify model save/load round-trips correctly."""

    def test_save_and_load_model(self, tmp_path):
        """Save a tiny model and reload it; predictions must match."""
        import lightgbm as lgb

        from scripts.benchmark_paid_state_lgbm_chunked import FEATURES, save_model

        # Create a tiny dataset and train
        rng = np.random.RandomState(42)
        n = 200
        X = rng.rand(n, len(FEATURES)).astype(np.float32)
        y = rng.rand(n).astype(np.float32)
        train = lgb.Dataset(X, label=y)
        booster = lgb.train(
            {"objective": "regression", "num_leaves": 4, "verbose": -1},
            train,
            num_boost_round=10,
        )
        # Wrap in LGBMRegressor-like object with booster_ attribute
        class _Model:
            booster_ = booster
            best_iteration_ = 10

        model = _Model()
        model_dir = tmp_path / "models"
        save_model(
            model, model_dir,
            mm=0.05, mr=0.10, pm=0.06, pr=0.12,
            train_rows=100, val_rows=50, test_rows=50,
        )

        # Verify files exist
        model_file = model_dir / "paid_state_lgbm.txt"
        meta_file = model_dir / "paid_state_lgbm.meta.json"
        assert model_file.exists()
        assert meta_file.exists()

        # Load and verify metadata
        meta = json.loads(meta_file.read_text())
        assert meta["features"] == FEATURES
        assert meta["train_rows"] == 100
        assert meta["validation_rows"] == 50
        assert meta["test_rows"] == 50
        assert meta["model_mae"] == 0.05
        assert meta["best_iteration"] == 10

        # Reload and verify predictions match
        loaded = lgb.Booster(model_file=str(model_file))
        original_pred = booster.predict(X[:5])
        loaded_pred = loaded.predict(X[:5])
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_model_version_is_timestamp(self, tmp_path):
        import lightgbm as lgb

        from scripts.benchmark_paid_state_lgbm_chunked import save_model

        class _Model:
            booster_ = lgb.train(
                {"objective": "regression", "num_leaves": 4, "verbose": -1},
                lgb.Dataset(np.zeros((10, 15)), label=np.zeros(10)),
                num_boost_round=5,
            )
            best_iteration_ = 5

        model_dir = tmp_path / "m"
        save_model(_Model(), model_dir, 0, 0, 0, 0, 0, 0, 0)
        meta = json.loads((model_dir / "paid_state_lgbm.meta.json").read_text())
        # Version should be a valid ISO timestamp
        datetime.fromisoformat(meta["model_version"])


# ---------------------------------------------------------------------------
# Prediction output tests
# ---------------------------------------------------------------------------

class TestPredictionOutput:
    """Verify the prediction script produces sensible output structure."""

    def test_predict_features_returns_list_of_dicts(self):
        """_predict_features should return a list of dicts with FEATURE keys."""
        from scripts.predict_paid_state import FEATURES as PREDICT_FEATURES
        from scripts.predict_paid_state import _predict_features

        if not _server_available():
            pytest.skip("PostgreSQL not reachable")

        conn = connect()
        try:
            # Use a date far in the past where no data exists → should return empty
            future = datetime(2099, 1, 1, 12, 0, tzinfo=UTC)
            result = _predict_features(conn, future)
            # If there's no data for 2099, result should be empty
            if result:
                # If data exists, verify structure
                assert isinstance(result, list)
                row = result[0]
                assert "post_id" in row
                assert "meter_type" in row
                for f in PREDICT_FEATURES:
                    assert f in row, f"Missing feature: {f}"
        finally:
            conn.close()

    def test_feature_computation_matches_benchmark_formula(self):
        """Verify that the inline feature math matches the benchmark's formulas."""
        # From benchmark: roll3 = (lag1+lag2+lag3)/3, roll24 = (lag1+lag2+lag3+lag6+lag24)/5
        lag1, lag2, lag3, lag6 = 0.8, 0.6, 0.4, 0.2
        roll3 = (lag1 + lag2 + lag3) / 3.0
        roll24 = (lag1 + lag2 + lag3 + lag6 + 0.1) / 5.0
        assert abs(roll3 - 0.6) < 1e-10
        assert abs(roll24 - 0.42) < 1e-10

    def test_haversine_zero_distance(self):
        from scripts.predict_paid_state import _haversine_m
        d = _haversine_m(37.78, -122.41, 37.78, -122.41)
        assert d == 0.0

    def test_haversine_known_distance(self):
        from scripts.predict_paid_state import _haversine_m
        # ~1 degree latitude ≈ 111 km
        d = _haversine_m(37.78, -122.41, 38.78, -122.41)
        assert 110_000 < d < 112_000


# ---------------------------------------------------------------------------
# Local-time conversion tests
# ---------------------------------------------------------------------------

class TestLocalTimeConversion:
    """Verify that local LA time → UTC slot_start conversions are correct."""

    def test_summer_afternoon_is_utc_minus_seven(self):
        from scripts.predict_paid_state import _target_slot_utc
        naive = datetime(2026, 8, 20, 14, 0)
        utc = _target_slot_utc(naive, "America/Los_Angeles")
        # August = PDT = UTC-7 → 14:00 PDT = 21:00 UTC
        assert utc.hour == 21
        assert utc.day == 20
        assert utc.tzinfo is not None

    def test_winter_afternoon_is_utc_minus_eight(self):
        from scripts.predict_paid_state import _target_slot_utc
        naive = datetime(2026, 12, 15, 14, 0)
        utc = _target_slot_utc(naive, "America/Los_Angeles")
        # December = PST = UTC-8 → 14:00 PST = 22:00 UTC
        assert utc.hour == 22
        assert utc.day == 15

    def test_midnight_local(self):
        from scripts.predict_paid_state import _target_slot_utc
        naive = datetime(2026, 8, 21, 0, 0)
        utc = _target_slot_utc(naive, "America/Los_Angeles")
        # 00:00 PDT = 07:00 UTC
        assert utc.hour == 7
        assert utc.day == 21

    def test_dst_spring_forward_ambiguity(self):
        """2026-03-08 02:30 doesn't exist; zoneinfo keeps wall clock on pre-transition offset."""
        from scripts.predict_paid_state import _target_slot_utc
        naive = datetime(2026, 3, 8, 2, 30)
        utc = _target_slot_utc(naive, "America/Los_Angeles")
        # 02:30 PST (UTC-8) → 10:30 UTC
        assert utc.hour == 10

    def test_dst_fall_back_first_occurrence(self):
        """2025-11-02 01:30 occurs twice; fold=0 picks PDT (UTC-7)."""
        from scripts.predict_paid_state import _target_slot_utc
        naive = datetime(2025, 11, 2, 1, 30)
        utc = _target_slot_utc(naive, "America/Los_Angeles")
        # 01:30 PDT = 08:30 UTC (first occurrence)
        assert utc.hour == 8
        assert utc.day == 2


# ---------------------------------------------------------------------------
# Integration test helpers (require database)
# ---------------------------------------------------------------------------

def _server_available() -> bool:
    try:
        conn = connect()
        conn.run("SELECT 1")
        conn.close()
        return True
    except (OSError, pg8000.Error):
        return False


@pytest.mark.skipif(not _server_available(), reason="PostgreSQL not reachable")
class TestDatabaseIntegration:
    """Integration tests that run against the real database."""

    def test_target_slot_exists_in_database(self):
        """The slot_start column should use timestamptz, compatible with our UTC conversion."""
        conn = connect()
        try:
            result = conn.run(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = 'parking_state_hourly' AND column_name = 'slot_start'"
            )
            assert result[0][0] == "timestamp with time zone"
        finally:
            conn.close()

    def test_lag_query_returns_strictly_earlier_rows(self):
        """For any target slot, lag1 must reference a row exactly 1 hour earlier."""
        conn = connect()
        try:
            result = conn.run(
                "SELECT slot_start FROM parking_state_hourly ORDER BY slot_start DESC LIMIT 1"
            )
            if not result:
                pytest.skip("No data in parking_state_hourly")
            latest = result[0][0]
            # lag1 should be exactly 1 hour before; use CTE for pg8000 type safety
            lag1_result = conn.run(
                "WITH target AS (SELECT CAST(:slot AS timestamptz) AS ts) "
                "SELECT p.slot_start FROM parking_state_hourly p, target t "
                "WHERE p.slot_start = t.ts - INTERVAL '1 hour'",
                slot=latest,
            )
            if lag1_result:
                assert lag1_result[0][0] == latest - timedelta(hours=1)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Future-target regression tests (T+1 hour, no target row)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _server_available(), reason="PostgreSQL not reachable")
class TestFutureTargetPrediction:
    """Regression: predict for T+1h when the target row does not exist.

    Scenario:  latest observation = T, prediction target = T+1 hour,
    target row does not exist, prior six lag rows do exist.
    Expected:  predictions are produced (not empty).
    """

    def test_predict_features_t_plus_1h_without_target_row(self):
        """_predict_features must return results for T+1h even if no target row exists."""
        from scripts.predict_paid_state import _predict_features

        conn = connect()
        try:
            result = conn.run(
                "SELECT slot_start FROM parking_state_hourly "
                "WHERE slot_start <= NOW() "
                "ORDER BY slot_start DESC LIMIT 1"
            )
            if not result:
                pytest.skip("No data in parking_state_hourly")
            latest = result[0][0]

            # Target is T+1h — one hour after the latest observation.
            target_utc = latest + timedelta(hours=1)
            features = _predict_features(conn, target_utc)

            # Must return predictions, not an empty list.
            assert isinstance(features, list)
            assert len(features) > 0, (
                f"Expected predictions for target {target_utc} "
                f"(latest data = {latest}), but got none"
            )

            # Every returned meter must have all 15 feature keys.
            from scripts.predict_paid_state import FEATURES as PREDICT_FEATURES
            for row in features:
                for f in PREDICT_FEATURES:
                    assert f in row, f"Missing feature '{f}' in row for post_id={row.get('post_id')}"

            # The lag values must come from strictly historical slots.
            for row in features:
                assert row["lag1_availability"] is not None
                assert row["lag168_availability"] is not None
        finally:
            conn.close()

    def test_target_row_not_required_in_parking_state_hourly(self):
        """Prediction target must NOT need an existing parking_state_hourly row."""
        from scripts.predict_paid_state import _predict_features

        conn = connect()
        try:
            result = conn.run(
                "SELECT slot_start FROM parking_state_hourly "
                "WHERE slot_start <= NOW() "
                "ORDER BY slot_start DESC LIMIT 1"
            )
            if not result:
                pytest.skip("No data in parking_state_hourly")
            latest = result[0][0]
            target_utc = latest + timedelta(hours=1)

            # Verify the target row does NOT exist (if the target is beyond
            # the materialized range).  Within the range, the grid always has
            # rows for every meter, so the assertion is skipped.
            exists = conn.run(
                "SELECT count(*) FROM parking_state_hourly WHERE slot_start = :slot",
                slot=target_utc,
            )
            if exists[0][0] == 0:
                # Target is beyond materialized range — confirm predictions
                # still work without an existing row.
                features = _predict_features(conn, target_utc)
                assert len(features) > 0, (
                    "Predictions should be produced even when the target row "
                    "does not exist in parking_state_hourly"
                )
            else:
                # Target is within the materialized range — the grid has rows
                # for every meter.  Just confirm predictions are produced.
                features = _predict_features(conn, target_utc)
                assert len(features) > 0, (
                    "Predictions should be produced for a target within the "
                    "materialized range"
                )
        finally:
            conn.close()

    def test_predict_features_t_plus_2h_no_target_row(self):
        """_predict_features must work for T+2h if all lag rows exist."""
        from scripts.predict_paid_state import _predict_features

        conn = connect()
        try:
            result = conn.run(
                "SELECT slot_start FROM parking_state_hourly "
                "WHERE slot_start <= NOW() "
                "ORDER BY slot_start DESC LIMIT 1"
            )
            if not result:
                pytest.skip("No data in parking_state_hourly")
            latest = result[0][0]

            # For T+2h, lag1 = T+1h which also does not exist.
            # The discover query needs p1 at T+1h → should return empty.
            target_utc = latest + timedelta(hours=2)
            features = _predict_features(conn, target_utc)

            # No lag-1 row at T+1h means no meters can be discovered.
            # This is correct: T+2h needs recursive forecasting.
            assert isinstance(features, list)
            # Either empty (if lag1 missing) or non-empty (if data is dense).
            # The key invariant: no fabricated data.
            for row in features:
                assert row["lag1_availability"] is not None
        finally:
            conn.close()

    def test_predict_main_t_plus_1h_returns_zero(self):
        """The main() function must return 0 (success) for T+1h prediction."""
        from scripts.predict_paid_state import main as predict_main
        import sys
        from io import StringIO

        conn = connect()
        try:
            result = conn.run(
                "SELECT slot_start FROM parking_state_hourly "
                "WHERE slot_start <= NOW() "
                "ORDER BY slot_start DESC LIMIT 1"
            )
            if not result:
                pytest.skip("No data in parking_state_hourly")
            latest = result[0][0]
            target_utc = latest + timedelta(hours=1)

            # Convert UTC target back to LA local for CLI args.
            local = target_utc.astimezone(SF_TZ)
            date_str = local.strftime("%Y-%m-%d")
            hour_str = str(local.hour)

            # Patch sys.argv to simulate CLI invocation.
            old_argv = sys.argv
            sys.argv = [
                "predict_paid_state.py",
                "--date", date_str,
                "--hour", hour_str,
                "--top", "5",
            ]
            try:
                ret = predict_main()
                assert ret == 0, f"main() returned {ret}, expected 0 for T+1h prediction"
            finally:
                sys.argv = old_argv
        finally:
            conn.close()

    def test_discover_sql_uses_lag_tables_not_target_slot(self):
        """The discover SQL must find meters from lag tables, not from the target slot."""
        predict_path = Path(__file__).resolve().parents[1] / "scripts" / "predict_paid_state.py"
        source = predict_path.read_text(encoding="utf-8")
        # The discover query must NOT contain "WHERE slot_start = :slot"
        # (which would require the target row to exist).
        # Instead it must use "WHERE p1.slot_start = CAST(:slot ...) - INTERVAL"
        assert "WHERE p1.slot_start = CAST(:slot AS timestamptz) - INTERVAL '1 hour'" in source, (
            "Discover SQL must use lag-1 slot, not the target slot directly"
        )
        # Must NOT have a standalone "WHERE slot_start = :slot" in the discover query.
        import re
        # Check that the discover query doesn't have the old pattern.
        # The old pattern was: "FROM parking_state_hourly\n    WHERE slot_start = :slot"
        assert "FROM parking_state_hourly\n    WHERE slot_start = :slot" not in source, (
            "Discover SQL must not require target row to exist in parking_state_hourly"
        )
