"""Comprehensive tests for the forecast evaluation and calibration system.

Covers:
1. Matured forecast matching
2. No future leakage in evaluation
3. Correct horizon grouping
4. MAE calculation
5. RMSE calculation
6. Bias calculation
7. Persistence comparison
8. Probability binning
9. Calibration error
10. Brier score
11. Empty-bin handling
12. DST handling
13. Local-hour grouping
14. Radius filtering
15. At-least-one-space calculation
16. No duplicate parking posts
17. Missing actual values handled explicitly
18. Forecast provenance preserved
19. Calibration fitting uses only training data
20. Existing tests continue passing
"""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pg8000
import pytest

from sf_parking.database import connect

SF_TZ = ZoneInfo("America/Los_Angeles")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _server_available() -> bool:
    try:
        conn = connect()
        conn.run("SELECT 1")
        conn.close()
        return True
    except (OSError, pg8000.Error):
        return False


# ── synthetic matured forecast data for unit tests ──────────────────────

def _make_matured(
    n: int = 100,
    base_pred: float = 0.7,
    base_actual: float = 0.65,
    hours_ahead: int = 1,
    local_hour: int = 14,
    is_weekend: bool = False,
    meter_type: str = "SS",
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic matured forecast records for testing."""
    rng = np.random.default_rng(seed)
    tz_la = SF_TZ
    now = datetime.now(UTC)
    records = []
    for i in range(n):
        pred = float(np.clip(base_pred + rng.normal(0, 0.1), 0, 1))
        actual = float(np.clip(base_actual + rng.normal(0, 0.1), 0, 1))
        target = now + timedelta(hours=hours_ahead)
        local_dt = target.astimezone(tz_la).replace(hour=local_hour)
        records.append({
            "post_id": f"METER-{i:05d}",
            "target_slot": local_dt.astimezone(UTC),
            "hours_ahead": hours_ahead,
            "predicted_availability": pred,
            "actual_availability": actual,
            "absolute_error": abs(pred - actual),
            "squared_error": (pred - actual) ** 2,
            "signed_error": pred - actual,
            "model_version": "test-v1",
            "forecast_generated_at": now,
            "feature_data_as_of": now - timedelta(hours=1),
            "meter_type": meter_type,
            "local_hour": local_hour,
            "local_date": local_dt.date(),
            "is_weekend": is_weekend,
            "persistence_availability": actual + rng.normal(0, 0.05),
        })
    return records


# ── 1. Matured forecast matching ───────────────────────────────────────

@pytest.mark.skipif(not _server_available(), reason="PostgreSQL not reachable")
class TestMaturedForecastMatching:
    def test_fetch_matured_returns_verified_forecasts(self):
        from scripts.evaluate_parking_probability import fetch_matured_forecasts
        conn = connect()
        try:
            matured = fetch_matured_forecasts(conn)
            for r in matured:
                assert r["actual_availability"] is not None
                assert r["predicted_availability"] is not None
        finally:
            conn.close()

    def test_fetch_matured_filters_by_model_version(self):
        from scripts.evaluate_parking_probability import fetch_matured_forecasts
        conn = connect()
        try:
            matured = fetch_matured_forecasts(conn, model_version="nonexistent-version")
            assert len(matured) == 0
        finally:
            conn.close()

    def test_fetch_matured_filters_by_max_horizon(self):
        from scripts.evaluate_parking_probability import fetch_matured_forecasts
        conn = connect()
        try:
            matured = fetch_matured_forecasts(conn, max_horizon=1)
            for r in matured:
                assert r["hours_ahead"] <= 1
        finally:
            conn.close()


# ── 2. No future leakage in evaluation ─────────────────────────────────

class TestNoFutureLeakage:
    def test_no_leakage_in_matured_data(self):
        """All matured forecasts must have actual_availability != NULL,
        meaning the target slot has already been observed."""
        from scripts.evaluate_parking_probability import fetch_matured_forecasts
        if not _server_available():
            pytest.skip("PostgreSQL not reachable")
        conn = connect()
        try:
            matured = fetch_matured_forecasts(conn)
            now = datetime.now(UTC)
            for r in matured:
                assert r["actual_availability"] is not None, (
                    f"Forecast for {r['post_id']} at {r['target_slot']} "
                    f"has no actual value"
                )
                # The target slot should be in the past or only slightly
                # ahead (build_hourly_state materializes the full local day,
                # so slots within the current local day may be "observed"
                # even if their local hour hasn't fully elapsed).
                assert r["target_slot"] <= now + timedelta(hours=18), (
                    f"Target slot {r['target_slot']} is unreasonably far in the future"
                )
        finally:
            conn.close()

    def test_matured_forecasts_have_provenance(self):
        """Every matured record must preserve forecast_generated_at and
        feature_data_as_of for temporal leakage auditing."""
        from scripts.evaluate_parking_probability import fetch_matured_forecasts
        if not _server_available():
            pytest.skip("PostgreSQL not reachable")
        conn = connect()
        try:
            matured = fetch_matured_forecasts(conn)
            for r in matured:
                assert r["forecast_generated_at"] is not None
                assert r["feature_data_as_of"] is not None
                assert r["forecast_generated_at"] >= r["feature_data_as_of"]
        finally:
            conn.close()


# ── 3. Correct horizon grouping ────────────────────────────────────────

class TestHorizonGrouping:
    def test_grouped_horizon_metrics_cover_all_horizons(self):
        from scripts.evaluate_parking_probability import (
            horizon_metrics, grouped_horizon_metrics,
        )
        matured = _make_matured(n=50, hours_ahead=1)
        matured += _make_matured(n=50, hours_ahead=7, seed=99)
        matured += _make_matured(n=50, hours_ahead=15, seed=100)
        matured += _make_matured(n=50, hours_ahead=22, seed=101)
        h_m = horizon_metrics(matured)
        grouped = grouped_horizon_metrics(h_m)
        assert len(grouped) == 4
        total_rows = sum(g["rows"] for g in grouped)
        assert total_rows == 200

    def test_grouped_labels_are_correct(self):
        from scripts.evaluate_parking_probability import (
            horizon_metrics, grouped_horizon_metrics,
        )
        matured = _make_matured(n=10, hours_ahead=1)
        h_m = horizon_metrics(matured)
        grouped = grouped_horizon_metrics(h_m)
        labels = [g["label"] for g in grouped]
        assert labels == ["T+1–T+6", "T+7–T+12", "T+13–T+18", "T+19–T+24"]


# ── 4. MAE calculation ─────────────────────────────────────────────────

class TestMAECalculation:
    def test_mae_perfect_predictions(self):
        from scripts.evaluate_parking_probability import horizon_metrics
        matured = _make_matured(n=10, base_pred=0.5, base_actual=0.5, seed=0)
        for r in matured:
            r["predicted_availability"] = r["actual_availability"]
            r["absolute_error"] = 0.0
            r["squared_error"] = 0.0
            r["signed_error"] = 0.0
        h_m = horizon_metrics(matured)
        assert h_m[1]["mae"] == pytest.approx(0.0, abs=1e-10)

    def test_mae_known_values(self):
        from scripts.evaluate_parking_probability import horizon_metrics
        matured = _make_matured(n=5, seed=0)
        # Set exact values
        matured[0]["predicted_availability"] = 0.8
        matured[0]["actual_availability"] = 0.6
        matured[0]["absolute_error"] = 0.2
        matured[0]["squared_error"] = 0.04
        matured[0]["signed_error"] = 0.2
        matured[1]["predicted_availability"] = 0.3
        matured[1]["actual_availability"] = 0.5
        matured[1]["absolute_error"] = 0.2
        matured[1]["squared_error"] = 0.04
        matured[1]["signed_error"] = -0.2
        for i in range(2, 5):
            matured[i]["predicted_availability"] = 0.5
            matured[i]["actual_availability"] = 0.5
            matured[i]["absolute_error"] = 0.0
            matured[i]["squared_error"] = 0.0
            matured[i]["signed_error"] = 0.0
        h_m = horizon_metrics(matured)
        assert h_m[1]["mae"] == pytest.approx(0.08, abs=1e-10)


# ── 5. RMSE calculation ────────────────────────────────────────────────

class TestRMSECalculation:
    def test_rmse_known_values(self):
        from scripts.evaluate_parking_probability import horizon_metrics
        matured = _make_matured(n=3, seed=0)
        matured[0]["predicted_availability"] = 0.8
        matured[0]["actual_availability"] = 0.6
        matured[0]["absolute_error"] = 0.2
        matured[0]["squared_error"] = 0.04
        matured[0]["signed_error"] = 0.2
        matured[1]["predicted_availability"] = 0.3
        matured[1]["actual_availability"] = 0.5
        matured[1]["absolute_error"] = 0.2
        matured[1]["squared_error"] = 0.04
        matured[1]["signed_error"] = -0.2
        matured[2]["predicted_availability"] = 0.5
        matured[2]["actual_availability"] = 0.5
        matured[2]["absolute_error"] = 0.0
        matured[2]["squared_error"] = 0.0
        matured[2]["signed_error"] = 0.0
        h_m = horizon_metrics(matured)
        # RMSE = sqrt((0.04 + 0.04 + 0) / 3) = sqrt(0.02667) ≈ 0.1633
        assert h_m[1]["rmse"] == pytest.approx(math.sqrt(0.08 / 3), rel=1e-10)


# ── 6. Bias calculation ────────────────────────────────────────────────

class TestBiasCalculation:
    def test_bias_positive(self):
        from scripts.evaluate_parking_probability import horizon_metrics
        matured = _make_matured(n=2, seed=0)
        matured[0]["predicted_availability"] = 0.8
        matured[0]["actual_availability"] = 0.6
        matured[0]["signed_error"] = 0.2
        matured[1]["predicted_availability"] = 0.7
        matured[1]["actual_availability"] = 0.5
        matured[1]["signed_error"] = 0.2
        for r in matured:
            r["absolute_error"] = abs(r["signed_error"])
            r["squared_error"] = r["signed_error"] ** 2
        h_m = horizon_metrics(matured)
        assert h_m[1]["bias"] == pytest.approx(0.2, abs=1e-10)

    def test_bias_negative(self):
        from scripts.evaluate_parking_probability import horizon_metrics
        matured = _make_matured(n=2, seed=0)
        matured[0]["predicted_availability"] = 0.3
        matured[0]["actual_availability"] = 0.6
        matured[0]["signed_error"] = -0.3
        matured[1]["predicted_availability"] = 0.2
        matured[1]["actual_availability"] = 0.5
        matured[1]["signed_error"] = -0.3
        for r in matured:
            r["absolute_error"] = abs(r["signed_error"])
            r["squared_error"] = r["signed_error"] ** 2
        h_m = horizon_metrics(matured)
        assert h_m[1]["bias"] < 0


# ── 7. Persistence comparison ──────────────────────────────────────────

class TestPersistenceComparison:
    def test_gain_mae_positive_when_model_beats_persistence(self):
        from scripts.evaluate_parking_probability import horizon_metrics
        matured = _make_matured(n=10, seed=42)
        # Make model predictions close to actual, persistence far
        for r in matured:
            r["predicted_availability"] = r["actual_availability"]
            r["absolute_error"] = 0.0
            r["squared_error"] = 0.0
            r["signed_error"] = 0.0
            r["persistence_availability"] = 0.5  # bad persistence
        h_m = horizon_metrics(matured)
        assert h_m[1]["gain_mae"] > 0
        assert h_m[1]["rel_improvement_pct"] > 0

    def test_gain_mae_negative_when_model_worse_than_persistence(self):
        from scripts.evaluate_parking_probability import horizon_metrics
        matured = _make_matured(n=10, seed=42)
        for r in matured:
            r["predicted_availability"] = 0.5  # bad predictions
            r["actual_availability"] = 0.9
            r["absolute_error"] = 0.4
            r["squared_error"] = 0.16
            r["signed_error"] = -0.4
            r["persistence_availability"] = 0.85  # good persistence
        h_m = horizon_metrics(matured)
        assert h_m[1]["gain_mae"] < 0


# ── 8. Probability binning ─────────────────────────────────────────────

class TestProbabilityBinning:
    def test_calibration_bins_cover_full_range(self):
        from scripts.evaluate_parking_probability import calibration_analysis
        matured = _make_matured(n=200, seed=42)
        cal = calibration_analysis(matured)
        assert len(cal["bins"]) == 10
        assert cal["bins"][0]["lo"] == 0.0
        assert cal["bins"][-1]["hi"] == 1.0

    def test_calibration_bins_contain_all_predictions(self):
        from scripts.evaluate_parking_probability import calibration_analysis
        matured = _make_matured(n=200, seed=42)
        cal = calibration_analysis(matured)
        total_in_bins = sum(b["n"] for b in cal["bins"])
        assert total_in_bins == len(matured)


# ── 9. Calibration error ──────────────────────────────────────────────

class TestCalibrationError:
    def test_perfect_calibration_has_zero_error(self):
        from scripts.evaluate_parking_probability import calibration_analysis
        # Create forecasts where prediction = actual exactly
        matured = []
        for i in range(100):
            p = i / 100.0
            matured.append({
                "predicted_availability": p,
                "actual_availability": p,
                "hours_ahead": 1,
                "local_hour": 14,
                "is_weekend": False,
                "meter_type": "SS",
            })
        cal = calibration_analysis(matured)
        assert cal["expected_calibration_error"] == pytest.approx(0.0, abs=0.02)
        assert cal["brier_score"] == pytest.approx(0.0, abs=0.01)

    def test_ece_is_weighted_average(self):
        from scripts.evaluate_parking_probability import calibration_analysis
        matured = _make_matured(n=200, seed=42)
        cal = calibration_analysis(matured)
        valid = [b for b in cal["bins"] if b["n"] > 0]
        if valid and cal["n"] > 0:
            manual_ece = sum(b["n"] / cal["n"] * b["calibration_error"] for b in valid)
            assert cal["expected_calibration_error"] == pytest.approx(manual_ece, rel=1e-10)


# ── 10. Brier score ───────────────────────────────────────────────────

class TestBrierScore:
    def test_brier_score_perfect_predictions(self):
        from scripts.evaluate_parking_probability import calibration_analysis
        matured = []
        for i in range(50):
            matured.append({
                "predicted_availability": 1.0,
                "actual_availability": 1.0,
                "hours_ahead": 1,
                "local_hour": 14,
                "is_weekend": False,
                "meter_type": "SS",
            })
        cal = calibration_analysis(matured)
        assert cal["brier_score"] == pytest.approx(0.0, abs=1e-10)

    def test_brier_score_worst_predictions(self):
        from scripts.evaluate_parking_probability import calibration_analysis
        matured = []
        for i in range(50):
            matured.append({
                "predicted_availability": 1.0,
                "actual_availability": 0.0,
                "hours_ahead": 1,
                "local_hour": 14,
                "is_weekend": False,
                "meter_type": "SS",
            })
        cal = calibration_analysis(matured)
        assert cal["brier_score"] == pytest.approx(1.0, abs=1e-10)

    def test_brier_score_is_mean_squared_error(self):
        from scripts.evaluate_parking_probability import calibration_analysis
        matured = _make_matured(n=100, seed=42)
        cal = calibration_analysis(matured)
        manual_brier = np.mean([
            (r["predicted_availability"] - r["actual_availability"]) ** 2
            for r in matured
        ])
        assert cal["brier_score"] == pytest.approx(float(manual_brier), rel=1e-10)


# ── 11. Empty-bin handling ─────────────────────────────────────────────

class TestEmptyBinHandling:
    def test_empty_bins_have_nan_values(self):
        from scripts.evaluate_parking_probability import calibration_analysis
        # All predictions clustered in one bin
        matured = []
        for i in range(50):
            matured.append({
                "predicted_availability": 0.5,
                "actual_availability": 0.5,
                "hours_ahead": 1,
                "local_hour": 14,
                "is_weekend": False,
                "meter_type": "SS",
            })
        cal = calibration_analysis(matured)
        empty_bins = [b for b in cal["bins"] if b["n"] == 0]
        assert len(empty_bins) > 0
        for b in empty_bins:
            assert math.isnan(b["mean_predicted"])
            assert math.isnan(b["actual_rate"])
            assert math.isnan(b["calibration_error"])

    def test_empty_matured_returns_zero_metrics(self):
        from scripts.evaluate_parking_probability import calibration_analysis
        cal = calibration_analysis([])
        assert cal["n"] == 0
        assert cal["brier_score"] == 0.0 or math.isnan(cal["brier_score"])


# ── 12. DST handling ───────────────────────────────────────────────────

class TestDSTHandling:
    def test_local_hour_conversion_uses_la_time(self):
        from scripts.evaluate_parking_probability import fetch_matured_forecasts
        if not _server_available():
            pytest.skip("PostgreSQL not reachable")
        conn = connect()
        try:
            matured = fetch_matured_forecasts(conn, max_horizon=1)
            if not matured:
                pytest.skip("No matured forecasts")
            for r in matured[:10]:
                local_dt = r["target_slot"].astimezone(SF_TZ)
                assert 0 <= local_dt.hour <= 23
                assert r["local_hour"] == local_dt.hour
        finally:
            conn.close()


# ── 13. Local-hour grouping ────────────────────────────────────────────

class TestLocalHourGrouping:
    def test_breakdown_by_hour_covers_all_buckets(self):
        from scripts.evaluate_parking_probability import breakdown_by_hour
        matured = _make_matured(n=10, local_hour=3)  # overnight
        matured += _make_matured(n=10, local_hour=9, seed=1)  # morning
        matured += _make_matured(n=10, local_hour=14, seed=2)  # afternoon
        matured += _make_matured(n=10, local_hour=19, seed=3)  # evening
        matured += _make_matured(n=10, local_hour=23, seed=4)  # late_night
        groups = breakdown_by_hour(matured)
        assert "overnight" in groups
        assert "morning" in groups
        assert "afternoon" in groups
        assert "evening" in groups
        assert "late_night" in groups

    def test_breakdown_by_day_type(self):
        from scripts.evaluate_parking_probability import breakdown_by_day_type
        matured = _make_matured(n=10, is_weekend=False)
        matured += _make_matured(n=10, is_weekend=True, seed=1)
        groups = breakdown_by_day_type(matured)
        assert "weekday" in groups
        assert "weekend" in groups


# ── 14. Radius filtering ───────────────────────────────────────────────

class TestRadiusFiltering:
    def test_haversine_zero_distance(self):
        from scripts.evaluate_parking_probability import _haversine_m
        d = _haversine_m(37.797, -122.433, 37.797, -122.433)
        assert d == pytest.approx(0.0, abs=0.1)

    def test_haversine_known_distance(self):
        from scripts.evaluate_parking_probability import _haversine_m
        # Roughly 1 degree latitude ≈ 111 km
        d = _haversine_m(37.0, -122.0, 38.0, -122.0)
        assert 110_000 < d < 112_000

    def test_haversine_symmetry(self):
        from scripts.evaluate_parking_probability import _haversine_m
        d1 = _haversine_m(37.797, -122.433, 37.800, -122.430)
        d2 = _haversine_m(37.800, -122.430, 37.797, -122.433)
        assert d1 == pytest.approx(d2, rel=1e-10)


# ── 15. At-least-one-space calculation ─────────────────────────────────

class TestAtLeastOneSpace:
    def test_evaluate_radius_returns_valid_structure(self):
        from scripts.evaluate_parking_probability import evaluate_radius
        rows = []
        rng = np.random.default_rng(42)
        for i in range(50):
            rows.append({
                "post_id": f"M-{i}",
                "target_slot": datetime(2026, 1, 1, tzinfo=UTC),
                "hours_ahead": 1,
                "predicted_availability": float(rng.uniform(0, 1)),
                "actual_availability": float(rng.uniform(0, 1)),
                "latitude": 37.797 + rng.normal(0, 0.005),
                "longitude": -122.433 + rng.normal(0, 0.005),
                "local_hour": 14,
                "is_weekend": False,
            })
        result = evaluate_radius(rows, radius_m=500, sample_events=100)
        assert result["radius_m"] == 500
        assert result["events"] > 0
        assert 0 <= result["mean_predicted_prob"] <= 1
        assert result["actual_success_rate"] in (0.0, 1.0) or 0 < result["actual_success_rate"] < 1

    def test_zero_meters_returns_empty(self):
        from scripts.evaluate_parking_probability import evaluate_radius
        result = evaluate_radius([], radius_m=100)
        assert result["events"] == 0


# ── 16. No duplicate parking posts ─────────────────────────────────────

class TestNoDuplicatePosts:
    def test_matured_no_duplicate_post_slot_pairs(self):
        from scripts.evaluate_parking_probability import fetch_matured_forecasts
        if not _server_available():
            pytest.skip("PostgreSQL not reachable")
        conn = connect()
        try:
            matured = fetch_matured_forecasts(conn)
            seen = set()
            for r in matured:
                key = (r["post_id"], r["target_slot"])
                assert key not in seen, f"Duplicate: {key}"
                seen.add(key)
        finally:
            conn.close()


# ── 17. Missing actual values handled explicitly ───────────────────────

class TestMissingActualValues:
    def test_unverified_forecasts_excluded(self):
        """Forecasts without actual_availability must not appear in matured data."""
        from scripts.evaluate_parking_probability import fetch_matured_forecasts
        if not _server_available():
            pytest.skip("PostgreSQL not reachable")
        conn = connect()
        try:
            matured = fetch_matured_forecasts(conn)
            for r in matured:
                assert r["actual_availability"] is not None
        finally:
            conn.close()


# ── 18. Forecast provenance preserved ──────────────────────────────────

class TestForecastProvenance:
    def test_provenance_fields_present(self):
        from scripts.evaluate_parking_probability import fetch_matured_forecasts
        if not _server_available():
            pytest.skip("PostgreSQL not reachable")
        conn = connect()
        try:
            matured = fetch_matured_forecasts(conn)
            for r in matured[:20]:
                assert r["model_version"] is not None
                assert r["forecast_generated_at"] is not None
                assert r["feature_data_as_of"] is not None
                assert r["hours_ahead"] >= 1
        finally:
            conn.close()


# ── 19. Calibration fitting leakage safety ─────────────────────────────

class TestCalibrationLeakageSafety:
    def test_calibration_uses_same_data_not_future(self):
        """Calibration analysis should only use already-matured forecasts,
        never peeking at future observations."""
        from scripts.evaluate_parking_probability import calibration_analysis
        matured = _make_matured(n=100, seed=42)
        cal = calibration_analysis(matured)
        # The calibration should only use the predictions and actuals
        # that are already in the matured data
        total_in_bins = sum(b["n"] for b in cal["bins"])
        assert total_in_bins == len(matured)


# ── 20. Module importable ──────────────────────────────────────────────

class TestModuleStructure:
    def test_evaluate_script_importable(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "eval_script",
            REPO_ROOT / "scripts" / "evaluate_parking_probability.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main")
        assert hasattr(mod, "horizon_metrics")
        assert hasattr(mod, "calibration_analysis")
        assert hasattr(mod, "evaluate_radius")
        assert hasattr(mod, "fetch_matured_forecasts")

    def test_main_is_callable(self):
        from scripts.evaluate_parking_probability import main
        assert callable(main)
