"""Tests for the period-evaluation script.

Covers hour-bucket classification, weekday/weekend detection, metric
computations, and group-report aggregation.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest


def _import_eval_module():
    """Dynamically import the evaluation module."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "evaluate",
        Path(__file__).resolve().parents[1] / "scripts" / "evaluate_paid_state_by_period.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── hour bucket tests ───────────────────────────────────────────────────

class TestHourBucket:
    """Verify every hour maps to the correct operational period."""

    def test_overnight(self):
        mod = _import_eval_module()
        for h in range(0, 6):
            assert mod.hour_bucket(h) == "overnight", f"hour={h}"

    def test_morning(self):
        mod = _import_eval_module()
        for h in range(6, 12):
            assert mod.hour_bucket(h) == "morning", f"hour={h}"

    def test_afternoon(self):
        mod = _import_eval_module()
        for h in range(12, 17):
            assert mod.hour_bucket(h) == "afternoon", f"hour={h}"

    def test_evening(self):
        mod = _import_eval_module()
        for h in range(17, 22):
            assert mod.hour_bucket(h) == "evening", f"hour={h}"

    def test_late_night(self):
        mod = _import_eval_module()
        for h in range(22, 24):
            assert mod.hour_bucket(h) == "late_night", f"hour={h}"

    def test_all_hours_covered(self):
        mod = _import_eval_module()
        buckets = {mod.hour_bucket(h) for h in range(24)}
        assert buckets == {"overnight", "morning", "afternoon", "evening", "late_night"}

    def test_boundary_hour_05(self):
        mod = _import_eval_module()
        assert mod.hour_bucket(5) == "overnight"

    def test_boundary_hour_06(self):
        mod = _import_eval_module()
        assert mod.hour_bucket(6) == "morning"

    def test_boundary_hour_16(self):
        mod = _import_eval_module()
        assert mod.hour_bucket(16) == "afternoon"

    def test_boundary_hour_17(self):
        mod = _import_eval_module()
        assert mod.hour_bucket(17) == "evening"

    def test_boundary_hour_21(self):
        mod = _import_eval_module()
        assert mod.hour_bucket(21) == "evening"

    def test_boundary_hour_22(self):
        mod = _import_eval_module()
        assert mod.hour_bucket(22) == "late_night"


# ── weekday / weekend tests ─────────────────────────────────────────────

class TestWeekend:
    """Verify ISO weekday classification."""

    def test_monday_is_weekday(self):
        mod = _import_eval_module()
        assert not mod.is_weekend(date(2026, 8, 24))  # Monday

    def test_friday_is_weekday(self):
        mod = _import_eval_module()
        assert not mod.is_weekend(date(2026, 8, 28))  # Friday

    def test_saturday_is_weekend(self):
        mod = _import_eval_module()
        assert mod.is_weekend(date(2026, 8, 29))  # Saturday

    def test_sunday_is_weekend(self):
        mod = _import_eval_module()
        assert mod.is_weekend(date(2026, 8, 30))  # Sunday


# ── metric computation tests ────────────────────────────────────────────

class TestGroupReport:
    """Verify group_report produces correct metric values."""

    def test_perfect_predictions(self):
        mod = _import_eval_module()
        y = np.array([0.0, 0.5, 1.0, 0.3])
        pred = y.copy()
        persistence = np.array([0.1, 0.6, 0.9, 0.2])
        r = mod.group_report(y, pred, persistence)
        assert r["rows"] == 4
        assert r["model_mae"] == pytest.approx(0.0)
        assert r["model_rmse"] == pytest.approx(0.0)
        assert r["gain_mae"] > 0  # model beats persistence

    def test_empty_group(self):
        mod = _import_eval_module()
        r = mod.group_report(np.array([]), np.array([]), np.array([]))
        assert r["rows"] == 0
        assert np.isnan(r["model_mae"])

    def test_known_mae(self):
        mod = _import_eval_module()
        y = np.array([0.0, 1.0])
        pred = np.array([0.2, 0.8])
        r = mod.group_report(y, pred, y)  # persistence = truth
        assert r["model_mae"] == pytest.approx(0.2)
        assert r["persist_mae"] == pytest.approx(0.0)
        assert r["gain_mae"] == pytest.approx(-0.2)  # model is worse

    def test_known_rmse(self):
        mod = _import_eval_module()
        y = np.array([0.0, 1.0])
        pred = np.array([0.3, 0.7])
        r = mod.group_report(y, pred, y)
        expected_rmse = np.sqrt(np.mean([(0.3) ** 2, (0.3) ** 2]))
        assert r["model_rmse"] == pytest.approx(expected_rmse)

    def test_relative_improvement_formula(self):
        mod = _import_eval_module()
        y = np.array([0.5, 0.5, 0.5])
        pred = np.array([0.5, 0.5, 0.5])
        persistence = np.array([0.0, 1.0, 0.5])
        r = mod.group_report(y, pred, persistence)
        # persistence MAE = mean(|0.0-0.5|, |1.0-0.5|, |0.5-0.5|) = (0.5+0.5+0)/3 ≈ 0.3333
        # model MAE = 0
        # gain = 0.3333, rel = 100%
        assert r["rel_improvement_pct"] == pytest.approx(100.0, abs=0.1)

    def test_single_row(self):
        mod = _import_eval_module()
        y = np.array([0.7])
        pred = np.array([0.6])
        persistence = np.array([0.8])
        r = mod.group_report(y, pred, persistence)
        assert r["rows"] == 1
        assert r["model_mae"] == pytest.approx(0.1)
        assert r["persist_mae"] == pytest.approx(0.1)
        assert r["gain_mae"] == pytest.approx(0.0)
