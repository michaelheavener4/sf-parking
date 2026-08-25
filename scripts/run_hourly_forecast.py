"""Closed-loop hourly forecasting pipeline with recursive T+N horizon.

Executes the full cycle:

    ingest transactions → build hourly state → verify freshness
    → generate T+1 … T+N forecasts (recursive) → verify → evaluate matured

Usage::

    python scripts/run_hourly_forecast.py
    python scripts/run_hourly_forecast.py --dry-run
    python scripts/run_hourly_forecast.py --max-horizon 24
    python scripts/run_hourly_forecast.py --max-data-age-minutes 120
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
DEFAULT_MAX_AGE_MINUTES = 90


# ── pipeline run logging ─────────────────────────────────────────────────

def _create_run(conn) -> int:
    """Insert a new pipeline run row and return its id."""
    result = conn.run(
        "INSERT INTO forecast_pipeline_runs DEFAULT VALUES "
        "RETURNING id"
    )
    conn.run("COMMIT")
    return int(result[0][0])


def _update_run(conn, run_id: int, **fields) -> None:
    """Update a pipeline run row with the given fields."""
    if not fields:
        return
    set_parts = []
    params: dict[str, object] = {"rid": run_id}
    for i, (k, v) in enumerate(fields.items()):
        param = f"v{i}"
        set_parts.append(f"{k} = :{param}")
        params[param] = v
    sql = f"UPDATE forecast_pipeline_runs SET {', '.join(set_parts)} WHERE id = :rid"
    conn.run(sql, **params)
    conn.run("COMMIT")


# ── step A: ingestion ────────────────────────────────────────────────────

def _run_ingestion(dry_run: bool) -> tuple[bool, str]:
    """Run transaction ingestion. Returns (success, message)."""
    if dry_run:
        return True, "dry-run: ingestion skipped"

    cmd = [
        sys.executable,
        str(SCRIPTS / "run_ingestion.py"),
        "--source", "sfmta_meter_transactions",
        "--config", str(REPO_ROOT / "config" / "sources.yaml"),
        "--schema", str(REPO_ROOT / "db" / "schema.sql"),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=600,
            env={
                "PYTHONPATH": str(REPO_ROOT / "src"),
                "PATH": "/usr/bin:/bin",
                "HOME": "/tmp",
            },
        )
        if result.returncode != 0:
            return False, f"ingestion exited {result.returncode}: {result.stderr[:500]}"
        return True, "ingestion succeeded"
    except subprocess.TimeoutExpired:
        return False, "ingestion timed out after 600s"
    except Exception as exc:
        return False, f"ingestion error: {exc}"


def _run_build_hourly_state(dry_run: bool, target_date: str) -> tuple[bool, str, int]:
    """Run build_hourly_state.py for the given date. Returns (success, message, rows)."""
    if dry_run:
        return True, "dry-run: build_hourly_state skipped", 0

    cmd = [
        sys.executable,
        str(SCRIPTS / "build_hourly_state.py"),
        "--start", target_date,
        "--end", target_date,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=300,
            env={
                "PYTHONPATH": str(REPO_ROOT / "src"),
                "PATH": "/usr/bin:/bin",
                "HOME": "/tmp",
            },
        )
        if result.returncode != 0:
            return False, f"build_hourly_state exited {result.returncode}: {result.stderr[:500]}", 0

        # Parse row count from output.
        rows = 0
        for line in result.stdout.splitlines():
            if "state rows; total=" in line:
                try:
                    part = line.split("total=")[1].strip().rstrip(",")
                    rows = int(part)
                except (IndexError, ValueError):
                    pass
        return True, "build_hourly_state succeeded", rows
    except subprocess.TimeoutExpired:
        return False, "build_hourly_state timed out after 300s", 0
    except Exception as exc:
        return False, f"build_hourly_state error: {exc}", 0


# ── step B: freshness check ──────────────────────────────────────────────

def _check_freshness(conn, max_age_minutes: float) -> tuple[bool, datetime | None, float | None]:
    """Check data freshness.

    Uses ``slot_start <= NOW()`` to exclude future rows that
    ``build_hourly_state.py`` materializes for the remainder of the
    local day.
    """
    result = conn.run(
        "SELECT max(slot_start) FROM parking_state_hourly "
        "WHERE slot_start <= NOW()"
    )
    if not result or result[0][0] is None:
        return False, None, None

    latest_slot = result[0][0]
    now = datetime.now(UTC)
    age = (now - latest_slot).total_seconds() / 60.0
    return age <= max_age_minutes, latest_slot, age


# ── step C: generate forecast ────────────────────────────────────────────

def _generate_forecast(
    conn,
    *,
    dry_run: bool,
    max_horizon: int,
) -> tuple[bool, str, int, datetime | None, str | None]:
    """Generate and store forecasts from T+1 through T+max_horizon.

    Each step is sequential: T+1 is stored before T+2 is generated,
    so T+2 can use T+1 forecasts as lag overrides.  Returns
    (success, message, total_rows, last_target_slot, model_version).
    """
    if dry_run:
        return True, "dry-run: forecast skipped", 0, None, None

    try:
        from sf_parking.forecasting import (
            FEATURES,
            _build_features,
            latest_observed_slot,
            load_model,
            store_forecasts,
        )

        import numpy as np
        import pandas as pd

        model, meta = load_model(None)
        model_version = meta.get("model_version", "unknown")
        model_path = str(
            (REPO_ROOT / "models" / "paid_state_lgbm.txt").resolve()
        )

        db_latest = latest_observed_slot(conn)
        feature_data_as_of = db_latest
        total_stored = 0
        last_target = None

        # Import override collection from forecast_paid_state.
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from forecast_paid_state import _collect_forecast_overrides

        # Timing accumulators
        total_feature_time = 0.0
        total_predict_time = 0.0
        total_write_time = 0.0
        total_db_time = 0.0
        total_db_queries = 0

        print()
        print("  {:<5} {:>7}  {:>8}  {:>8}  {:>8}  {:>8}".format(
            "T+?", "rows", "feat", "predict", "write", "total"
        ))
        print("  " + "-" * 55)

        for horizon in range(1, max_horizon + 1):
            target_slot = db_latest + timedelta(hours=horizon)
            t_horizon_start = time.monotonic()

            # Phase 1: collect overrides
            t0 = time.monotonic()
            overrides = _collect_forecast_overrides(
                conn, target_slot, horizon, db_latest=db_latest,
            )
            t_overrides = time.monotonic() - t0

            # Phase 2: build features
            t0 = time.monotonic()
            features = _build_features(conn, target_slot, overrides)
            t_features = time.monotonic() - t0

            if not features:
                print(f"  T+{horizon:>2}: no meters with sufficient history — stopping.")
                break

            # Phase 3: predict
            t0 = time.monotonic()
            df = pd.DataFrame(features)
            feature_cols = [c for c in FEATURES if c in df.columns]
            preds = np.clip(model.predict(df[feature_cols]), 0.0, 1.0)
            df["predicted_availability"] = preds
            t_predict = time.monotonic() - t0

            # Phase 4: write forecasts
            t0 = time.monotonic()
            rows = [
                {"post_id": r["post_id"], "predicted_availability": float(r["predicted_availability"])}
                for _, r in df.iterrows()
            ]

            stored = store_forecasts(
                conn,
                target_slot=target_slot,
                hours_ahead=horizon,
                model_version=model_version,
                model_path=model_path,
                feature_data_as_of=feature_data_as_of,
                rows=rows,
            )
            t_write = time.monotonic() - t0

            total_stored += stored
            last_target = target_slot
            t_horizon = time.monotonic() - t_horizon_start

            total_feature_time += t_features
            total_predict_time += t_predict
            total_write_time += t_write

            print("  T+{h:>2}: {r:>7,}  {f:>7.2f}s  {p:>7.2f}s  {w:>7.2f}s  {t:>7.2f}s".format(
                h=horizon, r=stored, f=t_features, p=t_predict, w=t_write, t=t_horizon,
            ))

        if total_stored == 0:
            return False, "no forecasts generated across any horizon", 0, last_target, model_version

        print()
        print("  Summary")
        print("  " + "-" * 40)
        print(f"  Feature build time : {total_feature_time:.2f}s")
        print(f"  LightGBM predict   : {total_predict_time:.2f}s")
        print(f"  Forecast write     : {total_write_time:.2f}s")
        print(f"  Total rows stored  : {total_stored:,}")

        return True, f"generated T+1…T+{max_horizon}", total_stored, last_target, model_version

    except Exception as exc:
        return False, f"forecast error: {exc}", 0, None, None


# ── step D: verify persistence ───────────────────────────────────────────

def _verify_persistence(
    conn,
    *,
    target_slot: datetime,
    model_version: str,
    expected_rows: int,
    max_horizon: int = 1,
) -> tuple[bool, str]:
    """Verify that forecasts were stored correctly.

    For multi-horizon runs, verifies the total count across all generated
    target slots, not just the last one.
    """
    if max_horizon > 1:
        # Multi-horizon: verify total count across all slots.
        result = conn.run(
            "SELECT count(*) "
            "FROM parking_state_forecasts "
            "WHERE model_version = :mv "
              "AND target_slot >= :t_start "
              "AND target_slot <= :t_end",
            mv=model_version,
            t_start=target_slot - timedelta(hours=max_horizon - 1),
            t_end=target_slot,
        )
        if not result or result[0][0] is None:
            return False, "verification query returned no results"
        count = result[0][0]
        if count != expected_rows:
            return False, f"expected {expected_rows} total rows across T+1…T+{max_horizon}, found {count}"
        return True, f"verified {count:,} forecasts across T+1…T+{max_horizon}"
    else:
        # Single horizon: verify the specific slot.
        result = conn.run(
            "SELECT count(*), "
            "       min(hours_ahead), max(hours_ahead), "
            "       min(forecast_generated_at), max(model_version) "
            "FROM parking_state_forecasts "
            "WHERE target_slot = :t AND model_version = :mv",
            t=target_slot, mv=model_version,
        )
        if not result or result[0][0] is None:
            return False, "verification query returned no results"

        count, min_ha, max_ha, _, mv = result[0]
        if count != expected_rows:
            return False, f"expected {expected_rows} rows, found {count}"
        if min_ha != max_ha:
            return False, f"inconsistent hours_ahead: {min_ha}..{max_ha}"

        return True, f"verified {count:,} forecasts"


# ── step E: evaluate matured forecasts ───────────────────────────────────

def _evaluate_matured(conn, dry_run: bool) -> tuple[int, str]:
    """Verify forecasts against newly-observed state. Returns (count, message)."""
    if dry_run:
        return 0, "dry-run: evaluation skipped"

    try:
        from sf_parking.forecasting import verify_forecasts
        updated = verify_forecasts(conn)
        return updated, f"verified {updated:,} matured forecasts"
    except Exception as exc:
        return 0, f"evaluation error: {exc}"


# ── main pipeline ────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Closed-loop hourly forecasting pipeline with recursive T+N horizon.",
    )
    p.add_argument(
        "--max-horizon", type=int, default=24, choices=range(1, 25),
        metavar="[1-24]",
        help="Maximum forecast horizon in hours (default: 24)",
    )
    p.add_argument(
        "--max-data-age-minutes", type=float, default=DEFAULT_MAX_AGE_MINUTES,
        help=f"Maximum acceptable data age in minutes (default: {DEFAULT_MAX_AGE_MINUTES})",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Inspect state without performing mutations",
    )
    args = p.parse_args()
    started = time.monotonic()

    from sf_parking.database import connect, transaction

    conn = connect()
    try:
        # ── create pipeline run record ──────────────────────────────────
        if not args.dry_run:
            run_id = _create_run(conn)
        else:
            run_id = -1

        print("SF PARKING — HOURLY FORECAST PIPELINE")
        print("══════════════════════════════════════════════════════════════")
        print(f"  Dry run     : {args.dry_run}")
        print(f"  Max horizon : T+{args.max_horizon}")
        print(f"  Max age     : {args.max_data_age_minutes} minutes")
        if run_id > 0:
            print(f"  Pipeline run: #{run_id}")
        print()

        # ── Step A: Ingest ──────────────────────────────────────────────
        print("[Step A] Ingesting fresh transaction data...")
        ing_ok, ing_msg = _run_ingestion(args.dry_run)
        print(f"  Result: {ing_msg}")

        if not ing_ok:
            print(f"\n❌ INGESTION FAILED: {ing_msg}")
            if not args.dry_run:
                _update_run(conn, run_id, status="ingestion_failed", error_message=ing_msg)
            return 1

        # Build hourly state for today and yesterday.
        today = datetime.now(UTC).astimezone(
            __import__("zoneinfo", fromlist=["ZoneInfo"]).ZoneInfo("America/Los_Angeles")
        ).date()
        yesterday = today - timedelta(days=1)

        for day_label, day_date in [("yesterday", yesterday), ("today", today)]:
            day_str = day_date.isoformat()
            print(f"\n  Building hourly state for {day_label} ({day_str})...")
            bh_ok, bh_msg, bh_rows = _run_build_hourly_state(args.dry_run, day_str)
            print(f"  Result: {bh_msg}")
            if not bh_ok:
                print(f"\n❌ HOURLY STATE BUILD FAILED: {bh_msg}")
                if not args.dry_run:
                    _update_run(conn, run_id, status="ingestion_failed", error_message=bh_msg)
                return 1

        # ── Step B: Verify freshness ────────────────────────────────────
        print("\n[Step B] Checking data freshness...")
        is_fresh, latest_slot, age_min = _check_freshness(conn, args.max_data_age_minutes)

        if latest_slot is None:
            msg = "parking_state_hourly is empty"
            print(f"  ⚠️  {msg}")
            if not args.dry_run:
                _update_run(conn, run_id, status="data_stale", error_message=msg)
            return 1

        print(f"  Latest observed : {latest_slot.isoformat()}")
        print(f"  Data age        : {age_min:.1f} minutes")
        print(f"  Fresh           : {'yes' if is_fresh else 'NO'}")

        if not is_fresh:
            msg = (
                f"Data is {age_min:.1f} minutes old (max allowed: {args.max_data_age_minutes}). "
                f"Latest slot: {latest_slot.isoformat()}"
            )
            print(f"\n⚠️  DATA STALE — forecast skipped.")
            print(f"    {msg}")
            if not args.dry_run:
                _update_run(
                    conn, run_id,
                    status="data_stale",
                    latest_observed_slot=latest_slot,
                    data_age_minutes=round(age_min, 1),
                    error_message=msg,
                )
            return 0

        # ── Step C: Generate T+1…T+N forecasts ─────────────────────────
        print(f"\n[Step C] Generating T+1…T+{args.max_horizon} forecasts...")
        fc_ok, fc_msg, fc_rows, target_slot, model_version = _generate_forecast(
            conn, dry_run=args.dry_run, max_horizon=args.max_horizon,
        )
        print(f"  Result: {fc_msg}")

        if not fc_ok:
            print(f"\n❌ FORECAST FAILED: {fc_msg}")
            if not args.dry_run:
                _update_run(
                    conn, run_id,
                    status="forecast_failed",
                    latest_observed_slot=latest_slot,
                    data_age_minutes=round(age_min, 1),
                    error_message=fc_msg,
                )
            return 1

        if target_slot is not None:
            print(f"  Target slot    : {target_slot.isoformat()}")
            print(f"  Rows forecast  : {fc_rows:,}")
            print(f"  Model version  : {model_version}")

        # ── Step D: Verify persistence ──────────────────────────────────
        if not args.dry_run and target_slot is not None and fc_rows > 0:
            print("\n[Step D] Verifying forecast persistence...")
            v_ok, v_msg = _verify_persistence(
                conn,
                target_slot=target_slot,
                model_version=model_version,
                expected_rows=fc_rows,
                max_horizon=args.max_horizon,
            )
            print(f"  Result: {v_msg}")
            if not v_ok:
                print(f"\n❌ VERIFICATION FAILED: {v_msg}")
                _update_run(
                    conn, run_id,
                    status="verification_failed",
                    latest_observed_slot=latest_slot,
                    forecast_target_slot=target_slot,
                    hours_ahead=args.max_horizon,
                    rows_forecast=fc_rows,
                    data_age_minutes=round(age_min, 1),
                    model_version=model_version,
                    error_message=v_msg,
                )
                return 1
        else:
            print("\n[Step D] Skipping persistence verification (dry-run or no forecasts).")

        # ── Step E: Evaluate matured forecasts ──────────────────────────
        print("\n[Step E] Evaluating matured forecasts...")
        ev_count, ev_msg = _evaluate_matured(conn, args.dry_run)
        print(f"  Result: {ev_msg}")

        # ── record success ──────────────────────────────────────────────
        if not args.dry_run:
            _update_run(
                conn, run_id,
                status="success",
                completed_at=datetime.now(UTC),
                latest_observed_slot=latest_slot,
                forecast_target_slot=target_slot,
                hours_ahead=args.max_horizon,
                rows_forecast=fc_rows,
                forecasts_evaluated=ev_count,
                data_age_minutes=round(age_min, 1),
                model_version=model_version,
            )

        elapsed = int(time.monotonic() - started)
        print(f"\n✅ PIPELINE COMPLETE — elapsed {elapsed}s")
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
