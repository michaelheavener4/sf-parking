"""Evaluate stored forecasts against observed parking state.

Compares predictions in ``parking_state_forecasts`` against actual values
in ``parking_state_hourly`` once they become available.  Reports MAE, RMSE,
coverage, and breakdowns by hour bucket, day type, and meter type.

Usage::

    python scripts/evaluate_forecasts.py
    python scripts/evaluate_forecasts.py --hours-ahead 1
    python scripts/evaluate_forecasts.py --model-version 20260825T002130Z
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timedelta

from sf_parking.database import connect
from sf_parking.forecasting import (
    evaluate_verified_forecasts,
    fetch_unverified_forecasts,
    verify_forecasts,
    _HOUR_BUCKETS,
)


def _print_section(title: str, data: dict, col_width: int = 14) -> None:
    """Pretty-print a metrics breakdown table."""
    if not data:
        return
    hdr = (
        f"{'Group':<16} {'Rows':>8} {'Model MAE':>{col_width}} "
        f"{'Model RMSE':>{col_width}} {'Coverage':>8}"
    )
    print(f"\n{'─' * len(hdr)}")
    print(f"  {title}")
    print(f"{'─' * len(hdr)}")
    print(hdr)
    print("─" * len(hdr))
    for label, m in sorted(data.items()):
        print(
            f"  {label:<14} {m['rows']:>8,} "
            f"{m['model_mae']:>{col_width}.6f} "
            f"{m['model_rmse']:>{col_width}.6f} "
            f"{m['coverage']*100:>7.1f}%"
        )
    print("─" * len(hdr))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Evaluate stored forecasts against observed parking state.",
    )
    p.add_argument(
        "--hours-ahead", type=int, default=None,
        help="Filter to a specific hours_ahead value",
    )
    p.add_argument(
        "--model-version", type=str, default=None,
        help="Filter to a specific model version",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show unverified forecasts without updating",
    )
    args = p.parse_args()
    started = time.monotonic()

    conn = connect()
    try:
        # ── step 1: verify forecasts that now have observed values ───────
        print("SF PARKING — FORECAST EVALUATION")
        print("══════════════════════════════════════════════════════════════")

        unverified = fetch_unverified_forecasts(conn)
        print(f"Unverified forecasts: {len(unverified):,}")

        if unverified and not args.dry_run:
            updated = verify_forecasts(conn)
            print(f"Newly verified:       {updated:,}")
        elif args.dry_run:
            print("Dry run — not updating actual values.")
        else:
            print("Nothing to verify.")

        # ── step 2: evaluate verified forecasts ─────────────────────────
        print("\nEvaluating verified forecasts...")
        results = evaluate_verified_forecasts(
            conn,
            hours_ahead=args.hours_ahead,
            model_version=args.model_version,
        )

        overall = results["overall"]
        if overall["rows"] == 0:
            print("No verified forecasts found to evaluate.")
            return 0

        print(f"\nTotal verified predictions: {overall['rows']:,}")
        print(f"Overall MAE:    {overall['model_mae']:.6f}")
        print(f"Overall RMSE:   {overall['model_rmse']:.6f}")
        print(f"Coverage:       {overall['coverage']*100:.1f}%")

        _print_section("BY HOUR BUCKET", results["by_hour_bucket"])
        _print_section("BY DAY TYPE", results["by_day_type"])
        _print_section("BY METER TYPE", results["by_meter_type"])

        elapsed = int(time.monotonic() - started)
        print(f"\nEvaluation complete — elapsed {elapsed}s")
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
