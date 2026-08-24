"""Run the parking backtest with an automatically clamped observation frontier.

Usage:
    python scripts/safe_backtest.py --model hour_conditioned_v1 --eval-days 7
    python scripts/safe_backtest.py --model deterministic_v0 --until 2026-08-24T07:00:00Z

The requested cutoff is clamped to the latest time for which the complete
outcome horizon is observable in the local database. This prevents accidental
right-edge evaluation when the CLI is run without an explicit --until.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from sf_parking.backtest import MODELS, run_backtest
from sf_parking.database import connect
from sf_parking.research_frontier import observation_frontier


def _parse_until(value: str) -> datetime:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _print_report(report) -> None:
    print(
        f"Backtest — method={report.method} eval_days={report.eval_days} "
        f"history_window={report.history_window_days}d until={report.until.isoformat()}"
    )
    print(
        f"observations generated={report.observations_generated} "
        f"predicted={report.predictions_made} "
        f"skipped(no history)={report.skipped_no_history}"
    )
    o = report.overall
    print(
        f"overall: n={o.n} MAE={o.mae} RMSE={o.rmse} Brier={o.brier} "
        f"mean_score={o.mean_score} proxy_avail_rate={o.proxy_availability_rate}"
    )
    for hour, metric in report.by_hour.items():
        if metric.n >= report.min_samples:
            print(f"by_hour: {hour}: n={metric.n} MAE={metric.mae}")
    for name, metric in report.by_weekday.items():
        if metric.n >= report.min_samples:
            print(f"by_weekday: {name}: n={metric.n} MAE={metric.mae}")
    for name, metric in report.by_meter_type.items():
        if metric.n >= report.min_samples:
            print(f"by_meter_type: {name}: n={metric.n} MAE={metric.mae}")
    for name, metric in report.by_evidence_days_bucket.items():
        if metric.n >= report.min_samples:
            print(f"by_evidence_days: {name}: n={metric.n} MAE={metric.mae}")
    if report.calibration:
        print("calibration(vs binary proxy):")
        for row in report.calibration:
            print(
                f"  {row.get('bucket')}: n={row.get('n')} "
                f"pred={row.get('mean_pred')} obs={row.get('observed_rate')}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until", type=_parse_until, default=None)
    parser.add_argument("--eval-days", type=int, default=7)
    parser.add_argument("--window-days", type=int, default=28)
    parser.add_argument("--model", choices=sorted(MODELS), default="deterministic_v0")
    parser.add_argument("--hour", action="append", type=int, dest="hours")
    parser.add_argument("--post-id", action="append", dest="post_ids")
    parser.add_argument("--max-meters", type=int, default=None)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--include-observations", action="store_true")
    args = parser.parse_args(argv)

    conn = connect()
    try:
        frontier = observation_frontier(conn, horizon_minutes=60)
        if frontier is None:
            print("No observable transaction frontier exists; refusing to run.")
            return 2

        requested = args.until or datetime.now(UTC)
        effective = frontier.clamp(requested)
        if effective is None:
            print("No safe evaluation cutoff exists; refusing to run.")
            return 2

        print(f"requested_until={requested.isoformat()}")
        print(f"max_session_end={frontier.max_session_end.isoformat()}")
        print(f"horizon={frontier.horizon}")
        print(f"effective_until={effective.isoformat()}")
        if effective != requested:
            print("WARNING: requested cutoff exceeded observable data; clamped safely.")

        report = run_backtest(
            conn,
            until=effective,
            eval_days=args.eval_days,
            history_window_days=args.window_days,
            hours=tuple(args.hours) if args.hours else None,
            post_ids=args.post_ids,
            max_meters=args.max_meters,
            include_observations=args.include_observations,
            min_samples=args.min_samples,
            model=MODELS[args.model],
        )
        _print_report(report)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
