"""Run a narrated, leakage-safe parking backtest with clipboard output."""

from __future__ import annotations

import argparse
import io
import subprocess
import threading
import time
from datetime import UTC, datetime

from sf_parking.backtest import MODELS, run_backtest
from sf_parking.database import connect
from sf_parking.research_frontier import observation_frontier

HEARTBEAT_FRAMES = ("🐌", "🐢", "🦥", "🚗", "🚕", "🚌", "🚙")


def _parse_until(value: str) -> datetime:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _heartbeat(stop: threading.Event, started: float) -> None:
    i = 0
    while not stop.wait(5.0):
        elapsed = int(time.monotonic() - started)
        mins, secs = divmod(elapsed, 60)
        print(
            f"\n{HEARTBEAT_FRAMES[i % len(HEARTBEAT_FRAMES)]} "
            f"Still working after {mins:02d}:{secs:02d}. "
            "The backtest is processing historical meter observations; "
            "this is expected and no input is required.",
            flush=True,
        )
        i += 1


def _copy_to_clipboard(text: str) -> bool:
    try:
        subprocess.run(["pbcopy"], input=text, text=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _run_and_capture_report(report) -> str:
    out = io.StringIO()
    out.write(
        f"Backtest — method={report.method} eval_days={report.eval_days} "
        f"history_window={report.history_window_days}d until={report.until.isoformat()}\n"
    )
    out.write(
        f"observations generated={report.observations_generated} "
        f"predicted={report.predictions_made} "
        f"skipped(no history)={report.skipped_no_history}\n"
    )
    o = report.overall
    out.write(
        f"overall: n={o.n} MAE={o.mae} RMSE={o.rmse} Brier={o.brier} "
        f"mean_score={o.mean_score} proxy_avail_rate={o.proxy_availability_rate}\n"
    )
    for hour, metric in report.by_hour.items():
        if metric.n >= report.min_samples:
            out.write(f"by_hour: {hour}: n={metric.n} MAE={metric.mae}\n")
    for name, metric in report.by_weekday.items():
        if metric.n >= report.min_samples:
            out.write(f"by_weekday: {name}: n={metric.n} MAE={metric.mae}\n")
    for name, metric in report.by_meter_type.items():
        if metric.n >= report.min_samples:
            out.write(f"by_meter_type: {name}: n={metric.n} MAE={metric.mae}\n")
    for name, metric in report.by_evidence_days_bucket.items():
        if metric.n >= report.min_samples:
            out.write(f"by_evidence_days: {name}: n={metric.n} MAE={metric.mae}\n")
    if report.calibration:
        out.write("calibration(vs binary proxy):\n")
        for row in report.calibration:
            out.write(
                f"  {row.get('bucket')}: n={row.get('n')} "
                f"pred={row.get('mean_pred')} obs={row.get('observed_rate')}\n"
            )
    return out.getvalue()


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

    transcript: list[str] = []
    started = time.monotonic()

    def log(message: str = "") -> None:
        transcript.append(message)
        print(message, flush=True)

    log("🌉 SF PARKING — NARRATED SAFE BACKTEST")
    log("═" * 72)
    log("[1/5] Connecting to PostgreSQL.")
    log("      I am opening the local database because the historical transactions are the model's source of truth.")
    conn = connect()
    log("      ✅ Database connection established.")

    try:
        log("[2/5] Finding the safe observation frontier.")
        log("      I am checking the latest observed session end so we never score an hour whose complete outcome is still unseen.")
        frontier = observation_frontier(conn, horizon_minutes=60)
        if frontier is None:
            log("      ❌ No observable transaction frontier exists. Refusing to run.")
            return 2

        requested = args.until or datetime.now(UTC)
        effective = frontier.clamp(requested)
        if effective is None:
            log("      ❌ No safe evaluation cutoff exists. Refusing to run.")
            return 2

        log(f"      Latest observed session end: {frontier.max_session_end.isoformat()}")
        log(f"      Forecast horizon: {frontier.horizon}")
        log(f"      Requested cutoff: {requested.isoformat()}")
        log(f"      Safe cutoff: {effective.isoformat()}")
        if effective != requested:
            log("      ⚠️ Requested cutoff was beyond observed data, so it was clamped backward.")
        else:
            log("      ✅ Requested cutoff is inside the observed data frontier.")

        log("[3/5] Preparing the experiment.")
        log(f"      Model: {args.model}")
        log(f"      Evaluation window: {args.eval_days} days")
        log(f"      Historical lookback: {args.window_days} days")
        log(f"      Hours: {sorted(args.hours) if args.hours else 'all local clock hours'}")
        log(f"      Meters: {args.max_meters:,} maximum" if args.max_meters else "      Meters: every meter with usable history")
        log("      The target is the paid-session overlap proxy, not physical ground-truth occupancy.")

        log("[4/5] Running the backtest.")
        log("      The system is now rebuilding what was knowable at each prediction time, generating predictions, and scoring them against the observed future interval.")
        log("      A live heartbeat will appear every 5 seconds. If you see the heartbeat, the process is alive.")

        stop = threading.Event()
        heartbeat = threading.Thread(target=_heartbeat, args=(stop, started), daemon=True)
        heartbeat.start()
        try:
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
        finally:
            stop.set()
            heartbeat.join(timeout=1.0)

        log("      ✅ Backtest computation finished.")
        log("[5/5] Calculating and displaying the results.")
        report_text = _run_and_capture_report(report).rstrip("\n")
        for line in report_text.splitlines():
            log(f"      {line}")

        elapsed = int(time.monotonic() - started)
        mins, secs = divmod(elapsed, 60)
        log("═" * 72)
        log(f"✅ COMPLETE — total elapsed time {mins:02d}:{secs:02d}")
        log("📋 Copying the complete transcript to your macOS clipboard...")
        final_text = "\n".join(transcript) + "\n"
        copied = _copy_to_clipboard(final_text)
        print("📋 Clipboard: copied successfully." if copied else "⚠️ Clipboard copy failed; pbcopy is unavailable.", flush=True)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
