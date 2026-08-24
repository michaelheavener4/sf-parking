"""Run a narrated, leakage-safe parking backtest with clipboard output.

The command explains each major operation in plain English, keeps a visible
heartbeat while the model works, and copies the complete final transcript to
macOS' clipboard so it can be pasted directly into ChatGPT.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import threading
import time
from contextlib import redirect_stdout
from datetime import UTC, datetime

from sf_parking.backtest import MODELS, run_backtest
from sf_parking.database import connect
from sf_parking.research_frontier import observation_frontier


HEARTBEAT_FRAMES = ("🐌", "🐢", "🦥", "🐌", "🚗", "🚕", "🚌", "🚙")


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


def _heartbeat(stop: threading.Event, started: float, message: str) -> None:
    i = 0
    while not stop.wait(5.0):
        elapsed = int(time.monotonic() - started)
        mins, secs = divmod(elapsed, 60)
        frame = HEARTBEAT_FRAMES[i % len(HEARTBEAT_FRAMES)]
        print(
            f"\n{frame} Still working after {mins:02d}:{secs:02d}. {message}",
            flush=True,
        )
        i += 1


def _copy_to_clipboard(text: str) -> bool:
    try:
        subprocess.run(["pbcopy"], input=text, text=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


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

    transcript = io.StringIO()
    started = time.monotonic()

    with redirect_stdout(transcript):
        print("🌉 SF PARKING — NARRATED SAFE BACKTEST")
        print("═" * 72)
        print(f"[1/5] Connecting to PostgreSQL so we can inspect the local parking history.")
        conn = connect()
        print("      ✅ Database connection established.")

        try:
            print("[2/5] Finding the observation frontier.")
            print("      I am checking the latest observed session end time because we must not score a forecast whose full outcome is still unseen.")
            frontier = observation_frontier(conn, horizon_minutes=60)
            if frontier is None:
                print("      ❌ There is no usable transaction frontier. Refusing to run.")
                return 2

            requested = args.until or datetime.now(UTC)
            effective = frontier.clamp(requested)
            if effective is None:
                print("      ❌ No safe evaluation cutoff exists. Refusing to run.")
                return 2

            print(f"      Latest observed session end: {frontier.max_session_end.isoformat()}")
            print(f"      Forecast horizon: {frontier.horizon}")
            print(f"      Requested cutoff: {requested.isoformat()}")
            print(f"      Safe cutoff: {effective.isoformat()}")
            if effective != requested:
                print("      ⚠️ Requested cutoff was beyond the observable data, so it was clamped backward.")
            else:
                print("      ✅ Requested cutoff is already inside the observable data frontier.")

            print("[3/5] Preparing the experiment configuration.")
            print(f"      Model: {args.model}")
            print(f"      Evaluation window: {args.eval_days} days")
            print(f"      Historical lookback: {args.window_days} days")
            if args.hours:
                print(f"      Restricting to local hours: {sorted(args.hours)}")
            else:
                print("      Evaluating all local clock hours.")
            if args.max_meters:
                print(f"      Restricting to first {args.max_meters:,} meters for this run.")
            else:
                print("      Evaluating every meter with usable transaction history.")

            print("[4/5] Running the backtest.")
            print("      The model is now reconstructing what was knowable at each prediction time, generating predictions, and comparing them with the observed paid-session proxy.")
            print("      ⏳ A heartbeat will appear every 5 seconds so a long run is visibly alive.")
            stop = threading.Event()
            heartbeat = threading.Thread(
                target=_heartbeat,
                args=(
                    stop,
                    started,
                    "PostgreSQL and the model are still processing the historical evaluation window; no new terminal input is required.",
                ),
                daemon=True,
            )
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

            print("      ✅ Backtest computation finished.")
            print("[5/5] Summarizing the experiment.")
            _print_report(report)
            elapsed = int(time.monotonic() - started)
            mins, secs = divmod(elapsed, 60)
            print("═" * 72)
            print(f"✅ COMPLETE — total elapsed time {mins:02d}:{secs:02d}")
            print("📋 The complete transcript will be copied to the macOS clipboard now.")
        finally:
            conn.close()

    final_text = transcript.getvalue()
    copied = _copy_to_clipboard(final_text)
    print(final_text, end="")
    print("📋 Clipboard: copied successfully." if copied else "⚠️ Clipboard copy failed; pbcopy is unavailable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
