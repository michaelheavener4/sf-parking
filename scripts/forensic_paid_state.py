"""Forensic audit of paid-state persistence and upstream observability.

Read-only diagnostic. It never mutates production tables. Transition metrics use
an exact T-1 hour self-join, not the previous *recorded* row, so missing hours
cannot be silently treated as persistence.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from sf_parking.database import connect

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "models" / "paid_state_forensics.json"
THRESHOLDS = (.01, .05, .10, .15, .25)


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    return (
        {f"p{q:g}": float(np.quantile(values, q / 100)) for q in (50, 75, 90, 95, 99, 99.9)}
        if len(values)
        else {}
    )


def inventory(conn) -> list[dict]:
    rows = conn.run("""
      SELECT table_schema,table_name,column_name,data_type
      FROM information_schema.columns
      WHERE table_schema NOT IN ('pg_catalog','information_schema')
      ORDER BY table_schema,table_name,ordinal_position
    """)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for schema, table, col, dtype in rows:
        grouped.setdefault((schema, table), []).append({"column": col, "type": dtype})
    keywords = ("transaction", "session", "payment", "occup", "parking", "meter")
    return [
        {"schema": s, "table": t, "columns": cols}
        for (s, t), cols in grouped.items()
        if any(k in (t + " " + " ".join(c["column"] for c in cols)).lower() for k in keywords)
    ]


def parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def exact_hour_state_audit(conn, start: datetime, end: datetime) -> dict:
    """Measure transitions only where T-1h exists for the same post."""
    rows = conn.run("""
      WITH current_state AS (
        SELECT post_id, slot_start, paid_availability_probability AS y,
               transaction_count, local_date, local_hour
        FROM parking_state_hourly
        WHERE slot_start >= :start
          AND slot_start <= :end
      ),
      pairs AS (
        SELECT
          c.post_id,
          c.slot_start,
          c.y,
          c.transaction_count AS tx,
          c.local_date,
          c.local_hour,
          p.slot_start AS prev_slot_start,
          p.paid_availability_probability AS prev_y,
          p.transaction_count AS prev_tx
        FROM current_state c
        JOIN parking_state_hourly p
          ON p.post_id = c.post_id
         AND p.slot_start = c.slot_start - INTERVAL '1 hour'
      )
      SELECT *
      FROM pairs
      ORDER BY slot_start, post_id
    """, start=start, end=end)

    if not rows:
        return {
            "n_exact_hour_pairs": 0,
            "window_start": str(start),
            "window_end": str(end),
        }

    df = pd.DataFrame(
        rows,
        columns=[
            "post_id", "slot_start", "y", "tx", "local_date", "local_hour",
            "prev_slot_start", "prev_y", "prev_tx",
        ],
    )
    y = pd.to_numeric(df.y, errors="coerce").to_numpy(float)
    prev = pd.to_numeric(df.prev_y, errors="coerce").to_numpy(float)
    tx = pd.to_numeric(df.tx, errors="coerce").fillna(0).to_numpy(float)
    prev_tx = pd.to_numeric(df.prev_tx, errors="coerce").fillna(0).to_numpy(float)
    finite = np.isfinite(y) & np.isfinite(prev)
    y, prev, tx, prev_tx = y[finite], prev[finite], tx[finite], prev_tx[finite]

    delta = y - prev
    abs_delta = np.abs(delta)
    tx_changed = tx != prev_tx

    return {
        "n_exact_hour_pairs": int(len(delta)),
        "window_start": str(start),
        "window_end": str(end),
        "mean_abs_delta": float(np.mean(abs_delta)),
        "median_abs_delta": float(np.median(abs_delta)),
        "std_delta": float(np.std(delta)),
        "fraction_exactly_unchanged": float(np.mean(abs_delta == 0)),
        "fractions_abs_delta": {
            str(t): float(np.mean(abs_delta >= t)) for t in THRESHOLDS
        },
        "quantiles_abs_delta": quantiles(abs_delta),
        "transaction_nonzero_fraction": float(np.mean(tx > 0)),
        "transaction_changed_fraction": float(np.mean(tx_changed)),
        "mean_abs_delta_when_transaction_changes": (
            float(np.mean(abs_delta[tx_changed])) if tx_changed.any() else None
        ),
        "mean_abs_delta_when_transaction_unchanged": (
            float(np.mean(abs_delta[~tx_changed])) if (~tx_changed).any() else None
        ),
        "transition_counts": {
            str(t): int(np.sum(abs_delta >= t)) for t in (.05, .10, .15, .25)
        },
    }


def coverage_audit(conn, start: datetime, end: datetime) -> dict:
    """Quantify how often an hourly target has an exact previous-hour row."""
    row = conn.run("""
      WITH targets AS (
        SELECT post_id, slot_start
        FROM parking_state_hourly
        WHERE slot_start >= :start
          AND slot_start <= :end
      )
      SELECT
        count(*) AS target_rows,
        count(*) FILTER (
          WHERE EXISTS (
            SELECT 1 FROM parking_state_hourly p
            WHERE p.post_id = targets.post_id
              AND p.slot_start = targets.slot_start - INTERVAL '1 hour'
          )
        ) AS rows_with_exact_prev_hour
      FROM targets
    """, start=start, end=end)[0]
    targets, paired = int(row[0]), int(row[1])
    return {
        "target_rows": targets,
        "rows_with_exact_prev_hour": paired,
        "missing_exact_prev_hour": targets - paired,
        "exact_prev_hour_coverage": paired / targets if targets else None,
    }


def run_window(conn, start: datetime, end: datetime) -> dict:
    return {
        "coverage": coverage_audit(conn, start, end),
        "state": exact_hour_state_audit(conn, start, end),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--train-start")
    ap.add_argument("--train-end")
    ap.add_argument("--validation-start")
    ap.add_argument("--validation-end")
    ap.add_argument("--test-start")
    ap.add_argument("--test-end")
    args = ap.parse_args()

    conn = connect()
    try:
        first, latest = conn.run(
            "SELECT min(slot_start),max(slot_start) FROM parking_state_hourly WHERE slot_start<=NOW()"
        )[0]
        if first is None:
            raise RuntimeError("parking_state_hourly is empty")

        latest = min(latest, datetime.now(latest.tzinfo) if getattr(latest, "tzinfo", None) else latest)
        default_start = max(first, latest - timedelta(days=args.days - 1))

        windows: dict[str, dict] = {
            "audit": run_window(conn, default_start, latest),
        }

        split_args = [
            ("train", args.train_start, args.train_end),
            ("validation", args.validation_start, args.validation_end),
            ("test", args.test_start, args.test_end),
        ]
        for name, start_raw, end_raw in split_args:
            if start_raw and end_raw:
                windows[name] = run_window(conn, parse_dt(start_raw), parse_dt(end_raw))
    finally:
        conn.close()

    report = {
        "version": 2,
        "transition_definition": "same post_id at exact target_slot_start - 1 hour",
        "windows": windows,
        "candidate_upstream_tables": inventory(connect()),
    }

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("🧪 SF PARKING — PAID-STATE EXACT-HOUR FORENSICS")
    print("Transition definition: exact same-post T-1 hour")
    for name, w in windows.items():
        c, s = w["coverage"], w["state"]
        print(f"\n{name}: {s.get('window_start')} → {s.get('window_end')}")
        print(f"  target rows: {c['target_rows']:,}")
        print(f"  exact T-1h pairs: {s.get('n_exact_hour_pairs', 0):,}")
        print(f"  missing exact T-1h: {c['missing_exact_prev_hour']:,}")
        print(f"  exact-prev coverage: {c['exact_prev_hour_coverage']:.2%}" if c['exact_prev_hour_coverage'] is not None else "  exact-prev coverage: n/a")
        if s.get("n_exact_hour_pairs"):
            print(f"  Mean |Δ|: {s['mean_abs_delta']:.8f}")
            print(f"  Unchanged: {s['fraction_exactly_unchanged']:.2%}")
            for t, f in s["fractions_abs_delta"].items():
                print(f"  |Δ| >= {t}: {f:.2%}")
            print(f"  Transition counts: {s['transition_counts']}")

    print("\nCandidate upstream tables:")
    for t in report["candidate_upstream_tables"]:
        print(f"  {t['schema']}.{t['table']}: " + ", ".join(c['column'] for c in t['columns']))
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
