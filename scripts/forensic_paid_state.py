"""Forensic audit of paid-state persistence and upstream observability.

Read-only diagnostic. It never mutates production tables. It inventories the
live schema, measures target persistence, transaction coupling, and timestamp
cadence so we can distinguish genuine stability from target-construction loss.
"""
from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from sf_parking.database import connect

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "models" / "paid_state_forensics.json"


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    return {f"p{q:g}": float(np.quantile(values, q / 100)) for q in (50, 75, 90, 95, 99, 99.9)} if len(values) else {}


def inventory(conn) -> list[dict]:
    rows = conn.run("""
      SELECT table_schema,table_name,column_name,data_type
      FROM information_schema.columns
      WHERE table_schema NOT IN ('pg_catalog','information_schema')
      ORDER BY table_schema,table_name,ordinal_position
    """)
    grouped: dict[tuple[str,str], list[dict]] = {}
    for schema, table, col, dtype in rows:
        grouped.setdefault((schema, table), []).append({"column": col, "type": dtype})
    keywords = ("transaction", "session", "payment", "occup", "parking", "meter")
    return [
        {"schema": s, "table": t, "columns": cols}
        for (s, t), cols in grouped.items()
        if any(k in (t + " " + " ".join(c["column"] for c in cols)).lower() for k in keywords)
    ]


def state_audit(conn, days: int) -> dict:
    first, latest = conn.run("SELECT min(slot_start),max(slot_start) FROM parking_state_hourly WHERE slot_start<=NOW()")[0]
    if first is None:
        raise RuntimeError("parking_state_hourly is empty")
    start = max(first, latest - timedelta(days=days - 1))
    rows = conn.run("""
      WITH x AS (
        SELECT post_id,slot_start,paid_availability_probability AS y,transaction_count,
               lag(paid_availability_probability) OVER (PARTITION BY post_id ORDER BY slot_start) AS prev_y,
               lag(transaction_count) OVER (PARTITION BY post_id ORDER BY slot_start) AS prev_tx
        FROM parking_state_hourly
        WHERE slot_start>=:start AND slot_start<=:latest
      ) SELECT * FROM x WHERE prev_y IS NOT NULL ORDER BY slot_start,post_id
    """, start=start, latest=latest)
    if not rows:
        return {"n": 0, "window_start": str(start), "window_end": str(latest)}
    df = pd.DataFrame(rows, columns=["post_id","slot_start","y","tx","prev_y","prev_tx"])
    y = pd.to_numeric(df.y, errors="coerce").to_numpy(float)
    prev = pd.to_numeric(df.prev_y, errors="coerce").to_numpy(float)
    tx = pd.to_numeric(df.tx, errors="coerce").fillna(0).to_numpy(float)
    prev_tx = pd.to_numeric(df.prev_tx, errors="coerce").fillna(0).to_numpy(float)
    finite = np.isfinite(y) & np.isfinite(prev)
    delta = y[finite] - prev[finite]
    abs_delta = np.abs(delta)
    tx = tx[finite]; prev_tx = prev_tx[finite]
    tx_changed = tx != prev_tx
    return {
        "n": int(len(delta)), "window_start": str(start), "window_end": str(latest),
        "mean_abs_delta": float(np.mean(abs_delta)), "median_abs_delta": float(np.median(abs_delta)),
        "std_delta": float(np.std(delta)), "fraction_exactly_unchanged": float(np.mean(abs_delta == 0)),
        "fractions_abs_delta": {str(t): float(np.mean(abs_delta >= t)) for t in (.01,.05,.10,.15,.25)},
        "quantiles_abs_delta": quantiles(abs_delta),
        "transaction_nonzero_fraction": float(np.mean(tx > 0)),
        "transaction_changed_fraction": float(np.mean(tx_changed)),
        "mean_abs_delta_when_transaction_changes": float(np.mean(abs_delta[tx_changed])) if tx_changed.any() else None,
        "mean_abs_delta_when_transaction_unchanged": float(np.mean(abs_delta[~tx_changed])) if (~tx_changed).any() else None,
        "transition_counts": {str(t): int(np.sum(abs_delta >= t)) for t in (.05,.10,.15,.25)},
    }


def cadence(conn) -> dict:
    r = conn.run("SELECT count(*),count(DISTINCT post_id),count(DISTINCT slot_start),min(slot_start),max(slot_start) FROM parking_state_hourly")[0]
    gaps = conn.run("""
      WITH d AS (SELECT post_id,slot_start-slot_start_prev AS gap FROM (
        SELECT post_id,slot_start,lag(slot_start) OVER(PARTITION BY post_id ORDER BY slot_start) slot_start_prev
        FROM parking_state_hourly) q WHERE slot_start_prev IS NOT NULL)
      SELECT gap,count(*) FROM d GROUP BY gap ORDER BY count(*) DESC LIMIT 20
    """)
    return {"rows": int(r[0]), "meters": int(r[1]), "slots": int(r[2]), "first": str(r[3]), "latest": str(r[4]), "common_gaps": [{"gap": str(x[0]), "count": int(x[1])} for x in gaps]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    conn = connect()
    try:
        report = {"version": 1, "state": state_audit(conn, args.days), "cadence": cadence(conn), "candidate_upstream_tables": inventory(conn)}
    finally:
        conn.close()
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    s = report["state"]
    print("🧪 SF PARKING — PAID-STATE FORENSICS")
    print(f"Rows analyzed: {s.get('n',0):,}")
    if s.get("n"):
        print(f"Mean |Δ|: {s['mean_abs_delta']:.8f}")
        print(f"Unchanged: {s['fraction_exactly_unchanged']:.2%}")
        for t, f in s["fractions_abs_delta"].items(): print(f"|Δ| >= {t}: {f:.2%}")
        print(f"|Δ| when transaction count changes: {s['mean_abs_delta_when_transaction_changes']}")
        print(f"|Δ| when transaction count is unchanged: {s['mean_abs_delta_when_transaction_unchanged']}")
        print(f"Transition counts: {s['transition_counts']}")
    print("\nCandidate upstream tables:")
    for t in report["candidate_upstream_tables"]:
        print(f"  {t['schema']}.{t['table']}: " + ", ".join(c['column'] for c in t['columns']))
    print(f"\nReport: {path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
