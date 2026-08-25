"""Fit an empirical probability calibrator from matured forecasts."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from sf_parking.calibration import fit_isotonic, save_calibrator
from sf_parking.database import connect

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "models" / "paid_state_probability_calibrator.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=0.50, help="availability threshold defining a successful space")
    ap.add_argument("--min-hours-ahead", type=int, default=1)
    ap.add_argument("--max-hours-ahead", type=int, default=24)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    conn = connect()
    try:
        rows = conn.run("""
            SELECT predicted_availability, actual_availability, target_slot
            FROM parking_state_forecasts
            WHERE actual_availability IS NOT NULL
              AND hours_ahead BETWEEN :lo AND :hi
            ORDER BY target_slot
        """, lo=args.min_hours_ahead, hi=args.max_hours_ahead)
    finally:
        conn.close()
    if len(rows) < 200:
        print(f"Only {len(rows)} matured forecasts; need at least 200 for calibration.")
        return 2
    scores = np.asarray([float(r[0]) for r in rows])
    actual = np.asarray([float(r[1]) for r in rows])
    events = (actual >= args.threshold).astype(float)
    cal = fit_isotonic(scores, events)
    start = str(rows[0][2]); end = str(rows[-1][2])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_calibrator(cal, args.out, event=f"actual_availability >= {args.threshold:.3f}", training_window=(start,end))
    pred = cal.predict(scores)
    brier = float(np.mean((pred-events)**2))
    print("SF PARKING — PROBABILITY CALIBRATION")
    print(f"Rows: {len(rows):,}")
    print(f"Event: actual_availability >= {args.threshold:.1%}")
    print(f"Brier: {brier:.6f}")
    print(f"Wrote: {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
