"""Decision-oriented parking finder using calibrated forecast probabilities.

Unlike the original finder, this command reports both meter-level choices and
a correlated radius-level success estimate. It never silently substitutes a
later forecast slot.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

from sf_parking.calibration import load_calibrator
from sf_parking.database import connect
from sf_parking.decision import ParkingCandidate, radius_probability, rank_candidates

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATOR = ROOT / "models" / "paid_state_probability_calibrator.json"


def local_to_utc(date_s: str, hour: int) -> datetime:
    from zoneinfo import ZoneInfo
    from datetime import datetime, timezone
    dt = datetime.strptime(date_s, "%Y-%m-%d").replace(hour=hour, tzinfo=ZoneInfo("America/Los_Angeles"))
    return dt.astimezone(timezone.utc)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--hour", type=int, required=True)
    p.add_argument("--radius", type=int, default=250)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--model-version", default=None)
    p.add_argument("--calibrator", type=Path, default=DEFAULT_CALIBRATOR)
    args = p.parse_args(argv)
    if not (-90 <= args.lat <= 90 and -180 <= args.lon <= 180):
        print("ERROR: invalid coordinates", file=sys.stderr); return 2
    if not 0 <= args.hour <= 23:
        print("ERROR: --hour must be 0..23", file=sys.stderr); return 2

    target = local_to_utc(args.date, args.hour)
    conn = connect()
    try:
        version_clause = "AND f.model_version = :model_version" if args.model_version else ""
        sql = f"""
        SELECT f.post_id, m.street_number, m.street_name, m.meter_type,
               ST_Distance(m.location, ST_SetSRID(ST_MakePoint(:lon,:lat),4326)::geography),
               f.predicted_availability, f.target_slot, f.hours_ahead, f.model_version
        FROM parking_state_forecasts f
        JOIN parking_meters m ON m.post_id=f.post_id
        WHERE f.target_slot=:target AND m.location IS NOT NULL
          AND ST_DWithin(m.location, ST_SetSRID(ST_MakePoint(:lon,:lat),4326)::geography,:radius)
          {version_clause}
        ORDER BY f.predicted_availability DESC, m.location <-> ST_SetSRID(ST_MakePoint(:lon,:lat),4326)::geography
        LIMIT :top
        """
        rows = conn.run(sql, target=target, lon=args.lon, lat=args.lat, radius=args.radius, top=args.top, **({"model_version": args.model_version} if args.model_version else {}))
        if not rows:
            available = conn.run("SELECT target_slot, count(*) FROM parking_state_forecasts GROUP BY target_slot ORDER BY target_slot")
            print(f"ERROR: No forecast exists for {target.isoformat()}", file=sys.stderr)
            print("Available forecast slots:", file=sys.stderr)
            for slot, n in available[-12:]: print(f"  {slot.isoformat()} ({n:,} meters)", file=sys.stderr)
            return 1
        candidates = []
        for r in rows:
            candidates.append(ParkingCandidate(
                post_id=r[0], availability=float(r[5]), distance_m=float(r[4]),
                street=f"{r[1] or ''} {r[2] or ''}".strip(), meter_type=r[3]
            ))
        if args.calibrator.exists():
            cal = load_calibrator(args.calibrator)
            ps = cal.predict([c.availability for c in candidates])
            candidates = [ParkingCandidate(c.post_id,c.availability,c.distance_m,c.street,c.meter_type,float(ps[i])) for i,c in enumerate(candidates)]
        ranked = rank_candidates(candidates)
        rp = radius_probability(candidates)
        print("SF PARKING — INTELLIGENT DECISION")
        print("="*72)
        print(f"Arrival: {args.date} {args.hour:02d}:00 PDT")
        print(f"Radius:  {args.radius}m")
        print(f"Radius success estimate: {rp*100:.1f}%")
        print()
        for i, r in enumerate(ranked, 1):
            print(f"{i:>2}. {r.probability*100:6.1f}%  {r.candidate.distance_m:6.1f}m  {r.confidence:6s}  {r.candidate.street or r.candidate.post_id}")
        return 0
    finally:
        conn.close()

if __name__ == "__main__":
    raise SystemExit(main())
