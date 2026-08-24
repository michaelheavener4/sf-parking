"""Print deterministic per-meter parking features and availability baselines.

Usage:
    python scripts/meter_features.py                       # features only
    python scripts/meter_features.py --at "2026-08-24T12:00:00-07:00"
    python scripts/meter_features.py --post-id 363-04151 --baseline
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime

from sf_parking.database import connect
from sf_parking.features import availability_baseline, meter_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-days", type=int, default=28)
    parser.add_argument(
        "--at",
        help="target instant for the baseline (any ISO-8601 offset form)",
    )
    parser.add_argument("--post-id", help="restrict to one meter")
    parser.add_argument(
        "--limit", type=int, default=20, help="meters to print (default 20)"
    )
    args = parser.parse_args()

    at = datetime.fromisoformat(args.at) if args.at else None

    conn = connect()
    try:
        feats = meter_features(conn, window_days=args.window_days, now=at)
        if args.post_id:
            feats = [f for f in feats if f.post_id == args.post_id]
        payload = {
            "features": [asdict(f) for f in feats[: args.limit]],
            "summary": {
                "window_days": args.window_days,
                "meters_with_history": len(feats),
            },
        }
        if at is not None or args.post_id:
            targets = [args.post_id] if args.post_id else [f.post_id for f in feats[:5]]
            payload["availability_baseline"] = [
                asdict(availability_baseline(conn, p, at=at, window_days=args.window_days))
                for p in targets
            ]
    finally:
        conn.close()

    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
