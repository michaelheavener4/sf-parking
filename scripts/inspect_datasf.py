"""Fetch a small sample of current SFMTA data for schema inspection.

Run from the repository root with:
    python scripts/inspect_datasf.py
"""

from pprint import pprint

from sf_parking.datasf import DataSFClient


if __name__ == "__main__":
    with DataSFClient() as client:
        meters = client.parking_meters(limit=3)
        policies = client.meter_policies(limit=3)

    print("Parking meter sample:")
    pprint(meters)
    print("\nMeter policy sample:")
    pprint(policies)
