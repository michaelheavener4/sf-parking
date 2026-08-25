"""Compatibility runner for the causal blockface dynamics benchmark.

V4 normalizes blockface_id to text at the mapping boundary so joins between
parking_meters, parking_spaces, transaction mappings, and temporary target
tables have one unambiguous SQL type.
"""
from __future__ import annotations

from scripts import benchmark_blockface_transaction_dynamics_v3 as v3


def normalized_mapping_sql() -> str:
    return (
        "SELECT DISTINCT post_id::text AS post_id, "
        "blockface_id::text AS blockface_id "
        "FROM parking_meters "
        "WHERE post_id IS NOT NULL AND blockface_id IS NOT NULL"
    )


v3.mapping_sql = normalized_mapping_sql


if __name__ == "__main__":
    raise SystemExit(v3.main())
