"""Compatibility runner for the causal blockface dynamics benchmark.

V4 normalizes blockface_id to text at the mapping boundary and supports direct
invocation as `python3 scripts/benchmark_blockface_transaction_dynamics_v4.py`.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "benchmark_blockface_transaction_dynamics_v3",
    HERE / "benchmark_blockface_transaction_dynamics_v3.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load V3 benchmark module")
v3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v3)


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
