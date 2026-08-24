"""Regression tests for SFMTA timestamp semantics (source → PostgreSQL).

DataSF exposes ``session_start_dt``/``session_end_dt`` on dataset
``imvp-dq3v`` as Socrata *floating* timestamps: wall-clock times with no UTC
offset. Evidence for the America/Los_Angeles convention:

1. DataSF/Socrata define floating timestamps as local time in the operating
   agency's clock; SFMTA operates in San Francisco.
2. Observed session-start histogram over 320k real transactions peaks at
   09:00-16:00 with a hard stop at 18:00 — exactly SF metered hours on a
   Pacific clock; interpreted as UTC they would be ~2-11 AM Pacific.
3. The meters inventory's ``data_as_of`` (floating) vs ``data_loaded_at``
   (UTC) differ by the Pacific offset plus processing lag.

The adapter attaches ``America/Los_Angeles`` before insert so timestamptz
columns hold true instants, including across DST transitions.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sf_parking.adapters.datasf import SOURCE_TZ, normalize_transaction, parse_socrata_timestamp


def _row(start: str, end: str | None = None) -> dict:
    row = {
        "transmission_datetime": "tx_1_x",
        "post_id": "665-01003",
        "session_start_dt": start,
    }
    if end is not None:
        row["session_end_dt"] = end
    return row


def test_summer_timestamp_is_pdt_utc_minus_seven() -> None:
    parsed = parse_socrata_timestamp("2026-08-17T04:31:23.000")
    assert parsed == datetime(2026, 8, 17, 4, 31, 23, tzinfo=SOURCE_TZ)
    assert parsed.astimezone(UTC) == datetime(2026, 8, 17, 11, 31, 23, tzinfo=UTC)


def test_winter_timestamp_is_pst_utc_minus_eight() -> None:
    parsed = parse_socrata_timestamp("2022-11-18T07:00:00.000")
    assert parsed.utcoffset().total_seconds() == -8 * 3600
    assert parsed.astimezone(UTC) == datetime(2022, 11, 18, 15, 0, tzinfo=UTC)


def test_dst_fall_back_ambiguous_time_resolves_to_first_occurrence() -> None:
    # 01:30 happens twice on 2025-11-02; fold=0 selects the first (PDT).
    parsed = parse_socrata_timestamp("2025-11-02T01:30:00")
    assert parsed.fold == 0
    assert parsed.utcoffset().total_seconds() == -7 * 3600
    assert parsed.astimezone(UTC) == datetime(2025, 11, 2, 8, 30, tzinfo=UTC)


def test_dst_spring_forward_nonexistent_time_keeps_wall_clock() -> None:
    # 02:30 does not exist on 2026-03-08; PEP 495 keeps the wall clock on the
    # pre-transition offset (-8), landing after the jump once converted to UTC.
    parsed = parse_socrata_timestamp("2026-03-08T02:30:00")
    assert parsed.fold == 0
    assert parsed.utcoffset().total_seconds() == -8 * 3600
    assert parsed.astimezone(UTC) == datetime(2026, 3, 8, 10, 30, tzinfo=UTC)


def test_duration_spans_real_instant_not_wall_clock_across_fall_back() -> None:
    # Wall clock diff is 4h but the real elapsed time across the fall-back
    # transition is 5h; duration must use absolute instants.
    record = normalize_transaction(_row("2025-11-02T00:30:00", "2025-11-02T04:30:00"))
    assert record.values["duration_minutes"] == 300


def test_duration_spans_real_instant_across_spring_forward() -> None:
    # Wall clock diff is 2h but 02:00-03:00 does not exist on 2026-03-08;
    # real elapsed time is 1h.
    record = normalize_transaction(_row("2026-03-08T01:30:00", "2026-03-08T03:30:00"))
    assert record.values["duration_minutes"] == 60


def test_normalize_produces_aware_datetimes_end_to_end() -> None:
    record = normalize_transaction(_row("2026-08-17T04:31:23.000"))
    assert record.values["session_start"].tzinfo is not None
    assert record.source_timestamp.tzinfo is not None


def test_parse_preserves_explicit_offsets_if_source_adds_them() -> None:
    aware = parse_socrata_timestamp("2026-08-17T04:31:23-07:00")
    assert aware.astimezone(UTC) == datetime(2026, 8, 17, 11, 31, 23, tzinfo=UTC)
