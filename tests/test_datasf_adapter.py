"""Unit tests for the DataSF meter transactions adapter normalization."""

from __future__ import annotations

from datetime import datetime

import pytest

from sf_parking.adapters import DataSFMeterTransactionsAdapter
from sf_parking.adapters.datasf import _window_where, normalize_transaction
from sf_parking.ingestion.framework import IngestionRecord, InvalidRecord

RAW_ROW = {
    "transmission_datetime": "999999996_9_11182022070000",
    "post_id": "665-01003",
    "street_block": "STEUART ST 100",
    "payment_type": "PAY BY CELL",
    "session_start_dt": "2022-11-18T07:00:00.000",
    "session_end_dt": "2022-11-18T09:00:00.000",
    "meter_event_type": "NS",
    "gross_paid_amt": "2",
}


def test_normalize_real_shaped_row() -> None:
    record = normalize_transaction(RAW_ROW)

    assert isinstance(record, IngestionRecord)
    assert record.key == ("999999996_9_11182022070000", "665-01003")
    values = record.values
    assert values["post_id"] == "665-01003"
    assert values["session_start"] == datetime.fromisoformat("2022-11-18T07:00:00")
    assert values["session_end"] == datetime.fromisoformat("2022-11-18T09:00:00")
    assert values["duration_minutes"] == 120
    assert values["gross_paid_amt"] == 2.0
    assert values["payment_type"] == "PAY BY CELL"
    assert values["meter_event_type"] == "NS"
    assert record.source_timestamp == datetime.fromisoformat("2022-11-18T07:00:00")


def test_normalize_computes_fractional_duration() -> None:
    record = normalize_transaction(
        {
            **RAW_ROW,
            "session_start_dt": "2025-11-01T15:40:00.000",
            "session_end_dt": "2025-11-01T17:10:00.000",
            "gross_paid_amt": "2.6",
        }
    )
    assert record.values["duration_minutes"] == 90
    assert record.values["gross_paid_amt"] == 2.6


def test_normalize_tolerates_missing_optional_fields() -> None:
    record = normalize_transaction(
        {
            "transmission_datetime": "x_1_01012024000000",
            "post_id": "100-00001",
            "session_start_dt": "2024-01-01T10:00:00.000",
        }
    )
    assert record.values["session_end"] is None
    assert record.values["duration_minutes"] is None
    assert record.values["gross_paid_amt"] is None


def test_normalize_missing_post_id_raises() -> None:
    with pytest.raises(ValueError, match="identity"):
        normalize_transaction({**RAW_ROW, "post_id": ""})


def test_normalize_missing_session_start_raises() -> None:
    with pytest.raises(ValueError, match="no session start"):
        normalize_transaction({**RAW_ROW, "session_start_dt": None})


def test_normalize_unparseable_timestamp_raises() -> None:
    with pytest.raises(ValueError):
        normalize_transaction({**RAW_ROW, "session_start_dt": "not-a-date"})


class _FakeClient:
    def __init__(self, pages: list[list[dict]]) -> None:
        self.pages = pages
        self.calls: list[dict] = []

    def iter_rows(self, dataset_id: str, **kwargs):
        self.calls.append({"dataset_id": dataset_id, **kwargs})
        yield from self.pages.pop(0)


def test_fetch_skips_malformed_rows_and_keeps_good_ones() -> None:
    client = _FakeClient([[RAW_ROW, {"garbage": True}, {**RAW_ROW, "post_id": None}]])
    adapter = DataSFMeterTransactionsAdapter(client=client)

    results = list(adapter.fetch({}))

    valid = [r for r in results if isinstance(r, IngestionRecord)]
    invalid = [r for r in results if isinstance(r, InvalidRecord)]
    assert len(valid) == 1
    assert len(invalid) == 2
    assert invalid[0].error


def test_fetch_passes_window_and_dataset_to_client() -> None:
    client = _FakeClient([[]])
    adapter = DataSFMeterTransactionsAdapter(client=client)

    list(adapter.fetch({"window_days": 3}))

    call = client.calls[0]
    assert call["dataset_id"] == "imvp-dq3v"
    assert "session_start_dt >=" in call["where"]
    assert call["order"] == ":id"


def test_window_where_combines_filter_and_explicit_where() -> None:
    where = _window_where(
        {"window_days": 1, "where": "payment_type = 'COINS'"},
        now=datetime.fromisoformat("2026-08-23T12:00:00"),
    )
    assert "session_start_dt >= '2026-08-22T12:00:00'" in where
    assert "(payment_type = 'COINS')" in where
