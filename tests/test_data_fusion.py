import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_source_manifest_has_canonical_current_sources():
    cfg = json.loads((ROOT / "config/data_fusion_sources.json").read_text())
    assert cfg["sources"]["parking_meters"]["dataset_id"] == "8vzz-qzz9"
    assert cfg["sources"]["meter_policies"]["dataset_id"] == "qq7v-hds4"
    assert cfg["sources"]["parking_citations"]["dataset_id"] == "ab4h-6ztd"
    assert "PARKING_SPACE_ID".lower() in cfg["notes"][3].lower()


def test_sensor_schema_has_only_documented_hourly_target_fields():
    text = (ROOT / "db/migrations/2026-08-25_data_fusion.sql").read_text()
    for field in (
        "start_time_local",
        "total_occupied_time",
        "total_vacant_time",
        "gmp_occupied_time",
        "gmp_vacant_time",
    ):
        assert field in text


def test_payment_import_requires_session_start_and_end():
    script = (ROOT / "scripts/bootstrap_data_fusion.py").read_text()
    assert "session_start_utc" in script
    assert "session_end_utc" in script
    assert "session_end_utc >= session_start_utc" in (ROOT / "db/migrations/2026-08-25_sfpark_historical_sessions.sql").read_text()


def test_feature_view_uses_prior_observation_not_future_observation():
    sql = (ROOT / "db/migrations/2026-08-25_sfpark_historical_sessions.sql").read_text()
    assert "ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING" in sql
    assert "ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING" in sql
    assert "LAG(b.occupancy_total)" in sql
