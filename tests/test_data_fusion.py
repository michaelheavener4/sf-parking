import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_manifest_sources():
    cfg=json.loads((ROOT/"config/data_fusion_sources.json").read_text())
    assert cfg["sources"]["parking_meters"]["dataset_id"]=="8vzz-qzz9"
    assert cfg["sources"]["meter_policies"]["dataset_id"]=="qq7v-hds4"
    assert cfg["sources"]["parking_citations"]["dataset_id"]=="ab4h-6ztd"

def test_historical_target_is_physical_occupancy():
    sql=(ROOT/"db/migrations/2026-08-25_sfpark_historical_sessions.sql").read_text()
    assert "occupancy_total" in sql
    assert "occupancy_gmp" in sql
    assert "LAG(b.occupancy_total)" in sql

def test_causal_windows_exclude_current_row():
    sql=(ROOT/"db/migrations/2026-08-25_sfpark_historical_sessions.sql").read_text()
    assert "ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING" in sql
    assert "ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING" in sql

def test_bootstrap_streams_large_csvs():
    script=(ROOT/"scripts/bootstrap_data_fusion.py").read_text()
    assert "subprocess.Popen" in script
    assert "SFpark Parking Sensor Data Hourly Occupancy 2011 - 2013" in script
