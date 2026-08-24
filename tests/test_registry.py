"""Unit tests for the source registry loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from sf_parking.adapters import DataSFMeterTransactionsAdapter
from sf_parking.ingestion import RegistryError, load_sources, resolve_adapter

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_loads_registry_from_config() -> None:
    sources = load_sources(REPO_ROOT / "config" / "sources.yaml")

    assert "sfmta_meter_transactions" in sources
    transactions = sources["sfmta_meter_transactions"]
    assert transactions.provider == "datasf"
    assert transactions.dataset_id == "imvp-dq3v"
    assert transactions.freshness_hours > 0
    assert transactions.options.get("window_days", 0) > 0


def test_registry_declares_all_known_sources() -> None:
    sources = load_sources(REPO_ROOT / "config" / "sources.yaml")
    assert {"parking_meters", "meter_policies", "sfmta_meter_transactions"} <= set(sources)


def test_missing_registry_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="not found"):
        load_sources(tmp_path / "does_not_exist.yaml")


def test_source_without_adapter_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "sources.yaml"
    path.write_text("sources:\n  broken:\n    provider: datasf\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="adapter"):
        load_sources(path)


def test_resolve_adapter_returns_registered_adapter() -> None:
    sources = load_sources(REPO_ROOT / "config" / "sources.yaml")
    adapter = resolve_adapter(sources["sfmta_meter_transactions"])
    assert isinstance(adapter, DataSFMeterTransactionsAdapter)
    assert adapter.name == "sfmta_meter_transactions"
    assert adapter.target_table == "meter_transactions"


def test_resolve_adapter_unknown_key_raises() -> None:
    from sf_parking.ingestion.registry import SourceDefinition

    definition = SourceDefinition(name="ghost", provider="x", adapter="nope")
    with pytest.raises(RegistryError, match="unknown adapter"):
        resolve_adapter(definition)
