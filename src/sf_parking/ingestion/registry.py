"""Source registry: definitions loaded from ``config/sources.yaml``."""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config/sources.yaml")


class RegistryError(Exception):
    """Raised for missing/invalid registry entries or unknown adapters."""


@dataclass(frozen=True, slots=True)
class SourceDefinition:
    name: str
    provider: str
    adapter: str
    dataset_id: str | None = None
    description: str = ""
    freshness_hours: float = 24.0
    options: dict[str, Any] = field(default_factory=dict)


def load_sources(path: Path | str | None = None) -> dict[str, SourceDefinition]:
    """Load source definitions from the YAML registry."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise RegistryError(f"source registry not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, dict) or not raw_sources:
        raise RegistryError(f"no sources defined in {config_path}")

    sources: dict[str, SourceDefinition] = {}
    for name, raw in raw_sources.items():
        if not isinstance(raw, dict):
            raise RegistryError(f"source {name!r} must be a mapping")
        adapter_key = raw.get("adapter")
        if not adapter_key:
            raise RegistryError(f"source {name!r} is missing an 'adapter' key")
        freshness = raw.get("freshness_hours", 24)
        if not isinstance(freshness, numbers.Real) or freshness <= 0:
            raise RegistryError(f"source {name!r} has invalid freshness_hours: {freshness!r}")
        sources[name] = SourceDefinition(
            name=name,
            provider=str(raw.get("provider", "unknown")),
            adapter=str(adapter_key),
            dataset_id=raw.get("dataset_id"),
            description=str(raw.get("description", "")),
            freshness_hours=float(freshness),
            options=dict(raw.get("options") or {}),
        )
    return sources


@cache
def _adapter_registry() -> dict[str, type]:
    # Imported lazily so provider-specific code stays out of the framework.
    from ..adapters import ADAPTERS

    return dict(ADAPTERS)


def resolve_adapter(definition: SourceDefinition):
    adapters = _adapter_registry()
    adapter_cls = adapters.get(definition.adapter)
    if adapter_cls is None:
        raise RegistryError(
            f"source {definition.name!r} references unknown adapter {definition.adapter!r}"
        )
    return adapter_cls()
