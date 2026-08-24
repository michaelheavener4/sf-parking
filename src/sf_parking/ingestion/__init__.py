"""Generic ingestion framework with provenance (GitHub issue #1)."""

from .framework import (
    BATCH_SIZE,
    Adapter,
    IngestionRecord,
    InvalidRecord,
    RunResult,
    run_ingestion,
)
from .health import SourceHealth, source_health
from .registry import (
    RegistryError,
    SourceDefinition,
    load_sources,
    resolve_adapter,
)

__all__ = [
    "BATCH_SIZE",
    "Adapter",
    "IngestionRecord",
    "InvalidRecord",
    "RegistryError",
    "RunResult",
    "SourceDefinition",
    "SourceHealth",
    "load_sources",
    "resolve_adapter",
    "run_ingestion",
    "source_health",
]
