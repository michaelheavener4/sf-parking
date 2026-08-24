"""Provider-specific source adapters.

Adapters are the only place where provider/API knowledge lives; the generic
framework (sf_parking.ingestion.framework) consumes them via the Adapter
protocol. Register new adapters here so the YAML registry can resolve them.
"""

from .datasf import DataSFMeterTransactionsAdapter

ADAPTERS = {
    DataSFMeterTransactionsAdapter.registry_key: DataSFMeterTransactionsAdapter,
}

__all__ = ["ADAPTERS", "DataSFMeterTransactionsAdapter"]
