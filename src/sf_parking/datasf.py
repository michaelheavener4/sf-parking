"""Small client for the public DataSF Socrata API."""

from __future__ import annotations

from typing import Any

import httpx

BASE_URL = "https://data.sfgov.org/resource"

PARKING_METERS_DATASET = "8vzz-qzz9"
METER_POLICIES_DATASET = "qq7v-hds4"
PARKING_REGULATIONS_DATASET = "hi6h-neyh"


class DataSFClient:
    """Read-only client for DataSF datasets used by the parking engine."""

    def __init__(self, *, timeout: float = 30.0) -> None:
        self._client = httpx.Client(base_url=BASE_URL, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "DataSFClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def query(
        self,
        dataset_id: str,
        *,
        select: str = "*",
        where: str | None = None,
        limit: int = 1000,
        offset: int = 0,
        order: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run a parameterized SoQL query and return JSON rows."""
        if limit < 1 or limit > 50_000:
            raise ValueError("limit must be between 1 and 50,000")
        params: dict[str, Any] = {
            "$select": select,
            "$limit": limit,
            "$offset": offset,
        }
        if where:
            params["$where"] = where
        if order:
            params["$order"] = order

        response = self._client.get(f"/{dataset_id}.json", params=params)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise TypeError("DataSF returned a non-list JSON response")
        return payload

    def parking_meters(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.query(PARKING_METERS_DATASET, **kwargs)

    def meter_policies(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.query(METER_POLICIES_DATASET, **kwargs)

    def parking_regulations(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.query(PARKING_REGULATIONS_DATASET, **kwargs)
