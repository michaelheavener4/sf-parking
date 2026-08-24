"""Spatial features and graph construction for parking intelligence."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Point:
    post_id: str
    latitude: float
    longitude: float
    blockface_id: str | None = None


@dataclass(frozen=True, slots=True)
class Edge:
    source: str
    target: str
    distance_meters: float
    weight: float


@dataclass(frozen=True, slots=True)
class SpatialGraph:
    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]

    def neighbors(self, post_id: str) -> tuple[Edge, ...]:
        return tuple(e for e in self.edges if e.source == post_id)

    def neighbor_mean(self, values: dict[str, float], post_id: str) -> float | None:
        vals = [
            values[e.target]
            for e in self.neighbors(post_id)
            if e.target in values
        ]
        return None if not vals else sum(vals) / len(vals)


def haversine_meters(a: Point, b: Point) -> float:
    r = 6_371_008.8
    lat1, lat2 = radians(a.latitude), radians(b.latitude)
    dlat = lat2 - lat1
    dlon = radians(b.longitude - a.longitude)
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(h))


def build_knn_graph(
    points: Iterable[Point],
    *,
    k: int = 8,
    radius_meters: float = 250.0,
) -> SpatialGraph:
    """Build a deterministic directed kNN/radius graph.

    Each node links to up to k nearest other nodes inside radius_meters.
    Edge weights decay with distance. No external ML dependency is needed.
    """
    pts = sorted(points, key=lambda p: p.post_id)
    if k <= 0 or radius_meters <= 0:
        raise ValueError("k and radius_meters must be positive")
    edges: list[Edge] = []
    for p in pts:
        candidates: list[tuple[float, str]] = []
        for q in pts:
            if p.post_id == q.post_id:
                continue
            d = haversine_meters(p, q)
            if d <= radius_meters:
                candidates.append((d, q.post_id))
        candidates.sort(key=lambda x: (x[0], x[1]))
        for d, target in candidates[:k]:
            weight = 1.0 / (1.0 + d)
            edges.append(Edge(p.post_id, target, round(d, 6), round(weight, 12)))
    return SpatialGraph(
        nodes=tuple(p.post_id for p in pts),
        edges=tuple(edges),
    )


def spatial_neighbor_features(
    graph: SpatialGraph,
    values: dict[str, float],
    *,
    prefix: str = "neighbor",
) -> dict[str, dict[str, float | None]]:
    """Create interpretable neighbor mean/max/min features."""
    result: dict[str, dict[str, float | None]] = {}
    for node in graph.nodes:
        vals = [
            values[e.target]
            for e in graph.neighbors(node)
            if e.target in values
        ]
        if not vals:
            result[node] = {
                f"{prefix}_mean": None,
                f"{prefix}_min": None,
                f"{prefix}_max": None,
            }
            continue
        result[node] = {
            f"{prefix}_mean": sum(vals) / len(vals),
            f"{prefix}_min": min(vals),
            f"{prefix}_max": max(vals),
        }
    return result
