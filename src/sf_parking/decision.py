"""Parking decision layer: turn meter forecasts into actionable choices."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ParkingCandidate:
    post_id: str
    availability: float
    distance_m: float
    street: str | None = None
    meter_type: str | None = None
    calibrated_probability: float | None = None


@dataclass(frozen=True)
class RankedParking:
    candidate: ParkingCandidate
    score: float
    probability: float
    confidence: str


def _confidence(p: float) -> str:
    if p >= 0.85:
        return "high"
    if p >= 0.60:
        return "medium"
    return "low"


def rank_candidates(candidates: list[ParkingCandidate], *, walking_penalty_per_m: float = 0.0015) -> list[RankedParking]:
    """Rank by calibrated success probability first, walking distance second."""
    ranked: list[RankedParking] = []
    for c in candidates:
        p = c.calibrated_probability if c.calibrated_probability is not None else max(0.0, min(1.0, c.availability))
        score = p * 100.0 - c.distance_m * walking_penalty_per_m
        ranked.append(RankedParking(c, score, p, _confidence(p)))
    ranked.sort(key=lambda x: (-x.score, x.candidate.distance_m, x.candidate.post_id))
    return ranked


def radius_probability(candidates: list[ParkingCandidate], *, correlation: float = 0.35) -> float:
    """Estimate P(at least one usable candidate) without assuming independence.

    A naive product would massively overstate probability because neighboring
    parking spaces are correlated.  ``correlation`` shrinks the effective
    number of independent opportunities.  This is a decision-layer estimate,
    not a calibrated claim; calibration should be learned from historical
    arrival events by the evaluation pipeline.
    """
    if not candidates:
        return 0.0
    ps = [max(0.0, min(1.0, c.calibrated_probability if c.calibrated_probability is not None else c.availability)) for c in candidates]
    ps.sort(reverse=True)
    # Keep the best local opportunities and discount correlated neighbors.
    log_no = 0.0
    weight = 1.0
    for p in ps[:25]:
        log_no += weight * math.log(max(1e-9, 1.0-p))
        weight *= max(0.05, min(0.95, 1.0-correlation))
    return max(0.0, min(1.0, 1.0-math.exp(log_no)))
