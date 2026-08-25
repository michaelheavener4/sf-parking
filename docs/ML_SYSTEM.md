# SF Parking Intelligence — ML System

This document defines the production ML path beyond the original 15-feature LightGBM.

## Objective

The optimization target is not merely meter-level MAE. The product objective is:

> Given a destination, arrival time, and walking/search radius, maximize the calibrated probability that a driver finds a usable space within that radius and time budget.

The system therefore has four layers:

1. State estimation — infer paid-state availability from observed transactions.
2. Forecasting — predict future meter/block neighborhood state.
3. Calibration — make probability values empirically trustworthy.
4. Decision ranking — convert predictions into the best parking action.

## Model ladder

Every new model must beat the previous model on the same point-in-time held-out windows:

- persistence
- hour/day climatology
- existing paid-state LightGBM
- spatial LightGBM
- spatial + dynamics LightGBM
- horizon-specific models
- calibrated probability model

A model is not promoted because its aggregate MAE improves. Promotion requires improvement on difficult regimes (30–90% availability, high transition velocity, peak demand) and no unacceptable regression on easy regimes.

## Feature groups

### Temporal

- lag 1/2/3/6/24/168 availability
- lagged transaction counts
- rolling means/stddevs over 3/6/12/24 hours
- first differences and second differences
- same-hour previous day/week
- hour/day cyclical encoding

### Spatial

For each meter and historical cutoff:

- neighbor mean/median/min/max/std availability
- neighbor occupied fraction
- neighbor transaction intensity
- neighbor 1h/3h velocity
- 50m/100m/250m neighborhood counts
- distance-weighted neighborhood availability
- local grid-cell occupancy

Spatial features are computed from prior slots only. The target slot is never used.

### Context

The feature layer is designed to accept future source adapters for traffic, weather, events, closures and policy. These are optional until their data are available and are represented as nullable features rather than fabricated values.

## Probability formulation

The production decision model should eventually predict:

`P(at least one usable space within radius R at arrival time T)`

for R in {50, 100, 250, 500, 1000} meters.

Calibration is evaluated with Brier score, ECE, reliability bins, and event-conditioned calibration. The 90–100% bin must not dominate the conclusion when difficult examples are sparse.

## Leakage rules

1. Feature cutoff is strictly before target slot.
2. Inventory geometry is selected using placement validity at the cutoff when historical placement data are available.
3. Forecast overrides are only previous forecasts, never future observations.
4. Training/validation/test windows are chronological and disjoint.
5. Normalization and calibration parameters are fitted on training data only.
6. Any external source must carry retrieval/provenance timestamps.

## Promotion gates

A candidate may become the production model only if:

- point-in-time leakage tests pass;
- test MAE improves over persistence and current production on the difficult regime;
- Brier/ECE improve or remain statistically non-inferior;
- performance is stable across meter type and local-hour buckets;
- no horizon exhibits catastrophic recursive error growth;
- inference latency remains within the hourly pipeline budget.

## Product architecture

```text
DataSF transactions ─┐
SFMTA inventory ──────┤
Policies ─────────────┤
Traffic/weather/etc ──┤
                      ▼
               canonical state
                      ▼
             point-in-time features
                      ▼
        ┌─────────────┴─────────────┐
        │                           │
   state estimator             forecast models
        │                           │
        └─────────────┬─────────────┘
                      ▼
                calibration
                      ▼
              decision engine
                      ▼
       "Where should I park right now?"
```
