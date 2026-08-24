# SF Parking Intelligence Architecture

This project now separates three scientific layers instead of treating payment
transactions as direct ground-truth occupancy.

## 1. Observation

SFMTA/DataSF transactions are paid-use observations. They are not direct vehicle
occupancy sensors. The backtesting framework therefore labels the existing
target as a **paid-session overlap proxy**.

Every historical experiment must respect an observation frontier: a target hour
is eligible only when the database contains enough data beyond the prediction
cutoff to observe the complete outcome horizon. `sf_parking.research_frontier`
provides that invariant.

## 2. State inference

`sf_parking.occupancy` converts paid transactions into a bounded latent
**paid-occupancy probability** using a noisy-OR combination of transaction
interval evidence. This is deliberately named paid occupancy rather than true
occupancy until an independent occupancy label is available.

For multi-space meters, concurrent transactions are not treated as one physical
space. Physical capacity should be incorporated later when authoritative space
counts are available.

## 3. Supervised forecasting

`sf_parking.forecast` provides:

- deterministic temporal train/validation/test splitting;
- leakage assertions;
- Brier and MAE metrics;
- a dependency-free logistic smoke model;
- an optional LightGBM production model.

LightGBM is the recommended first serious ML baseline because the data are
heterogeneous, sparse, mixed-type, and high-cardinality. Its categorical-feature
and missing-value support are documented by the project dependency notes.

## 4. Spatial intelligence

`sf_parking.spatial` builds a deterministic nearest-neighbor graph from meter
locations and exposes neighbor mean/min/max features.

The initial spatial model should remain interpretable: blockface and nearby
meter features are explicit. A neural graph model should only be promoted after
it beats the tree/feature baseline on a strictly future holdout.

## 5. Spatial-temporal neural model

`sf_parking.graph_model` contains an optional PyTorch GNN + GRU architecture.
It accepts `[batch, time, nodes, features]` snapshots and an edge list, performs
message passing, and forecasts next-slot paid occupancy.

This is deliberately downstream of the classical ML baseline. It is not a
license to train a neural model on the current six-day dataset; it is the
architecture reserved for when enough history exists.

## Scientific order of operations

1. Fix and enforce the observation frontier.
2. Freeze deterministic_v0 and blockface_hourly as benchmarks.
3. Validate the paid-occupancy inference layer on historical data.
4. Build time-lag + spatial features.
5. Train a leakage-safe LightGBM model on historical snapshots.
6. Compare against all frozen baselines on a future holdout.
7. Add richer exogenous datasets: weather, traffic, events, street cleaning,
   meter policy and curb inventory.
8. Only then evaluate the GNN/GRU model.

The objective is not to maximize one metric on one week. It is to build a
reproducible city-scale prediction system whose target, temporal availability,
and uncertainty are explicit.
