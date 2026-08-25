# SF Parking Data Fusion V1

## Why this exists

The project now treats **measured occupancy** as the calibration target instead of attempting to infer physical occupancy from payment-derived availability alone.

SFMTA's SFpark pilot published hourly block-level occupancy from parking sensors for April 2011 through July 2013, with explicit definitions for total, operational, general-metered-parking (GMP), and commercial occupancy. The published guide also documents the historical sensor outages and coverage rules.

SFMTA also published smart-meter transaction data for the pilot, including parking-space/post identifiers, payment time, session start and session end. This gives the project a historical paired dataset:

`transaction behavior -> measured physical occupancy`

That is the calibration layer.

## Canonical current spatial identity

The current DataSF Parking Meters dataset contains `PARKING_SPACE_ID`, `POST_ID`, and `BLOCKFACE_ID`. SFMTA identifies `PARKING_SPACE_ID` as the key for joining a meter to the physical parking-space inventory.

The project therefore treats:

- `PARKING_SPACE_ID` as the physical-space identity;
- `POST_ID` as the meter/payment identity;
- `BLOCKFACE_ID` as the spatial aggregation identity.

No fallback from missing parking-space IDs to arbitrary capacity assumptions is used in the data-fusion layer.

## Sources wired into the registry

- Parking Meters: DataSF dataset `8vzz-qzz9`
- Meter Policies: DataSF dataset `qq7v-hds4`
- Parking Citations & Fines: DataSF dataset `ab4h-6ztd`
- SFpark historical hourly sensor occupancy: official SFMTA SFpark evaluation data
- SFpark historical smart-meter payment sessions: official SFMTA SFpark evaluation data
- Historical SFpark events, weather, roadway sensors, and transit are registered as optional exogenous sources for later enrichment.

## Causality contract

For a prediction at time `T`, a feature may use only information with an observation timestamp `<= T-1` (or an equivalent clearly defined as-of timestamp). Historical `session_end` is allowed for learning a duration distribution, and observed sensor occupancy at `T-1` is allowed as a state feature. Target-hour values must never be used as predictors.

## Bootstrap

```bash
python3 scripts/bootstrap_data_fusion.py --download-historical
```

The historical sensor and payment files are large and are streamed into PostgreSQL; they are not committed to git.

Then:

```bash
python3 scripts/train_fused_occupancy.py
```

## What this experiment answers

The first scientific question is deliberately narrow:

> Does historical transaction behavior add predictive information beyond one-hour physical-occupancy persistence?

If yes, the next production model can use the calibrated transaction-to-occupancy relationship on current data. If no, the transaction stream is not sufficiently informative at this resolution and the project should prioritize direct availability observations / richer exogenous signals instead of more elaborate state extrapolation.
