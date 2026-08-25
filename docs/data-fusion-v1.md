# Parking Data Fusion V1

This branch resets the forecasting target around **measured physical occupancy**.

## Ground truth

SFMTA's SFpark pilot published hourly block-level occupancy from parking sensors for April 2011 through July 2013. The sensor guide defines total, operational, general-metered-parking (GMP), and commercial occupancy and documents how unknown time is excluded from the occupancy denominator.

## Historical paired features

SFMTA also published smart-meter payment sessions with parking-space/post identifiers plus payment, session-start, and session-end timestamps. Those records can be joined to the sensor data through the historical `street_block` field and provide the transaction-side features.

The result is a clean calibration problem:

`transaction behavior -> measured physical occupancy`

## Current physical-space identity

The current DataSF Parking Meters dataset contains `PARKING_SPACE_ID`, `POST_ID`, and `BLOCKFACE_ID`; SFMTA identifies `PARKING_SPACE_ID` as the physical-space join key. The data-fusion layer does not invent capacity when that field is unavailable.

## Causality

For a target hour T:

- measured occupancy at T is the target;
- occupancy at T-1 is allowed;
- transaction history ending at or before T-1 is allowed;
- duration/session-end is used only as historical training information, never as a future fact for an open session;
- target-hour transaction or policy values are not predictors.

## Bootstrap

The historical files are large and are streamed through `psql`; they are never committed to git.

```bash
python3 scripts/bootstrap_data_fusion.py --download-historical
python3 scripts/train_fused_occupancy.py
```

If the SFMTA landing page changes its link text, download the two official historical CSVs manually and pass them with `--sensor-csv` and `--smart-payments-csv`.

## Why this ends the previous modeling loop

The previous experiments tried to infer a physical occupancy process from a derived payment-based probability. This branch creates the missing calibration target first. Only after this benchmark answers whether transactions add information beyond persistence should we introduce richer current-day signals such as citations, events, weather, traffic, and transit.
